# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Uniqueness of the sharded checkpoint an expert adapter describes, across a whole world.

A `_extra_state` entry becomes a `ShardedObject`, and a `ShardedObject` carries no shard offsets:
the only thing distinguishing one rank's copy from another's is `replica_id`. Megatron requires
exactly one rank per key to be the main replica, so a `replica_id` that collapses several ranks
onto the all-zero value makes `validate_sharding_integrity` raise and takes the training run down
at its first save. Every assertion here is about that global picture, which means the interesting
part cannot be seen from a single rank -- the tests build one adapter per rank of a simulated
world and look at the shardings together.

Only the ranks are simulated. The `ShardedObject`s and `ShardedTensor`s come from megatron's own
`make_sharded_tensors_for_checkpoint`, so the convention the adapter is checked against is
megatron's convention rather than a constant transcribed into this file, and the verdict comes
from megatron's own `validate_sharding_integrity`.
"""

import datetime
import os
from collections import Counter
from dataclasses import dataclass
from unittest.mock import Mock, patch

import pytest
import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.dist_checkpointing.dict_utils import nested_values
from megatron.core.dist_checkpointing.mapping import ShardedObject, apply_factories, is_main_replica
from megatron.core.dist_checkpointing.utils import extract_sharded_base
from megatron.core.dist_checkpointing.validation import validate_sharding_integrity
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint

from megatron.bridge.peft.utils import ParallelLinearAdapter


# The grouped-expert path: `mlp.experts` is the TEGroupedMLP, whose sharded_state_dict prepends a
# second `experts.` before its own submodule names. LoRA on Nemotron-3 hangs adapters here.
PREFIX = "decoder.layers.1.mlp.experts.experts.linear_fc1.adapter."
IN_FEATURES, DIM, OUT_FEATURES = 16, 8, 16


@dataclass(frozen=True)
class World:
    """The parallel sizes an expert adapter's replica_id has to tell apart."""

    etp_size: int
    ep_size: int
    edp_size: int

    def ranks(self) -> list["Rank"]:
        return [
            Rank(etp, ep, edp, self)
            for edp in range(self.edp_size)
            for ep in range(self.ep_size)
            for etp in range(self.etp_size)
        ]


@dataclass(frozen=True)
class Rank:
    """One rank's coordinates within a `World`."""

    etp: int
    ep: int
    edp: int
    world: World

    @property
    def dp_size(self) -> int:
        return self.world.ep_size * self.world.edp_size

    @property
    def dp_rank(self) -> int:
        # Megatron orders ranks tp-cp-ep-dp-pp, so EP is the faster-varying index inside DP.
        return self.edp * self.world.ep_size + self.ep


class _FakeProcessGroup:
    """As much of a process group as megatron's `get_pg_rank`/`get_pg_size` ever ask for."""

    def __init__(self, rank: int, size: int) -> None:
        self._rank, self._size = rank, size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


def _base_linear_sharded_state_dict(prefix: str, rows: int, cols: int, rank: Rank) -> dict:
    """What megatron-core's ColumnParallelLinear emits for `rank`.

    Mirrors ColumnParallelLinear.sharded_state_dict: axis-0 sharding for the weight, no bias, and
    the `_extra_state` that megatron adds for Transformer Engine compatibility. An expert linear
    passes its expert-tensor-parallel group as `tp_group`, which is what puts the ETP rank into the
    `_extra_state` replica_id.
    """
    state_dict = {"weight": torch.zeros(rows, cols), "_extra_state": None}
    return make_sharded_tensors_for_checkpoint(
        state_dict,
        prefix,
        {"weight": 0},
        (),
        tp_group=_FakeProcessGroup(rank.etp, rank.world.etp_size),
        dp_cp_group=_FakeProcessGroup(rank.dp_rank, rank.dp_size),
    )


def _adapter_shardings(rank: Rank, is_expert: bool = True) -> list:
    """Build the adapter as `rank` would and return the shardings it contributes to the save.

    The two inner linears are stubbed so the adapter can be built without a real process group,
    but what they return is real megatron sharded state for this rank. Factories are applied and
    the result flattened exactly as `dist_checkpointing.save` does before it validates.
    """
    linear_in, linear_out = Mock(), Mock()
    linear_in.sharded_state_dict.side_effect = lambda prefix, offsets, metadata: _base_linear_sharded_state_dict(
        prefix, DIM // rank.world.etp_size, IN_FEATURES, rank
    )
    linear_out.sharded_state_dict.side_effect = lambda prefix, offsets, metadata: _base_linear_sharded_state_dict(
        prefix, OUT_FEATURES // rank.world.etp_size, DIM, rank
    )

    config = ModelParallelConfig(
        tensor_model_parallel_size=rank.world.etp_size,
        expert_tensor_parallel_size=rank.world.etp_size,
        expert_model_parallel_size=rank.world.ep_size,
    )
    with (
        patch("megatron.bridge.peft.utils.ColumnParallelLinear", side_effect=[linear_in, linear_out]),
        patch("megatron.bridge.peft.utils.RowParallelLinear"),
    ):
        adapter = ParallelLinearAdapter(
            in_features=IN_FEATURES,
            out_features=OUT_FEATURES,
            dim=DIM,
            base_linear_name="experts.linear_fc1",
            is_expert=is_expert,
            model_parallel_config=config,
        )

    # Every expert coordinate this rank could be asked for, so the assertions are about what the
    # adapter does with them rather than about which of them it happens to read.
    with (
        patch.object(parallel_state, "get_expert_model_parallel_rank", return_value=rank.ep),
        patch.object(parallel_state, "get_expert_model_parallel_world_size", return_value=rank.world.ep_size),
        patch.object(parallel_state, "get_expert_tensor_parallel_rank", return_value=rank.etp),
        patch.object(parallel_state, "get_expert_tensor_parallel_world_size", return_value=rank.world.etp_size),
        patch.object(parallel_state, "get_expert_data_parallel_rank", return_value=rank.edp),
        patch.object(parallel_state, "get_expert_data_parallel_world_size", return_value=rank.world.edp_size),
        patch.object(parallel_state, "get_data_parallel_rank", return_value=rank.dp_rank),
        patch.object(parallel_state, "get_data_parallel_world_size", return_value=rank.dp_size),
    ):
        sharded_state_dict = adapter.sharded_state_dict(prefix=PREFIX)

    apply_factories(sharded_state_dict)
    sharded, _ = extract_sharded_base(sharded_state_dict)
    return list(nested_values(sharded))


def _global_metadata(world: World, is_expert: bool = True) -> list[list]:
    """The per-rank shardings megatron gathers on rank 0 before validating a save."""
    return [_adapter_shardings(rank, is_expert) for rank in world.ranks()]


def _main_replica_object_keys(global_metadata: list[list]) -> list[str]:
    return [
        sharding.unique_key
        for rank_shardings in global_metadata
        for sharding in rank_shardings
        if isinstance(sharding, ShardedObject) and is_main_replica(sharding.replica_id)
    ]


def _extra_state_replica_ids(shardings: list) -> dict[str, tuple]:
    return {s.key: s.replica_id for s in shardings if isinstance(s, ShardedObject)}


EXPECTED_OBJECT_KEYS = {
    f"{PREFIX}linear_in._extra_state/shard_0_1",
    f"{PREFIX}linear_out._extra_state/shard_0_1",
}


@pytest.fixture(scope="module", autouse=True)
def single_rank_process_group():
    """Megatron's `get_pg_rank` short-circuits to 0 unless torch.distributed is initialized.

    Without a live backend every simulated rank would read back as rank 0 and the tests would
    assert against a world that does not exist.
    """
    created = not dist.is_initialized()
    if created:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29511")
        dist.init_process_group(backend="gloo", world_size=1, rank=0, timeout=datetime.timedelta(minutes=5))
    yield
    if created and dist.is_initialized():
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "world",
    [
        # The configuration of the run that this reproduces: TP = ETP = 4, EP = 1, DP = 2.
        pytest.param(World(etp_size=4, ep_size=1, edp_size=2), id="etp4-ep1-edp2"),
        # The same run without data parallelism, which used to take a separate code path.
        pytest.param(World(etp_size=4, ep_size=1, edp_size=1), id="etp4-ep1-edp1"),
        # EP alone still has to be told apart, since the adapter key holds no expert index.
        pytest.param(World(etp_size=1, ep_size=4, edp_size=2), id="etp1-ep4-edp2"),
        pytest.param(World(etp_size=2, ep_size=2, edp_size=2), id="etp2-ep2-edp2"),
        pytest.param(World(etp_size=1, ep_size=1, edp_size=1), id="single-rank"),
    ],
)
def test_every_extra_state_object_has_exactly_one_main_replica(world: World) -> None:
    """One rank per key writes it, and no key goes unwritten."""
    keys = _main_replica_object_keys(_global_metadata(world))
    duplicates = sorted(key for key, count in Counter(keys).items() if count > 1)
    assert not duplicates, f"{len(keys) - len(set(keys))} ranks over-claim: {duplicates}"
    # Uniqueness is trivially satisfiable by writing nothing, so pin the set as well.
    assert set(keys) == EXPECTED_OBJECT_KEYS


@pytest.mark.parametrize(
    "world",
    [
        pytest.param(World(etp_size=4, ep_size=1, edp_size=2), id="etp4-ep1-edp2"),
        pytest.param(World(etp_size=2, ep_size=2, edp_size=2), id="etp2-ep2-edp2"),
    ],
)
def test_megatron_accepts_the_gathered_sharding(world: World) -> None:
    """The check that actually runs during a save, over weights and extra state alike."""
    validate_sharding_integrity(_global_metadata(world))


def test_expert_tensor_parallel_rank_survives_in_replica_id() -> None:
    """Slot 1 keeps meaning the expert-TP rank, as TEGroupedLinear leaves it.

    This is the property whose loss produced the duplicate keys, asserted directly so a future
    change that reintroduces it fails here and not only in the world-level test above.
    """
    world = World(etp_size=4, ep_size=1, edp_size=2)
    for rank in world.ranks():
        replica_ids = _extra_state_replica_ids(_adapter_shardings(rank))
        assert set(replica_ids) == {f"{PREFIX}linear_in._extra_state", f"{PREFIX}linear_out._extra_state"}
        for key, replica_id in replica_ids.items():
            assert replica_id[1] == rank.etp, f"{key} on {rank} lost the expert-TP rank: {replica_id}"


def test_expert_parallel_identity_survives_the_optimizer_truncation() -> None:
    """Whatever separates EP ranks has to sit in the first two slots of replica_id.

    The distributed optimizer builds its shardings from these, keeping `replica_id[:2]` and
    overwriting the last slot with its instance id. An EP identity parked in the last slot is
    therefore erased, and the optimizer state collides on keys where the model state did not.
    """
    world = World(etp_size=2, ep_size=2, edp_size=2)
    heads: dict[str, dict[tuple[int, int], tuple]] = {}
    for rank in world.ranks():
        for key, replica_id in _extra_state_replica_ids(_adapter_shardings(rank)).items():
            heads.setdefault(key, {})[(rank.ep, rank.etp)] = replica_id[:2]
    for key, by_rank in heads.items():
        assert len(set(by_rank.values())) == len(by_rank), f"{key} collapses EP/ETP ranks: {by_rank}"


def test_non_expert_adapter_keeps_megatron_replica_ids() -> None:
    """The expert branch is the only thing that rewrites replica_id."""
    rank = Rank(etp=2, ep=0, edp=1, world=World(etp_size=4, ep_size=1, edp_size=2))
    replica_ids = _extra_state_replica_ids(_adapter_shardings(rank, is_expert=False))
    assert set(replica_ids.values()) == {(0, rank.etp, rank.dp_rank)}
