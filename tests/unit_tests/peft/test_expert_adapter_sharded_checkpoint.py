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

"""Global checkpoint identity for grouped-expert adapters.

The tests simulate a complete EP/ETP/EDP world and validate the resulting global
expert-axis sharding with Megatron's own integrity checker. Tensor shards must be
distinct across EP partitions while unsharded extra state has one main replica.
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
from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedTensor, apply_factories, is_main_replica
from megatron.core.dist_checkpointing.utils import extract_sharded_base
from megatron.core.dist_checkpointing.validation import validate_sharding_integrity
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint

from megatron.bridge.peft.utils import ParallelLinearAdapter, _enable_legacy_shared_expert_adapter_loading


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


def _make_adapter(
    rank: Rank,
    *,
    is_expert: bool = True,
    base_linear_name: str = "experts.linear_fc1",
    gated_linear_unit: bool = True,
) -> ParallelLinearAdapter:
    """Build an adapter whose inner linears emit real Megatron sharding metadata."""
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
    config.num_moe_experts = rank.world.ep_size * 2
    config.gated_linear_unit = gated_linear_unit
    with (
        patch("megatron.bridge.peft.utils.ColumnParallelLinear", side_effect=[linear_in, linear_out]),
        patch("megatron.bridge.peft.utils.RowParallelLinear"),
    ):
        return ParallelLinearAdapter(
            in_features=IN_FEATURES,
            out_features=OUT_FEATURES,
            dim=DIM,
            base_linear_name=base_linear_name,
            is_expert=is_expert,
            model_parallel_config=config,
        )


def _adapter_shardings(
    rank: Rank,
    *,
    is_expert: bool = True,
    base_linear_name: str = "experts.linear_fc1",
    gated_linear_unit: bool = True,
) -> list:
    """Return the adapter shardings contributed by a simulated rank."""
    adapter = _make_adapter(
        rank,
        is_expert=is_expert,
        base_linear_name=base_linear_name,
        gated_linear_unit=gated_linear_unit,
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
    return [_adapter_shardings(rank, is_expert=is_expert) for rank in world.ranks()]


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


def test_extra_state_is_kept_only_on_the_main_expert_rank() -> None:
    """Only EP0/ETP0 contributes unsharded adapter extra state."""
    world = World(etp_size=4, ep_size=1, edp_size=2)
    for rank in world.ranks():
        replica_ids = _extra_state_replica_ids(_adapter_shardings(rank))
        if rank.ep == 0 and rank.etp == 0:
            assert set(replica_ids) == {f"{PREFIX}linear_in._extra_state", f"{PREFIX}linear_out._extra_state"}
        else:
            assert replica_ids == {}


def test_expert_parallel_identity_is_a_global_tensor_axis() -> None:
    """EP-local values occupy distinct global expert offsets rather than replica ids."""
    world = World(etp_size=2, ep_size=2, edp_size=2)
    offsets_by_ep = {}
    for rank in world.ranks():
        if rank.etp != 0 or rank.edp != 0:
            continue
        tensor_offsets = {
            tuple(sharding.global_offset)
            for sharding in _adapter_shardings(rank)
            if isinstance(sharding, ShardedTensor)
        }
        offsets_by_ep[rank.ep] = tensor_offsets
    assert offsets_by_ep[0].isdisjoint(offsets_by_ep[1])


def test_resharding_distinct_expert_slots_fails_closed() -> None:
    """A shared runtime adapter cannot represent unequal expert-slot values."""
    rank = Rank(etp=0, ep=0, edp=0, world=World(etp_size=1, ep_size=2, edp_size=1))
    adapter = _make_adapter(rank)
    with (
        patch.object(parallel_state, "get_expert_model_parallel_rank", return_value=rank.ep),
        patch.object(parallel_state, "get_expert_model_parallel_world_size", return_value=rank.world.ep_size),
    ):
        factory = adapter._apply_expert_axis_factory(
            _base_linear_sharded_state_dict("adapter.", DIM, IN_FEATURES, rank)["adapter.weight"], ()
        )

    with pytest.raises(RuntimeError, match="Cannot merge distinct global expert slots"):
        factory.merge_fn([torch.zeros(DIM, IN_FEATURES), torch.ones(DIM, IN_FEATURES)])


def test_legacy_schema_detection_fails_closed_on_missing_metadata() -> None:
    """A PEFT resume never guesses the schema of an unresolved grouped adapter."""
    rank = Rank(etp=0, ep=0, edp=0, world=World(etp_size=1, ep_size=2, edp_size=1))
    adapter = _make_adapter(rank)
    with (
        patch.object(parallel_state, "get_expert_model_parallel_rank", return_value=rank.ep),
        patch.object(parallel_state, "get_expert_model_parallel_world_size", return_value=rank.world.ep_size),
        patch.object(parallel_state, "get_expert_tensor_parallel_rank", return_value=rank.etp),
        patch.object(parallel_state, "get_expert_data_parallel_rank", return_value=rank.edp),
    ):
        state_dict = {"model": adapter.sharded_state_dict(prefix=PREFIX)}
        with patch("megatron.bridge.peft.utils.dist_checkpointing.load_tensors_metadata", return_value={}):
            with pytest.raises(RuntimeError, match="missing from checkpoint tensor metadata"):
                _enable_legacy_shared_expert_adapter_loading(adapter, state_dict, "/checkpoint")


def test_global_expert_axis_factory_preserves_etp_swiglu_values() -> None:
    """ETP-local gate/up halves retain fused ordering through factory build and merge."""
    rank = Rank(etp=1, ep=0, edp=0, world=World(etp_size=2, ep_size=2, edp_size=1))
    adapter = _make_adapter(rank)
    local_rows = OUT_FEATURES // rank.world.etp_size
    source = torch.cat(
        (
            torch.full((local_rows // 2, DIM), 3.0),
            torch.full((local_rows // 2, DIM), 7.0),
        )
    )
    sharded_tensor = _base_linear_sharded_state_dict("adapter.", local_rows, DIM, rank)["adapter.weight"]
    with (
        patch.object(parallel_state, "get_expert_model_parallel_rank", return_value=rank.ep),
        patch.object(parallel_state, "get_expert_model_parallel_world_size", return_value=rank.world.ep_size),
    ):
        factory = adapter._apply_expert_axis_factory(sharded_tensor, (), split_swiglu=True)

    built = factory.build_fn(factory.key, source, factory.replica_id, None)
    assert len(built) == 4
    for expert_index in range(2):
        gate, up = built[expert_index * 2 : expert_index * 2 + 2]
        torch.testing.assert_close(gate.data, torch.full_like(gate.data, 3.0))
        torch.testing.assert_close(up.data, torch.full_like(up.data, 7.0))
    torch.testing.assert_close(factory.merge_fn([shard.data for shard in built]), source)


def test_non_expert_adapter_keeps_megatron_replica_ids() -> None:
    """The expert branch is the only thing that rewrites replica_id."""
    rank = Rank(etp=2, ep=0, edp=1, world=World(etp_size=4, ep_size=1, edp_size=2))
    replica_ids = _extra_state_replica_ids(_adapter_shardings(rank, is_expert=False))
    assert set(replica_ids.values()) == {(0, rank.etp, rank.dp_rank)}


def test_non_grouped_expert_adapter_preserves_legacy_sharding() -> None:
    """Per-expert adapters retain flattened EP/ETP identity and FC1 splitting."""
    rank = Rank(etp=1, ep=1, edp=0, world=World(etp_size=2, ep_size=2, edp_size=2))
    shardings = _adapter_shardings(
        rank,
        base_linear_name="mlp.experts.local_experts.1.linear_fc1",
        gated_linear_unit=False,
    )
    replica_ids = _extra_state_replica_ids(shardings)
    assert set(replica_ids.values()) == {(0, rank.ep * rank.world.etp_size + rank.etp, rank.dp_rank)}
    linear_out_shards = [
        sharding
        for sharding in shardings
        if isinstance(sharding, ShardedTensor) and sharding.key == f"{PREFIX}linear_out.weight"
    ]
    assert len(linear_out_shards) == 2
