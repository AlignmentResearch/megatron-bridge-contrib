# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Four-rank DCP round trips for EP-local grouped-expert LoRA.

Run with:
uv run python -m torch.distributed.run --standalone --nproc-per-node=4 -m pytest -q \
    tests/unit_tests/peft/test_expert_adapter_dcp_distributed.py
"""

import os
import shutil
import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from megatron.core import dist_checkpointing, parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedTensorFactory
from megatron.core.dist_checkpointing.strategies.fully_parallel import (
    FullyParallelLoadStrategyWrapper,
    FullyParallelSaveStrategyWrapper,
)
from megatron.core.dist_checkpointing.strategies.torch import (
    TorchDistLoadShardedStrategy,
    TorchDistSaveShardedStrategy,
)
from megatron.core.model_parallel_config import ModelParallelConfig

from megatron.bridge.peft.utils import (
    ParallelLinearAdapter,
    _disable_legacy_shared_expert_adapter_loading,
    _enable_legacy_shared_expert_adapter_loading,
)


EP_SIZE = 4
NUM_EXPERTS = 8
PREFIX = "decoder.layers.0.mlp.experts.linear_fc2.adapter."


def _make_adapter() -> ParallelLinearAdapter:
    config = ModelParallelConfig(
        tensor_model_parallel_size=1,
        expert_model_parallel_size=EP_SIZE,
        expert_tensor_parallel_size=1,
        params_dtype=torch.float32,
        gradient_accumulation_fusion=False,
        use_cpu_initialization=True,
    )
    config.num_moe_experts = NUM_EXPERTS
    config.gated_linear_unit = True
    adapter = ParallelLinearAdapter(
        in_features=8,
        out_features=8,
        dim=4,
        base_linear_name="decoder.layers.0.mlp.experts.linear_fc2",
        activation="identity",
        input_is_parallel=True,
        is_expert=True,
        model_parallel_config=config,
    )
    if torch.cuda.is_available():
        adapter = adapter.to(torch.device("cuda", int(os.environ["LOCAL_RANK"])))
    return adapter


def _metadata() -> dict:
    return {"dp_cp_group": parallel_state.get_data_parallel_group(with_context_parallel=True)}


def _sharded_state(adapter: ParallelLinearAdapter) -> dict:
    return {"model": adapter.sharded_state_dict(prefix=PREFIX, metadata=_metadata())}


def _set_weights(adapter: ParallelLinearAdapter, linear_in: float, linear_out: float) -> None:
    with torch.no_grad():
        adapter.linear_in.weight.fill_(linear_in)
        adapter.linear_out.weight.fill_(linear_out)


def _load_adapter_state(adapter: ParallelLinearAdapter, loaded: dict) -> None:
    local_state = {key.removeprefix(PREFIX): value for key, value in loaded["model"].items()}
    adapter.load_state_dict(local_state, strict=False)


def _fully_parallel_save(state_dict: dict, checkpoint_dir: Path) -> None:
    strategy = FullyParallelSaveStrategyWrapper(
        TorchDistSaveShardedStrategy(),
        parallel_state.get_data_parallel_group(with_context_parallel=True),
        True,
    )
    dist_checkpointing.save(state_dict, str(checkpoint_dir), strategy)


def _fully_parallel_load(state_dict: dict, checkpoint_dir: Path) -> dict:
    strategy = FullyParallelLoadStrategyWrapper(
        TorchDistLoadShardedStrategy(),
        parallel_state.get_data_parallel_group(with_context_parallel=True),
    )
    return dist_checkpointing.load(state_dict, str(checkpoint_dir), strategy)


@pytest.fixture(scope="module", autouse=True)
def ep_topology():
    if int(os.environ.get("WORLD_SIZE", "1")) != EP_SIZE:
        pytest.skip("requires a four-rank torchrun launch")

    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        backend = "nccl"
    else:
        backend = "gloo"
    dist.init_process_group(backend)
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=EP_SIZE,
        expert_tensor_parallel_size=1,
    )
    yield
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


@pytest.fixture
def shared_checkpoint_dir() -> Path:
    paths = [tempfile.mkdtemp(prefix="expert-lora-dcp-") if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(paths, src=0)
    path = Path(paths[0])
    yield path
    dist.barrier()
    if dist.get_rank() == 0:
        shutil.rmtree(path)
    dist.barrier()


def test_distinct_ep_values_roundtrip_through_global_expert_axis(shared_checkpoint_dir: Path) -> None:
    ep_rank = parallel_state.get_expert_model_parallel_rank()
    source = _make_adapter()
    _set_weights(source, linear_in=ep_rank + 1.0, linear_out=ep_rank + 11.0)

    state_dict = _sharded_state(source)
    assert isinstance(state_dict["model"][f"{PREFIX}linear_in.weight"], ShardedTensorFactory)
    _fully_parallel_save(state_dict, shared_checkpoint_dir)

    fresh = _make_adapter()
    _set_weights(fresh, linear_in=-1.0, linear_out=-1.0)
    loaded = _fully_parallel_load(_sharded_state(fresh), shared_checkpoint_dir)
    _load_adapter_state(fresh, loaded)

    torch.testing.assert_close(fresh.linear_in.weight, torch.full_like(fresh.linear_in.weight, ep_rank + 1.0))
    torch.testing.assert_close(fresh.linear_out.weight, torch.full_like(fresh.linear_out.weight, ep_rank + 11.0))


def test_legacy_2d_loads_identically_then_future_save_uses_global_axis(shared_checkpoint_dir: Path) -> None:
    external_checkpoint_dir = os.environ.get("LEGACY_EXPERT_ADAPTER_CHECKPOINT_DIR")
    legacy_checkpoint_dir = Path(external_checkpoint_dir) if external_checkpoint_dir else shared_checkpoint_dir
    if external_checkpoint_dir is None:
        source = _make_adapter()
        _set_weights(source, linear_in=3.0, linear_out=7.0)
        source._use_legacy_shared_expert_adapter_checkpoint = True
        _fully_parallel_save(_sharded_state(source), legacy_checkpoint_dir)

    fresh = _make_adapter()
    current_state = _sharded_state(fresh)
    legacy_adapters = _enable_legacy_shared_expert_adapter_loading(fresh, current_state, legacy_checkpoint_dir)
    assert legacy_adapters == (fresh,)
    try:
        loaded = _fully_parallel_load(_sharded_state(fresh), legacy_checkpoint_dir)
    finally:
        _disable_legacy_shared_expert_adapter_loading(legacy_adapters)
    _load_adapter_state(fresh, loaded)

    torch.testing.assert_close(fresh.linear_in.weight, torch.full_like(fresh.linear_in.weight, 3.0))
    torch.testing.assert_close(fresh.linear_out.weight, torch.full_like(fresh.linear_out.weight, 7.0))
    assert isinstance(_sharded_state(fresh)["model"][f"{PREFIX}linear_in.weight"], ShardedTensorFactory)

    migrated_paths = [tempfile.mkdtemp(prefix="expert-lora-migrated-dcp-") if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(migrated_paths, src=0)
    migrated_checkpoint_dir = Path(migrated_paths[0])
    try:
        _fully_parallel_save(_sharded_state(fresh), migrated_checkpoint_dir)

        migrated = _make_adapter()
        _set_weights(migrated, linear_in=-1.0, linear_out=-1.0)
        migrated_state = _fully_parallel_load(_sharded_state(migrated), migrated_checkpoint_dir)
        _load_adapter_state(migrated, migrated_state)
        torch.testing.assert_close(migrated.linear_in.weight, torch.full_like(migrated.linear_in.weight, 3.0))
        torch.testing.assert_close(migrated.linear_out.weight, torch.full_like(migrated.linear_out.weight, 7.0))
    finally:
        dist.barrier()
        if dist.get_rank() == 0:
            shutil.rmtree(migrated_checkpoint_dir)
        dist.barrier()
