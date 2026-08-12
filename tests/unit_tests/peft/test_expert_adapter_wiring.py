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

"""Regression tests for the expert-adapter wiring in ParallelLinearAdapter itself.

The algebra files (test_expert_row_parallel_adapter_algebra.py,
test_expert_column_parallel_adapter_backward_algebra.py) prove the intended arithmetic in
closed form. These tests pin the SHIPPED code to that arithmetic: the construction predicates
(routed-expert and shared-expert flavors), the forced gather_output=False, the zero-embed
offsets and group selection computed by the real method, the placement of the collective
completions inside forward(), gradient flow through the pad/unpad helpers, and LoRAMerge's
group selection. Reverting any of those in src/ fails here, which no closed-form model can
do. The collectives themselves are replaced by spies; their distributed behaviour needs a real
process group and was verified manually on 2 GPUs (see the PR).
"""

from unittest.mock import Mock, patch

import pytest
import torch

from megatron.bridge.peft.dora import DoRA
from megatron.bridge.peft.lora import LoRAMerge
from megatron.bridge.peft.utils import (
    AdapterAttributes,
    ParallelLinearAdapter,
    pad_seq_to_mult,
    unpad_seq_to_mult,
)


IN, DIM, OUT, ETP = 16, 4, 8, 4
OUT_SHARD = OUT // ETP


class _Config:
    """Minimal ModelParallelConfig stand-in (mirrors test_utils.MockModelParallelConfig)."""

    def __init__(self):
        self.sequence_parallel = False
        self.tensor_model_parallel_size = 1
        self.bf16 = False
        self.fp16 = False
        self.cpu_offloading = False
        self.cpu_offloading_activations = False
        self.expert_model_parallel_size = 1
        self.expert_tensor_parallel_size = ETP
        self.pipeline_model_parallel_size = 1
        self.virtual_pipeline_model_parallel_size = None


def _make_adapter(mock_row, mock_col, linear_in, linear_out, config=None, **kwargs):
    """Construct a real ParallelLinearAdapter around mocked child linears."""
    mock_row.reset_mock(side_effect=True)
    mock_col.reset_mock(side_effect=True)
    if kwargs.get("input_is_parallel", False):
        mock_row.return_value = linear_in
        mock_col.return_value = linear_out
    else:
        mock_col.side_effect = [linear_in, linear_out]
    return ParallelLinearAdapter(
        IN,
        OUT,
        DIM,
        base_linear_name="test",
        activation="identity",
        model_parallel_config=config if config is not None else _Config(),
        **kwargs,
    )


def test_pad_and_unpad_stay_in_the_autograd_graph():
    """Detaching either helper zeroes the adapter's gradients while the forward still moves."""
    x = torch.randn(7, 4, requires_grad=True)
    padded, pad_len = pad_seq_to_mult(x, ETP)
    assert pad_len == 1
    out = unpad_seq_to_mult(padded, pad_len)
    assert out.grad_fn is not None
    out.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))

    # The divisible branch returns x itself and must also keep the graph.
    y = torch.randn(8, 4, requires_grad=True)
    padded, pad_len = pad_seq_to_mult(y, ETP)
    assert pad_len == 0
    assert unpad_seq_to_mult(padded, pad_len) is y


@patch("megatron.bridge.peft.utils.parallel_state")
@patch("megatron.bridge.peft.utils.ColumnParallelLinear")
@patch("megatron.bridge.peft.utils.RowParallelLinear")
def test_embed_zero_embeds_the_local_shard_at_the_rank_offset(mock_row, mock_col, mock_ps):
    """The shipped offset formula, not a transcription of it: rank r owns columns [r*w, (r+1)*w)."""
    mock_ps.get_expert_tensor_parallel_world_size.return_value = ETP
    adapter = _make_adapter(mock_row, mock_col, Mock(), Mock(), is_expert=True, input_is_parallel=True)
    assert adapter._expert_row_parallel

    shards = [torch.randn(5, OUT_SHARD, dtype=torch.float64) for _ in range(ETP)]
    embedded = []
    for rank, shard in enumerate(shards):
        mock_ps.get_expert_tensor_parallel_rank.return_value = rank
        out = adapter._embed_row_parallel_shard(shard)
        assert out.shape == (5, OUT)
        assert torch.equal(out[:, rank * OUT_SHARD : (rank + 1) * OUT_SHARD], shard)
        mask = torch.ones(OUT, dtype=torch.bool)
        mask[rank * OUT_SHARD : (rank + 1) * OUT_SHARD] = False
        assert torch.all(out[:, mask] == 0)
        embedded.append(out)

    # The dispatcher's sum then reconstructs the concatenation exactly once.
    assert torch.equal(sum(embedded), torch.cat(shards, dim=-1))


@patch("megatron.bridge.peft.utils.parallel_state")
@patch("megatron.bridge.peft.utils.ColumnParallelLinear")
@patch("megatron.bridge.peft.utils.RowParallelLinear")
def test_expert_row_parallel_predicate_and_gather_flag(mock_row, mock_col, mock_ps):
    """gather_output must be False exactly when the dispatcher will sum the adapter's output."""
    cases = [
        # (is_expert, input_is_parallel, base_linear_is_parallel, disable_tp_comm)
        #   -> (expert_predicate, shared_predicate, gather_output)
        ((True, True, True, False), (True, False, False)),
        ((False, True, True, False), (False, False, True)),
        ((True, False, True, False), (False, False, False)),
        ((True, True, False, False), (False, False, True)),
        # Shared-expert fc2 under moe_shared_expert_overlap: base comm suppressed, dense TP
        # sums the full-width partials downstream.
        ((False, True, True, True), (False, True, False)),
        # The same suppression on an expert linear stays the EXPERT flavor.
        ((True, True, True, True), (True, False, False)),
    ]
    for (is_expert, iip, blip, dtc), (want_expert, want_shared, want_gather) in cases:
        adapter = _make_adapter(
            mock_row,
            mock_col,
            Mock(),
            Mock(),
            is_expert=is_expert,
            input_is_parallel=iip,
            base_linear_is_parallel=blip,
            disable_tensor_parallel_comm=dtc,
        )
        case = (is_expert, iip, blip, dtc)
        assert adapter._expert_row_parallel is want_expert, case
        assert adapter._shared_expert_row_parallel is want_shared, case
        # linear_out is always the last ColumnParallelLinear constructed.
        assert mock_col.call_args.kwargs["gather_output"] is want_gather, case


@patch("megatron.bridge.peft.utils.parallel_state")
@patch("megatron.bridge.peft.utils.ColumnParallelLinear")
@patch("megatron.bridge.peft.utils.RowParallelLinear")
def test_a2a_with_sequence_parallel_constructs_for_expert_row_parallel(mock_row, mock_col, mock_ps):
    """a2a is inert for experts (forward's SP blocks are `not is_expert`-gated), so this
    combination must construct and take the same zero-embed path as the non-a2a case."""
    config = _Config()
    config.sequence_parallel = True
    adapter = _make_adapter(
        mock_row,
        mock_col,
        Mock(),
        Mock(),
        config=config,
        is_expert=True,
        input_is_parallel=True,
        a2a_experimental=True,
    )
    assert adapter._expert_row_parallel
    assert mock_col.call_args.kwargs["gather_output"] is False


@patch("megatron.bridge.peft.utils.reduce_from_tensor_model_parallel_region")
@patch("megatron.bridge.peft.utils.copy_to_tensor_model_parallel_region")
@patch("megatron.bridge.peft.utils.parallel_state")
@patch("megatron.bridge.peft.utils.ColumnParallelLinear")
@patch("megatron.bridge.peft.utils.RowParallelLinear")
def test_forward_wiring_for_expert_row_parallel(mock_row, mock_col, mock_ps, mock_copy, mock_reduce):
    """forward() must complete A@h (copy_to then reduce_from) BEFORE the activation/linear_out
    and zero-embed AFTER linear_out, with pad/unpad live around it all."""
    etp_group = object()
    mock_ps.get_expert_tensor_parallel_world_size.return_value = ETP
    mock_ps.get_expert_tensor_parallel_rank.return_value = 2
    mock_ps.get_expert_tensor_parallel_group.return_value = etp_group
    # Distinguishable, differentiable stand-ins: swapped or dropped calls change the numbers.
    mock_copy.side_effect = lambda t, g: t + 1.0
    mock_reduce.side_effect = lambda t, g: t * 2.0

    t1 = torch.randn(8, DIM, dtype=torch.float64)
    t2 = torch.randn(8, OUT_SHARD, dtype=torch.float64)
    linear_in, linear_out = Mock(return_value=(t1, None)), Mock(return_value=(t2, None))
    adapter = _make_adapter(mock_row, mock_col, linear_in, linear_out, is_expert=True, input_is_parallel=True)

    out = adapter(torch.randn(7, IN, dtype=torch.float64))  # 7 tokens: pad path live

    # pad: linear_in saw the padded 8 rows.
    assert linear_in.call_args.args[0].shape[0] == 8
    # copy_to feeds reduce_from, both on the ETP group, and their composition feeds linear_out.
    assert mock_copy.call_count == 1 and mock_copy.call_args.args[1] is etp_group
    assert mock_reduce.call_count == 1 and mock_reduce.call_args.args[1] is etp_group
    assert torch.allclose(linear_out.call_args.args[0], (t1 + 1.0) * 2.0)
    # embed at rank 2, scale alpha/dim == 1, unpad back to 7 rows.
    assert out.shape == (7, OUT)
    assert torch.allclose(out[:, 2 * OUT_SHARD : 3 * OUT_SHARD], t2[:7])
    mask = torch.ones(OUT, dtype=torch.bool)
    mask[2 * OUT_SHARD : 3 * OUT_SHARD] = False
    assert torch.all(out[:, mask] == 0)


@patch("megatron.bridge.peft.utils.reduce_from_tensor_model_parallel_region")
@patch("megatron.bridge.peft.utils.copy_to_tensor_model_parallel_region")
@patch("megatron.bridge.peft.utils.parallel_state")
@patch("megatron.bridge.peft.utils.ColumnParallelLinear")
@patch("megatron.bridge.peft.utils.RowParallelLinear")
def test_forward_wiring_for_expert_column_parallel(mock_row, mock_col, mock_ps, mock_copy, mock_reduce):
    """The fc1 case needs only copy_to (backward all-reduce); reduce_from must NOT run, and the
    output is the plain linear_out shard -- no embed."""
    mock_ps.get_expert_tensor_parallel_world_size.return_value = ETP
    mock_ps.get_expert_tensor_parallel_group.return_value = object()
    mock_copy.side_effect = lambda t, g: t + 1.0

    t1 = torch.randn(8, DIM, dtype=torch.float64)
    t2 = torch.randn(8, OUT, dtype=torch.float64)
    linear_in, linear_out = Mock(return_value=(t1, None)), Mock(return_value=(t2, None))
    adapter = _make_adapter(mock_row, mock_col, linear_in, linear_out, is_expert=True, input_is_parallel=False)

    out = adapter(torch.randn(8, IN, dtype=torch.float64))

    assert mock_copy.call_count == 1
    mock_reduce.assert_not_called()
    assert torch.allclose(linear_out.call_args.args[0], t1 + 1.0)
    assert torch.allclose(out, t2)


@patch("megatron.bridge.peft.utils.reduce_from_tensor_model_parallel_region")
@patch("megatron.bridge.peft.utils.copy_to_tensor_model_parallel_region")
@patch("megatron.bridge.peft.utils.parallel_state")
@patch("megatron.bridge.peft.utils.ColumnParallelLinear")
@patch("megatron.bridge.peft.utils.RowParallelLinear")
def test_forward_is_untouched_for_non_expert_and_etp1(mock_row, mock_col, mock_ps, mock_copy, mock_reduce):
    """Strict no-op everywhere else: no collectives for a non-expert adapter, none at ETP=1."""
    t2 = torch.randn(8, OUT, dtype=torch.float64)

    for is_expert, etp in [(False, ETP), (True, 1)]:
        mock_ps.get_expert_tensor_parallel_world_size.return_value = etp
        config = _Config()
        config.expert_tensor_parallel_size = etp
        linear_in = Mock(return_value=(torch.randn(8, DIM, dtype=torch.float64), None))
        linear_out = Mock(return_value=(t2, None))
        adapter = _make_adapter(
            mock_row,
            mock_col,
            linear_in,
            linear_out,
            config=config,
            is_expert=is_expert,
            input_is_parallel=True,
        )
        out = adapter(torch.randn(8, IN, dtype=torch.float64))
        mock_copy.assert_not_called()
        mock_reduce.assert_not_called()
        assert torch.allclose(out, t2)


def test_dora_refuses_expert_linears():
    """DoRA builds its adapter without is_expert, so transforming an expert linear under any
    model parallelism would recreate the exact miscounted-delta defect the LoRA path fixes."""
    dora = DoRA()
    module = torch.nn.Linear(4, 4)
    with pytest.raises(NotImplementedError, match="expert linears"):
        dora.transform(module, name="linear_fc2", prefix="decoder.layers.0.mlp.experts.0")


class _DummyDoRALinear(torch.nn.Module):
    def __init__(self, to_wrap, adapter):
        super().__init__()


@patch("megatron.bridge.peft.dora.DoRALinear", new=_DummyDoRALinear)
@patch("megatron.bridge.peft.dora.ParallelLinearDoRAAdapter")
@patch("megatron.bridge.peft.dora.get_adapter_attributes_from_linear")
def test_dora_refuses_suppressed_comm_parallel_bases(mock_attrs, mock_adapter):
    """Shared-expert fc2 under moe_shared_expert_overlap: DoRA's per-rank magnitude norm does
    not compose with the downstream TP sum, so refuse instead of training silently wrong. A
    replicated base (duplicated TELinear) also reports suppressed comm but has no downstream
    TP reduction, so it must stay allowed."""

    def attrs(dtc, blip):
        return AdapterAttributes(
            input_is_parallel=True,
            in_features=4,
            out_features=4,
            disable_tensor_parallel_comm=dtc,
            disable_sequence_parallel_comm=True,
            base_linear_is_parallel=blip,
        )

    dora = DoRA()
    module = torch.nn.Linear(4, 4)

    mock_attrs.return_value = attrs(dtc=True, blip=True)
    with pytest.raises(NotImplementedError, match="suppressed"):
        dora.transform(module, name="linear_fc2", prefix="decoder.layers.0.mlp.shared_experts")

    mock_attrs.return_value = attrs(dtc=True, blip=False)
    dora.transform(module, name="linear_fc2", prefix="decoder.layers.0.mlp.shared_experts")


@patch("megatron.bridge.peft.utils.parallel_state")
@patch("megatron.bridge.peft.utils.ColumnParallelLinear")
@patch("megatron.bridge.peft.utils.RowParallelLinear")
def test_shared_expert_embed_uses_the_dense_tp_group(mock_row, mock_col, mock_ps):
    """The shared-expert flavor must read the DENSE TP rank/size, not the expert group's:
    SharedExpertMLP.post_forward_comm sums across TP."""
    tp = 2
    tp_shard = OUT // tp
    mock_ps.get_tensor_model_parallel_world_size.return_value = tp
    # Poison the expert getters: reading them would produce wrong offsets.
    mock_ps.get_expert_tensor_parallel_world_size.return_value = 1
    mock_ps.get_expert_tensor_parallel_rank.return_value = 0

    adapter = _make_adapter(
        mock_row,
        mock_col,
        Mock(),
        Mock(),
        is_expert=False,
        input_is_parallel=True,
        disable_tensor_parallel_comm=True,
    )
    assert adapter._shared_expert_row_parallel
    assert mock_col.call_args.kwargs["gather_output"] is False

    shards = [torch.randn(5, tp_shard, dtype=torch.float64) for _ in range(tp)]
    embedded = []
    for rank, shard in enumerate(shards):
        mock_ps.get_tensor_model_parallel_rank.return_value = rank
        out = adapter._embed_row_parallel_shard(shard)
        assert out.shape == (5, OUT)
        assert torch.equal(out[:, rank * tp_shard : (rank + 1) * tp_shard], shard)
        embedded.append(out)
    # post_forward_comm's sum reconstructs the concatenation exactly once.
    assert torch.equal(sum(embedded), torch.cat(shards, dim=-1))


@patch("megatron.bridge.peft.utils.reduce_from_tensor_model_parallel_region")
@patch("megatron.bridge.peft.utils.copy_to_tensor_model_parallel_region")
@patch("megatron.bridge.peft.utils.parallel_state")
@patch("megatron.bridge.peft.utils.ColumnParallelLinear")
@patch("megatron.bridge.peft.utils.RowParallelLinear")
def test_forward_wiring_for_shared_expert_row_parallel(mock_row, mock_col, mock_ps, mock_copy, mock_reduce):
    """The shared-expert flavor needs the embed only: the adapter's linear_in is a normal
    row-parallel linear (is_expert=False) that performs its own reduction, so neither
    completion helper may fire, and no expert padding applies."""
    tp = 2
    tp_shard = OUT // tp
    mock_ps.get_tensor_model_parallel_world_size.return_value = tp
    mock_ps.get_tensor_model_parallel_rank.return_value = 1

    t1 = torch.randn(7, DIM, dtype=torch.float64)
    t2 = torch.randn(7, tp_shard, dtype=torch.float64)
    linear_in, linear_out = Mock(return_value=(t1, None)), Mock(return_value=(t2, None))
    adapter = _make_adapter(
        mock_row,
        mock_col,
        linear_in,
        linear_out,
        is_expert=False,
        input_is_parallel=True,
        disable_tensor_parallel_comm=True,
    )

    out = adapter(torch.randn(7, IN, dtype=torch.float64))  # 7 rows: no expert pad applies

    mock_copy.assert_not_called()
    mock_reduce.assert_not_called()
    assert linear_in.call_args.args[0].shape[0] == 7
    assert out.shape == (7, OUT)
    assert torch.allclose(out[:, 1 * tp_shard : 2 * tp_shard], t2)
    assert torch.all(out[:, :tp_shard] == 0)


@patch("megatron.bridge.peft.lora.dist")
@patch("megatron.bridge.peft.lora.parallel_state")
def test_lora_merge_gathers_over_the_expert_group_for_expert_adapters(mock_ps, mock_dist):
    """LoRAMerge must size and group its gathers by the adapter's own sharding: the expert
    TP group for is_expert adapters, the dense TP group otherwise. With TP != ETP the dense
    settings would misclassify the shapes and merge a wrong (or deadlocked) weight."""
    etp, tp = 2, 4
    dense_group, expert_group = object(), object()
    mock_ps.get_tensor_model_parallel_world_size.return_value = tp
    mock_ps.get_tensor_model_parallel_group.return_value = dense_group
    mock_ps.get_expert_tensor_parallel_world_size.return_value = etp
    mock_ps.get_expert_tensor_parallel_group.return_value = expert_group

    def fake_all_gather(tensor_list, tensor, group=None):
        for i in range(len(tensor_list)):
            tensor_list[i].copy_(tensor)

    mock_dist.all_gather.side_effect = fake_all_gather

    out_features, in_features, dim, alpha = 8, 12, 4, 4
    base = torch.randn(out_features, in_features // etp, dtype=torch.float64)
    linear_out = torch.randn(out_features // etp, dim, dtype=torch.float64)  # B shard
    linear_in = torch.randn(dim, in_features // etp, dtype=torch.float64)  # A shard

    merged = LoRAMerge().merge(base, linear_out, linear_in, alpha, dim, is_expert=True)

    # The row-parallel branch fired over the EXPERT group with ETP chunks.
    group_used = mock_dist.all_gather.call_args.kwargs.get("group", None)
    if group_used is None and len(mock_dist.all_gather.call_args.args) > 2:
        group_used = mock_dist.all_gather.call_args.args[2]
    assert group_used is expert_group
    assert len(mock_dist.all_gather.call_args.args[0]) == etp

    expected = base + alpha / dim * (torch.cat([linear_out, linear_out], dim=0) @ linear_in)
    assert torch.allclose(merged, expected)
