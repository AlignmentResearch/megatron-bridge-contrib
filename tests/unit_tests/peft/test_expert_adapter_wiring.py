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
closed form. These tests pin the SHIPPED code to that arithmetic: the construction predicate,
the forced gather_output=False, the zero-embed offsets computed by the real method, the
placement of the collective completions inside forward(), and gradient flow through the
pad/unpad helpers. Reverting any of those in src/ fails here, which no closed-form model can
do. The collectives themselves are replaced by spies; their distributed behaviour needs a real
process group and was verified manually on 2 GPUs (see the PR).
"""

from unittest.mock import Mock, patch

import pytest
import torch

from megatron.bridge.peft.dora import DoRA
from megatron.bridge.peft.utils import ParallelLinearAdapter, pad_seq_to_mult, unpad_seq_to_mult


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
        out = adapter._embed_expert_row_parallel_shard(shard)
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
        # (is_expert, input_is_parallel, base_linear_is_parallel) -> (predicate, gather_output)
        ((True, True, True), (True, False)),
        ((False, True, True), (False, True)),
        ((True, False, True), (False, False)),
        ((True, True, False), (False, True)),
    ]
    for (is_expert, iip, blip), (want_predicate, want_gather) in cases:
        adapter = _make_adapter(
            mock_row,
            mock_col,
            Mock(),
            Mock(),
            is_expert=is_expert,
            input_is_parallel=iip,
            base_linear_is_parallel=blip,
        )
        assert adapter._expert_row_parallel is want_predicate, (is_expert, iip, blip)
        # linear_out is always the last ColumnParallelLinear constructed.
        assert mock_col.call_args.kwargs["gather_output"] is want_gather, (is_expert, iip, blip)


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
