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

import math
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Callable, Dict, Final, Optional, Tuple

import packaging
import torch
import torch.nn as nn
from megatron.core import ModelParallelConfig, dist_checkpointing, parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedStateDict, ShardedTensor, ShardedTensorFactory
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from megatron.core.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
    reduce_from_tensor_model_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.mlp import apply_swiglu_sharded_factory
from megatron.core.transformer.moe.router import TopKRouter

from megatron.bridge.utils.import_utils import safe_import_from


TEColumnParallelLinear, HAVE_TE_COL_LINEAR = safe_import_from(
    "megatron.core.extensions.transformer_engine", "TEColumnParallelLinear"
)
TELayerNormColumnParallelLinear, HAVE_TE_LN_COL_LINEAR = safe_import_from(
    "megatron.core.extensions.transformer_engine",
    "TELayerNormColumnParallelLinear",
)
TEColumnParallelGroupedLinear, HAVE_TE_COL_GRP_LINEAR = safe_import_from(
    "megatron.core.extensions.transformer_engine", "TEColumnParallelGroupedLinear"
)
TERowParallelLinear, HAVE_TE_ROW_LINEAR = safe_import_from(
    "megatron.core.extensions.transformer_engine", "TERowParallelLinear"
)
TERowParallelGroupedLinear, HAVE_TE_ROW_GRP_LINEAR = safe_import_from(
    "megatron.core.extensions.transformer_engine", "TERowParallelGroupedLinear"
)
TELinear, HAVE_TE_LINEAR = safe_import_from("megatron.core.extensions.transformer_engine", "TELinear")
HAVE_TE = all(
    (
        HAVE_TE_COL_LINEAR,
        HAVE_TE_LN_COL_LINEAR,
        HAVE_TE_ROW_LINEAR,
        HAVE_TE_LINEAR,
        HAVE_TE_COL_GRP_LINEAR,
        HAVE_TE_ROW_GRP_LINEAR,
    )
)

MixedFusedLayerNorm, HAVE_APEX = safe_import_from("apex.normalization.fused_layer_norm", "MixedFusedLayerNorm")

TECL = (TEColumnParallelLinear, TELayerNormColumnParallelLinear, TEColumnParallelGroupedLinear)
TERL = (TERowParallelLinear, TERowParallelGroupedLinear)

LEGACY_EXPERT_ADAPTER_CHECKPOINT_SCHEMA: Final = "legacy_shared_2d"
EXPERT_ADAPTER_CHECKPOINT_SCHEMA: Final = "global_expert_axis_v1"


@dataclass(frozen=True)
class AdapterAttributes:
    """Container for base linear adapter attributes."""

    input_is_parallel: bool
    in_features: int
    out_features: int
    disable_tensor_parallel_comm: bool
    disable_sequence_parallel_comm: bool
    base_linear_is_parallel: bool


def get_adapter_attributes_from_linear(m: nn.Module, is_expert: bool = False) -> AdapterAttributes:
    """Returns attributes from the base layer as an AdapterAttributes dataclass.

    input_is_parallel, in_features, out_features, disable_tensor_parallel_comm,
    disable_sequence_parallel_comm, base_linear_is_parallel

    This function analyzes a linear module and extracts key attributes needed for adapter configuration,
    particularly for PEFT adapters in distributed training scenarios.

    Args:
        m: The linear module to analyze (should have a config attribute).

    Returns:
        AdapterAttributes containing:
            - input_is_parallel: Whether the input is already parallelized
            - in_features: Input feature dimension
            - out_features: Output feature dimension
            - disable_tensor_parallel_comm: Whether to disable tensor parallel communication
            - disable_sequence_parallel_comm: Whether to disable sequence parallel communication
            - base_linear_is_parallel: Whether the base linear layer uses parallelization

    Raises:
        NotImplementedError: If the layer type is not recognized for LoRA adaptation.
    """
    disable_sequence_parallel_comm = not m.config.sequence_parallel
    base_linear_is_parallel = True

    # In some modules (notably MoE shared_experts when moe_shared_expert_overlap is enabled),
    # Megatron disables TP-related communications on the base linear layer by
    # setting `parallel_mode=None` (TE) or `explicit_expert_comm=True` (legacy).
    # https://github.com/NVIDIA/Megatron-LM/blob/5b1ef0703184299fbf71f6131bf2f9a5331e7238/megatron/core/transformer/moe/shared_experts.py#L95-L104
    # The weights are still TP-sharded though, so we must keep using the real TP size
    disable_tensor_parallel_comm = getattr(m, "parallel_mode", "") is None or getattr(m, "explicit_expert_comm", False)
    if disable_tensor_parallel_comm:
        disable_sequence_parallel_comm = True

    if is_expert:
        tp_size = parallel_state.get_expert_tensor_parallel_world_size()
    else:
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
    if isinstance(m, TopKRouter):
        input_is_parallel = False
        in_features = m.weight.shape[1]
        out_features = m.weight.shape[0]
        base_linear_is_parallel = False
        disable_sequence_parallel_comm = True
    elif HAVE_TE and any(isinstance(m, te_column_parallel) for te_column_parallel in TECL):
        input_is_parallel = False
        # m.in_features and m.out_features are divided by tp_size already,
        # but in_features and out_features passed to ParallelLinearAdapter are not.
        in_features = m.in_features
        out_features = m.out_features * tp_size

        if isinstance(m, TELayerNormColumnParallelLinear):
            # LoRA is applied after layernorm, so layernorm output must be returned
            m.return_layernorm_output = True
            # perf optimization for LoRA + SP
            if hasattr(m, "ub_overlap_ag"):
                ub_overlap_ag = m.ub_overlap_ag
            elif hasattr(m, "ub_overlap_ag_fprop"):
                ub_overlap_ag = m.ub_overlap_ag_fprop
            else:
                ub_overlap_ag = False
            if hasattr(m, "config") and m.config.sequence_parallel and not ub_overlap_ag:
                m.return_layernorm_output_gathered = True
                te_version = packaging.version.Version(version("transformer-engine"))
                if te_version >= packaging.version.Version("1.5.0dev") and (
                    not getattr(m.config, "tp_comm_overlap", False)
                    or getattr(m.config, "tp_comm_overlap_disable_qkv", False)
                ):
                    # TE 1.5 introduces the option `return_layernorm_output_gathered`, so the all gather
                    # in the forward method is not needed, so disable sp communications
                    # unless TP communication overlap is used
                    disable_sequence_parallel_comm = True
    elif HAVE_TE and any(isinstance(m, te_row_parallel) for te_row_parallel in TERL):
        input_is_parallel = True
        in_features = m.in_features * tp_size
        out_features = m.out_features
    elif HAVE_TE and isinstance(m, TELinear):  # parallel_mode="duplicated"
        input_is_parallel = False
        in_features = m.in_features
        out_features = m.out_features
        base_linear_is_parallel = False
    elif isinstance(m, ColumnParallelLinear):
        input_is_parallel = False
        in_features = m.input_size
        out_features = m.output_size
    elif isinstance(m, RowParallelLinear):
        input_is_parallel = True
        in_features = m.input_size
        out_features = m.output_size
    else:
        raise NotImplementedError(f"Layer type is unrecognized for LoRA: {type(m)}")

    return AdapterAttributes(
        input_is_parallel=input_is_parallel,
        in_features=in_features,
        out_features=out_features,
        disable_tensor_parallel_comm=disable_tensor_parallel_comm,
        disable_sequence_parallel_comm=disable_sequence_parallel_comm,
        base_linear_is_parallel=base_linear_is_parallel,
    )


def is_expert_linear(fqn: str) -> bool:
    """Return whether the current base module is an expert linear module.

    This function checks if a fully qualified name (FQN) corresponds to an expert linear
    module in a Mixture of Experts (MoE) architecture.

    Args:
        fqn: Fully qualified name of the module.

    Returns:
        True if the module is an expert linear module, False otherwise.

    Example:
        >>> is_expert_linear("model.layers.0.mlp.experts.0.linear_fc1")
        True
        >>> is_expert_linear("model.layers.0.mlp.linear_fc1")
        False
    """
    return re.match(r".*mlp\..*experts.*\.linear_fc[1-2]$", fqn) is not None and not ".shared_experts." in fqn


def _iter_sharded_tensor_factories(state_dict: object) -> Iterator[ShardedTensorFactory]:
    """Yield tensor factories from a nested sharded state dict."""

    if isinstance(state_dict, ShardedTensorFactory):
        yield state_dict
    elif isinstance(state_dict, Mapping):
        for value in state_dict.values():
            yield from _iter_sharded_tensor_factories(value)
    elif isinstance(state_dict, (list, tuple)):
        for value in state_dict:
            yield from _iter_sharded_tensor_factories(value)


def _checkpoint_tensor_shape(checkpoint_metadata: Mapping[str, ShardedTensor], key: str) -> tuple[int, ...] | None:
    """Return a checkpoint tensor's global shape, tolerating model section prefixes."""

    for candidate in (key, f"model.{key}"):
        metadata = checkpoint_metadata.get(candidate)
        if metadata is not None:
            return tuple(metadata.global_shape)
    return None


def _shared_expert_adapter_factory_info(
    factory: ShardedTensorFactory,
) -> tuple[str, tuple[int, ...]] | None:
    """Return the adapter key and new-schema shape represented by a factory."""

    for suffix in (".linear_in.weight", ".linear_out.weight"):
        if not factory.key.endswith(suffix):
            continue
        built = factory.build()
        shards = built if isinstance(built, list) else [built]
        if not shards or not isinstance(shards[0], ShardedTensor):
            return None
        expected_shape = tuple(shards[0].global_shape)
        if len(expected_shape) == factory.data.ndim + 1:
            return factory.key[: -len(suffix)], expected_shape
    return None


def _matching_shared_expert_adapters(
    adapters_by_name: list[tuple[str, "ParallelLinearAdapter"]], adapter_key: str
) -> list["ParallelLinearAdapter"]:
    """Return model adapter candidates for a checkpoint adapter key."""

    exact_matches = [adapter for name, adapter in adapters_by_name if name == adapter_key]
    if exact_matches:
        return exact_matches

    adapter_base_key = adapter_key.removesuffix(".adapter")
    matches = []
    for module_name, module in adapters_by_name:
        module_base_key = module_name.removesuffix(".adapter")
        base_linear_name = module.base_linear_name
        if (
            adapter_key.endswith(module_name)
            or module_name.endswith(adapter_key)
            or adapter_base_key.endswith(module_base_key)
            or module_base_key.endswith(adapter_base_key)
            or adapter_base_key.endswith(base_linear_name)
            or base_linear_name.endswith(adapter_base_key)
        ):
            matches.append(module)
    return matches


def _enable_legacy_shared_expert_adapter_loading(
    megatron_model: list[nn.Module] | nn.Module,
    sharded_state_dict: ShardedStateDict,
    checkpoint_path: str | Path,
) -> tuple["ParallelLinearAdapter", ...]:
    """Select legacy 2D loading for grouped-expert adapters in an old checkpoint.

    The old schema stored one shared 2D adapter tensor and encoded EP ranks as
    replicas. The current schema adds a leading global-expert axis. This function
    inspects checkpoint metadata and temporarily marks only adapters whose saved
    shape is the old 2D form. Call :func:`_disable_legacy_shared_expert_adapter_loading`
    after loading so subsequent saves use the current schema.

    Args:
        megatron_model: Model module or pipeline model chunks containing adapters.
        sharded_state_dict: Current-schema state dict for the checkpoint load.
        checkpoint_path: Distributed checkpoint directory whose metadata is inspected.

    Returns:
        Adapters temporarily marked to emit legacy 2D sharding.
    """

    checkpoint_metadata = dist_checkpointing.load_tensors_metadata(str(checkpoint_path))
    models = megatron_model if isinstance(megatron_model, list) else [megatron_model]
    adapters_by_name = [
        (name.removeprefix("module."), module)
        for model in models
        for name, module in model.named_modules()
        if isinstance(module, ParallelLinearAdapter) and module._uses_grouped_expert_sharding()
    ]

    legacy_adapters: set[ParallelLinearAdapter] = set()
    for factory in _iter_sharded_tensor_factories(sharded_state_dict):
        factory_info = _shared_expert_adapter_factory_info(factory)
        if factory_info is None:
            continue
        adapter_key, expected_shape = factory_info
        checkpoint_shape = _checkpoint_tensor_shape(checkpoint_metadata, factory.key)
        if checkpoint_shape is None:
            raise RuntimeError(
                f"Grouped-expert adapter {factory.key} is missing from checkpoint tensor metadata; "
                "refusing to guess its checkpoint schema"
            )
        if checkpoint_shape == expected_shape:
            continue
        if checkpoint_shape != expected_shape[1:]:
            raise RuntimeError(
                f"Unsupported grouped-expert adapter checkpoint shape for {factory.key}: "
                f"checkpoint={checkpoint_shape}, expected current={expected_shape} or legacy={expected_shape[1:]}"
            )

        matches = _matching_shared_expert_adapters(adapters_by_name, adapter_key)
        if len(matches) != 1:
            raise RuntimeError(
                f"Legacy grouped-expert adapter key {adapter_key!r} matched {len(matches)} model adapters; "
                "refusing an ambiguous checkpoint migration"
            )
        legacy_adapters.add(matches[0])

    for adapter in legacy_adapters:
        adapter._use_legacy_shared_expert_adapter_checkpoint = True
    return tuple(legacy_adapters)


def _disable_legacy_shared_expert_adapter_loading(adapters: tuple["ParallelLinearAdapter", ...]) -> None:
    """Clear the temporary legacy-schema flag after loading.

    Args:
        adapters: Adapters returned by :func:`_enable_legacy_shared_expert_adapter_loading`.
    """

    for adapter in adapters:
        adapter._use_legacy_shared_expert_adapter_checkpoint = False


def wildcard_match(pattern: str, key: Optional[str]) -> Optional[bool]:
    """Return whether the pattern (target module to add LoRA) matches the key (model weight name).

    This function performs wildcard matching using '*' as a placeholder for any substring.

    Args:
        pattern: Pattern string with wildcards (*) to match against.
        key: Key string to test against the pattern.

    Returns:
        True if the pattern matches the key, False if it doesn't, None if key is None.

    Example:
        >>> wildcard_match("*.layers.0.*.linear_qkv", "decoder.layers.0.self_attention.linear_qkv")
        True
        >>> wildcard_match("*.layers.0.*.linear_qkv", "decoder.layers.1.self_attention.linear_qkv")
        False
    """
    if key is None:
        return None
    regex_pattern = re.compile("^" + pattern.replace("*", "(.*)") + "$")
    match = regex_pattern.match(key)
    return match is not None


def init_method_normal(sigma: float) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create an initialization method based on normal distribution N(0, sigma).

    Args:
        sigma: Standard deviation for the normal distribution.

    Returns:
        Initialization function that applies normal distribution to a tensor.
    """

    def init_(tensor: torch.Tensor) -> torch.Tensor:
        return nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_


def init_method_kaiming_uniform(val: float) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create an initialization method based on Kaiming uniform distribution.

    Args:
        val: The 'a' parameter for Kaiming uniform initialization.

    Returns:
        Initialization function that applies Kaiming uniform distribution to a tensor.
    """

    def init_(tensor: torch.Tensor) -> torch.Tensor:
        return nn.init.kaiming_uniform_(tensor, a=val)

    return init_


def init_method_const(val: float) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create an initialization method that sets all values to a constant.

    Args:
        val: Constant value to initialize the tensor with.

    Returns:
        Initialization function that sets tensor to constant value.
    """

    def init_(tensor: torch.Tensor) -> torch.Tensor:
        return nn.init.constant_(tensor, val)

    return init_


def pad_seq_to_mult(x: torch.Tensor, mult: int) -> Tuple[torch.Tensor, int]:
    """Pad sequence length to be a multiple of mult.

    This function pads the first dimension of the tensor to ensure it's divisible by mult.
    Used primarily for MoE (Mixture of Experts) operations that require specific sequence lengths.

    Args:
        x: Input tensor to pad.
        mult: Multiple that the sequence length should be divisible by.

    Returns:
        A tuple containing:
            - Padded tensor
            - Number of padding elements added
    """
    if x.shape[0] % mult == 0:
        return x, 0
    pad_len = mult - (x.shape[0] % mult)
    # Both this pad and the matching unpad must stay inside the autograd graph: they sit on
    # the adapter's activation path, and detaching either one zeroes dL/dA and dL/dB while
    # the delta still perturbs the forward.
    x = nn.functional.pad(x, (0, 0, 0, pad_len))
    return x, pad_len


def unpad_seq_to_mult(x: torch.Tensor, pad_len: int) -> torch.Tensor:
    """Remove sequence padding that was added by pad_seq_to_mult.

    Args:
        x: Padded tensor to unpad.
        pad_len: Number of padding elements to remove from the end.

    Returns:
        Unpadded tensor with pad_len elements removed from the first dimension.
    """
    if pad_len <= 0:
        return x
    return x[:-pad_len, :]


class _All2AllHp2Sp(torch.autograd.Function):
    """All-2-All from Hidden Parallel to Sequence Parallel.

    This is a temporary workaround for distributed communication patterns and can be updated in the future.
    It performs all-to-all communication to transform from hidden parallel to sequence parallel layout.

    TODO: Move the functionality to MCore
    """

    @staticmethod
    def forward(ctx, input_: torch.Tensor) -> torch.Tensor:
        """Forward pass: All-to-All from Hidden Parallel to Sequence Parallel.

        Args:
            ctx: Autograd context (unused but required by Function interface).
            input_: Input tensor in hidden parallel layout.

        Returns:
            Output tensor in sequence parallel layout.
        """
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        group = parallel_state.get_tensor_model_parallel_group()
        send_list = list(input_.chunk(world_size, dim=0))
        send_list = [tensor.contiguous() for tensor in send_list]
        receive_list = [torch.empty_like(send_list[0]) for _ in range(world_size)]
        torch.distributed.all_to_all(receive_list, send_list, group=group)
        x = torch.cat(receive_list, dim=-1)

        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """Backward pass: All-to-All from Sequence Parallel to Hidden Parallel.

        Args:
            ctx: Autograd context (unused but required by Function interface).
            grad_output: Gradient tensor in sequence parallel layout.

        Returns:
            Gradient tensor in hidden parallel layout.
        """
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        group = parallel_state.get_tensor_model_parallel_group()
        send_list = list(grad_output.chunk(world_size, dim=-1))
        send_list = [tensor.contiguous() for tensor in send_list]
        receive_list = [torch.empty_like(send_list[0]) for _ in range(world_size)]
        torch.distributed.all_to_all(receive_list, send_list, group=group)
        x = torch.cat(receive_list, dim=0)

        return x


def all2all_hp2sp(input_: torch.Tensor) -> torch.Tensor:
    """Perform All-to-All communication from Hidden Parallel to Sequence Parallel.

    Args:
        input_: Input tensor in hidden parallel layout.

    Returns:
        Output tensor in sequence parallel layout.
    """
    return _All2AllHp2Sp.apply(input_)


class ParallelLinearAdapter(nn.Module):
    """Parallel Linear Adapter for Parameter-Efficient Fine-Tuning (PEFT) in distributed settings.

    This adapter implements a low-rank adaptation pattern using two linear layers with configurable
    parallelization strategies. It supports both tensor and sequence parallelism patterns used in
    large language model training.

    The adapter follows the pattern: input -> linear_in -> activation -> linear_out -> scaling
    where linear_in and linear_out are parallelized according to the base layer configuration.

    Args:
        in_features: Input feature dimension.
        out_features: Output feature dimension.
        dim: Adapter bottleneck dimension (rank).
        base_linear_name: Name of the base linear layer being adapted.
        activation: Activation function name (default: 'swish').
        column_init_method: Initialization method for column parallel layer (default: 'xavier').
        row_init_method: Initialization method for row parallel layer (default: 'zero').
        input_is_parallel: Whether input is already parallelized (default: False).
        dropout: Dropout probability (default: 0.0).
        model_parallel_config: Configuration for model parallelism (default: None).
        alpha: Scaling factor for adapter output (default: None, uses dim).
        dropout_position: Where to apply dropout ('pre' or 'post', default: 'pre').
        a2a_experimental: Whether to use experimental all-to-all communication (default: False).
        is_expert: Whether this adapter is for expert layers in MoE (default: False).
        disable_sequence_parallel_comm: Whether to disable sequence parallel communication (default: True).
        base_linear_is_parallel: Whether the base linear layer uses parallelization (default: True).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dim: int,
        base_linear_name: str,
        activation: str = "swish",
        column_init_method: str = "xavier",
        row_init_method: str = "zero",
        input_is_parallel: bool = False,
        dropout: float = 0.0,
        model_parallel_config: Optional[ModelParallelConfig] = None,
        alpha: Optional[float] = None,
        dropout_position: str = "pre",
        a2a_experimental: bool = False,
        is_expert: bool = False,
        disable_tensor_parallel_comm: bool = False,
        disable_sequence_parallel_comm: bool = True,
        base_linear_is_parallel: bool = True,
        **kwargs,
    ) -> None:
        """Initialize the ParallelLinearAdapter.

        Args:
            in_features: Input feature dimension.
            out_features: Output feature dimension.
            dim: Adapter bottleneck dimension.
            base_linear_name: Name of the base linear layer.
            activation: Activation function name.
            column_init_method: Initialization for column parallel layers.
            row_init_method: Initialization for row parallel layers.
            input_is_parallel: Whether input is already parallelized.
            dropout: Dropout probability.
            model_parallel_config: Model parallelism configuration.
            alpha: Scaling factor (uses dim if None).
            dropout_position: When to apply dropout.
            a2a_experimental: Use experimental all-to-all communication.
            is_expert: Whether for expert layers in MoE.
            disable_tensor_parallel_comm: Disable tensor parallel communication.
            disable_sequence_parallel_comm: Disable sequence parallel communication.
            dropout_recompute: Use recomputation for dropout.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.base_linear_name = base_linear_name
        self.activation = self._get_activation_fn(activation)
        self.dim = dim
        self.alpha = alpha if alpha is not None else self.dim
        self.input_is_parallel = input_is_parallel
        self.dropout_position = dropout_position
        self.use_a2a = a2a_experimental
        self.is_expert = is_expert
        self._use_legacy_shared_expert_adapter_checkpoint = False

        # megatron_gpt_peft_models will provide this arg, but deprecated ones do not.
        # in case this arg is not provided, use the dummy default config.
        if model_parallel_config is None:
            model_parallel_config = ModelParallelConfig()
        _sequence_parallel = model_parallel_config.sequence_parallel
        model_parallel_config.sequence_parallel = False  # SP is irrelevant for the lora linear layer
        self.config = model_parallel_config

        # Ensure adapter parameters are initialized when creating adapter layers.
        # In some flows (e.g., after import), perform_initialization may be False to skip heavy init.
        if hasattr(model_parallel_config, "perform_initialization"):
            model_parallel_config.perform_initialization = True

        if input_is_parallel:
            self.linear_in = RowParallelLinear(
                in_features,
                dim,
                config=model_parallel_config,
                input_is_parallel=True,
                skip_bias_add=True,
                bias=False,
                init_method=self._get_init_fn(column_init_method),
                is_expert=is_expert,
            )
        else:
            self.linear_in = ColumnParallelLinear(
                in_features,
                dim,
                config=model_parallel_config,
                bias=False,
                gather_output=True,
                init_method=self._get_init_fn(column_init_method),
                disable_grad_reduce=_sequence_parallel,
                is_expert=is_expert,
            )

        # (@adithyare) we use this option to mirror the behavior
        # a column parallel layer with two low-rank column parallel layers
        # if the original column parallel layer uses gather_output=False,
        # then we will use the self.liner_out layer defined below.
        lin_out_gather_output = True if input_is_parallel else False
        if (
            self.use_a2a
            and input_is_parallel
            and _sequence_parallel
            or (disable_tensor_parallel_comm and not input_is_parallel)
        ):
            lin_out_gather_output = False

        if not base_linear_is_parallel:
            lin_out_gather_output = True

        # An expert linear whose input is already sharded (experts.linear_fc2) emits a
        # full-width partial that the MoE token dispatcher sums across the expert-tensor-
        # parallel group. The adapter must contribute in the same currency: keep linear_out's
        # local shard and zero-embed it into full width in forward(), so each output element
        # has exactly one non-zero contributor and the dispatcher's sum reconstructs B@z once.
        # Gathering here instead would hand every rank the same full-width delta, and the sum
        # would count it once per rank. A replicated base (base_linear_is_parallel=False)
        # holds the whole weight on every rank, so there is no shard to embed and gathering
        # remains right. `use_a2a` needs no special handling: every sequence-parallel block in
        # forward() is gated on `not self.is_expert`.
        self._expert_row_parallel = bool(is_expert and input_is_parallel and base_linear_is_parallel)
        # Shared-expert fc2 under moe_shared_expert_overlap is the same disease over the DENSE
        # TP group: the base's own collectives are suppressed (that is what
        # disable_tensor_parallel_comm reports) and SharedExpertMLP.post_forward_comm sums the
        # full-width partials across TP, so a gathered delta would be counted TP times.
        self._shared_expert_row_parallel = bool(
            not is_expert and disable_tensor_parallel_comm and input_is_parallel and base_linear_is_parallel
        )
        if self._expert_row_parallel or self._shared_expert_row_parallel:
            lin_out_gather_output = False

        self.linear_out = ColumnParallelLinear(
            dim,
            out_features,
            config=model_parallel_config,
            bias=False,
            gather_output=lin_out_gather_output,
            init_method=self._get_init_fn(row_init_method),
            is_expert=is_expert,
        )

        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

        # cast all parameters when using amp O2 training
        if model_parallel_config.bf16:
            self.bfloat16()
        elif model_parallel_config.fp16:
            self.half()

        # revert config change in case it is read elsewhere
        model_parallel_config.sequence_parallel = _sequence_parallel
        self.disable_sequence_parallel_comm = disable_sequence_parallel_comm
        if not _sequence_parallel:
            self.disable_sequence_parallel_comm = True

        if not base_linear_is_parallel:
            self.disable_sequence_parallel_comm = True

    def _get_activation_fn(self, activation: str) -> nn.Module:
        """Get activation function by name.

        Args:
            activation: Name of the activation function.

        Returns:
            PyTorch activation module.

        Note:
            Defaults to Identity if activation name is not recognized.
        """
        activation_map = {
            "identity": nn.Identity(),
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "swish": nn.SiLU(),
            "silu": nn.SiLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
        }
        return activation_map.get(activation, nn.Identity())

    def _get_init_fn(self, init_method: str) -> Callable[[torch.Tensor], torch.Tensor]:
        """Get initialization function by method name.

        Args:
            init_method: Name of the initialization method.

        Returns:
            Initialization function.

        Raises:
            NotImplementedError: If init_method is not supported.
        """
        if init_method == "xavier":
            init_fn = nn.init.xavier_normal_
        elif init_method == "normal":
            init_fn = init_method_normal(0.2)
        elif init_method == "kaiming":
            init_fn = init_method_kaiming_uniform(math.sqrt(5))
        elif init_method == "zero":
            init_fn = init_method_const(0.0)
        else:
            raise NotImplementedError("out_init_method should be zero, normal, kaiming or xavier")
        return init_fn

    def _reduce_expert_low_rank_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Complete `A @ h` across the expert-tensor-parallel group, in both directions.

        At the pinned megatron-core, `explicit_expert_comm` -- true for an `is_expert` linear
        when the ETP group has more than one rank or expert model parallelism is on --
        suppresses an expert linear's own collectives, because for the BASE layers the MoE
        token dispatcher owns that communication. An adapter is not routed through the
        dispatcher the same way, so two different completions are needed:

        - `input_is_parallel` (experts.linear_fc2): `linear_in` is row-parallel and its output
          all-reduce is suppressed, so `A_r @ h_r` is a per-rank partial that is never summed.
          `copy_to` is forward-identity with a backward all-reduce; `reduce_from` is a forward
          all-reduce with backward identity. Composing them gives all-reduce on BOTH passes,
          the adjoint pair for a sum whose result every rank consumes differently.
        - otherwise (experts.linear_fc1): `linear_in` is column-parallel and its all-gather is
          not suppressed, so the forward is already right -- but the gather's adjoint is a
          split, and `explicit_expert_comm` also forces `allreduce_dgrad=False`, which is what
          a non-expert column-parallel layer relies on to sum `dL/dz` across ranks. `copy_to`'s
          backward all-reduce restores exactly that sum.

        A bare `torch.distributed.all_reduce` would not work here: it is invisible to autograd.
        """
        if not self.is_expert:
            return x
        etp_size = parallel_state.get_expert_tensor_parallel_world_size()
        if etp_size <= 1:
            return x
        etp_group = parallel_state.get_expert_tensor_parallel_group()
        if self.input_is_parallel:
            return reduce_from_tensor_model_parallel_region(
                copy_to_tensor_model_parallel_region(x, etp_group), etp_group
            )
        return copy_to_tensor_model_parallel_region(x, etp_group)

    def _embed_row_parallel_shard(self, x: torch.Tensor) -> torch.Tensor:
        """Place this rank's hidden shard into a full-width buffer of zeros.

        The base emits a full-width partial and a downstream sum reduces those across a group
        -- the MoE token dispatcher over ETP for routed experts, and
        SharedExpertMLP.post_forward_comm over the dense TP group for shared-expert fc2 under
        moe_shared_expert_overlap -- so the adapter has to contribute in the same currency.
        Zero-padding rather than gathering means each output element has exactly one non-zero
        contributor, so the sum that follows reproduces `B @ z` once, with no `1/size` factor
        to derive. Padding is differentiable and its adjoint is the matching slice, so
        `dL/dB_r` needs no compensation either.
        """
        if self._expert_row_parallel:
            size = parallel_state.get_expert_tensor_parallel_world_size()
            rank = parallel_state.get_expert_tensor_parallel_rank()
        elif self._shared_expert_row_parallel:
            size = parallel_state.get_tensor_model_parallel_world_size()
            rank = parallel_state.get_tensor_model_parallel_rank()
        else:
            return x
        if size <= 1:
            return x
        shard_width = x.shape[-1]
        left = rank * shard_width
        right = (size - 1) * shard_width - left
        return nn.functional.pad(x, (left, right))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the parallel linear adapter.

        Performs the adaptation computation with proper handling of parallel communication
        patterns, dropout, and expert routing for MoE scenarios.

        Args:
            x: Input tensor.

        Returns:
            Adapted output tensor with scaling applied.
        """
        if self.dropout_position == "pre":
            x = self.dropout(x)

        pad_len = 0
        if self.is_expert:
            x, pad_len = pad_seq_to_mult(x, self.config.expert_tensor_parallel_size)

        if not self.disable_sequence_parallel_comm and not self.input_is_parallel and not self.is_expert:
            # for attention_qkv and linear_fc1
            # layernorm before lora is impacted by sequence parallel,
            # hence seq dim need to be gathered right before lora linear layers
            # this function also handles the backward pass correctly
            x = gather_from_sequence_parallel_region(x)

        if self.config.cpu_offloading and self.config.cpu_offloading_activations:
            x.activation_offloading = True
        x, _ = self.linear_in(x)  # (@adithyare) ColumnLinear returns output and bias, we are ignoring the bias term.

        x = self._reduce_expert_low_rank_activation(x)

        x = self.activation(x)

        if self.config.cpu_offloading and self.config.cpu_offloading_activations:
            x.activation_offloading = True
        x, _ = self.linear_out(x)

        x = self._embed_row_parallel_shard(x)

        if not self.disable_sequence_parallel_comm and self.input_is_parallel and not self.is_expert:
            # for attention_dense and linear_fc2
            # layernorm after lora is impacted by sequence parallel,
            # hence seq dim need to be scattered right after lora linear layers
            # this function also handles the backward pass correctly
            if self.use_a2a:
                # all2all hidden_size / TP to seq_len / TP
                x = all2all_hp2sp(x)
            else:
                x = scatter_to_sequence_parallel_region(x)

        # Add dropout if available
        if self.dropout_position == "post":
            x = self.dropout(x)

        x = x * (self.alpha / self.dim)

        if pad_len > 0:
            # Remove MoE padding.
            x = unpad_seq_to_mult(x, pad_len)

        return x

    def _uses_grouped_expert_sharding(self) -> bool:
        """Return whether this adapter spans a rank's grouped local experts."""

        return self.is_expert and ".local_experts." not in self.base_linear_name

    def local_experts_per_rank(self) -> int:
        """Return the number of global expert slots owned by this EP rank."""

        ep_size = parallel_state.get_expert_model_parallel_world_size()
        num_global_experts = getattr(self.config, "num_moe_experts", None)
        if num_global_experts is None:
            raise ValueError(
                f"num_moe_experts is required to checkpoint grouped expert adapter {self.base_linear_name}"
            )
        if int(num_global_experts) % ep_size != 0:
            raise ValueError(
                f"num_moe_experts={num_global_experts} must be divisible by expert_model_parallel_size={ep_size}"
            )
        return int(num_global_experts) // ep_size

    def _expert_axis_info(self, sharded_offsets: Tuple) -> tuple[int, int, int]:
        """Return the global expert-axis sharding metadata for this rank."""

        ep_rank = parallel_state.get_expert_model_parallel_rank()
        local_experts = self.local_experts_per_rank()
        num_global_experts = int(self.config.num_moe_experts)
        first_expert_slot = ep_rank * local_experts
        if first_expert_slot >= num_global_experts:
            raise ValueError(
                f"Invalid expert adapter sharding for {self.base_linear_name}: ep_rank={ep_rank}, "
                f"local_experts_per_rank={local_experts}, num_moe_experts={num_global_experts}"
            )
        return len(sharded_offsets), first_expert_slot, num_global_experts

    def _keep_expert_extra_state(self) -> bool:
        """Return whether this rank contributes unsharded adapter extra state."""

        return (
            parallel_state.get_expert_tensor_parallel_rank() == 0
            and parallel_state.get_expert_model_parallel_rank() == 0
        )

    def _set_expert_replica_ids(self, *state_dicts: ShardedStateDict) -> None:
        """Mark expert adapter replicas only across expert data-parallel ranks."""

        edp_rank = parallel_state.get_expert_data_parallel_rank()
        for state_dict in state_dicts:
            for value in state_dict.values():
                if not hasattr(value, "replica_id"):
                    continue
                replica_id = value.replica_id
                if isinstance(replica_id, int):
                    replica_id = (0, 0, replica_id)
                if len(replica_id) != 3:
                    raise ValueError(
                        f"Expected replica_id for {self.base_linear_name} in (PP, TP, DP) format, got {replica_id}"
                    )
                dp_replica_id = 0 if getattr(value, "is_data_parallel_fully_shard", False) else edp_rank
                value.replica_id = (*replica_id[:2], dp_replica_id)

    def _set_legacy_expert_replica_ids(self, *state_dicts: ShardedStateDict) -> None:
        """Reproduce the old EP-as-replica identity while loading a 2D checkpoint."""

        ep_rank = parallel_state.get_expert_model_parallel_rank()
        etp_size = parallel_state.get_expert_tensor_parallel_world_size()
        for state_dict in state_dicts:
            for value in state_dict.values():
                if not hasattr(value, "replica_id"):
                    continue
                replica_id = value.replica_id
                if isinstance(replica_id, int) or len(replica_id) != 3:
                    raise ValueError(
                        f"Expected legacy replica_id for {self.base_linear_name} in (PP, TP, DP) format, "
                        f"got {replica_id}"
                    )
                value.replica_id = (replica_id[0], ep_rank * etp_size + replica_id[1], replica_id[2])

    def _apply_expert_axis_factory(
        self,
        sharded_tensor: ShardedTensor,
        sharded_offsets: Tuple,
        *,
        split_swiglu: bool = False,
    ) -> ShardedTensorFactory:
        """Map one EP-local adapter tensor to the global expert slots it serves."""

        expert_axis, first_expert_slot, num_global_experts = self._expert_axis_info(sharded_offsets)
        local_experts = self.local_experts_per_rank()
        base_prepend_axis_num = len(sharded_offsets)
        output_prepend_axis_num = base_prepend_axis_num + 1
        swiglu_shard_axis = 0

        preserved_rank_offsets = []
        for axis, local_axis_shape in enumerate(sharded_tensor.local_shape):
            base_global_axis = axis + base_prepend_axis_num
            output_global_axis = base_global_axis + 1
            axis_fragments = sharded_tensor.axis_fragmentations[base_global_axis]
            if axis_fragments <= 1:
                continue
            global_offset = sharded_tensor.global_offset[base_global_axis]
            if global_offset % local_axis_shape != 0:
                raise ValueError(
                    f"Cannot preserve non-integral sharding for {sharded_tensor.key}: "
                    f"offset={global_offset}, local_axis_shape={local_axis_shape}"
                )
            preserved_rank_offsets.append((output_global_axis, global_offset // local_axis_shape, axis_fragments))

        base_swiglu_global_axis = swiglu_shard_axis + base_prepend_axis_num
        output_swiglu_global_axis = swiglu_shard_axis + output_prepend_axis_num
        swiglu_axis_frag = None
        swiglu_rank_offset = None
        if split_swiglu:
            local_axis_size = sharded_tensor.local_shape[swiglu_shard_axis]
            global_offset = sharded_tensor.global_offset[base_swiglu_global_axis]
            if global_offset % local_axis_size != 0:
                raise ValueError(
                    f"Cannot split SwiGLU tensor {sharded_tensor.key}: "
                    f"offset={global_offset}, local_axis_shape={local_axis_size}"
                )
            swiglu_rank_offset = global_offset // local_axis_size
            swiglu_axis_frag = sharded_tensor.axis_fragmentations[base_swiglu_global_axis]
            preserved_rank_offsets = [
                rank_offset for rank_offset in preserved_rank_offsets if rank_offset[0] != output_swiglu_global_axis
            ]

        @torch.no_grad()
        def build_fn(key: str, tensor: torch.Tensor, replica_id, flattened_range):
            if flattened_range is not None:
                raise ValueError(f"Flattened grouped-expert adapter tensors are unsupported for {key}")
            shards = []
            swiglu_parts = torch.chunk(tensor, 2, dim=swiglu_shard_axis) if split_swiglu else ()
            for expert_index in range(local_experts):
                expert_offset = (expert_axis, first_expert_slot + expert_index, num_global_experts)
                if not split_swiglu:
                    shards.append(
                        ShardedTensor.from_rank_offsets(
                            key,
                            tensor,
                            *sharded_offsets,
                            *preserved_rank_offsets,
                            expert_offset,
                            replica_id=replica_id,
                            prepend_axis_num=output_prepend_axis_num,
                        )
                    )
                    continue

                offset_w = (output_swiglu_global_axis, swiglu_rank_offset, swiglu_axis_frag * 2)
                offset_v = (
                    output_swiglu_global_axis,
                    swiglu_rank_offset + swiglu_axis_frag,
                    swiglu_axis_frag * 2,
                )
                for tensor_part, swiglu_offset in zip(swiglu_parts, (offset_w, offset_v)):
                    shards.append(
                        ShardedTensor.from_rank_offsets(
                            key,
                            tensor_part,
                            *sharded_offsets,
                            *preserved_rank_offsets,
                            expert_offset,
                            swiglu_offset,
                            replica_id=replica_id,
                            prepend_axis_num=output_prepend_axis_num,
                        )
                    )
            return shards

        def merge_fn(loaded_shards):
            shards = loaded_shards if isinstance(loaded_shards, list) else [loaded_shards]
            if split_swiglu:
                if len(shards) % 2 != 0:
                    raise ValueError(f"Expected paired SwiGLU shards for {sharded_tensor.key}")
                shards = [
                    torch.cat(shards[index : index + 2], dim=swiglu_shard_axis) for index in range(0, len(shards), 2)
                ]
            reference = shards[0]
            if any(not torch.equal(shard, reference) for shard in shards[1:]):
                raise RuntimeError(
                    f"Cannot merge distinct global expert slots into shared adapter {sharded_tensor.key}; "
                    "load with the checkpoint's EP degree; one rank-local adapter cannot represent unequal slots"
                )
            return reference

        return ShardedTensorFactory(
            sharded_tensor.key,
            sharded_tensor.data,
            build_fn,
            merge_fn,
            sharded_tensor.replica_id,
            flattened_range=sharded_tensor.flattened_range,
        )

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: Tuple = (),
        metadata: Optional[Dict] = None,
        mamba_dim_info: Optional[Dict] = None,
    ) -> ShardedStateDict:
        """Create sharded state dictionary for distributed checkpointing.

        Special treatment is given to the linear_fc1 adapter since tensor parallelism is
        sharded separately for the two logical matrices (gate and up) in SwiGLU.

        Args:
            prefix: Prefix for parameter names.
            sharded_offsets: Offsets for sharded parameters.
            metadata: Additional metadata for sharding.

        Returns:
            Sharded state dictionary for distributed checkpointing.
        """
        sharded_state_dict = {}
        use_expert_axis = (
            self._uses_grouped_expert_sharding() and not self._use_legacy_shared_expert_adapter_checkpoint
        )
        is_linear_fc1 = "linear_fc1" in self.base_linear_name
        split_swiglu = is_linear_fc1 and getattr(self.config, "gated_linear_unit", False)
        linear_in_sd = self.linear_in.sharded_state_dict(f"{prefix}linear_in.", sharded_offsets, metadata)
        linear_out_sd = self.linear_out.sharded_state_dict(f"{prefix}linear_out.", sharded_offsets, metadata)

        if use_expert_axis:
            if not self._keep_expert_extra_state():
                for state_dict in (linear_in_sd, linear_out_sd):
                    for key in list(state_dict):
                        if "_extra_state" in key:
                            del state_dict[key]
            for key, value in list(linear_in_sd.items()):
                if isinstance(value, ShardedTensor):
                    linear_in_sd[key] = self._apply_expert_axis_factory(value, sharded_offsets)
            for key, value in list(linear_out_sd.items()):
                if isinstance(value, ShardedTensor):
                    linear_out_sd[key] = self._apply_expert_axis_factory(
                        value, sharded_offsets, split_swiglu=split_swiglu
                    )
        elif self.is_expert:
            self._set_legacy_expert_replica_ids(linear_in_sd, linear_out_sd)

        if is_linear_fc1 and not use_expert_axis:
            for k, v in linear_out_sd.items():
                if k in (f"{prefix}linear_out.weight", f"{prefix}linear_out.bias"):
                    linear_out_sd[k] = apply_swiglu_sharded_factory(v, sharded_offsets)

        # Special handling for Mamba in_proj layer which needs to be split into 5 tensors
        if mamba_dim_info is not None:
            from megatron.core.ssm.mamba_mixer import _split_tensor_factory

            # Split linear_out.weight into 5 parts: z, x, B, C, dt
            # The in_proj output dimension is: d_inner * 2 + 2 * ngroups * d_state + nheads
            # After TP sharding: d_inner_local_tp * 2 + 2 * ngroups_local_tp * d_state + nheads_local_tp
            for k, v in linear_out_sd.items():
                if k == f"{prefix}linear_out.weight" and isinstance(v, ShardedTensor):
                    in_proj_dim_local = (
                        mamba_dim_info["d_inner_local_tp"] * 2
                        + 2 * mamba_dim_info["ngroups_local_tp"] * mamba_dim_info["d_state"]
                        + mamba_dim_info["nheads_local_tp"]
                    )
                    # Verify the dimension matches
                    if v.data.size(0) == in_proj_dim_local:
                        linear_out_sd[k] = _split_tensor_factory(
                            v,
                            [
                                mamba_dim_info["d_inner_local_tp"],  # z
                                mamba_dim_info["d_inner_local_tp"],  # x
                                mamba_dim_info["ngroups_local_tp"] * mamba_dim_info["d_state"],  # B
                                mamba_dim_info["ngroups_local_tp"] * mamba_dim_info["d_state"],  # C
                                mamba_dim_info["nheads_local_tp"],  # dt
                            ],
                            ["z", "x", "B", "C", "dt"],
                            0,  # split along dimension 0
                        )

        if use_expert_axis:
            self._set_expert_replica_ids(linear_in_sd, linear_out_sd)

        sharded_state_dict.update(linear_in_sd)
        sharded_state_dict.update(linear_out_sd)
        return sharded_state_dict
