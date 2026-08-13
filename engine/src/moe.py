"""Correctness-first routed MoE primitives.

The provider boundary deliberately says nothing about cache or disk policy. M1
uses resident tensors; M2 can materialize only selected experts behind the same
interface.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Protocol

import torch
import torch.nn.functional as F
import nvtx


@dataclass(frozen=True)
class MaterializedExperts:
    global_ids: torch.Tensor
    gate_proj: torch.Tensor
    up_proj: torch.Tensor
    down_proj: torch.Tensor

    def __post_init__(self):
        num_slots = self.global_ids.numel()
        if self.global_ids.ndim != 1:
            raise ValueError("global_ids must be one-dimensional")
        if self.gate_proj.ndim != 3 or self.up_proj.ndim != 3:
            raise ValueError("gate_proj and up_proj must have shape [E, N, H]")
        if self.down_proj.ndim != 3:
            raise ValueError("down_proj must have shape [E, H, N]")
        if not (
            self.gate_proj.shape[0]
            == self.up_proj.shape[0]
            == self.down_proj.shape[0]
            == num_slots
        ):
            raise ValueError("expert tensors and global_ids disagree on slot count")
        if self.gate_proj.shape != self.up_proj.shape:
            raise ValueError("gate_proj and up_proj shapes must match")
        if self.down_proj.shape[1] != self.gate_proj.shape[2]:
            raise ValueError("down_proj output size must match the hidden size")
        if self.down_proj.shape[2] != self.gate_proj.shape[1]:
            raise ValueError("down_proj input size must match the intermediate size")
        devices = {
            self.global_ids.device,
            self.gate_proj.device,
            self.up_proj.device,
            self.down_proj.device,
        }
        if len(devices) != 1:
            raise ValueError("expert IDs and tensors must be on the same device")
        dtypes = {self.gate_proj.dtype, self.up_proj.dtype, self.down_proj.dtype}
        if len(dtypes) != 1:
            raise ValueError("all expert projections must have the same dtype")


class ExpertProvider(Protocol):
    def materialize(self, selected_expert_ids):
        """Return weights covering every requested global expert ID."""
        ...


class ExpertProviderFactory(Protocol):
    def create(self, layer_id, device, dtype):
        """Create the provider that owns expert storage for one routed layer."""
        ...


class ResidentExpertProvider:
    """Expose an all-resident expert bank without copying selected weights."""

    def __init__(self, gate_proj, up_proj, down_proj):
        num_experts = gate_proj.shape[0]
        global_ids = torch.arange(
            num_experts, dtype=torch.long, device=gate_proj.device
        )
        self.experts = MaterializedExperts(
            global_ids=global_ids,
            gate_proj=gate_proj,
            up_proj=up_proj,
            down_proj=down_proj,
        )

    def materialize(self, selected_expert_ids):
        if selected_expert_ids.numel():
            min_id = int(selected_expert_ids.min().item())
            max_id = int(selected_expert_ids.max().item())
            if min_id < 0 or max_id >= self.experts.global_ids.numel():
                raise IndexError(
                    f"selected expert IDs [{min_id}, {max_id}] are outside the resident bank"
                )
        return self.experts


class TracingExpertProvider:
    """Optional JSONL routing trace without changing storage semantics."""

    def __init__(self, provider, layer_id, trace_path):
        self.provider = provider
        self.layer_id = int(layer_id)
        self.trace_path = Path(trace_path)
        self.call_id = 0

    def materialize(self, selected_expert_ids):
        selected = selected_expert_ids.detach().to("cpu")
        record = {
            "event": "EXPERT_MATERIALIZE",
            "layer": self.layer_id,
            "call": self.call_id,
            "selected_ids": selected.tolist(),
            "unique_ids": sorted(int(value) for value in torch.unique(selected).tolist()),
        }
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        self.call_id += 1
        return self.provider.materialize(selected_expert_ids)


@dataclass(frozen=True)
class RoutingResult:
    router_logits: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor


def route_qwen3_moe(
    hidden_states, router_weight, top_k, norm_topk_prob, router_fp32=False
):
    hidden_size = hidden_states.shape[-1]
    if router_weight.ndim != 2 or router_weight.shape[1] != hidden_size:
        raise ValueError("router_weight must have shape [num_experts, hidden_size]")
    if not 0 < top_k <= router_weight.shape[0]:
        raise ValueError("top_k must be between 1 and num_experts")

    flat_hidden = hidden_states.reshape(-1, hidden_size)
    if router_fp32:
        router_logits = F.linear(flat_hidden.float(), router_weight.float())
    else:
        router_logits = F.linear(flat_hidden, router_weight)
    routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(routing_weights, top_k, dim=-1)
    if norm_topk_prob:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    if not router_fp32:
        topk_weights = topk_weights.to(hidden_states.dtype)
    return RoutingResult(router_logits, topk_ids, topk_weights)


def qwen3_moe_forward(
    hidden_states,
    router_weight,
    expert_provider: ExpertProvider,
    top_k,
    norm_topk_prob,
    expert_clamp=False,
    router_fp32=False,
):
    """Match Qwen3MoeSparseMoeBlock for a fixed input tensor."""
    original_shape = hidden_states.shape
    hidden_size = original_shape[-1]
    flat_hidden = hidden_states.reshape(-1, hidden_size)
    routing = route_qwen3_moe(
        hidden_states, router_weight, top_k, norm_topk_prob,
        router_fp32=router_fp32,
    )
    experts = expert_provider.materialize(routing.topk_ids)
    accumulation_dtype = torch.float32 if router_fp32 else flat_hidden.dtype
    final_hidden = torch.zeros_like(flat_hidden, dtype=accumulation_dtype)

    for global_id_tensor in torch.unique(routing.topk_ids):
        global_id = int(global_id_tensor.item())
        slots = torch.where(experts.global_ids == global_id)[0]
        if slots.numel() != 1:
            raise RuntimeError(
                f"provider returned {slots.numel()} slots for expert {global_id}"
            )
        slot = int(slots[0].item())
        token_indices, topk_positions = torch.where(
            routing.topk_ids == global_id
        )
        with nvtx.annotate(f"COMPUTE_EXPERT expert={global_id}", color="green"):
            current = flat_hidden.index_select(0, token_indices)
            gate_linear = F.linear(current, experts.gate_proj[slot])
            up = F.linear(current, experts.up_proj[slot])
            if expert_clamp:
                gate_linear = torch.clamp(gate_linear, max=7.0)
                up = torch.clamp(up, min=-7.0, max=7.0)
            gate = F.silu(gate_linear)
            expert_output = F.linear(gate * up, experts.down_proj[slot])
            expert_output = expert_output.to(accumulation_dtype) * routing.topk_weights[
                token_indices, topk_positions, None
            ].to(accumulation_dtype)
            final_hidden.index_add_(0, token_indices, expert_output)

    return final_hidden.to(hidden_states.dtype).reshape(original_shape), routing


def rms_norm_reference(hidden_states, weight, eps):
    """Pure PyTorch RMSNorm used by the correctness-first MoE path."""
    input_dtype = hidden_states.dtype
    normalized = hidden_states.float()
    variance = normalized.square().mean(dim=-1, keepdim=True)
    normalized = normalized * torch.rsqrt(variance + eps)
    return (normalized.to(input_dtype) * weight.to(input_dtype))


def qwen3_moe_layer_forward(
    hidden_states,
    post_attention_norm,
    router_weight,
    expert_provider: ExpertProvider,
    top_k,
    norm_topk_prob,
    rms_norm_eps,
    expert_clamp=False,
    router_fp32=False,
):
    """Apply Qwen3-MoE post-attention norm, routed FFN, and residual."""
    residual = hidden_states
    normalized = rms_norm_reference(
        hidden_states, post_attention_norm, rms_norm_eps
    )
    moe_output, routing = qwen3_moe_forward(
        normalized,
        router_weight,
        expert_provider,
        top_k,
        norm_topk_prob,
        expert_clamp=expert_clamp,
        router_fp32=router_fp32,
    )
    return residual + moe_output, routing


def qwen3_moe_layer_forward_chunked(
    hidden_states,
    post_attention_norm,
    router_weight,
    expert_provider: ExpertProvider,
    top_k,
    norm_topk_prob,
    rms_norm_eps,
    token_chunk_size,
    expert_clamp=False,
    router_fp32=False,
):
    """Bound prefill temporaries while preserving token-independent semantics."""
    if token_chunk_size <= 0:
        raise ValueError("token_chunk_size must be positive")
    original_shape = hidden_states.shape
    flat_hidden = hidden_states.reshape(-1, original_shape[-1])
    if flat_hidden.shape[0] <= token_chunk_size:
        return qwen3_moe_layer_forward(
            hidden_states,
            post_attention_norm,
            router_weight,
            expert_provider,
            top_k,
            norm_topk_prob,
            rms_norm_eps,
            expert_clamp=expert_clamp,
            router_fp32=router_fp32,
        )

    outputs = []
    routings = []
    for start in range(0, flat_hidden.shape[0], token_chunk_size):
        output, routing = qwen3_moe_layer_forward(
            flat_hidden[start : start + token_chunk_size],
            post_attention_norm,
            router_weight,
            expert_provider,
            top_k,
            norm_topk_prob,
            rms_norm_eps,
            expert_clamp=expert_clamp,
            router_fp32=router_fp32,
        )
        outputs.append(output)
        routings.append(routing)
    routing = RoutingResult(
        router_logits=torch.cat([item.router_logits for item in routings]),
        topk_ids=torch.cat([item.topk_ids for item in routings]),
        topk_weights=torch.cat([item.topk_weights for item in routings]),
    )
    return torch.cat(outputs).reshape(original_shape), routing
