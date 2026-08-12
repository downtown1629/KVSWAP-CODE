"""Minimal model-family helpers for the first Qwen3-MoE integration slice."""

from dataclasses import dataclass
from enum import Enum


class FFNKind(str, Enum):
    DENSE = "dense"
    ROUTED_MOE = "routed_moe"


@dataclass(frozen=True)
class Qwen3MoeLayerSpec:
    layer_id: int
    num_experts: int
    top_k: int
    intermediate_size: int
    norm_topk_prob: bool


def is_qwen3_family(model_type):
    return model_type in ("qwen3", "qwen3_moe")


def get_ffn_kind(config, layer_id):
    """Return the FFN kind using the Qwen3-MoE layer scheduling rule."""
    if not 0 <= layer_id < config.num_hidden_layers:
        raise IndexError(
            f"layer_id={layer_id} is outside [0, {config.num_hidden_layers})"
        )

    if config.model_type != "qwen3_moe":
        return FFNKind.DENSE

    num_experts = int(config.num_experts)
    sparse_step = int(config.decoder_sparse_step)
    top_k = int(config.num_experts_per_tok)
    if num_experts <= 0:
        raise ValueError("num_experts must be positive for qwen3_moe")
    if sparse_step <= 0:
        raise ValueError("decoder_sparse_step must be positive")
    if not 0 < top_k <= num_experts:
        raise ValueError(
            f"num_experts_per_tok={top_k} must be in [1, {num_experts}]"
        )

    mlp_only_layers = set(config.mlp_only_layers or [])
    is_sparse = (
        layer_id not in mlp_only_layers
        and (layer_id + 1) % sparse_step == 0
    )
    return FFNKind.ROUTED_MOE if is_sparse else FFNKind.DENSE


def get_qwen3_moe_layer_spec(config, layer_id):
    if get_ffn_kind(config, layer_id) != FFNKind.ROUTED_MOE:
        return None
    return Qwen3MoeLayerSpec(
        layer_id=layer_id,
        num_experts=int(config.num_experts),
        top_k=int(config.num_experts_per_tok),
        intermediate_size=int(config.moe_intermediate_size),
        norm_topk_prob=bool(config.norm_topk_prob),
    )
