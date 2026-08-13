"""Small architecture contracts shared by routed-MoE engine adapters."""

from dataclasses import dataclass
from enum import Enum


class FFNKind(str, Enum):
    DENSE = "dense"
    ROUTED_MOE = "routed_moe"


@dataclass(frozen=True)
class RoutedMoeLayerSpec:
    layer_id: int
    num_experts: int
    top_k: int
    intermediate_size: int
    norm_topk_prob: bool


# Compatibility name retained for the Qwen3 M1/M2 call sites.
Qwen3MoeLayerSpec = RoutedMoeLayerSpec


def is_qwen3_family(model_type):
    return model_type in ("qwen3", "qwen3_moe")


def is_routed_moe_model(model_type):
    return model_type in ("qwen3_moe", "maple")


def has_qk_norm(model_type):
    return is_qwen3_family(model_type) or model_type == "maple"


def uses_rotary_position(config, layer_id):
    """Whether Q/K receive RoPE in this layer."""
    if not 0 <= layer_id < config.num_hidden_layers:
        raise IndexError(
            f"layer_id={layer_id} is outside [0, {config.num_hidden_layers})"
        )
    if config.model_type == "maple":
        return config.layer_types[layer_id] == "sliding_attention"
    return config.model_type != "opt"


def validate_maple_config(config):
    """Reject Maple variants whose semantics the initial adapter cannot match."""
    if getattr(config, "model_type", None) != "maple":
        raise ValueError("expected model_type=maple")
    positive = (
        "hidden_size", "head_dim", "num_attention_heads", "num_kv_heads",
        "num_hidden_layers", "moe_intermediate_size", "num_experts",
        "num_experts_per_tok", "vocab_size", "max_position_embeddings",
        "sliding_window",
    )
    for field in positive:
        value = getattr(config, field, None)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{field} must be a positive integer, got {value!r}")
    if config.num_attention_heads % config.num_kv_heads:
        raise ValueError("num_attention_heads must be divisible by num_kv_heads")
    if config.num_experts_per_tok > config.num_experts:
        raise ValueError("num_experts_per_tok cannot exceed num_experts")
    if getattr(config, "hidden_act", None) != "silu":
        raise ValueError("Maple adapter requires hidden_act=silu")
    if getattr(config, "attention_bias", None):
        raise ValueError("Maple adapter does not support attention projection bias")
    if getattr(config, "use_qk_norm", None) is not True:
        raise ValueError("Maple adapter requires Q/K RMSNorm")
    if getattr(config, "norm_topk_prob", None) is not True:
        raise ValueError("Maple adapter requires normalized top-k routing")
    if getattr(config, "router_fp32", None) is not True:
        raise ValueError("Maple adapter requires fp32 router computation")
    if getattr(config, "num_shared_experts", None) != 0:
        raise ValueError("Maple adapter does not support shared experts")
    if getattr(config, "partial_rotary_factor", None) != 0.5:
        raise ValueError("Maple adapter currently requires partial_rotary_factor=0.5")
    if getattr(config, "nope_on_global_attention", None) is not True:
        raise ValueError("Maple adapter requires NoPE global-attention layers")
    layer_types = getattr(config, "layer_types", None)
    if not isinstance(layer_types, list) or len(layer_types) != config.num_hidden_layers:
        raise ValueError("layer_types must contain one entry per Maple layer")
    if any(kind not in {"sliding_attention", "full_attention"} for kind in layer_types):
        raise ValueError("Maple layer_types contains an unsupported attention kind")
    epsilon = getattr(config, "rms_norm_eps", None)
    if not isinstance(epsilon, (int, float)) or isinstance(epsilon, bool) or epsilon <= 0:
        raise ValueError(f"rms_norm_eps must be positive, got {epsilon!r}")
    return config


def validate_maple_run(config, prompt_len, gen_len, lr_proj_mode):
    """Keep the initial adapter inside semantics implemented by the engine."""
    if config.model_type != "maple":
        return config
    if prompt_len + gen_len - 1 > config.sliding_window:
        raise ValueError(
            "the initial Maple adapter is correctness-gated only while "
            "prompt_len + gen_len - 1 <= sliding_window; layer-specific "
            "long-context SWA cache handling is not implemented"
        )
    if lr_proj_mode != "none":
        raise ValueError(
            "Maple KV selection requires a separately calibrated predictor; "
            "the initial architecture adapter supports --lr_proj_mode none"
        )
    return config


def validate_qwen3_moe_config(config):
    """Reject an inconsistent canonical Qwen3-MoE config before allocation."""
    if getattr(config, "model_type", None) != "qwen3_moe":
        raise ValueError("expected model_type=qwen3_moe")

    positive_integer_fields = (
        "hidden_size",
        "head_dim",
        "num_attention_heads",
        "num_kv_heads",
        "num_hidden_layers",
        "intermediate_size",
        "moe_intermediate_size",
        "num_experts",
        "num_experts_per_tok",
        "decoder_sparse_step",
        "vocab_size",
        "max_position_embeddings",
    )
    for field in positive_integer_fields:
        value = getattr(config, field, None)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{field} must be a positive integer, got {value!r}")

    if config.num_attention_heads % config.num_kv_heads != 0:
        raise ValueError("num_attention_heads must be divisible by num_kv_heads")
    if config.num_experts_per_tok > config.num_experts:
        raise ValueError("num_experts_per_tok cannot exceed num_experts")
    if config.max_position_embeddings < 32768:
        raise ValueError("max_position_embeddings must be at least 32768")

    epsilon = getattr(config, "rms_norm_eps", None)
    if not isinstance(epsilon, (int, float)) or isinstance(epsilon, bool) or epsilon <= 0:
        raise ValueError(f"rms_norm_eps must be positive, got {epsilon!r}")
    if not isinstance(getattr(config, "norm_topk_prob", None), bool):
        raise ValueError("norm_topk_prob must be boolean")

    mlp_only_layers = getattr(config, "mlp_only_layers", None)
    if not isinstance(mlp_only_layers, list):
        raise ValueError("mlp_only_layers must be a list")
    if any(
        not isinstance(layer, int)
        or isinstance(layer, bool)
        or not 0 <= layer < config.num_hidden_layers
        for layer in mlp_only_layers
    ):
        raise ValueError("mlp_only_layers contains an invalid layer index")
    if len(set(mlp_only_layers)) != len(mlp_only_layers):
        raise ValueError("mlp_only_layers contains duplicate layer indices")

    if not any(
        layer not in set(mlp_only_layers)
        and (layer + 1) % config.decoder_sparse_step == 0
        for layer in range(config.num_hidden_layers)
    ):
        raise ValueError("Qwen3-MoE config does not contain a routed MoE layer")
    return config


def get_ffn_kind(config, layer_id):
    """Return the FFN kind using the Qwen3-MoE layer scheduling rule."""
    if not 0 <= layer_id < config.num_hidden_layers:
        raise IndexError(
            f"layer_id={layer_id} is outside [0, {config.num_hidden_layers})"
        )

    if config.model_type == "maple":
        return FFNKind.ROUTED_MOE
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


def get_routed_moe_layer_spec(config, layer_id):
    if get_ffn_kind(config, layer_id) != FFNKind.ROUTED_MOE:
        return None
    return RoutedMoeLayerSpec(
        layer_id=layer_id,
        num_experts=int(config.num_experts),
        top_k=int(config.num_experts_per_tok),
        intermediate_size=int(config.moe_intermediate_size),
        norm_topk_prob=bool(config.norm_topk_prob),
    )


def get_qwen3_moe_layer_spec(config, layer_id):
    """Compatibility wrapper for the original Qwen3-named storage helpers."""
    return get_routed_moe_layer_spec(config, layer_id)
