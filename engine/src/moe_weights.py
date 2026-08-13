"""Streaming safetensors loading for resident Qwen3-MoE layers."""

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

from model_adapters import FFNKind, get_ffn_kind, get_qwen3_moe_layer_spec


SAFETENSORS_DTYPE = {
    torch.bfloat16: "BF16",
    torch.float16: "F16",
    torch.float32: "F32",
    torch.int8: "I8",
    torch.uint8: "U8",
}


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple
    dtype: str
    shard: Path

    @property
    def bytes(self):
        bits = {
            "BF16": 16,
            "F16": 16,
            "F32": 32,
            "I8": 8,
            "U8": 8,
        }.get(self.dtype)
        if bits is None:
            raise ValueError(f"unsupported safetensors dtype: {self.dtype}")
        return math.prod(self.shape) * bits // 8


class SafetensorIndex:
    """Read a sharded checkpoint index without requiring any weight shards."""

    def __init__(self, checkpoint_path):
        checkpoint_path = Path(checkpoint_path)
        index_files = sorted(checkpoint_path.glob("*.safetensors.index.json"))
        if len(index_files) != 1:
            raise FileNotFoundError(
                f"expected one safetensors index under {checkpoint_path}, "
                f"found {len(index_files)}"
            )
        self.path = index_files[0]
        index = json.loads(self.path.read_text())
        self.weight_map = index.get("weight_map")
        if not isinstance(self.weight_map, dict):
            raise ValueError(f"invalid weight_map in {self.path}")
        self.total_size = index.get("metadata", {}).get("total_size")
        if not isinstance(self.total_size, int) or self.total_size < 0:
            raise ValueError(f"invalid metadata.total_size in {self.path}")

    def validate_names(self, expected_names):
        expected_names = set(expected_names)
        actual_names = set(self.weight_map)
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        if missing or extra:
            raise ValueError(
                "safetensors index tensor names disagree with the model config: "
                f"missing={missing}, extra={extra}"
            )


class SafetensorCheckpoint:
    """Index tensor metadata without materializing checkpoint weights."""

    def __init__(self, checkpoint_path):
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.is_file():
            if checkpoint_path.suffix != ".safetensors":
                raise ValueError(f"not a safetensors file: {checkpoint_path}")
            shards = [checkpoint_path]
            declared_weight_map = None
        else:
            index_files = sorted(checkpoint_path.glob("*.safetensors.index.json"))
            if len(index_files) > 1:
                raise ValueError(
                    f"multiple safetensors indexes found under {checkpoint_path}"
                )
            if index_files:
                index = json.loads(index_files[0].read_text())
                declared_weight_map = index.get("weight_map")
                if not isinstance(declared_weight_map, dict):
                    raise ValueError(f"invalid weight_map in {index_files[0]}")
                shards = sorted(
                    {checkpoint_path / name for name in declared_weight_map.values()}
                )
            else:
                declared_weight_map = None
                shards = sorted(checkpoint_path.glob("*.safetensors"))

        if not shards:
            raise FileNotFoundError(f"no safetensors checkpoint found at {checkpoint_path}")
        missing = [str(shard) for shard in shards if not shard.is_file()]
        if missing:
            raise FileNotFoundError(f"missing safetensors shards: {missing}")

        self.tensor_specs = {}
        for shard in shards:
            with safe_open(str(shard), framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    if name in self.tensor_specs:
                        raise ValueError(f"duplicate tensor {name} in {shard}")
                    view = handle.get_slice(name)
                    self.tensor_specs[name] = TensorSpec(
                        shape=tuple(view.get_shape()),
                        dtype=view.get_dtype(),
                        shard=shard,
                    )

        if declared_weight_map is not None:
            actual_names = set(self.tensor_specs)
            declared_names = set(declared_weight_map)
            if actual_names != declared_names:
                missing_names = sorted(declared_names - actual_names)
                extra_names = sorted(actual_names - declared_names)
                raise ValueError(
                    "safetensors index disagrees with shards: "
                    f"missing={missing_names}, extra={extra_names}"
                )
            for name, shard_name in declared_weight_map.items():
                if self.tensor_specs[name].shard.name != shard_name:
                    raise ValueError(
                        f"index maps {name} to {shard_name}, but it is in "
                        f"{self.tensor_specs[name].shard.name}"
                    )

    @property
    def total_tensor_bytes(self):
        return sum(spec.bytes for spec in self.tensor_specs.values())

    def validate(self, expected):
        """Validate name -> (shape, torch.dtype) without allocating tensors."""
        errors = []
        for name, (shape, dtype) in expected.items():
            spec = self.tensor_specs.get(name)
            if spec is None:
                errors.append(f"missing tensor: {name}")
                continue
            expected_dtype = SAFETENSORS_DTYPE.get(dtype)
            if expected_dtype is None:
                errors.append(f"unsupported destination dtype for {name}: {dtype}")
            elif spec.dtype != expected_dtype:
                errors.append(
                    f"dtype mismatch for {name}: checkpoint={spec.dtype}, "
                    f"expected={expected_dtype}"
                )
            if tuple(shape) != spec.shape:
                errors.append(
                    f"shape mismatch for {name}: checkpoint={spec.shape}, "
                    f"expected={tuple(shape)}"
                )
        if errors:
            raise ValueError("checkpoint validation failed:\n" + "\n".join(errors))

    def load_into(self, destinations):
        """Load name -> destination, opening each shard at most once."""
        expected = {
            name: (tuple(destination.shape), destination.dtype)
            for name, destination in destinations.items()
        }
        self.validate(expected)
        by_shard = defaultdict(list)
        for name, destination in destinations.items():
            by_shard[self.tensor_specs[name].shard].append((name, destination))

        with torch.no_grad():
            for shard, entries in by_shard.items():
                with safe_open(str(shard), framework="pt", device="cpu") as handle:
                    for name, destination in entries:
                        source = handle.get_tensor(name)
                        destination.copy_(source)
                        del source


def validate_resident_weight_budget(required_bytes, limit_bytes):
    """Require an explicit weight-only budget before resident allocation."""
    if limit_bytes <= 0:
        raise ValueError(
            "Qwen3-MoE resident execution is disabled by default; set an explicit "
            "positive --moe_resident_weight_limit_gb after reserving memory for "
            "KV, activations, I/O, runtime, and the OS"
        )
    if required_bytes > limit_bytes:
        raise MemoryError(
            f"resident weights require {required_bytes / (1024 ** 3):.3f} GiB, "
            f"exceeding the configured resident weight limit "
            f"{limit_bytes / (1024 ** 3):.3f} GiB"
        )
    return required_bytes


def tensor_name_from_legacy_np_path(filename):
    """Recover the HF tensor key from an engine `weights-np` path."""
    path = Path(filename)
    parts = path.parts
    try:
        marker = parts.index("weights-np")
    except ValueError as error:
        raise ValueError(f"weight path does not contain weights-np: {filename}") from error
    tensor_parts = parts[marker + 1 :]
    if not tensor_parts:
        raise ValueError(f"weight path has no tensor name after weights-np: {filename}")
    return ".".join(tensor_parts)


@dataclass(frozen=True)
class Qwen3MoeLayerWeights:
    router: torch.Tensor
    post_attention_norm: torch.Tensor
    gate_proj: torch.Tensor
    up_proj: torch.Tensor
    down_proj: torch.Tensor


@dataclass(frozen=True)
class Qwen3MoeFixedWeights:
    router: torch.Tensor
    post_attention_norm: torch.Tensor


@dataclass(frozen=True)
class Qwen3MoeExpertBank:
    gate_proj: torch.Tensor
    up_proj: torch.Tensor
    down_proj: torch.Tensor


@dataclass(frozen=True)
class ResidentMemoryPlan:
    weights: int
    memory_kv: int
    gpu_kv: int
    memory_activations: int
    gpu_activations: int
    workspace: int
    staging: int
    system_headroom: int

    @property
    def system_required(self):
        return (
            self.weights
            + self.memory_kv
            + self.memory_activations
            + self.workspace
            + self.staging
            + self.system_headroom
        )

    @property
    def cuda_required(self):
        return (
            self.weights
            + self.gpu_kv
            + self.gpu_activations
            + self.workspace
        )


def qwen3_moe_layer_expected(config, layer_id, dtype=torch.bfloat16):
    spec = get_qwen3_moe_layer_spec(config, layer_id)
    if spec is None:
        raise ValueError(f"layer {layer_id} is not a routed Qwen3-MoE layer")
    hidden = int(config.hidden_size)
    intermediate = spec.intermediate_size
    prefix = f"model.layers.{layer_id}"
    expected = {
        f"{prefix}.mlp.gate.weight": ((spec.num_experts, hidden), dtype),
        f"{prefix}.post_attention_layernorm.weight": ((hidden,), dtype),
    }
    for expert_id in range(spec.num_experts):
        expert_prefix = f"{prefix}.mlp.experts.{expert_id}"
        expected[f"{expert_prefix}.gate_proj.weight"] = (
            (intermediate, hidden),
            dtype,
        )
        expected[f"{expert_prefix}.up_proj.weight"] = (
            (intermediate, hidden),
            dtype,
        )
        expected[f"{expert_prefix}.down_proj.weight"] = (
            (hidden, intermediate),
            dtype,
        )
    return expected


def qwen3_moe_layer_bytes(config, layer_id, dtype=torch.bfloat16):
    expected = qwen3_moe_layer_expected(config, layer_id, dtype=dtype)
    element_size = torch.empty((), dtype=dtype).element_size()
    return sum(
        math.prod(shape) * element_size
        for shape, _ in expected.values()
    )


def qwen3_moe_resident_expected(config, dtype=torch.bfloat16):
    """Describe every tensor materialized by the resident Qwen3-MoE engine."""
    hidden = int(config.hidden_size)
    head_dim = int(config.head_dim)
    query_width = int(config.num_attention_heads) * head_dim
    kv_width = int(config.num_kv_heads) * head_dim
    expected = {
        getattr(config, "embedding_weight_name", "model.embed_tokens.weight"): (
            (int(config.vocab_size), hidden), dtype
        ),
        "model.norm.weight": ((hidden,), dtype),
    }
    output_name = (
        getattr(config, "embedding_weight_name", "model.embed_tokens.weight")
        if config.tie_word_embeddings else "lm_head.weight"
    )
    expected[output_name] = ((int(config.vocab_size), hidden), dtype)

    for layer_id in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer_id}"
        expected.update({
            f"{prefix}.self_attn.q_proj.weight": ((query_width, hidden), dtype),
            f"{prefix}.self_attn.k_proj.weight": ((kv_width, hidden), dtype),
            f"{prefix}.self_attn.v_proj.weight": ((kv_width, hidden), dtype),
            f"{prefix}.self_attn.o_proj.weight": ((hidden, query_width), dtype),
            f"{prefix}.self_attn.q_norm.weight": ((head_dim,), dtype),
            f"{prefix}.self_attn.k_norm.weight": ((head_dim,), dtype),
            f"{prefix}.input_layernorm.weight": ((hidden,), dtype),
        })
        if config.attention_bias:
            expected.update({
                f"{prefix}.self_attn.q_proj.bias": ((query_width,), dtype),
                f"{prefix}.self_attn.k_proj.bias": ((kv_width,), dtype),
                f"{prefix}.self_attn.v_proj.bias": ((kv_width,), dtype),
            })

        if get_ffn_kind(config, layer_id) == FFNKind.ROUTED_MOE:
            expected.update(qwen3_moe_layer_expected(config, layer_id, dtype=dtype))
        else:
            intermediate = int(config.intermediate_size)
            expected.update({
                f"{prefix}.mlp.gate_proj.weight": ((intermediate, hidden), dtype),
                f"{prefix}.mlp.up_proj.weight": ((intermediate, hidden), dtype),
                f"{prefix}.mlp.down_proj.weight": ((hidden, intermediate), dtype),
                f"{prefix}.post_attention_layernorm.weight": ((hidden,), dtype),
            })
    return expected


def qwen3_moe_resident_bytes(config, dtype=torch.bfloat16):
    """Estimate weight bytes allocated by the current engine implementation."""
    element_size = torch.empty((), dtype=dtype).element_size()
    expected = qwen3_moe_resident_expected(config, dtype=dtype)
    total = sum(math.prod(shape) * element_size for shape, _ in expected.values())
    if config.tie_word_embeddings:
        # InputEmbed and OutputEmbed currently own separate buffers.
        total += int(config.vocab_size) * int(config.hidden_size) * element_size
    return total


def qwen3_moe_largest_tensor_bytes(config, dtype=torch.bfloat16):
    """Maximum CPU source tensor live beside its final resident destination."""
    element_size = torch.empty((), dtype=dtype).element_size()
    return max(
        math.prod(shape) * element_size
        for shape, _ in qwen3_moe_resident_expected(config, dtype=dtype).values()
    )


def estimate_qwen3_moe_resident_memory(
    config,
    gpu_batch_size,
    num_gpu_batches,
    prompt_len,
    gen_len,
    cache_gpu_percent,
    cache_cpu_percent,
    activation_gpu_percent,
    activation_cpu_percent,
    flash_attention,
    system_headroom_bytes,
    dtype=torch.bfloat16,
):
    """Conservatively estimate unified-memory demand before opening shards."""
    if min(gpu_batch_size, num_gpu_batches, prompt_len, gen_len) <= 0:
        raise ValueError("batch sizes and sequence lengths must be positive")
    percentages = (
        cache_gpu_percent,
        cache_cpu_percent,
        activation_gpu_percent,
        activation_cpu_percent,
    )
    if any(value < 0 or value > 100 for value in percentages):
        raise ValueError("memory percentages must be in [0, 100]")
    if cache_gpu_percent + cache_cpu_percent > 100:
        raise ValueError("cache GPU and CPU percentages exceed 100")
    if activation_gpu_percent + activation_cpu_percent > 100:
        raise ValueError("activation GPU and CPU percentages exceed 100")
    if system_headroom_bytes < 0:
        raise ValueError("system headroom must be non-negative")

    element_size = torch.empty((), dtype=dtype).element_size()
    total_batch = gpu_batch_size * num_gpu_batches
    sequence = prompt_len + gen_len - 1
    kv_bytes = (
        2
        * total_batch
        * sequence
        * config.num_hidden_layers
        * config.num_kv_heads
        * config.head_dim
        * element_size
    )
    activation_bytes = total_batch * sequence * config.hidden_size * element_size * 3
    memory_kv = math.ceil(
        kv_bytes * (cache_gpu_percent + cache_cpu_percent) / 100
    )
    gpu_kv = math.ceil(kv_bytes * cache_gpu_percent / 100)
    memory_activations = math.ceil(
        activation_bytes
        * (activation_gpu_percent + activation_cpu_percent)
        / 100
    )
    gpu_activations = math.ceil(
        activation_bytes * activation_gpu_percent / 100
    )

    tokens = gpu_batch_size * prompt_len
    moe_workspace = tokens * (
        config.num_experts * 4
        + config.num_experts_per_tok * (8 + element_size)
        + 3 * config.moe_intermediate_size * element_size
        + 3 * config.hidden_size * element_size
    )
    if flash_attention:
        attention_workspace = tokens * config.hidden_size * element_size * 4
    else:
        attention_workspace = (
            gpu_batch_size
            * config.num_attention_heads
            * prompt_len
            * prompt_len
            * element_size
        )
    workspace = max(moe_workspace, attention_workspace)
    return ResidentMemoryPlan(
        weights=qwen3_moe_resident_bytes(config, dtype=dtype),
        memory_kv=memory_kv,
        gpu_kv=gpu_kv,
        memory_activations=memory_activations,
        gpu_activations=gpu_activations,
        workspace=workspace,
        staging=qwen3_moe_largest_tensor_bytes(config, dtype=dtype),
        system_headroom=system_headroom_bytes,
    )


def read_linux_memory_capacity(meminfo_path="/proc/meminfo"):
    values = {}
    with open(meminfo_path) as handle:
        for line in handle:
            parts = line.split()
            name = parts[0].rstrip(":")
            if name not in {"MemAvailable", "MemTotal"}:
                continue
            if len(parts) != 3:
                raise ValueError(f"invalid {name} entry in {meminfo_path}: {line!r}")
            _, value, unit = parts
            if unit != "kB":
                raise ValueError(f"unexpected unit for {name}: {unit}")
            values[name] = int(value) * 1024
    try:
        return values["MemAvailable"], values["MemTotal"]
    except KeyError as error:
        raise ValueError(f"missing {error.args[0]} in {meminfo_path}") from error


def validate_resident_memory_capacity(
    plan,
    available_bytes,
    total_bytes,
    cuda_allocator_fraction=0.85,
):
    """Reject a resident run that exceeds system or CUDA allocation capacity."""
    if available_bytes <= 0 or total_bytes <= 0:
        raise ValueError("memory capacity values must be positive")
    if not 0 < cuda_allocator_fraction <= 1:
        raise ValueError("cuda_allocator_fraction must be in (0, 1]")
    errors = []
    if plan.system_required > available_bytes:
        errors.append(
            f"unified-memory plan requires {plan.system_required / (1024 ** 3):.3f} "
            f"GiB but MemAvailable is {available_bytes / (1024 ** 3):.3f} GiB"
        )
    cuda_limit = int(total_bytes * cuda_allocator_fraction)
    if plan.cuda_required > cuda_limit:
        errors.append(
            f"CUDA allocations require {plan.cuda_required / (1024 ** 3):.3f} "
            f"GiB but the {cuda_allocator_fraction:.0%} allocator limit is "
            f"{cuda_limit / (1024 ** 3):.3f} GiB"
        )
    if errors:
        raise MemoryError("resident capacity preflight failed: " + "; ".join(errors))
    return plan


def qwen3_moe_checkpoint_bytes(config, dtype=torch.bfloat16):
    """Return unique checkpoint bytes implied by a Qwen3-MoE config."""
    element_size = torch.empty((), dtype=dtype).element_size()
    return sum(
        math.prod(shape) * element_size
        for shape, _ in qwen3_moe_resident_expected(config, dtype=dtype).values()
    )


def validate_qwen3_moe_index(index, config, dtype=torch.bfloat16):
    """Validate names and total bytes using only config.json and the index."""
    expected = qwen3_moe_resident_expected(config, dtype=dtype)
    if not any(
        get_qwen3_moe_layer_spec(config, layer_id) is not None
        for layer_id in range(config.num_hidden_layers)
    ):
        raise ValueError("qwen3_moe config does not contain any routed MoE layers")
    index.validate_names(expected)
    expected_bytes = qwen3_moe_checkpoint_bytes(config, dtype=dtype)
    if index.total_size != expected_bytes:
        raise ValueError(
            "safetensors index byte count disagrees with the model config: "
            f"index={index.total_size}, expected={expected_bytes}"
        )
    return expected_bytes


def validate_qwen3_moe_checkpoint(checkpoint, config, dtype=torch.bfloat16):
    """Validate all resident weights before any tensor allocation."""
    routed_layers = []
    for layer_id in range(config.num_hidden_layers):
        spec = get_qwen3_moe_layer_spec(config, layer_id)
        if spec is None:
            continue
        routed_layers.append(layer_id)
    if not routed_layers:
        raise ValueError("qwen3_moe config does not contain any routed MoE layers")
    checkpoint.validate(qwen3_moe_resident_expected(config, dtype=dtype))
    return tuple(routed_layers)


def load_qwen3_moe_layer(
    checkpoint,
    config,
    layer_id,
    device="cpu",
    dtype=torch.bfloat16,
):
    """Compatibility helper combining the independently owned M1 weights."""
    fixed = load_qwen3_moe_fixed_weights(
        checkpoint, config, layer_id, device=device, dtype=dtype
    )
    experts = load_qwen3_moe_expert_bank(
        checkpoint, config, layer_id, device=device, dtype=dtype
    )
    return Qwen3MoeLayerWeights(
        router=fixed.router,
        post_attention_norm=fixed.post_attention_norm,
        gate_proj=experts.gate_proj,
        up_proj=experts.up_proj,
        down_proj=experts.down_proj,
    )


def load_qwen3_moe_fixed_weights(
    checkpoint,
    config,
    layer_id,
    device="cpu",
    dtype=torch.bfloat16,
):
    """Load resident router and post-attention norm outside expert storage."""
    spec = get_qwen3_moe_layer_spec(config, layer_id)
    if spec is None:
        raise ValueError(f"layer {layer_id} is not a routed Qwen3-MoE layer")
    hidden = int(config.hidden_size)
    prefix = f"model.layers.{layer_id}"
    expected = {
        f"{prefix}.mlp.gate.weight": ((spec.num_experts, hidden), dtype),
        f"{prefix}.post_attention_layernorm.weight": ((hidden,), dtype),
    }
    checkpoint.validate(expected)
    fixed = Qwen3MoeFixedWeights(
        router=torch.empty(
            (spec.num_experts, hidden), dtype=dtype, device=device
        ),
        post_attention_norm=torch.empty((hidden,), dtype=dtype, device=device),
    )
    checkpoint.load_into({
        f"{prefix}.mlp.gate.weight": fixed.router,
        f"{prefix}.post_attention_layernorm.weight": fixed.post_attention_norm,
    })
    return fixed


def load_qwen3_moe_expert_bank(
    checkpoint,
    config,
    layer_id,
    device="cpu",
    dtype=torch.bfloat16,
):
    """Load only the physical expert bank owned by an expert provider."""
    spec = get_qwen3_moe_layer_spec(config, layer_id)
    if spec is None:
        raise ValueError(f"layer {layer_id} is not a routed Qwen3-MoE layer")
    hidden = int(config.hidden_size)
    intermediate = spec.intermediate_size
    num_experts = spec.num_experts
    prefix = f"model.layers.{layer_id}"
    expected = {}
    for expert_id in range(num_experts):
        expert_prefix = f"{prefix}.mlp.experts.{expert_id}"
        expected[f"{expert_prefix}.gate_proj.weight"] = (
            (intermediate, hidden), dtype
        )
        expected[f"{expert_prefix}.up_proj.weight"] = (
            (intermediate, hidden), dtype
        )
        expected[f"{expert_prefix}.down_proj.weight"] = (
            (hidden, intermediate), dtype
        )
    checkpoint.validate(expected)
    weights = Qwen3MoeExpertBank(
        gate_proj=torch.empty(
            (num_experts, intermediate, hidden), dtype=dtype, device=device
        ),
        up_proj=torch.empty(
            (num_experts, intermediate, hidden), dtype=dtype, device=device
        ),
        down_proj=torch.empty(
            (num_experts, hidden, intermediate), dtype=dtype, device=device
        ),
    )

    destinations = {}
    for expert_id in range(num_experts):
        expert_prefix = f"{prefix}.mlp.experts.{expert_id}"
        destinations[f"{expert_prefix}.gate_proj.weight"] = weights.gate_proj[
            expert_id
        ]
        destinations[f"{expert_prefix}.up_proj.weight"] = weights.up_proj[
            expert_id
        ]
        destinations[f"{expert_prefix}.down_proj.weight"] = weights.down_proj[
            expert_id
        ]
    checkpoint.load_into(destinations)
    return weights
