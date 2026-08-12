"""Lossless Qwen3-MoE expert artifacts and synchronous demand loading.

M2 deliberately keeps this path independent from KVSwap's token-group disk
abstractions.  An expert is one aligned extent containing gate/up/down BF16
weights.  A factory shares one staging buffer and one bounded CUDA scratch bank
across sequential MoE layers, so memory does not grow with layer count.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import mmap
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import nvtx
from safetensors import safe_open

from model_adapters import FFNKind, get_ffn_kind, get_qwen3_moe_layer_spec
from moe import MaterializedExperts


FORMAT_VERSION = "kvswap-qwen3-moe-expert-store-v1"
REPRESENTATION = "bf16"
COMPONENTS = ("gate_proj", "up_proj", "down_proj")
DEFAULT_ALIGNMENT = 4096


def _align_up(value, alignment):
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("alignment must be a positive power of two")
    return (value + alignment - 1) // alignment * alignment


def _config_identity(config):
    fields = (
        "model_type",
        "hidden_size",
        "moe_intermediate_size",
        "num_hidden_layers",
        "num_experts",
        "num_experts_per_tok",
        "decoder_sparse_step",
        "mlp_only_layers",
    )
    identity = {name: getattr(config, name) for name in fields}
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return identity, hashlib.sha256(encoded).hexdigest()


def qwen3_expert_component_specs(config, layer_id, expert_id):
    spec = get_qwen3_moe_layer_spec(config, layer_id)
    if spec is None:
        raise ValueError(f"layer {layer_id} is not routed MoE")
    if not 0 <= expert_id < spec.num_experts:
        raise IndexError(f"expert {expert_id} is outside [0, {spec.num_experts})")
    prefix = f"model.layers.{layer_id}.mlp.experts.{expert_id}"
    return (
        (f"{prefix}.gate_proj.weight", (spec.intermediate_size, config.hidden_size)),
        (f"{prefix}.up_proj.weight", (spec.intermediate_size, config.hidden_size)),
        (f"{prefix}.down_proj.weight", (config.hidden_size, spec.intermediate_size)),
    )


def qwen3_routed_layer_ids(config):
    return tuple(
        layer_id
        for layer_id in range(config.num_hidden_layers)
        if get_ffn_kind(config, layer_id) == FFNKind.ROUTED_MOE
    )


def qwen3_expert_logical_bytes(config):
    return (
        3
        * int(config.hidden_size)
        * int(config.moe_intermediate_size)
        * torch.empty((), dtype=torch.bfloat16).element_size()
    )


def qwen3_expert_store_required_bytes(config, alignment=DEFAULT_ALIGNMENT):
    extent = _align_up(qwen3_expert_logical_bytes(config), alignment)
    return len(qwen3_routed_layer_ids(config)) * int(config.num_experts) * extent


def qwen3_moe_fixed_weight_bytes(config, dtype=torch.bfloat16):
    """Bytes kept resident in demand mode, excluding routed expert tensors."""
    from moe_weights import qwen3_moe_resident_expected

    element_size = torch.empty((), dtype=dtype).element_size()
    expected = {
        name: value
        for name, value in qwen3_moe_resident_expected(config, dtype=dtype).items()
        if ".mlp.experts." not in name
    }
    total = sum(math.prod(shape) * element_size for shape, _ in expected.values())
    if config.tie_word_embeddings:
        total += int(config.vocab_size) * int(config.hidden_size) * element_size
    return total


def qwen3_moe_largest_fixed_tensor_bytes(config, dtype=torch.bfloat16):
    from moe_weights import qwen3_moe_resident_expected

    element_size = torch.empty((), dtype=dtype).element_size()
    return max(
        math.prod(shape) * element_size
        for name, (shape, _) in qwen3_moe_resident_expected(
            config, dtype=dtype
        ).items()
        if ".mlp.experts." not in name
    )


def qwen3_checkpoint_digest(checkpoint, config, dtype=torch.bfloat16):
    """Stream a canonical digest over every checkpoint tensor and its metadata."""
    from moe_weights import qwen3_moe_resident_expected

    expected = qwen3_moe_resident_expected(config, dtype=dtype)
    checkpoint.validate(expected)
    digest = hashlib.sha256()
    for name in sorted(expected):
        spec = checkpoint.tensor_specs[name]
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(spec.dtype.encode("ascii") + b"\0")
        digest.update(json.dumps(spec.shape).encode("ascii") + b"\0")
        with safe_open(str(spec.shard), framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor(name)
        raw = tensor.contiguous().view(torch.uint8).numpy()
        digest.update(raw)
        del tensor, raw
    return digest.hexdigest()


def estimate_qwen3_moe_demand_memory(
    config,
    scratch_slots,
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
    alignment=DEFAULT_ALIGNMENT,
    dtype=torch.bfloat16,
):
    """Return the existing capacity-plan shape with fixed+scratch as weights."""
    from moe_weights import ResidentMemoryPlan, estimate_qwen3_moe_resident_memory

    if not isinstance(scratch_slots, int) or not 0 < scratch_slots <= config.num_experts:
        raise ValueError("scratch_slots must be in [1, num_experts]")
    resident_shape = estimate_qwen3_moe_resident_memory(
        config=config,
        gpu_batch_size=gpu_batch_size,
        num_gpu_batches=num_gpu_batches,
        prompt_len=prompt_len,
        gen_len=gen_len,
        cache_gpu_percent=cache_gpu_percent,
        cache_cpu_percent=cache_cpu_percent,
        activation_gpu_percent=activation_gpu_percent,
        activation_cpu_percent=activation_cpu_percent,
        flash_attention=flash_attention,
        system_headroom_bytes=system_headroom_bytes,
        dtype=dtype,
    )
    scratch = scratch_slots * qwen3_expert_logical_bytes(config)
    fixed = qwen3_moe_fixed_weight_bytes(config, dtype=dtype)
    staging = max(
        _align_up(qwen3_expert_logical_bytes(config), alignment),
        qwen3_moe_largest_fixed_tensor_bytes(config, dtype=dtype),
    )
    return ResidentMemoryPlan(
        weights=fixed + scratch,
        memory_kv=resident_shape.memory_kv,
        gpu_kv=resident_shape.gpu_kv,
        memory_activations=resident_shape.memory_activations,
        gpu_activations=resident_shape.gpu_activations,
        workspace=resident_shape.workspace,
        staging=staging,
        system_headroom=system_headroom_bytes,
    )


@dataclass(frozen=True)
class ExpertComponent:
    name: str
    shape: tuple
    offset: int
    length: int


@dataclass(frozen=True)
class ExpertExtent:
    layer_id: int
    expert_id: int
    file: str
    offset: int
    logical_bytes: int
    stored_bytes: int
    checksum: str
    components: tuple


class ExpertStore:
    """Strictly validated read-only expert extent index."""

    def __init__(self, root, config=None, expected_source_revision=None,
                 expected_checkpoint_digest=None):
        self.root = Path(root)
        manifest_path = self.root / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text())
        except FileNotFoundError as error:
            raise FileNotFoundError(f"missing expert manifest: {manifest_path}") from error
        if manifest.get("format") != FORMAT_VERSION:
            raise ValueError(f"unsupported expert store format: {manifest.get('format')!r}")
        if manifest.get("representation") != REPRESENTATION:
            raise ValueError("M2 supports only lossless BF16 expert stores")
        self.source_revision = manifest.get("source_revision")
        if not isinstance(self.source_revision, str) or not self.source_revision.strip():
            raise ValueError("expert store source_revision must be a non-empty string")
        if (
            expected_source_revision is not None
            and self.source_revision != expected_source_revision
        ):
            raise ValueError(
                "expert store source revision mismatch: "
                f"store={self.source_revision!r}, checkpoint={expected_source_revision!r}"
            )
        self.checkpoint_digest = manifest.get("checkpoint_sha256")
        if (
            not isinstance(self.checkpoint_digest, str)
            or len(self.checkpoint_digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.checkpoint_digest
            )
        ):
            raise ValueError("expert store checkpoint digest is invalid")
        if (
            expected_checkpoint_digest is not None
            and self.checkpoint_digest != expected_checkpoint_digest
        ):
            raise ValueError("expert store checkpoint digest mismatch")
        self.alignment = manifest.get("alignment")
        _align_up(0, self.alignment)
        if config is not None:
            identity, fingerprint = _config_identity(config)
            if manifest.get("config_fingerprint") != fingerprint:
                raise ValueError("expert store config fingerprint mismatch")
            if manifest.get("config") != identity:
                raise ValueError("expert store config identity mismatch")

        raw_extents = manifest.get("extents")
        if not isinstance(raw_extents, list) or not raw_extents:
            raise ValueError("expert manifest has no extents")
        self.extents = {}
        file_ranges = {}
        for raw in raw_extents:
            extent = self._parse_extent(raw)
            key = (extent.layer_id, extent.expert_id)
            if key in self.extents:
                raise ValueError(f"duplicate expert extent {key}")
            self.extents[key] = extent
            file_ranges.setdefault(extent.file, []).append(
                (extent.offset, extent.offset + extent.stored_bytes, key)
            )

        for filename, ranges in file_ranges.items():
            path = self.root / filename
            if not path.is_file():
                raise FileNotFoundError(f"missing expert data file: {path}")
            file_size = path.stat().st_size
            previous_end = 0
            for start, end, key in sorted(ranges):
                if start < previous_end:
                    raise ValueError(f"overlapping expert extent {key} in {filename}")
                if end > file_size:
                    raise ValueError(f"expert extent {key} exceeds {filename} size")
                previous_end = end
        if set(file_ranges) != {"experts-000.bin"}:
            raise ValueError("M2 v1 requires exactly one experts-000.bin data file")

        if config is not None:
            expected = {
                (layer_id, expert_id)
                for layer_id in qwen3_routed_layer_ids(config)
                for expert_id in range(config.num_experts)
            }
            actual = set(self.extents)
            if actual != expected:
                raise ValueError(
                    f"expert extent coverage mismatch: missing={sorted(expected - actual)}, "
                    f"extra={sorted(actual - expected)}"
                )
            logical_bytes = qwen3_expert_logical_bytes(config)
            for (layer_id, expert_id), extent in self.extents.items():
                expected_specs = qwen3_expert_component_specs(
                    config, layer_id, expert_id
                )
                expected_shapes = tuple(tuple(shape) for _, shape in expected_specs)
                actual_shapes = tuple(component.shape for component in extent.components)
                if actual_shapes != expected_shapes or extent.logical_bytes != logical_bytes:
                    raise ValueError(
                        f"expert extent shape mismatch for ({layer_id}, {expert_id})"
                    )

    def _parse_extent(self, raw):
        required = {
            "layer_id", "expert_id", "file", "offset", "logical_bytes",
            "stored_bytes", "checksum", "components",
        }
        if not isinstance(raw, dict) or set(raw) != required:
            raise ValueError("expert extent fields are invalid")
        integers = (
            raw["layer_id"], raw["expert_id"], raw["offset"],
            raw["logical_bytes"], raw["stored_bytes"],
        )
        if any(not isinstance(value, int) or isinstance(value, bool) for value in integers):
            raise ValueError("expert extent integer field has invalid type")
        if min(integers) < 0 or raw["logical_bytes"] <= 0:
            raise ValueError("expert extent sizes and IDs must be non-negative")
        if raw["offset"] % self.alignment or raw["stored_bytes"] % self.alignment:
            raise ValueError("expert extent is not aligned")
        if raw["stored_bytes"] < raw["logical_bytes"]:
            raise ValueError("stored expert extent is shorter than logical data")
        if not isinstance(raw["file"], str) or Path(raw["file"]).name != raw["file"]:
            raise ValueError("expert data file must be a basename")
        if (
            not isinstance(raw["checksum"], str)
            or len(raw["checksum"]) != 64
            or any(character not in "0123456789abcdef" for character in raw["checksum"])
        ):
            raise ValueError("expert checksum must be SHA-256 hex")

        components = []
        expected_offset = 0
        for raw_component, expected_name in zip(raw["components"], COMPONENTS):
            if set(raw_component) != {"name", "shape", "offset", "length"}:
                raise ValueError("expert component fields are invalid")
            if raw_component["name"] != expected_name:
                raise ValueError("expert components must be gate/up/down ordered")
            shape = raw_component["shape"]
            if (
                not isinstance(shape, list)
                or len(shape) != 2
                or any(not isinstance(dim, int) or dim <= 0 for dim in shape)
            ):
                raise ValueError("expert component shape is invalid")
            offset = raw_component["offset"]
            length = raw_component["length"]
            if (
                not isinstance(offset, int)
                or isinstance(offset, bool)
                or not isinstance(length, int)
                or isinstance(length, bool)
                or offset < 0
                or length <= 0
            ):
                raise ValueError("expert component byte range type is invalid")
            if offset != expected_offset or length != math.prod(shape) * 2:
                raise ValueError("expert component byte range is invalid")
            components.append(ExpertComponent(expected_name, tuple(shape), offset, length))
            expected_offset += length
        if len(components) != len(COMPONENTS) or expected_offset != raw["logical_bytes"]:
            raise ValueError("expert component coverage is incomplete")
        return ExpertExtent(
            raw["layer_id"], raw["expert_id"], raw["file"], raw["offset"],
            raw["logical_bytes"], raw["stored_bytes"], raw["checksum"],
            tuple(components),
        )

    def get(self, layer_id, expert_id):
        try:
            return self.extents[(int(layer_id), int(expert_id))]
        except KeyError as error:
            raise KeyError(f"expert ({layer_id}, {expert_id}) is not in the store") from error

    def verify_checksums(self):
        readers = {}
        try:
            for key, extent in sorted(self.extents.items()):
                reader = readers.get(extent.file)
                if reader is None:
                    reader = SynchronousExtentReader(
                        self.root / extent.file, direct=False
                    )
                    readers[extent.file] = reader
                data = reader.read(extent.offset, extent.logical_bytes, extent.stored_bytes)
                try:
                    actual = hashlib.sha256(data).hexdigest()
                finally:
                    data.release()
                if actual != extent.checksum:
                    raise ValueError(f"expert checksum mismatch for {key}")
        finally:
            for reader in readers.values():
                reader.close()
        return len(self.extents)


def pack_qwen3_expert_store(checkpoint, config, output_dir, alignment=DEFAULT_ALIGNMENT,
                            source_revision=None):
    """Stream one component at a time into an atomic, aligned expert artifact."""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not isinstance(source_revision, str) or not source_revision.strip():
        raise ValueError("source_revision must be a non-empty immutable revision")
    _align_up(0, alignment)
    routed_layers = qwen3_routed_layer_ids(config)
    if not routed_layers:
        raise ValueError("config contains no routed MoE layers")
    required_bytes = qwen3_expert_store_required_bytes(config, alignment)
    if shutil.disk_usage(output_dir).free < required_bytes:
        raise OSError(f"expert store needs {required_bytes} bytes of free space")

    expected = {}
    for layer_id in routed_layers:
        for expert_id in range(config.num_experts):
            for name, shape in qwen3_expert_component_specs(config, layer_id, expert_id):
                expected[name] = (shape, torch.bfloat16)
    checkpoint.validate(expected)
    checkpoint_digest = qwen3_checkpoint_digest(checkpoint, config)

    data_name = "experts-000.bin"
    temporary = output_dir / f".{data_name}.tmp-{os.getpid()}"
    final_data = output_dir / data_name
    extents = []
    published_data = False
    try:
        with open(temporary, "xb", buffering=0) as output:
            for layer_id in routed_layers:
                for expert_id in range(config.num_experts):
                    start = output.tell()
                    if start % alignment:
                        raise RuntimeError("internal expert packer alignment error")
                    digest = hashlib.sha256()
                    components = []
                    component_offset = 0
                    for component, (name, shape) in zip(
                        COMPONENTS,
                        qwen3_expert_component_specs(config, layer_id, expert_id),
                    ):
                        spec = checkpoint.tensor_specs[name]
                        with safe_open(str(spec.shard), framework="pt", device="cpu") as handle:
                            tensor = handle.get_tensor(name)
                        if tensor.dtype != torch.bfloat16 or tuple(tensor.shape) != tuple(shape):
                            raise ValueError(f"source tensor changed after validation: {name}")
                        raw = tensor.contiguous().view(torch.uint8).numpy().tobytes()
                        if output.write(raw) != len(raw):
                            raise OSError(f"short expert store write for {name}")
                        digest.update(raw)
                        components.append({
                            "name": component,
                            "shape": list(shape),
                            "offset": component_offset,
                            "length": len(raw),
                        })
                        component_offset += len(raw)
                        del tensor, raw
                    stored_bytes = _align_up(component_offset, alignment)
                    padding = b"\0" * (stored_bytes - component_offset)
                    if output.write(padding) != len(padding):
                        raise OSError("short expert store padding write")
                    extents.append({
                        "layer_id": layer_id,
                        "expert_id": expert_id,
                        "file": data_name,
                        "offset": start,
                        "logical_bytes": component_offset,
                        "stored_bytes": stored_bytes,
                        "checksum": digest.hexdigest(),
                        "components": components,
                    })
            output.flush()
            os.fsync(output.fileno())
        with SynchronousExtentReader(temporary, direct=False) as verifier:
            for raw_extent in extents:
                data = verifier.read(
                    raw_extent["offset"], raw_extent["logical_bytes"],
                    raw_extent["stored_bytes"],
                )
                try:
                    if hashlib.sha256(data).hexdigest() != raw_extent["checksum"]:
                        raise OSError(
                            "packed expert bytes failed verification for "
                            f"({raw_extent['layer_id']}, {raw_extent['expert_id']})"
                        )
                finally:
                    data.release()
        os.replace(temporary, final_data)
        published_data = True
        identity, fingerprint = _config_identity(config)
        manifest = {
            "format": FORMAT_VERSION,
            "representation": REPRESENTATION,
            "alignment": alignment,
            "config": identity,
            "config_fingerprint": fingerprint,
            "source_revision": source_revision,
            "checkpoint_sha256": checkpoint_digest,
            "extents": extents,
        }
        manifest_tmp = output_dir / f".manifest.json.tmp-{os.getpid()}"
        manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        with open(manifest_tmp, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(manifest_tmp, output_dir / "manifest.json")
        directory_fd = os.open(output_dir, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        if published_data and final_data.exists() and not (output_dir / "manifest.json").exists():
            final_data.unlink()
        for manifest_tmp in output_dir.glob(".manifest.json.tmp-*"):
            manifest_tmp.unlink()
        raise
    return ExpertStore(
        output_dir, config=config, expected_source_revision=source_revision,
        expected_checkpoint_digest=checkpoint_digest,
    )


class SynchronousExtentReader:
    """One reusable aligned mmap buffer with exact synchronous pread semantics."""

    def __init__(self, path, direct=False, pin_cuda=False):
        self.path = Path(path)
        self.direct = bool(direct)
        self.pin_cuda = bool(pin_cuda)
        if self.pin_cuda and not torch.cuda.is_available():
            raise ValueError("pin_cuda requires an available CUDA runtime")
        flags = os.O_RDONLY | (getattr(os, "O_DIRECT", 0) if self.direct else 0)
        if self.direct and not hasattr(os, "O_DIRECT"):
            raise OSError("O_DIRECT is unavailable on this platform")
        self.fd = os.open(self.path, flags)
        self.ring = None
        self.ring_initialized = False
        self.cqe = None
        self._liburing = None
        try:
            if self.direct:
                import liburing

                self._liburing = liburing
                self.ring = liburing.io_uring()
                result = liburing.io_uring_queue_init(2, self.ring)
                if result != 0:
                    raise RuntimeError(f"io_uring_queue_init failed with {result}")
                self.ring_initialized = True
                self.cqe = liburing.io_uring_cqe()
        except Exception:
            if self.ring_initialized:
                self._liburing.io_uring_queue_exit(self.ring)
            os.close(self.fd)
            self.fd = None
            raise
        self.buffer = None
        self.capacity = 0
        self._registered_address = None
        self.read_count = 0
        self.logical_bytes = 0
        self.stored_bytes = 0

    def _cuda_result(self, result, operation):
        code = result[0] if isinstance(result, tuple) else result
        if int(code) != 0:
            raise RuntimeError(f"{operation} failed with CUDA error {int(code)}")

    def _release_buffer(self):
        if self._registered_address is not None:
            self._cuda_result(
                torch.cuda.cudart().cudaHostUnregister(self._registered_address),
                "cudaHostUnregister",
            )
            self._registered_address = None
        if self.buffer is not None:
            self.buffer.close()
            self.buffer = None
            self.capacity = 0

    def _ensure_capacity(self, size):
        if size <= self.capacity:
            return
        self._release_buffer()
        self.capacity = _align_up(size, mmap.PAGESIZE)
        self.buffer = mmap.mmap(-1, self.capacity, access=mmap.ACCESS_WRITE)
        if self.pin_cuda:
            address = ctypes.addressof(ctypes.c_char.from_buffer(self.buffer))
            self._cuda_result(
                torch.cuda.cudart().cudaHostRegister(address, self.capacity, 0),
                "cudaHostRegister",
            )
            self._registered_address = address

    def read(self, offset, logical_bytes, stored_bytes=None):
        if self.fd is None:
            raise RuntimeError("extent reader is closed")
        if min(offset, logical_bytes) < 0 or logical_bytes <= 0:
            raise ValueError("extent offset/length is invalid")
        stored_bytes = logical_bytes if stored_bytes is None else stored_bytes
        if stored_bytes < logical_bytes:
            raise ValueError("stored extent is shorter than logical data")
        if self.direct and (
            offset % DEFAULT_ALIGNMENT or stored_bytes % DEFAULT_ALIGNMENT
        ):
            raise ValueError("direct extent reads require 4096-byte alignment")
        self._ensure_capacity(stored_bytes)
        target = memoryview(self.buffer)[:stored_bytes]
        try:
            if self.direct:
                sqe = self._liburing.io_uring_get_sqe(self.ring)
                if sqe is None:
                    raise RuntimeError("io_uring submission queue is full")
                self._liburing.io_uring_prep_read(
                    sqe, self.fd, target, stored_bytes, offset
                )
                with nvtx.annotate("EXPERT_READ_SUBMIT", color="orange"):
                    submitted = self._liburing.io_uring_submit(self.ring)
                if submitted != 1:
                    raise RuntimeError(f"io_uring submitted {submitted}, expected 1")
                with nvtx.annotate("EXPERT_READ_COMPLETE", color="yellow"):
                    result = self._liburing.io_uring_wait_cqe(self.ring, self.cqe)
                if result != 0:
                    raise RuntimeError(f"io_uring_wait_cqe failed with {result}")
                count = int(self.cqe.res)
                self._liburing.io_uring_cqe_seen(self.ring, self.cqe)
                if count < 0:
                    raise OSError(-count, os.strerror(-count), self.path)
            else:
                data = os.pread(self.fd, stored_bytes, offset)
                count = len(data)
                target[:count] = data
        finally:
            target.release()
        if count != stored_bytes:
            raise EOFError(
                f"short expert read from {self.path}: expected {stored_bytes}, got {count}"
            )
        self.read_count += 1
        self.logical_bytes += logical_bytes
        self.stored_bytes += stored_bytes
        return memoryview(self.buffer)[:logical_bytes]

    def close(self):
        if self.fd is None and self.ring is None and self.buffer is None:
            return
        error = None
        try:
            self._release_buffer()
        except Exception as caught:
            error = caught
        finally:
            try:
                if self.ring_initialized:
                    self._liburing.io_uring_queue_exit(self.ring)
                    self.ring_initialized = False
                if self.ring is not None:
                    self.ring = None
                    self.cqe = None
            finally:
                if self.fd is not None:
                    fd, self.fd = self.fd, None
                    os.close(fd)
        if error is not None:
            raise error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class SharedExpertWorkspace:
    """Single reader/staging allocation and bounded device bank for all layers."""

    def __init__(self, store, config, device, dtype, slots, direct=False,
                 verify_reads=False):
        if dtype != torch.bfloat16:
            raise ValueError("M2 demand experts require BF16 scratch")
        if not isinstance(slots, int) or slots <= 0 or slots > config.num_experts:
            raise ValueError("expert scratch slots must be in [1, num_experts]")
        self.store = store
        self.config = config
        self.device = torch.device(device)
        self.dtype = dtype
        self.slots = slots
        self.readers = {}
        self.direct = bool(direct)
        self.verify_reads = bool(verify_reads)
        hidden = int(config.hidden_size)
        intermediate = int(config.moe_intermediate_size)
        self.global_ids = torch.empty(slots, dtype=torch.long, device=self.device)
        self.gate_proj = torch.empty(
            (slots, intermediate, hidden), dtype=dtype, device=self.device
        )
        self.up_proj = torch.empty_like(self.gate_proj)
        self.down_proj = torch.empty(
            (slots, hidden, intermediate), dtype=dtype, device=self.device
        )
        self.materialize_calls = 0
        self.read_seconds = 0.0
        self.copy_seconds = 0.0
        self.layer_stats = {}

    @property
    def read_count(self):
        return sum(reader.read_count for reader in self.readers.values())

    @property
    def logical_bytes(self):
        return sum(reader.logical_bytes for reader in self.readers.values())

    @property
    def stored_bytes(self):
        return sum(reader.stored_bytes for reader in self.readers.values())

    def _reader(self, filename):
        if filename not in self.readers:
            self.readers[filename] = SynchronousExtentReader(
                self.store.root / filename,
                direct=self.direct,
                pin_cuda=self.device.type == "cuda",
            )
        return self.readers[filename]

    def materialize(self, layer_id, selected_expert_ids):
        unique_ids = torch.unique(selected_expert_ids.detach()).to("cpu").tolist()
        unique_ids = sorted(int(expert_id) for expert_id in unique_ids)
        if len(unique_ids) > self.slots:
            raise MemoryError(
                f"routing selected {len(unique_ids)} unique experts but scratch has "
                f"{self.slots} slots; reduce --moe_token_chunk_size or increase slots"
            )
        for slot, expert_id in enumerate(unique_ids):
            extent = self.store.get(layer_id, expert_id)
            read_start = time.perf_counter()
            with nvtx.annotate(
                f"EXPERT_READ layer={layer_id} expert={expert_id}", color="orange"
            ):
                raw = self._reader(extent.file).read(
                    extent.offset, extent.logical_bytes, extent.stored_bytes
                )
            read_elapsed = time.perf_counter() - read_start
            self.read_seconds += read_elapsed
            try:
                if (
                    self.verify_reads
                    and hashlib.sha256(raw).hexdigest() != extent.checksum
                ):
                    raise ValueError(
                        f"expert checksum mismatch for ({layer_id}, {expert_id})"
                    )
                copy_start = time.perf_counter()
                with nvtx.annotate(
                    f"EXPERT_COPY layer={layer_id} expert={expert_id}", color="blue"
                ):
                    destinations = (
                        self.gate_proj[slot], self.up_proj[slot],
                        self.down_proj[slot],
                    )
                    for component, destination in zip(extent.components, destinations):
                        component_view = raw[
                            component.offset : component.offset + component.length
                        ]
                        try:
                            source = torch.frombuffer(
                                component_view, dtype=torch.bfloat16
                            ).reshape(component.shape)
                            destination.copy_(source, non_blocking=False)
                            del source
                        finally:
                            component_view.release()
                    if self.device.type == "cuda":
                        torch.cuda.current_stream(self.device).synchronize()
                copy_elapsed = time.perf_counter() - copy_start
            finally:
                raw.release()
            self.copy_seconds += copy_elapsed
            self.global_ids[slot] = expert_id
            stats = self.layer_stats.setdefault(
                int(layer_id),
                {"calls": 0, "reads": 0, "logical_bytes": 0,
                 "stored_bytes": 0, "read_seconds": 0.0,
                 "copy_seconds": 0.0},
            )
            stats["reads"] += 1
            stats["logical_bytes"] += extent.logical_bytes
            stats["stored_bytes"] += extent.stored_bytes
            stats["read_seconds"] += read_elapsed
            stats["copy_seconds"] += copy_elapsed
        self.materialize_calls += 1
        self.layer_stats.setdefault(
            int(layer_id),
            {"calls": 0, "reads": 0, "logical_bytes": 0,
             "stored_bytes": 0, "read_seconds": 0.0,
             "copy_seconds": 0.0},
        )["calls"] += 1
        count = len(unique_ids)
        return MaterializedExperts(
            self.global_ids[:count], self.gate_proj[:count], self.up_proj[:count],
            self.down_proj[:count],
        )

    def close(self):
        for reader in self.readers.values():
            reader.close()
        self.readers.clear()


class DemandExpertProvider:
    def __init__(self, layer_id, workspace):
        self.layer_id = int(layer_id)
        self.workspace = workspace

    def materialize(self, selected_expert_ids):
        return self.workspace.materialize(self.layer_id, selected_expert_ids)


class Qwen3DemandExpertProviderFactory:
    """Factory sharing one bounded workspace across sequential routed layers."""

    def __init__(self, store, config, slots, direct=False, verify_reads=False):
        self.store = store
        self.config = config
        self.slots = slots
        self.direct = direct
        self.verify_reads = verify_reads
        self.workspace = None

    def create(self, layer_id, device, dtype):
        if self.workspace is None:
            self.workspace = SharedExpertWorkspace(
                self.store, self.config, device, dtype, self.slots, self.direct,
                self.verify_reads,
            )
        elif self.workspace.device != torch.device(device) or self.workspace.dtype != dtype:
            raise ValueError("all demand providers must share one device and dtype")
        return DemandExpertProvider(layer_id, self.workspace)

    def close(self):
        if self.workspace is not None:
            self.workspace.close()
