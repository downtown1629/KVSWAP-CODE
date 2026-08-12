# Milestone 2 구현 계획 — Synchronous Expert Demand Loading

## 1. 목표와 선행 조건

M2의 목표는 Qwen3-MoE의 non-expert weight를 memory에 유지하고, router가 선택한
routed expert만 persistent storage에서 동기적으로 읽어 계산한 뒤 논리적으로
폐기하는 것이다. 이는 성능 최적화가 아니라 이후 cache와 scheduler의 비교 기준인
`KVSwap + naive SSD experts` baseline이다.

```text
Attention → Router → exact top-k → blocking expert read → MoE compute → discard
```

M1의 resident provider는 비교 기준으로 계속 보존한다. 다만 실제 M2 통합 실행 전
다음 M1 gate를 먼저 닫아야 한다.

- tiny fixture에서 KVSwap KV 경로와 resident MoE 동시 실행
- dense Qwen3 smoke regression
- 실제 checkpoint의 resident/reference parity 또는 동일 weight에서 생성한 검증 artifact

Storage packer, manifest validator, CPU reader test는 이 gate와 독립적으로 개발할 수
있지만, 미완료 M1 항목을 M2 성과로 대체하지 않는다.

## 2. 범위

M2에 포함하는 기능은 BF16 Qwen3-MoE expert의 lossless packing, O(1) extent lookup,
synchronous demand read, bounded scratch memory, engine integration 및 I/O accounting이다.

다음은 명시적으로 제외한다.

- expert cache, eviction 또는 token 간 weight reuse
- prefetch, compute/I/O overlap 또는 unified scheduler
- dynamic/static quantization, compression, fused MoE kernel
- mapped-host zero-copy backend
- Qwen3.5, Maple 또는 Gemma 실행 adapter

사용자가 허용한 static INT8은 실제 하드웨어 feasibility lane으로 유효하지만, M2의
reference-preserving BF16 gate와 섞지 않는다. 필요하면 BF16 M2가 성립한 뒤 동일
expert ID를 사용하는 별도 static-INT8 lane에서 quality delta와 bytes/token을 측정한다.

한 호출 안에서 중복 선택된 expert는 한 번만 읽는다. 호출이 끝난 뒤 scratch allocation은
재사용할 수 있지만 그 안의 expert는 invalid로 취급하며, 다음 호출은 반드시 다시 읽는다.
따라서 allocator 재사용은 cache hit가 아니다.

## 3. 제안 구조

```text
MoEBlock
  ├── resident router / norm
  └── ExpertProviderFactory
        ├── ResidentExpertProvider       # M1/B0
        └── Qwen3DemandExpertProvider    # M2/B1
              ├── ExpertStoreIndex
              ├── SynchronousExtentReader
              ├── aligned pinned staging slots
              └── bounded CUDA expert scratch
```

`qwen3_moe_forward`와 routing arithmetic은 변경하지 않는다. Demand provider의
`materialize(selected_ids)`는 unique ID를 한 번 host로 전달하고, 선택 extent를 읽어
provider 소유 scratch에 채운 `MaterializedExperts` view를 반환한다. 이 view의 lifetime은
다음 `materialize()` 호출 전까지다. M2는 single-stream blocking execution만 허용한다.

기존 `TorchDisk` facade는 KV batch, token group, 전역 `DiskIO` 상태에 결합되어 있으므로
expert를 그 shape contract에 끼우지 않는다. 대신 기존 liburing의 aligned-buffer 및
batched submit/wait 원리를 재사용하는 작은 extent reader를 둔다. Canonical Jetson
backend는 `O_DIRECT + io_uring submit-and-wait`이며 CPU unit test에는 buffered reader를
사용한다. 둘은 같은 byte-range contract를 가져야 한다.

## 4. Expert storage artifact

Server/host에서 safetensors shard를 tensor 단위로 읽어 다음 artifact를 생성한다.

```text
expert-store/
  manifest.json
  experts-000.bin
  experts-001.bin       # 여러 storage path가 필요할 때만 분할
```

각 expert는 `gate_proj`, `up_proj`, `down_proj` 순서의 하나의 contiguous extent로
저장하고 extent 시작과 padded 길이를 4096 byte에 맞춘다. Manifest는 최소 다음을
기록한다.

```text
format/version, model identity/config fingerprint, source revision
logical dtype/representation
(layer, expert) → file, offset, stored bytes, logical bytes
component name, shape, byte offset, byte length
alignment, per-expert checksum
```

M2 runtime은 `representation=bf16`만 허용한다. Manifest 자체는 component list와
representation ID를 갖게 해 다른 MoE의 물리 표현을 추가할 수 있게 하되, 미래 format을
추측해 runtime contract를 일반화하지 않는다. Packer는 전체 모델을 RAM에 올리지 않고
expert component 하나씩 처리하며, 생성 직후 source와 byte-level checksum을 검증한다.
출력 disk capacity를 먼저 검사하고 임시 파일을 완성·동기화한 뒤 manifest를 마지막에
atomic rename하여 불완전 artifact가 정상 store로 보이지 않게 한다.

Startup validation은 config fingerprint, layer/expert coverage, shape/dtype, extent 범위,
정렬, 중복/겹침 및 실제 file size를 검사한다. 전체 58 GiB 재해시는 offline verify
mode로 두고, runtime hot path의 checksum은 fixture/debug mode에서만 수행한다.

## 5. Memory와 I/O semantics

M2 preflight는 M1의 전체 resident-weight 추정을 그대로 사용하지 않는다. 다음 항목을
모두 unified LPDDR budget에 포함한다.

```text
B_fixed                 resident embedding/attention/router/norm/dense FFN
B_expert_scratch        max materialized unique experts × expert bytes
B_IO                    min(unique experts, demand queue depth) × padded extent
B_KV + B_activation + B_workspace + B_headroom
```

`max_unique = min(num_experts, token_chunk_size × top_k)`를 보수적 상한으로 사용하고,
명시적 scratch-slot limit보다 크면 startup 또는 chunk 실행 전에 거부한다. Nano의 첫
실모델 lane은 decode/prefill 모두 작은 token chunk로 시작한다. Queue depth는 storage
측정값에 따라 최대 4를 허용하지만, 선택 expert batch를 모두 기다린 뒤 compute하므로
여전히 synchronous M2 semantics다.

Canonical path는 aligned pinned staging에서 CUDA scratch로 복사한다. Jetson에서는 두
allocation이 같은 LPDDR capacity를 소비하므로 둘 다 센다. GDS는 지원되지 않으며,
mapped pinned memory를 GEMM이 직접 읽는 zero-copy 실험은 uncached-access 성능을 별도로
검증할 후속 backend로 남긴다.

## 6. 구현 단계와 검증 gate

### A — Contract와 packer

- Qwen3 adapter에서 expert component spec과 fixed/expert byte를 분리한다.
- manifest schema, deterministic packer, offline verifier를 구현한다.
- tiny fixture를 pack하고 원 safetensors와 bitwise round-trip을 비교한다.

**Gate A:** 모든 `(layer, expert, component)`가 정확히 하나의 aligned extent에 매핑되고
누락·중복·손상·config mismatch가 allocation 전에 거부되어야 한다.

### B — Synchronous reader

- buffered CPU reader로 offset/length와 short-read/error handling을 검증한다.
- Jetson reader에 aligned pinned buffer, `O_DIRECT`, submit-and-wait 및 명시적 close를
  추가한다.
- `cudaHostUnregister`, `munlock`, fd/ring close까지 반복 lifecycle test로 확인한다.

**Gate B:** fixture의 cold read가 정확한 bytes를 반환하고, partial read나 alignment
위반이 조용히 통과하지 않으며 반복 open/read/close에서 RSS와 locked memory가 bounded여야 한다.

### C — Demand provider parity

- resident factory와 별도인 `Qwen3DemandExpertProviderFactory`를 추가한다.
- selected ID를 unique/sorted 처리하고 scratch slot에 deterministic하게 배치한다.
- resident와 demand provider에 동일 post-attention hidden을 입력한다.

**Gate C:** top-k ID는 exact하고 routing weight 및 MoE output은 resident reference와
같아야 한다. 같은 expert를 연속 요청해도 매 호출 read counter가 증가하고 cross-call
cache hit는 0이어야 한다.

### D — Engine와 memory gate

- `--expert_mode resident|demand`와 별도 expert-store path를 추가한다.
- demand mode에서는 startup에 전체 expert tensor나 shard page를 materialize하지 않는다.
- M2 capacity estimator와 token-chunk/scratch limit을 CUDA 초기화 전에 검사한다.
- resident mode의 기존 결과와 CLI를 그대로 보존한다.

**Gate D:** tiny full-KV fixture의 resident/demand greedy token이 반복 실행에서 일치하고,
peak memory가 계산된 bound 안에 있으며 expert store가 없거나 너무 큰 scratch 설정은
shard open 전에 실패해야 한다.

### E — KVSwap coexistence

- 동일 fixture와 동일 KVSwap configuration에서 resident/demand를 A/B 실행한다.
- expert reader는 `CacheManager`를 호출하지 않고 KV I/O code를 변경하지 않는다.
- expert와 KV가 같은 NVMe path를 사용하는 configuration도 synchronous baseline으로 기록한다.
- 이 단계에서 처음 selective KV lane과 expert demand read를 함께 켜고, M1의 regression
  gate를 joint-I/O accounting 검사로 확장한다.

**Gate E:** routing과 expert output이 resident path와 일치하고, KV trace는 해당 실행의
selection으로 설명되며, expert/KV byte 합계와 실제 read/write counter가 일치해야 한다.

### F — 실제 Qwen3-MoE lane

- metadata/manifest-only preflight 후 fixed weights, scratch, KV, I/O와 headroom이 맞을 때만
  Jetson 실행을 승인한다.
- prompt 64, batch 1, decode 1–2 token부터 시작하고 단계적으로 늘린다.
- large-memory reference에서 저장한 routing/token artifact와 first divergence를 비교한다.

**Gate F:** 전체 expert set을 LPDDR에 올리지 않고 generation이 성공하며, 반복 token에서
RSS가 증가하지 않고 모든 expert traffic이 selected unique expert와 extent size로 설명되어야
한다. 느린 성능은 failure가 아니다.

## 7. Observability와 평가

M2 필수 event는 `EXPERT_READ_SUBMIT`, `EXPERT_READ_COMPLETE`, `COMPUTE_EXPERT`,
`MEMORY_ALLOC/FREE`다. 기본 실행은 layer별 집계만 남기고 선택적 JSONL trace에 다음을
기록한다.

```text
request/token/chunk/layer, selected IDs, unique IDs
file/offset/logical bytes/padded bytes, request count
read service time, staging-to-device copy time, total materialize stall
scratch/staging peak, cold-or-warm-cache mode, error status
```

필수 지표는 expert bytes/token, requests/token, average request size, read latency,
materialize stall ratio, effective bandwidth, TPOT, resident-MoE B0 대비 slowdown 및 peak
RSS다. 기존 KVSwap-only 결과도 환경 기준점으로 함께 보존한다.
Canonical 성능 측정은 `O_DIRECT`로 page cache를 우회한다. Buffered mode를 측정할 경우
cold/warm cache를 분리하고 결과에 명시한다.

## 8. 테스트와 종료 조건

새 테스트는 기존 `engine/tests/test_moe_*.py`에 contract/provider 검사를 추가하고,
storage lifecycle은 별도 `test_expert_store.py`로 둔다. 실행 wrapper에는 다음 lane을
추가한다.

```bash
bash scripts/eval_moe_m2.sh static
bash scripts/eval_moe_m2.sh fixture-buffered
bash scripts/eval_moe_m2.sh fixture-direct       # Jetson, tiny allocation
bash scripts/eval_moe_m2.sh fixture-fullkv
bash scripts/eval_moe_m2.sh fixture-kvswap
bash scripts/eval_moe_m2.sh real-demand          # explicit memory approval required
```

M2 완료 조건은 다음과 같다.

- [x] M1의 resident/KVSwap/dense 선행 gate가 닫혔다.
- [x] expert store가 lossless하며 O(1) lookup과 strict validation을 제공한다.
- [x] startup에 routed expert 전체가 LPDDR에 materialize되지 않는다.
- [x] selected unique expert만 동기적으로 읽고 호출 간 cache/reuse가 없다.
- [x] resident와 demand path의 routing, MoE output 및 fixture token이 일치한다.
- [x] 반복 generation의 memory/locked-memory/fd 수가 bounded다.
- [x] expert 및 KV traffic이 trace와 byte accounting으로 설명된다.
- [x] 이전 resident baseline과 모든 M1 test가 유지된다.

위 항목은 tiny deterministic fixture에서 확인한 1차 구현 gate다. 실제
Qwen3-30B-A3B의 manifest/preflight 및 허용 가능한 하드웨어에서의 generation은 Gate F의
최종 evidence로 별도 기록하며, 이를 완료하기 전에는 전체 M2를 closed로 선언하지 않는다.

M3는 이 storage index와 reader를 재사용해 bounded LRU를 추가한다. M2에는 cache slot
metadata, eviction policy, async lease, prediction 또는 unified I/O task abstraction을
미리 구현하지 않는다.
