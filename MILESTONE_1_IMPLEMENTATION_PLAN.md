# Milestone 1 구현 계획 — Fully-Resident MoE Functional Integration

## 1. 목표와 완료 상태

이 계획의 목표는 기존 KVSwap `engine`이 dense MLP뿐 아니라 **정확한 routed MoE layer**를 실행하게 만드는 것이다. M1 완료 시 attention과 KV cache는 기존 KVSwap 경로를 그대로 사용하고, router와 모든 routed expert weight는 unified LPDDR/GPU memory에 상주한다.

완료 결과는 다음과 같다.

- `model_type=qwen3_moe`인 단일 canonical architecture 지원
- dense layer와 sparse MoE layer가 섞인 checkpoint 지원
- prefill의 다중 token dispatch와 decode의 single/few-token dispatch 지원
- router logits, top-k expert ID, routing weight, expert output, MoE output 검증 가능
- full-KV와 KVSwap KV policy 모두에서 resident MoE 경로 실행
- 이후 M2가 resident provider를 synchronous demand provider로 교체할 수 있는 명확한 경계 확보

M1 correctness path에서는 expert offload, cache, eviction, prefetch, quantization, compression 및 fused/custom kernel 최적화를 구현하지 않는다. AGX Orin에서의 static expert-only INT8은 BF16 correctness와 분리된 M1.5 feasibility lane으로 다룬다.

## 2. 선행 결정

### 2.1 Canonical architecture

첫 adapter는 Hugging Face Transformers 4.51 계열의 `Qwen3MoeForCausalLM`/`Qwen3MoeSparseMoeBlock` semantics를 대상으로 한다. 모델명 문자열이 아니라 `AutoConfig.model_type == "qwen3_moe"`로 판별한다. 실제 checkpoint 후보는 설계 노트의 Qwen3-30B-A3B이며, core runtime에 이 checkpoint의 파일명 규칙을 직접 넣지 않는다.

현재 Qwen3-MoE 구현은 routed expert만 가진다. Shared-expert 필드는 나중에 추가할 수 있지만 그 연산과 테스트를 M1에서 미리 구현하지 않는다. 두 번째 MoE architecture 지원은 M1 범위가 아니다.

### 2.2 두 개의 검증 lane

Qwen3-30B-A3B BF16 all-resident 실행은 약 61 GB의 weight memory가 필요하므로 8 GB Orin Nano에서 M1 reference 실행이 불가능하다. 현재 engine도 `cuda:0` 단일 장치만 지원한다. 이를 숨기지 않고 다음 lane을 분리한다.

1. **Orin Nano integration lane:** 동일 Qwen3-MoE 구조의 작은 deterministic BF16 fixture로 engine/KVSwap 통합을 검증한다.
2. **Single-GPU reference lane:** 단일 A100/H100 80 GB 이상에서 BF16 engine과 HF reference를 순차 실행해 실제 checkpoint parity를 검증한다.
3. **Multi-GPU reference-only lane:** 2×48 GB 환경은 HF trace/reference 생성에만 사용하며 engine parity를 주장하지 않는다.
4. **AGX Orin feasibility lane:** 충분한 memory가 있는 AGX Orin에서 router·attention·non-expert는 BF16, routed expert는 static INT8로 resident 실행하고 BF16 대비 품질을 별도 측정한다.

Fixture는 2 layers, kernel-compatible head dimension과 attention/KV head 수, 8 experts, top-2, `max_position_embeddings >= 32768`을 사용한다. Router margin이 충분히 크도록 고정 seed weight/input을 만들고 중간 tensor를 reference로 보존한다. Token text hash는 보조 smoke로만 사용하며 실제 model weight는 저장소에 커밋하지 않는다.

## 3. 현재 코드의 변경 지점

| 현재 가정 | 영향 | 계획된 변경 |
|---|---|---|
| `model_config.py`가 모델 디렉터리 이름으로 dense Qwen을 판별 | `qwen3_moe` config가 dense Qwen으로 오인될 수 있음 | HF `model_type` 기반 adapter 선택 및 MoE 필드 정규화 |
| `LM`이 모든 block에 `SelfAttention + MLP`를 생성 | routed layer를 표현할 수 없음 | FFN layer factory로 `MLP` 또는 `MoEBlock` 선택 |
| `MLP.init_weight()`가 세 projection만 가정 | router/expert bank를 읽을 수 없음 | adapter가 weight 이름·shape·layer kind 제공 |
| Qwen3 분기가 여러 곳에서 정확히 `'qwen3'`만 검사 | q/k norm, BF16, rotary path 누락 가능 | `qwen3` family predicate를 한 곳으로 통합 |
| `make_np_weights.py`가 전체 model을 CPU에 올리고 FP16 NPY로 변환 | 원본 BF16과 router 경계가 달라지고 Jetson 변환이 불가능 | server-side safetensors shard streaming 또는 direct loading |
| `llama_mlp*`만 존재 | dispatch/aggregation 없음 | 별도의 reference MoE primitive 추가 |

KV selection, `CacheManager`, PagedAttention block table, rolling/reuse buffer 및 disk I/O 코드는 M1에서 변경하지 않는다.

## 4. 제안 설계

### 4.1 Model adapter contract

새 `engine/src/model_adapters.py`에는 Qwen3-MoE에 지금 필요한 책임만 둔다.

```text
model family / dtype / RMSNorm epsilon
layer index -> DENSE 또는 ROUTED_MOE
layer -> router weight spec
layer -> expert count, top-k, normalization option
expert -> gate/up/down projection specs
checkpoint name -> canonical runtime tensor name
```

`model_config.py`는 HF config를 runtime namespace로 복사하되 `num_experts`, `num_experts_per_tok`, `moe_intermediate_size`, `norm_topk_prob`, `decoder_sparse_step`, `mlp_only_layers`, `rms_norm_eps`를 명시적으로 검증한다. 누락되거나 모순된 config는 weight allocation 전에 실패시킨다.

### 4.2 Expert provider 경계

MoE primitive는 물리적 storage layout을 직접 알지 않고 다음의 얇은 경계를 사용한다.

```text
ExpertProvider.materialize(layer, selected_expert_ids)
    -> global expert IDs + gate/up/down tensors
```

M1의 `ResidentExpertProvider`는 이미 상주한 tensor의 reference만 반환하며 선택할 때 weight를 복제하지 않는다. M2는 같은 interface의 `DemandExpertProvider`로 교체한다. M1에서 packed `w1/w2`, expert별 file, flat binary 중 하나를 영구 format으로 고정하지 않는다. Router와 post-attention norm은 provider 밖의 fixed layer weight다. Startup 시 fixed/attention/router/expert/KV/workspace 예상 byte를 출력하고 headroom을 넘으면 inference 전에 거부한다.

### 4.3 Correctness-first MoE 연산

새 `engine/src/moe.py`에는 Transformers reference와 같은 순서의 순수 PyTorch 구현을 둔다.

1. `MoEBlock`에서 config epsilon을 사용한 post-attention RMSNorm
2. FP32 softmax를 사용하는 router logits 계산
3. 원본 top-k 선택 및 `norm_topk_prob` 적용
4. provider에 선택 expert materialization 요청
5. 선택 expert별 token grouping
6. SwiGLU `down(silu(gate(x)) * up(x))`
7. routing weight 곱과 `index_add_` 누적
8. `MoEBlock`에서 residual 추가

Prefill은 여러 token이 같은 expert로 가는 경우를 유지하며 chunk 단위 실행과 unchunked 실행이 같아야 한다. Decode도 동일 primitive를 사용한다. vLLM `fused_experts`는 M1의 필수 경로가 아니다. 추가한다면 `moe_impl=reference|fused`로 분리하고 reference parity를 통과한 뒤에만 기본 후보가 될 수 있다.

`RoutingResult(router_logits, topk_ids, topk_weights)`를 debug/trace mode에서 반환하되 production generation에서는 불필요한 tensor를 보존하지 않는다.

### 4.4 Engine 연결

`engine/src/main.py`의 layer 생성부를 FFN factory로 바꾸고 dense layer는 기존 `MLP`를 그대로 사용한다. `MoEBlock`은 기존 layer protocol인 `init_weight`, `load_weight`, `forward`, cache no-op 메서드를 구현하여 generation loop의 순서와 KV prefetch 타이밍을 바꾸지 않는다.

Qwen3-MoE attention은 dense Qwen3와 같은 q/k norm 및 rotary 처리를 사용하도록 family predicate를 적용한다. RMSNorm epsilon은 하드코딩하지 않고 config에서 전달한다. `layer_type()`과 NVTX label은 `MoE` 및 `Router`를 구분한다.

## 5. 구현 단계와 검증 게이트

### 단계 A — Fixture와 checkpoint contract

- isolated tensor fixture로 router margin과 MoE semantics를 먼저 검증
- 이후 `engine/scripts/make_tiny_qwen3_moe.py`로 kernel-compatible checkpoint 생성
- `safetensors.safe_open` 기반 shard streaming 또는 direct loading으로 BF16 보존
- 변환은 server/host에서 수행하고 Jetson은 완성된 fixture/artifact만 사용
- loader가 missing/unexpected tensor와 shape/dtype mismatch를 보고

**Gate A:** 원본 HF tensor와 engine에 적재된 tensor의 logical dtype/value가 동일하며, 예상 resident byte와 실제 allocation 차이가 설명 가능해야 한다.

### 단계 B — Adapter와 layer schedule

- Qwen3-MoE config adapter 구현
- dense/MoE layer 판정 및 weight mapping 구현
- dense Qwen3 기존 경로 회귀 테스트 추가

**Gate B:** fixture의 모든 parameter가 정확히 한 runtime object에 매핑되고 미사용/중복 parameter가 없어야 한다.

### 단계 C — Isolated MoE primitive

- router, expert dispatch, weighted aggregation 구현
- prefill, decode, duplicated expert selection, 사용되지 않은 expert, top-k=1/2 사례 테스트
- chunked/unchunked 결과 비교

**Gate C:** 같은 input과 weight에서 HF block 대비 top-k ID가 100% 일치해야 한다.

### 단계 D — LM integration

- FFN factory와 `MoEBlock` 연결
- resident weight load/buffer lifecycle 구현
- 선택 layer/token의 debug capture와 aggregated NVTX counter 추가
- full-KV fixture prefill 및 greedy decode 실행

**Gate D:** fixture에서 HF와 16-token greedy sequence가 정확히 일치하고 반복 실행 결과가 deterministic해야 한다.

### 단계 E — KVSwap coexistence

- 동일 fixture를 `lr_proj_mode=none`으로 먼저 실행
- shape-compatible deterministic predictor artifact를 생성하거나 selector 결과를 주입해 KVSwap 경로 실행
- KVSwap attention 직후의 동일 hidden state를 HF MoE block에도 입력하여 MoE 부분만 비교
- dense Qwen3 Nano smoke test를 다시 실행해 회귀 확인

**Gate E:** expert path가 `CacheManager`를 직접 호출하지 않고, 선택 group에서 계산한 expected KV bytes와 trace가 일치하며, expert storage I/O event가 0이어야 한다. MoE 통합 전후의 selection ID 동일성은 요구하지 않는다.

### 단계 F — 실제 checkpoint 검증

- 대용량 GPU에서 한 layer부터 비교 후 short-prompt 전체 prefill과 decode로 확대
- 메모리 추정치를 출력하고 승인을 받은 환경에서만 full-model load
- router mismatch가 발생하면 first divergent layer/token/expert와 logits margin을 저장

**Gate F:** 같은 device/backend에서는 router margin이 수치 오차보다 큰 위치의 top-k가 일치해야 한다. 실제 checkpoint는 layer별 hidden/logit error, routing boundary margin 및 first divergence를 기록하며 short greedy generation 결과도 함께 비교한다.

## 6. 테스트 계획

새 테스트는 `engine/tests/`에 두고 다음처럼 실행한다.

```bash
cd engine
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_moe_*.py'
bash scripts/eval_moe_m1.sh fixture-fullkv
bash scripts/eval_moe_m1.sh fixture-kvswap
bash scripts/eval_moe_m1.sh real-parity   # large-memory server only
```

권장 테스트 파일:

- `test_moe_adapter.py`: config, layer schedule, name/shape mapping
- `test_moe_weights.py`: streaming/direct load, BF16 round-trip, memory estimate
- `test_moe_forward.py`: router/expert/aggregation reference parity
- `test_moe_engine.py`: prefill, decode, residual/norm, generation parity

동일 input/weight/device의 isolated primitive는 exact top-k를 요구한다. End-to-end 비교는 dtype과 tensor scale별 max absolute error, relative norm, routing boundary margin 및 first divergence를 기록한다. Fixture는 의도적으로 큰 router margin을 사용해 token exact를 요구할 수 있지만 서로 다른 backend의 실모델 전체에 단일 `rtol/atol`을 강제하지 않는다. NaN/Inf는 허용하지 않는다.

필수 입력 matrix는 batch 1 decode, batch 2 few-token decode, 여러 token이 같은 expert를 고르는 fixed-length prefill, dense/MoE mixed layer schedule을 포함한다. Variable-length padded prompt는 별도 engine capability로 연기한다.

## 7. Observability와 산출물

M1은 timing perturbation을 피하기 위해 선택 layer/token의 debug capture와 layer별 집계만 기록한다.

```text
COMPUTE_ROUTER: token, layer, input tokens, num experts, top-k, duration
COMPUTE_EXPERT: layer, expert id, aggregated dispatched tokens, duration
MEMORY_ALLOC: object=router|expert|workspace, layer, bytes, dtype
```

최종 산출물은 다음과 같다.

- Qwen3-MoE adapter 및 resident `MoEBlock`
- dtype-preserving streaming/direct loader
- tiny deterministic fixture 생성기
- unit/parity test와 `eval_moe_m1.sh`
- `engine/moe_m1.md` 실행 및 결과 해석 문서
- fixture full-KV, fixture KVSwap, real-model parity 결과 요약/로그

## 8. 위험과 대응

- **메모리 초과:** 실제 model을 Jetson에서 resident 실행하지 않는다. Startup estimator와 두 검증 lane을 강제한다.
- **BF16 변환으로 top-k 변경:** FP16 중간 변환을 제거하고 checkpoint round-trip을 Gate A로 둔다.
- **Qwen3 분기 누락:** 문자열 비교를 흩어 추가하지 않고 adapter/family predicate로 통합한다.
- **Prefill temporary 폭증:** stacked expert bank 직접 적재와 token chunking을 사용하고 transient peak를 trace한다.
- **연산 순서 차이:** fused kernel보다 Transformers와 같은 reference loop를 먼저 구현한다.
- **KVSwap 근사와 MoE 오류 혼동:** KVSwap이 생성한 동일 post-attention hidden을 양쪽 MoE 구현에 넣어 FFN 오차를 분리한다.

## 9. M1 종료 체크리스트

- [x] canonical Qwen3-MoE config와 mixed dense/MoE schedule을 읽는다.
- [x] 모든 router/expert weight가 lossless하게 매핑되고 resident memory에 한 번만 존재한다.
- [x] tiny fixture의 prefill과 decode에서 router top-k ID가 reference와 모두 일치한다.
- [x] tiny fixture의 aggregate MoE output이 reference와 일치한다.
- [x] full-KV fixture의 short greedy output이 HF reference와 일치한다.
- [ ] 실제 checkpoint의 short greedy output이 HF reference와 일치한다.
- [ ] KVSwap KV 경로와 resident MoE가 함께 실행된다.
- [ ] 기존 dense Qwen3 smoke test에 기능 회귀가 없다.
- [x] expert I/O/cache/prefetch 코드가 M1에 섞이지 않았다.
- [x] memory estimate, peak usage와 재현 명령이 보존되었다.

모든 항목을 통과해야 Qwen3-MoE M1을 완료한 것으로 본다. 다음 우선순위는 Qwen3-MoE의 실제 Jetson synchronous demand loading(M2)이며, 가능하면 bounded cache(M3)까지 vertical slice를 확보한 뒤 아래 architecture 확장을 시작한다. 추가 모델에는 각자의 resident-parity gate를 통과하기 전 expert offloading을 활성화하지 않는다.

## 10. Qwen3-MoE 성공 이후 architecture 확장

Qwen3-MoE는 공통 routed-expert contract를 검증하는 첫 구현이다. Qwen3 M2/M3 이후 다음 대상을 순차 discovery하고, 서로 독립적인 구현 gate로 관리한다.

| 순서 | 대상 | Qwen3-MoE 대비 핵심 차이 | 첫 지원 범위 |
|---|---|---|---|
| E1 | Qwen3.5-MoE | 3:1 Gated DeltaNet/Gated Attention hybrid와 recurrent state | text-only hybrid sequence engine |
| E2 | [`deepgrove/maple-preview`](https://huggingface.co/deepgrove/maple-preview) | preview checkpoint, custom/ternary expert representation 가능성, revision 변화 위험 | pinned-revision discovery 후 범위 결정 |
| E3 | Gemma 4 MoE `google/gemma-4-26B-A4B` | local/global attention, 병렬 dense MLP+routed MoE, layer별 KV/RoPE schema | text-only 26B-A4B reference |

Qwen3.5-MoE의 첫 실모델은 `Qwen/Qwen3.5-35B-A3B`로 한다. 공식 Transformers 문서 기준 이 모델은 hidden size 2048, 40 layers, 256 experts, top-8 및 shared expert를 사용한다. Gemma 4 26B-A4B는 공식 model card 기준 25.2B total/3.8B active parameter, 30 layers, 256K context 구조다. 두 모델 모두 multimodal frontend가 있지만, KV/expert offloading 연구에서는 text decoder를 먼저 검증하고 vision/audio path는 별도 milestone로 미룬다.

### 10.1 공통 확장 원칙

- 기존 `model_adapters.py` contract를 사용하고 core generation loop에 checkpoint별 이름 분기를 추가하지 않는다.
- Qwen3 M1에서 미래 capability taxonomy를 미리 구현하지 않고 각 discovery 결과에 따라 필요한 protocol만 추가한다.
- 각 모델은 자체 tiny fixture, 원본-format round-trip, architecture-specific router/FFN, layer 및 generation parity test를 갖는다.
- attention architecture가 달라지는 경우 expert adapter와 attention adapter를 별도 변경으로 나누어 first-divergence를 추적한다.
- text-only resident parity를 먼저 통과한 뒤 같은 모델에 M2 demand loading을 연결한다.
- Qwen3-MoE와 기존 dense Qwen3 회귀 suite는 모든 adapter 추가 후 계속 실행한다.

### 10.2 E1 — Qwen3.5-MoE

Qwen3.5-MoE는 adapter 추가가 아니라 `HYBRID_SEQUENCE_ENGINE` milestone로 취급한다. 먼저 Gated DeltaNet recurrent state, full-attention layer schedule, multimodal RoPE 중 text position 처리 및 shared expert arithmetic을 HF reference와 분리 검증한다. DeltaNet layer에는 KVSwap KV object를 억지로 만들지 않고 full-attention layer에만 KVSwap 적용을 검토한다. 기존 engine의 Transformers 4.51 pin은 제자리 upgrade하지 않고 별도 reference environment를 사용한다.

**E1 gate:** DeltaNet state를 포함한 prefill/decode hidden state, routed top-k, shared-expert output 및 16-token greedy sequence가 reference와 일치해야 한다.

### 10.3 E2 — Maple Preview

E2는 구현 milestone이 아니라 discovery gate로 시작한다. Preview repository의 Hugging Face commit SHA, `config.json`, modeling code, tokenizer, weight index 및 license를 기록하고 `trust_remote_code` 필요 여부와 expert의 logical dtype/physical encoding을 조사한다. 이 gate가 지연되어도 Gemma discovery를 막지 않는다. Ternary/custom representation을 BF16 expert bank로 무조건 펼치면 모델의 핵심 memory 특성을 잃을 수 있으므로 다음 두 단계를 분리한다.

1. 원 checkpoint representation을 읽는 correctness reference
2. representation-aware resident tensor와 compute primitive

**E2 gate:** checkpoint revision이 고정되고, 원본 표현을 손실 없이 해석하며, router/expert/layer/generation parity와 실제 resident byte accounting을 통과해야 한다. Preview API가 변하면 adapter를 추측으로 보정하지 않고 새 revision을 별도 대상으로 취급한다.

### 10.4 E3 — Gemma 4 MoE 26B-A4B

Gemma adapter는 text decoder만 우선 구현하되 local/global attention schedule, sliding window, 병렬 dense MLP와 routed expert bank, router norm/scale, layer별 head/KV/RoPE schema가 기존 KVSwap group layout과 호환되는지 먼저 분석한다. Model card의 “shared expert” 표현을 Qwen식 shared-expert tensor로 가정하지 않는다. p-RoPE 및 global-layer KV semantics가 확인되기 전에는 Qwen용 KV predictor를 그대로 연결하지 않는다.

**E3 gate:** full-resident full-KV text generation parity를 먼저 통과하고, local/global layer별 KV layout 검증 후 KVSwap predictor를 연결한다. 병렬 dense MLP는 fixed resident parameter로 시작하며 routed expert와 함께 offload하지 않는다.

### 10.5 확장 완료 조건

각 adapter는 다음 상태를 독립적으로 표시한다.

```text
CONFIG_PARSED
RESIDENT_LAYER_PARITY
RESIDENT_GENERATION_PARITY
KVSWAP_COMPATIBLE
EXPERT_OFFLOAD_COMPATIBLE
```

앞 단계가 실패한 모델은 다음 상태를 선언할 수 없다. 이 방식으로 “checkpoint를 읽을 수 있음”, “MoE가 정확함”, “KVSwap이 정확히 결합됨”, “expert offloading이 동작함”을 서로 다른 성과로 관리한다.
