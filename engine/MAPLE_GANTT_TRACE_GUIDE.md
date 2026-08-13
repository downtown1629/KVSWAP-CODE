# Maple KVSwap–MoE Gantt Trace 해석 가이드

이 문서는 `maple-p8-g3-20260813-164819` Nsight Systems 결과를 해석하는 데
필요한 모델 구조, 실행 순서, SVG 표기, 현재 구현 범위와 병목을 한곳에
정리한다. 이 측정은 Jetson Orin Nano에서 `deepgrove/maple-preview`의 BF16
가중치를 사용한 짧은 입력 프로파일이다. Hugging Face logit parity 또는 실제
장문 KVSwap 성능 결과로 해석하면 안 된다.

## 측정 구성과 토큰의 의미

- 모델: Maple Preview, 24 layers, 256 experts/layer, top-8 routing
- 입력/출력: prompt 8 tokens, generation 3 tokens, batch 1
- 출력: `We need to`
- attention: FlashAttention enabled, PagedAttention disabled
- expert: BF16, synchronous direct-I/O demand loading, scratch 64 slots
- MoE prefill chunk: 8 tokens
- 프로파일러: non-root Nsight Systems CUDA/NVTX software trace only

`gen_len=3`에는 prefill이 만든 첫 출력도 포함된다.

| Trace range | 실제로 처리하는 입력 | 만들어지는 토큰 |
|---|---|---|
| `prefill step=0` | prompt token 1–8 전체를 병렬 처리 | output #1 `We` |
| `decode step=1` | `We` 한 토큰 | output #2 `need` |
| `decode step=2` | `need` 한 토큰 | output #3 `to` |

따라서 prefill 6.20초를 “prompt의 특정 한 토큰 시간”으로 볼 수 없다. 단순
상각값은 약 775 ms/prompt-token이지만, 실제로는 8개 토큰을 함께 계산한 한
prefill이다. Decode SVG 두 장은 각각 실제 단일-token step이며 1.420초와
1.411초가 걸렸다. 전체 trace latency는 9.032초, peak RSS는 3.998 GiB였다.

## 왜 모델 레이어가 두 번 나타나는가

Maple의 Transformer layer 하나는 attention과 routed FFN(MoE)을 순서대로
실행한다. 엔진은 이 둘을 별도 실행 블록으로 펼치므로 상세 SVG에는 다음 네
행이 반복된다.

1. `L0 attention_swa` 또는 `L0 attention_global`: layer 0 attention 전체
2. 들여쓴 `sub-stage`: 바로 위 attention의 내부 연산
3. `L0 moe`: layer 0 routed FFN 전체
4. 들여쓴 `sub-stage`: 바로 위 MoE의 내부 연산

24개 모델 레이어는 엔진에서 input + 24×(attention, MoE) + output, 총 50개
블록이 된다. Maple은 `SWA, SWA, SWA, global` 패턴을 반복한다. 레이어
3/7/11/15/19/23은 global NoPE attention이고 나머지 18개는 partial-RoPE
SWA이다. SWA의 의미상 window는 512이며 persistent cache에는 최대 511개의
이전 KV만 남긴다. Global layer는 전체 이력을 유지할 수 있다.

## SVG 계층과 시간 읽는 법

NVTX 계층은 다음과 같다.

```text
KVSWAP_TOKEN
  KVSWAP_LAYER
    KVSWAP_STAGE
      KVSWAP_ATTENTION_STAGE 또는 KVSWAP_MOE_STAGE
```

부모와 자식 범위는 중첩된다. 예를 들어 `L0 moe`의 `compute` 안에
`router`, `materialize`, `dispatch_compute`, `residual`이 들어 있으므로 이
값들을 부모 `compute`와 다시 더하면 이중 계산이다. 서로 같은 계층의 형제
단계만 합산하거나 비교해야 한다.

`token-gantt.svg`는 모든 레이어의 절대 위치를 보존한다. 상세 SVG는 한 개
또는 몇 개 모델 레이어의 시작–종료 시각으로 X축을 잘라 내부 단계를 확대한다.
2 px보다 짧은 이벤트는 발견할 수 있도록 2 px로 그리지만, 정확한 시간은
hover tooltip과 `.nvtx.csv`에 기록된다. 따라서 막대의 최소 폭을 실제 duration으로
역산해서는 안 된다. SVG는 CPU-side NVTX wall time이며 비동기 CUDA kernel과의
정확한 상관관계는 `.nsys-rep`에서 확인한다.

색은 성공/실패나 GPU 사용률이 아니라 단계 종류를 구분한다. 각 SVG 상단의
범례가 그 페이지의 최종 기준이다. 주요 색은 다음과 같다.

| 색/범례 | 의미 |
|---|---|
| light/dark blue `attention_swa/global` | attention 블록 종류 |
| purple `moe` | routed FFN 블록 |
| red `compute` | 엔진이 해당 블록의 forward를 호출한 부모 범위 |
| orange `load/store/prefetch_cache` | 엔진의 weight/KV 이동 경계 |
| orange-red `materialize` | 선택된 expert 가중치 준비 |
| green `dispatch_compute` | expert별 projection과 누적 |
| lime `router` | top-k expert 선택 |
| teal `qkv_projection` | Q/K/V projection |
| blue `attention/attention_output` | attention 핵심 계산 |
| yellow `norm/rope` | 정규화 또는 positional transform |

## 상위 엔진 단계

각 attention/MoE 행에 공통으로 보이는 `KVSWAP_STAGE`의 의미는 다음과 같다.

- `load_weight`: legacy weight 이동 경계. 현재 MoE fixed weights는 resident이고
  expert는 provider가 관리하므로 MoE에서는 사실상 no-op이다.
- `load_cache`: 해당 attention 블록의 KV/cache 입력 준비. MoE에는 KV cache가 없다.
- `load_hidden`: 이전 블록 hidden state를 compute device로 가져온다.
- `rope`: 엔진 수준 positional-embedding 준비 경계이다.
- `sync_kv`: decode KV prefetch가 있을 때 동기화한다.
- `compute`: attention 또는 MoE forward 전체를 감싼 부모 범위이다.
- `store_hidden`: 다음 블록을 위해 결과 hidden state를 저장한다.
- `store_cache`: 새 K/V를 layer cache에 반영한다. MoE에서는 no-op이다.
- `prefetch_cache`: 다음 attention을 위한 KV 선행 읽기 경계이다.

아주 짧은 no-op 경계도 계측 구조를 일정하게 유지하기 위해 SVG에 남아 있다.

## Attention sub-stage

Prefill에서는 다음 순서가 주로 보인다.

- `norm`: attention 입력 RMSNorm 및 Q/K norm 준비
- `qkv_projection`: Q/K/V 선형변환과 head reshape
- `rope`: SWA layer의 partial RoPE; global layer는 NoPE
- `attention`: causal SWA 또는 full attention. Prefill chunk 범위가 내부에 중첩됨
- `output_projection`: attention 출력을 hidden size로 투영하고 residual 결합
- `cache_pack`: 새 K/V를 layer cache 형식으로 정리

Decode에서는 `qkv_projection`, `kv_concat`, `attention_output`, `cache_pack`이
중심이다. `prefetch_sync`/`prefetch_wait`는 비동기 KV retrieval과의 동기화
경계이고 `speculate`는 KVSwap predictor 경계이다. 이 trace는 predictor를 끈
상태라 해당 범위는 없거나 사실상 no-op이다.

## MoE sub-stage와 현재의 naive demand 경로

- `norm`: post-attention RMSNorm
- `router`: FP32 router softmax 후 256개 중 token별 top-8 expert 선택 및 정규화
- `materialize`: 현재 chunk에서 선택된 unique expert를 준비
- `dispatch_compute`: unique expert별로 token을 모아 gate/up/down SwiGLU projection을
  실행하고 routing weight로 합산
- `residual`: MoE 결과를 attention residual에 더함

`materialize`는 현재 최적화된 cache가 아니라 M2 correctness baseline이다.
36 GiB expert store에서 선택된 expert extent를 `O_DIRECT`와 synchronous
io_uring으로 하나의 CUDA-registered host buffer에 읽고, 공유 BF16 CUDA scratch
bank로 복사한 뒤 stream synchronize한다. 한 expert는 gate/up/down을 합쳐
6 MiB이다. 읽기와 복사가 끝난 후에만 `dispatch_compute`가 시작된다.

현재 구현에는 다음이 없다.

- expert cache 또는 이전 token/layer의 expert reuse
- 다음 layer expert prefetch
- expert I/O와 현재 layer compute의 overlap
- INT8/ternary expert representation
- fused router/dispatch/expert kernel
- layer 간 concurrent expert workspace

Scratch bank 하나를 모든 레이어가 순차 공유하며 다음 materialize call에서 이전
내용을 논리적으로 폐기한다. Python unique-expert loop도 유지된다. 이 구조는
메모리를 제한하고 정확한 demand I/O 기준선을 만드는 데 목적이 있으며 성능
완성형이 아니다.

## 이 trace에서 KVSwap이 실제로 하는 일과 하지 않는 일

이 실행은 KVSwap 엔진 위에서 Maple attention과 MoE expert offloading이 기존
경로를 깨지 않고 공존하는지를 보여준다. 하지만 실제 long-context KVSwap
selection 성능 측정은 아니다.

실행 옵션은 `lr_proj_mode=none`, `use_token_cache=0`, KV cache 100% GPU
placement이며 로그도 `No swap`, `No reuse`이다. Prompt가 8 tokens라 KV cache는
0.47 MiB뿐이다. SWA/global별 cache policy는 적용되지만 global KV를 low-rank
predictor로 선택해 NVMe에서 가져오는 경로는 작동하지 않는다. 현재 Maple용
global-layer predictor가 보정되지 않았기 때문에 이 기능은 fail-closed 상태다.

따라서 여기서 발생한 대규모 NVMe traffic은 KV가 아니라 MoE expert traffic이다.
KV와 expert는 별도 manager/provider, scratch, trace counter를 사용하며 아직
공통 I/O scheduler나 bandwidth arbitration도 없다. 향후 실제 coexistence 실험은
보정된 global-layer predictor, 긴 prompt, global KV NVMe placement를 사용해 expert
read와 KV read의 경합 및 overlap을 별도로 측정해야 한다.

## 현재 병목

동일 계층의 sub-stage만 비교하면 결과는 다음과 같다.

| 구간 | Expert materialize | Expert compute | Attention 내부 합계 |
|---|---:|---:|---:|
| Prefill, 8 prompt tokens | 3,549.7 ms (57.3%) | 1,943.3 ms (31.3%) | 약 225 ms (3.6%) |
| Decode step 1 | 824.3 ms (58.0%) | 425.5 ms (30.0%) | 약 62 ms (4.4%) |
| Decode step 2 | 831.5 ms (58.9%) | 438.7 ms (31.1%) | 약 49 ms (3.5%) |

Decode steady-state에서 모델 레이어 하나당 평균은 materialize 약 34.5 ms,
expert compute 약 18 ms, attention 약 2 ms 이하이다. 즉 현재 token latency의 약
58%는 synchronous expert 준비, 약 30%는 unfused expert 계산이며 SWA KV cache가
주 병목은 아니다. 엔진 로그의 전체 expert 계측은 72 materialize calls, 1,159
expert reads, 7,291,797,504 logical bytes, read 3.309초, copy 1.588초이다.

Prefill L0의 QKV/router/compute가 다른 레이어보다 큰 현상은 첫 CUDA 실행의
lazy initialization과 allocator warm-up이 포함된 cold-start 효과다. 정상 상태
최적화 우선순위는 decode step 2를 기준으로 판단하는 편이 안전하다.

현재 결과가 지시하는 다음 성능 작업 순서는 다음과 같다.

1. expert reuse cache로 반복 NVMe read 제거
2. 다음 MoE layer prefetch 및 read/copy/compute pipeline overlap
3. expert representation 축소(INT8 등)로 I/O와 scratch bytes 감소
4. expert dispatch/GEMM fusion 및 Python/CUDA synchronization 감소
5. 이후 보정된 KVSwap global-KV retrieval과 같은 NVMe에서 joint profiling

## 결과 파일과 재생성

기본 결과는 `data/kvswap_logs/maple-preview/nsys/` 아래에 있다.

- `.nsys-rep`: CUDA kernel/API와 NVTX를 함께 보는 원본 Nsight Gantt
- `.sqlite`: SVG/CSV 재생성용 database
- `.token-gantt.svg`: 전체 token/layer 절대 timeline
- `.layers-XX-YY.gantt.svg`: 확대된 상세 timeline
- `.nvtx.csv`: 모든 range의 정확한 시작과 duration
- `.stage-summary.csv`: token별 동일 stage 합계/평균
- `.stats.txt`: Nsight CUDA/NVTX 통계
- `.log`: 엔진 설정, I/O counters, 출력, latency와 memory

레이어별 SVG 72개는 `nsys/layer-by-layer/`에 있다. 기존 database에서 GPU를
재실행하지 않고 다시 만들 수 있다.

```bash
cd engine
.venv/bin/python scripts/export_nsys_gantt.py TRACE.sqlite \
  --output-prefix OUTPUT_DIR/RUN_NAME --layers-per-page 1
```

Nsight 자체의 계측 오버헤드가 있으므로 이 수치는 unprofiled throughput 보고값과
직접 동일시하지 않는다. 이 Jetson에서는 `ncu`, Nsight GPU metrics, HWPM/Tegra
counter를 사용하지 않는다. 허용된 범위는 non-root CUDA/NVTX software trace이며
하드웨어 counter 장애 기록은 `NSIGHT_JETSON_INCIDENT.md`를 따른다.
