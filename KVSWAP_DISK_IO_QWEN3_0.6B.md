# KVSwap 디스크 KV 캐시 용량 / R-W 분석 (Qwen3-0.6B, 코드 기반)

> 실제로 KVSwap을 구동하지 않고, `engine/src` 코드만 정적으로 분석한 결과입니다.
> reuse 적중률처럼 런타임 데이터에 의존하는 값은 이론적 범위로만 제시합니다.

## 0. 분석 대상 설정

- 모델: Qwen3-0.6B (`data/model_weights_hf/Qwen3-0.6B/config.json`)
  - `num_hidden_layers=28`, `num_attention_heads=16`, `num_key_value_heads=8`, `head_dim=128`
- 실행 파라미터: `engine/scripts/eval_nano.sh`의 kvswap 모드 기본값
  - context 16K (`prompt_len=16284`, `gen_len=100`), `token_group(G)=4`, `max_num_kv(M)=400`, `reuse_budget=400`

## 1. 토큰 1개당 디스크 점유 용량

디스크 파일 shape은 `pytorch_backend.py:836`에서 `num_kv_heads * head_dim * 2(K,V)`로 잡히고,
`model_config.py:6-9`의 `cache_bytes()` 공식과 일치합니다.

- 레이어 1개, 토큰 1개: `num_kv_heads(8) × head_dim(128) × 2(K,V) × itemsize(2B, fp16)` = **4096 B**
- 전체 28개 레이어: `4096 × 28` = **114,688 B ≈ 112 KB / 토큰**

디스크에는 항상 이 **압축되지 않은 원본 크기**로 저장됩니다. KVSwap의 저랭크 압축(`lr_proj`)은 GPU에 상주하는
K-cache 요약본에만 적용되고, NVMe에 쓰이는 원본 KV는 FlexGen과 동일한 풀 사이즈입니다.

## 2. 모델 종류·파라미터 수에 따라 달라지나요? → **예, 달라집니다. 단, 비례하지는 않습니다.**

토큰당 KV 캐시 크기는 아래 3개 값에만 의존합니다 (`model_config.py:6-9`):

```
레이어당 바이트 = num_kv_heads × head_dim × 2 × itemsize
전체 바이트/토큰 = 레이어당 바이트 × num_hidden_layers
```

- **관여하는 값**: `num_key_value_heads`(GQA 그룹 수에 따라 결정), `head_dim`, `num_hidden_layers`
- **관여하지 않는 값**: `hidden_size`, `intermediate_size`(FFN/MLP 크기), `vocab_size` 등

즉 모델의 **총 파라미터 수**와 KV 캐시 크기는 직접 비례하지 않습니다. 총 파라미터의 대부분은
FFN/MLP 가중치(`intermediate_size`)와 임베딩(`vocab_size × hidden_size`)에 들어있는데, 이 값들은
KV 캐시 크기 공식에 아예 등장하지 않기 때문입니다. 반면 같은 계열이라도
- 레이어 수가 늘면(더 "깊은" 모델) 선형으로 증가
- GQA로 KV head 수를 줄이면(예: MHA→GQA 전환, `num_kv_groups` 증가) 같은 파라미터 수라도 캐시가 줄어듦
- `head_dim`이 커지면 증가

`model_config.py`가 `llama3`/`qwen2`/`qwen3` 등 모델 계열별로 HF config에서 이 값들을 뽑아오므로,
다른 모델(Qwen3-1.7B 등)을 넣으면 위 공식에 그 모델의 `num_key_value_heads`/`head_dim`/`num_hidden_layers`를
대입한 값이 나옵니다 — 코드 구조상 모델이 바뀌면 자동으로 재계산되는 값이지, 하드코딩된 상수가 아닙니다.

## 3. "스텝(step)"이 의미하는 것

`main.py`의 생성 루프는 `gen_len` 길이만큼 반복되며 (`main.py:1407` 등, `i in range(gen_len)` 형태),

- **`i = 0`**: **prefill** — 프롬프트 전체(`prompt_len`개 토큰)를 한 번에 처리해서 KV 캐시를 만드는 단계.
  루프가 아니라 단발성 처리입니다.
- **`i = 1, 2, ..., gen_len-1`**: **decode step** — 매 반복마다 새 토큰을 1개씩 autoregressive하게
  생성하는 단위입니다. 즉 "1 스텝 = 1개의 새 토큰을 생성하는 1회 반복"이며, 그 안에서
  레이어마다 predict(`speculate_attention`) → fetch(`get_load_buffer`) → compute(PagedAttention) →
  new KV 저장(rolling buffer) 이 순서대로 일어납니다.

예: `gen_len=100`이면 prefill 1회 + decode step 100회가 실행됩니다. 앞서 말한 "decode 스텝당 R/W"는
바로 이 100번의 반복 중 한 번에 실제로 오가는 디스크 I/O 양을 뜻합니다.

## 4. Prefill 단계 R/W

`main.py:558-563`(`load_cache`는 `i==0`이면 즉시 return, 즉 prefill에는 캐시 매니저가 개입하지 않음)와
`uring_io.py:139-166`(`write(..., prefill=True, prefill_mode='all_seq')`):

- **Read: 0** — prefill에는 아무 것도 읽지 않습니다(아직 캐시가 없으므로).
- **Write**: 레이어당 프롬프트 전체를 한 번에 씁니다. 크기 = `prompt_len × 4096 B`.
  - `prompt_len=16284`일 때: 레이어당 ≈63.6 MB, 28개 레이어 합계 = **≈1.74 GiB / 시퀀스** (batch=1)
  - batch_size에 선형 비례 (batch=8 → ≈13.9 GiB)

## 5. Decode 단계 R/W (스텝 1회당, 레이어별)

predictor(`methods.py`)가 매 스텝 `M=400`개 토큰(=`M/G=100`개 그룹)을 예측 선택하고,
`cache_manager.py:96-135`(`get_load_buffer`)가 이 100개 그룹을 reuse buffer(FIFO, 크기도 100그룹)와
대조해 **miss만** NVMe에서 읽습니다.

- **Read**: miss 그룹 수 × (`token_group(4) × 4096 B = 16 KB`)
  - 이론적 worst case(reuse율 0%): 레이어당 100그룹 × 16 KB = 1.6 MB → 28레이어 합계 ≈ **44.8 MB/스텝**
  - reuse율이 높을수록(직전 스텝들과 선택된 그룹이 겹칠수록) 실제 read는 훨씬 작아짐 —
    정확한 수치는 `reuse_info`(런타임 통계)에 의존하므로 코드 정적 분석만으로는 고정값을 낼 수 없습니다.
    이 부분이 KVSwap이 InfiniGen* 대비 I/O를 줄이는 핵심 포인트입니다.
- **Write**: 새로 생성된 토큰은 즉시 쓰지 않고 레이어별 rolling buffer(`cache_manager.py:73-88`)에
  쌓이다가 `G=4`개가 다 차는 시점에만 flush됩니다(`uring_io.py:194-236`).
  - flush 1회 = 레이어당 `4 × 4096 B = 16 KB` → 28레이어 = 448 KB, **4스텝마다 1번**
  - 스텝당 상각(평균) 값 = 114,688 B / 4 ≈ **28.7 KB/스텝** (토큰 1개의 전체 크기와 동일 — 결국 매 토큰이
    한 번은 기록되므로 당연한 결과)

## 6. 디스크 R/W 블록 크기

블록 크기를 결정하는 값은 `token_group`(그룹 크기 `G`)입니다. `cache_manager.py:20`의
`hdg_size = num_kv_heads * head_dim * block_size`와 `uring_io.py:62`의
`buffer_size = token_group * self.hd_bytes`가 실제 디스크 R/W 단위이며, 여기서 `block_size`는
`CacheManager` 생성자에 전달되는 `policy.token_group`(`main.py:529`)입니다.

- 레이어 1개, 토큰 1개 = 4096 B (K+V 합산, §1)
- **블록(그룹) 1개 = `token_group × hd_bytes`** — decode 단계에서 레이어 1개에 대해 issue되는
  최소 read/write 단위
- kvswap 기본값(`eval_nano.sh`, `token_group=4`) 기준: **4 × 4096 B = 16 KB/블록**

### Decode 단계: 블록 단위로 R/W

- **Read**: `cache_manager.py:96-135`의 `get_load_buffer`가 예측된 그룹 중 reuse buffer에 없는
  miss 그룹만 16 KB 단위로 읽어옵니다 (`uring_io.py:239` 이하 `read()`).
- **Write**: rolling buffer(`cache_manager.py:73-88`)에 새 토큰이 쌓이다가 `token_group`개(4개)가
  다 차야 16 KB 블록 하나로 flush됩니다 (`uring_io.py:194-236`, `write_size = self.token_group * self.hd_bytes`).

즉 디스크에는 토큰 1개 단위가 아니라 **`token_group`(기본 4)개 토큰 = 16 KB짜리 묶음**이 최소 단위입니다.
디바이스 블록 정렬도 이 크기를 기준으로 검증됩니다 (`uring_io.py:68-69`:
`buffer_size % BLOCK_DEV_SIZE(512) == 0` 어설션).

### Prefill은 예외 — 블록 단위가 아니라 통짜(bulk) 쓰기

Prefill(`uring_io.py:139-166`, `prefill_mode='all_seq'`)에서는 `token_group` 그룹 단위가 아니라
프롬프트 전체(`prompt_len`개 토큰)를 레이어당 한 번의 write로 통째로 씁니다. 그룹 단위 I/O는
decode 단계에서만 적용됩니다 (§4 참고).

### 모드별 블록 크기 비교

`token_group` 자체가 baseline마다 다르게 설정됩니다 (`eval_nano.sh`):

| 모드 | `token_group(G)` | 블록 크기 |
|---|---|---|
| flexgen | 1 | 4 KB (사실상 토큰 단위, 압축/예측 없음) |
| infinigen / infinigen_ru | 1 | 4 KB |
| infinigen_ru_gp | 4 (기본) | 16 KB |
| **kvswap** | **4 (기본)** | **16 KB** |

`token_group`을 늘리면 그룹당 I/O 요청 수는 줄어들지만(레이턴시/오버헤드 감소), 예측이 틀렸을 때
불필요하게 읽어오는 토큰(그룹 내 나머지 토큰)도 늘어나는 트레이드오프가 있습니다 — 논문 §3.5/§6.1이
말하는 `token_group`의 처리량-정확도 트레이드오프입니다.

## 7. 최소 요구 통합메모리(unified memory) — OOM 안 나는 기준선

> 목적: Orin Nano(8GB 통합메모리)에서 어느 배치/설정까지 OOM 없이 돌릴 수 있는지 코드 기반으로
> 하한선을 잡는다. 단, CUDA 컨텍스트·allocator 오버헤드·activation 피크는 정적 분석만으로 정확히
> 알 수 없으므로 이 값은 **하한선**이며, 실제 안전 기준선은 §7.4의 실측으로 확정해야 한다.

### 7.1 GPU(통합메모리)에 실제로 상주하는 항목

`--percent 100 0 0 0 100 0` (weight 100% GPU, cache 0% GPU→100% disk, activation 100% GPU,
`main.py:1673-1688`의 `Policy`) 기준, 코드에서 찾을 수 있는 GPU 상주 텐서는 4가지입니다.

| 항목 | 코드 위치 | 크기 공식 |
|---|---|---|
| 모델 가중치 | `w_gpu_percent=100`이면 전량 GPU | 파라미터 수 × 2B (fp16/bf16) |
| CacheManager paged-attention 풀 (예측·reuse된 KV만 상주 — KVSwap이 적은 메모리로 도는 핵심) | `cache_manager.py:32` `self.kv_cache = torch.zeros(2, num_blocks, num_kv_heads, head_dim, block_size, ...)` | `num_blocks = ((reuse_budget/token_group)×layer_num + layer_num) × batch_size`; 바이트 = `2 × num_blocks × num_kv_heads × head_dim × token_group × 2B` |
| lr_k 버퍼 (레이어별 저랭크 K 예측치, 시퀀스 전체 길이만큼 사전할당) | `main.py:519-523` `self.lr_k = torch.empty(prompt_len+gen_len-1, batch, rank, ...)` | `(prompt_len+gen_len-1) × batch × rank × 2B`, **레이어마다 별도 할당** (`start_layer=0-curr-emb`면 전 28개 레이어) |
| 활성화(activation) 버퍼 | `act_gpu_percent=100` | 대략 `hidden_bytes()` = `batch × seq_len × hidden_size × 2B` 오더 |

(디스크 I/O용 pinned 버퍼 — `diskio_base.py:41-77`의 `read_tensor`/`wr_tensor`/`prefill_tensor` — 는
`posix_memalign`+`cudaHostRegister`로 잡는 host RAM이지만, Jetson은 unified memory라 물리적으로 같은
풀을 씁니다. 크기는 `max_num_kv × batch × hd_bytes` 오더로 위 4가지보다 작습니다.)

### 7.2 Qwen3-0.6B, 16K ctx, kvswap 기본값으로 대입한 예시

`token_group=4`, `max_num_kv=reuse_budget=400`, rank=128(`lowrank_proj_..._mh_1/lr_kproj_*.pt` shape `[1024,128]`, 실측 확인):

- **가중치**: fp16 파일 실측 크기 = **1.2 GB** (고정, batch 무관)
- **CacheManager 풀**: `reuse_blk_num=100`, `num_blocks/batch=100×28+28=2828` → `2×2828×8×128×4×2B ≈ 44.2 MB/batch`
- **lr_k 버퍼**: 레이어당 `16383×128×2B≈4.0MB/batch`, 28개 레이어 합 ≈ **117.6 MB/batch**
- **activation**: `hidden_bytes` 오더로 ≈ **34 MB/batch**

| batch | GPU 텐서 합 (가중치+캐시+lr_k+act) | CUDA 컨텍스트(≈300~500MB)·allocator 오버헤드 포함 추정 |
|---|---|---|
| 1 | 1.2 + 0.044+0.118+0.034 ≈ **1.4 GB** | ≈1.8~2.0 GB |
| 4 | 1.2 + 4×0.196 ≈ **2.0 GB** | ≈2.4~2.6 GB |
| 8 | 1.2 + 8×0.196 ≈ **2.8 GB** | ≈3.2~3.6 GB |

### 7.3 OOM 안 나는 기준선 제안

Orin Nano의 8GB는 GPU/CPU/컴파일이 전부 공유하는 통합메모리이고, OS·jtop_logger·기타 프로세스가
이미 일부(보통 1~1.5GB 안팎)를 점유합니다. 위 추정치(batch=8에서 ≈3.2~3.6GB)를 감안하면:

- 이 문서 기준 설정(Qwen3-0.6B, 16K ctx, kvswap 기본값)은 **batch=1~8 전 구간에서 이론상 여유가
  충분**하고(8GB 중 최대 ~45% 사용), 이게 논문 Table 5가 이 조합(Qwen3-0.6B/1.7B, 16K, batch 1~8)을
  Orin Nano 기준으로 잡은 이유와도 맞아떨어집니다.
- 압박이 커지는 방향은 (a) `max_num_kv`/`reuse_budget`을 크게 올리는 경우(CacheManager 풀이
  선형으로 커짐), (b) context 길이를 늘려 `prompt_len+gen_len`이 커지는 경우(lr_k 버퍼가 선형으로
  커짐 — rank가 큰 어댑터(ratio=1)일수록 특히), (c) `gpu_batch_size`를 키우는 경우(위 표의 모든
  항목이 배치에 비례) 입니다. `lr_proj_mode=none`(FlexGen 순정)이나 `ratio` 낮은 어댑터(0.25)를 쓰면
  lr_k 버퍼가 그만큼 줄어듭니다.
- 실전 안전 마진: 정적 하한선 위에 OS/기타 프로세스 점유분(≈1~1.5GB)과 여유분(10~20%)을 더 얹어서
  판단하는 것을 권장합니다 — 즉 위 표의 "CUDA 컨텍스트 포함 추정" 값에 다시 +1~1.5GB 정도를 더한
  값이 실제 8GB 중 남는 공간과 비교할 기준선입니다.

### 7.4 정적 추정의 한계 — 실측으로 확정하기

CUDA 컨텍스트 로드 자체, allocator 단편화, `main.py`의 이중버퍼(`weight_read_buf`,
`cache_read_buf` 등 `ValueHolder`)는 코드만 읽어서는 정확한 바이트 수를 특정하기 어렵습니다.
그래서 정적 계산은 **하한선**으로만 쓰고, 실제 OOM 기준선은 실측으로 확정해야 합니다. §8에서
이 실측 계측을 `main.py`/`analyze_nano_results.py`에 실제로 구현했습니다.

## 8. 프로세스 단위 Peak 메모리 계측 (구현됨)

`main.py`는 원래 프로세스별 메모리를 전혀 계측하지 않았고(§ "무엇이 로깅되나" 참고,
`eval_nano.sh`의 `jtop_logger.py`는 **시스템 전체** RAM만 기록), `torch.cuda`의 자체 통계도
쓰지 않고 있었습니다. 이번에 코드 두 곳에 계측을 추가했습니다.

### 8.1 왜 두 가지 지표를 같이 재는지

디스크 오프로딩 데이터가 GPU 메모리로 들어오는 경로를 코드로 추적해보면, **경로에 따라
torch.cuda 통계에 잡히는지 여부가 갈립니다**:

| 구간 | torch.cuda에 잡히는가 |
|---|---|
| NVMe → pinned host 버퍼 (`diskio_base.py:63-80`의 `read_tensor`, `posix_memalign`+`cudaHostRegister`로 만든 raw ctypes 메모리를 `torch.from_numpy`로 감싼 CPU 텐서) | **안 잡힘** — torch의 CUDA caching allocator를 거치지 않음 |
| pinned 버퍼 → CacheManager 상주 GPU 버퍼, `reuse_budget=0` 경로 (`pytorch_backend.py:1042`, pinned 소스에서 이미 할당된 GPU 목적지로 직접 `copy_(non_blocking=True)`) | **안 잡힘** — 새 할당 없이 기존 버퍼에 덮어씀 |
| pinned 버퍼 → CacheManager 상주 GPU 버퍼, `reuse_budget>0` 경로 (`pytorch_backend.py:1039`, `output[g].cuda()`) — **kvswap/infinigen_ru/infinigen_ru_gp 기본 설정이 바로 이 경우** | **잡힘** — boolean mask scatter 전에 명시적으로 `.cuda()` 호출해서 임시 GPU 텐서 생성 |
| ShadowKV baseline (`shadowkv/models/disk_cache.py:121`, `.cuda()` 없이 pinned→상주 GPU 버퍼로 바로 `copy_()`) | **안 잡힘** |

즉 torch.cuda 통계만 보면 pinned staging 버퍼 자체와(모드에 따라서는) 오프로딩 read 경로 상당 부분을
놓칩니다. 그래서 커널이 추적하는 프로세스 전체 peak RSS(`VmHWM`, 방법1)와 torch.cuda의 자체 peak
통계(방법2)를 **함께** 로그에 남기도록 구현했습니다.

### 8.2 `engine/src/main.py` 변경

- `get_peak_rss_kb()` 헬퍼 추가 (임포트 직후, `main.py` 상단): `/proc/self/status`의
  `VmHWM`(커널이 프로세스 시작부터 계속 추적하는 peak RSS, 별도 샘플링 루프 불필요)을 읽어 KB로 반환.
- `run_flexgen()`의 기존 `Throughput Total: ...` 출력 직후에 아래 한 줄을 추가로 출력:
  ```
  Peak Memory (GB) RSS: 1.842 TorchAllocated: 1.401 TorchReserved: 1.520
  ```
  - `RSS`: `get_peak_rss_kb()` — 프로세스 전체 실측 peak (pinned diskio 버퍼, CUDA 컨텍스트,
    파이썬 오버헤드까지 전부 포함, GPU 없는 실행이면 `n/a`)
  - `TorchAllocated`: `torch.cuda.max_memory_allocated()` — torch가 `cuda` 디바이스에 할당한
    텐서(가중치, CacheManager 풀, lr_k 버퍼, activation 등, §7.1 4가지 항목)의 peak 합
  - `TorchReserved`: `torch.cuda.max_memory_reserved()` — allocator가 실제로 확보(캐싱 포함)한
    풀 크기. `RSS`와 `TorchReserved`의 차이가 대략 pinned staging 버퍼 + CUDA 컨텍스트 몫입니다.

기존 `Throughput Total`/`Latency Total` 로그 라인과 같은 파일, 같은 run 안에 이어서 찍히므로
`eval_nano.sh`가 만드는 `<run>.log` 하나만 보면 처리량과 peak 메모리를 같이 확인할 수 있습니다.

### 8.3 `engine/scripts/analyze_nano_results.py` 변경

- `RE_PEAKMEM` 정규식 추가: 위 `Peak Memory (GB) RSS: ... TorchAllocated: ... TorchReserved: ...`
  라인을 파싱.
- `parse_engine_log()`(main.py/ShadowKV 로그를 dict 하나로 만드는 함수)에서, main.py 로그일 때만
  `peak_rss_gb`/`peak_torch_alloc_gb`/`peak_torch_reserved_gb` 세 컬럼을 row에 추가. (ShadowKV·vLLM은
  이 계측이 없는 별도 코드 경로라 해당 컬럼이 비어(NaN) 나옵니다.)
- 별도 배관(plumbing) 없이 `pd.DataFrame(rows)`가 새 키를 자동으로 컬럼화하므로, **기존
  `summary.csv`에 이 세 컬럼이 그대로 추가**됩니다 — `decode_tps`/`swap_avg_*`/`latency_*` 등
  기존 통계와 나란히, 같은 행(같은 run)에 기록됩니다. `jtop_summary.csv`의 시스템 전체
  `ram_used_peak_gb`와 대조해보면 "이 프로세스만의 peak(RSS)" vs "시스템 전체 peak"의 차이(다른
  프로세스 점유분)도 바로 비교됩니다.

### 8.4 사용법 (실험은 사용자가 직접 실행)

```bash
cd engine
bash scripts/eval_nano.sh kvswap 16384 "1 2 4 8"     # 기존과 동일하게 실행
# 로그에 Peak Memory 줄이 자동으로 남음 (코드 변경 외 추가 조작 불필요)

.venv/bin/python scripts/analyze_nano_results.py --model Qwen3-0.6B
# summary.csv에서 peak_rss_gb / peak_torch_alloc_gb / peak_torch_reserved_gb 확인
```

`reuse_budget=0`(예: `infinigen` 모드, `--reuse_budget 0`)과 `reuse_budget>0`(kvswap 기본) 두 설정을
같은 batch로 돌려서 `peak_torch_alloc_gb`를 비교하면, §8.1에서 설명한 "`.cuda()` 경로가 실제로
torch 통계에 잡히는지"를 직접 확인할 수 있습니다 — `reuse_budget>0`일 때만 `peak_torch_alloc_gb`가
CacheManager 풀 정적 추정치(§7.2)보다 눈에 띄게 더 크게 나오면, 그 초과분이 scatter 경로의
임시 `.cuda()` 할당이 peak에 기여했다는 뜻입니다.

## 요약 표

| 항목 | 크기 |
|---|---|
| 토큰 1개, 전 레이어 K+V 디스크 점유 | 112 KB (4 KB/레이어 × 28) |
| 모델/파라미터 의존성 | `num_kv_heads`, `head_dim`, `num_hidden_layers`에만 의존 — 총 파라미터 수와 비례하지 않음 |
| "스텝" 정의 | decode 루프 1회 반복 = 새 토큰 1개 생성 (prefill은 스텝이 아니라 별도 1회성 처리) |
| Decode R/W 블록 크기 (kvswap 기본값) | 16 KB/블록 (`token_group(4) × 4096 B`, 레이어당) |
| Prefill write (16K ctx, batch=1) | ≈1.74 GiB, read 없음 (블록 단위 아닌 통짜 write) |
| Decode read/스텝 (miss 기준, worst case) | ≤44.8 MB/스텝 (실제는 reuse율만큼 감소) |
| Decode write/스텝 (상각) | ≈28.7 KB/스텝 (4스텝마다 448 KB 실제 flush) |
| 최소 통합메모리 (16K ctx, kvswap 기본값, 정적 하한선) | batch=1 ≈1.4GB, batch=8 ≈2.8GB (컨텍스트 오버헤드 별도, §7 참고) |
| 프로세스 단위 peak 메모리 실측 | `main.py`가 run마다 `Peak Memory (GB) RSS/TorchAllocated/TorchReserved` 로그, `summary.csv`에 자동 반영 (§8, 구현 완료 — 실행은 사용자가 직접) |
