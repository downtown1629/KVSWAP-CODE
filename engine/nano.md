# Jetson Orin Nano(8GB, NVMe 전용)에서 실행하기

이 디렉터리의 `README.md`는 논문이 목표로 하는 플랫폼인 **Jetson Orin AGX**(64GB 통합 RAM, eMMC와
NVMe 모두 탑재)를 기준으로 문서화되어 있다. `scripts/setup.sh`는 이 하드웨어 조건을 직접
강제한다 — `/proc/device-tree/model`을 읽어서 `"AGX Orin"` 문자열이 포함되어 있지 않으면
종료한다 — 그리고 `scripts/eval.sh`와 `fig-*.sh`/`tab-4.sh` 스윕들은 기본값으로 그 머신에
맞춰진 모델과 배치당 KV 예산을 사용하며, `EMMC_OFFLOAD_DIR`/`EMMC_DEV_NAME`과
`NVME_OFFLOAD_DIR`/`NVME_DEV_NAME`을 무조건 모두 요구한다.

이 중 어느 것도 **RAM 8GB에 eMMC가 없는 Jetson Orin Nano**에는 맞지 않는다. 이 문서는 대신 그
머신을 위해 작성된 `scripts/*_nano.sh` 변형본들을 다룬다. 이들은 기존 파일을 수정한 것이 아니라
새로 추가된 파일이다 — `setup.sh`/`eval.sh` 등은 그대로 유지되며 여전히 AGX Orin 경로를
설명한다.

이 종류의 하드웨어에 대한 논문 자체의 참조 지점은 **§5.2.2 / Table 5**다: Orin Nano, NVMe
전용, **Qwen3-0.6B와 Qwen3-1.7B**, 컨텍스트 길이 16K, 배치 크기 1–8. 여기 있는 스크립트들은
기본적으로 이 설정을 사용하며, 실제로 이 장치에서 실행되어 Qwen3-0.6B에 대해 배치 1–4 구간에서
Table 5의 flexgen/InfiniGen*/KVSwap/ShadowKV/vLLM 베이스라인을 재현한 바 있다(배치 8은
디스크 오프로딩 모드들에 있어 8GB에서 여유가 빠듯하다 — 아래 "알려진 한계" 참고).

## AGX Orin 스크립트와의 차이점

| | AGX Orin (`scripts/*.sh`) | Orin Nano (`scripts/*_nano.sh`) |
|---|---|---|
| 하드웨어 확인 | device-tree 모델에 `"AGX Orin"`이 없으면 강제 종료 | 감지된 모델을 출력하고 인식되지 않으면 경고만; `STRICT_HW_CHECK=1`일 때만 치명적 오류 |
| 디스크 | eMMC **와** NVMe 둘 다 필수 | NVMe만 사용 — `EMMC_*` 변수 불필요 |
| 기본 모델 | Llama-3.1-8B-Instruct / Qwen3-14B | Qwen3-0.6B |
| 전원 모드 / 클록 확인 | `nvpmodel`이 정확히 `MAXN`을 보고하지 않거나 CPU/GPU devfreq가 하드코딩된 sysfs 경로에 고정되어 있지 않으면 강제 종료 | 확인한 값을 출력하고 종료 대신 경고만; `eval_nano.sh`가 각 배치 실행 전 `sudo jetson_clocks`를 직접 추가로 실행(권고 사항 — 비밀번호 없는 sudo가 필요하며, 없으면 클록은 그대로 둠) |
| 어댑터 | 저장소에 `engine/data/adapters/`로 포함되어 배포됨 | Qwen3-0.6B/1.7B용이 `$MODEL_PATH_BASE/local_adapters`에 배포됨(저장소에 커밋됨, 아래 참고); `prepare_adapter_nano.sh`로 추가 생성 가능 |
| vLLM 베이스라인 | `run_vllm.sh`가 `setup.sh`처럼 강제 종료; `run_vllm.py`는 `gpu_memory_utilization=0.85`/`max_model_len=32768`을 하드코딩 | `run_vllm_nano.sh`; 두 값 모두 환경변수로 오버라이드 가능(`VLLM_GPU_MEM_UTIL`/`VLLM_MAX_MODEL_LEN`) — AGX 기본값은 8GB에 맞지 않기 때문 |
| ShadowKV 베이스라인 | `src/shadowkv/run_shadowkv.sh`가 `setup.sh`처럼 강제 종료 | `run_shadowkv_nano.sh`, 동등한 소프트 체크, 동일한 `test/e2e_jetson.py` 드라이버 사용 |
| 리소스 로깅 | 없음 | `eval_nano.sh`가 `scripts/jtop_logger.py`를 자동으로 연결 — 실행당 초당 1회 CPU/GPU/EMC/RAM/전력/디스크 I/O 샘플 |

## 파일 구성

- **`scripts/nano_common.sh`** — 다른 스크립트들이 공유하며 소스로 불러온다.
  `check_hardware_soft`, `check_powermode_soft`, `check_jetson_clocks_soft`를 정의한다 —
  `setup.sh`/`eval.sh`에 있는 검사들의 권고용(advisory) 버전이다. 직접 실행하는 용도가 아니다.
- **`scripts/download_models_nano.sh`** — `Qwen/Qwen3-0.6B`를 `$MODEL_PATH_BASE_HF`로
  가져온다(Qwen3-1.7B도 존재하지만 주석 처리되어 있음 — 논문의 두 번째 Nano 데이터 포인트를
  위해서는 주석을 해제).
- **`scripts/setup_nano.sh`** — `setup.sh`와 동일한 wheel/의존성 설치 과정을 수행하지만
  (`wheel_pkgs/`의 사전 빌드된 aarch64 wheel들은 AGX 전용이 아니라 JetPack/CUDA 버전에
  종속되므로 여기서 달라지는 부분은 없다), 디스크 설정은 NVMe만 마운트/확인하며, 가중치 변환
  (`scripts/make_np_weights.py`, 모델과 무관하게 항상 fp16으로 저장함)은 `NANO_MODEL_LIST`를
  통해 기본적으로 Qwen3-0.6B를 대상으로 한다.
- **`scripts/prepare_adapter_nano.sh`** — `engine/data/adapters/`에 아직 없는 모델을 위해
  KVSwap 저랭크(low-rank) 어댑터(그리고 `--with-infinigen` 옵션 시 InfiniGen* skew 어댑터도)를
  생성한다. `quality/src/prepare_adapter.py` — 논문 자체의 오프라인 어댑터 튜닝 코드(PAPER.pdf
  §3.5) — 를 *이* 디렉터리의 `.venv`를 통해 실행한다. `quality/scripts/install.sh`는 Jetson에서
  전혀 동작하지 않는 x86_64 PyPI torch wheel을 받아오기 때문이다. `$MODEL_PATH_BASE/local_adapters`
  에 기록되며, 이는 `link_adapters.sh`가 심볼릭 링크로 연결하는 Git-LFS 관리 트리인
  `engine/data/adapters/`와는 별개의 트리이지만(그 트리와 달리) `engine/data/adapters/`의
  자체 `.pt` 파일들과 마찬가지로 순수 바이너리로 여기에 커밋되어 있다. 현재 Qwen3-0.6B(KVSwap
  저랭크 비율 1.0/0.25 + InfiniGen* skew 비율 0.125)와 Qwen3-1.7B(KVSwap 저랭크 비율
  1.0/0.25)에 대해 채워져 있다.
- **`scripts/eval_nano.sh`** — `src/main.py`용 NVMe 전용 드라이버로, 다섯 가지 모드가 있다:
  - `flexgen` — 예측 없는 풀-KV 베이스라인, **어댑터 불필요**. 가장 먼저 실행할 것.
  - `infinigen` — InfiniGen* 스타일의 인덱스 선택형 예측기. skew 어댑터가 필요
    (`prepare_adapter_nano.sh --with-infinigen`).
  - `infinigen_ru` / `infinigen_ru_gp` — 동일한 InfiniGen* 예측기에 KVSwap의 재사용 버퍼
    (`+ru`)를 추가하고, 그 위에 그룹 I/O(`+ru+gp`)까지 추가한 것 — 논문의 InfiniGen*/+ru/+ru+gp
    ablation 체인(§4.2)을 `kvswap` 모드가 쓰는 것과 동일한 `--reuse_budget`/`--token_group`
    플래그로 재현한다(두 플래그 모두 `main.py`에 일반적인 것이며 KVSwap 전용이 아니다).
    `infinigen`과 동일한 skew 어댑터가 필요하다.
  - `kvswap` — 실제 KVSwap 저랭크 예측기. KVSwap 어댑터 필요
    (`prepare_adapter_nano.sh`, 별도 플래그 불필요).

  또한 각 배치 실행 전에 `sudo jetson_clocks`를 적용하고(건너뛰려면 `APPLY_JETSON_CLOCKS=0`
  설정), 배치 사이에 페이지 캐시를 비우며(권고 사항, 비밀번호 없는 sudo 필요), 각 실행 전후로
  `scripts/jtop_logger.py`를 시작/중지한다(아래 참고) — jetson-stats가 기본 경로
  `/home/jetson/.local/share/jtop/bin/python`에 없다면 `JTOP_PY`로 경로를 오버라이드할 것.
- **`scripts/run_vllm_nano.sh`** — `scripts/run_vllm.sh`의 소프트 체크 버전. vLLM "오프로딩
  없음" 베이스라인을 8GB에 맞추려면 `VLLM_GPU_MEM_UTIL`/`VLLM_MAX_MODEL_LEN`을 설정할 것
  (AGX 기본값인 0.85/32768은 여기서 시작에 실패한다); `VLLM_MAX_MODEL_LEN`의 기본값은 테스트 중인
  가장 큰 seqlen이다.
- **`scripts/run_shadowkv_nano.sh`** — `src/shadowkv/run_shadowkv.sh`의 소프트 체크 버전.
  ShadowKV CUDA 확장이 미리 빌드되어 있어야 한다
  (`cd src/shadowkv && MAX_JOBS=1 python setup.py build_ext --inplace` — `MAX_JOBS=1`은 8GB
  통합 메모리에서 중요하다, "알려진 한계" 참고). `budget`/`chunk_size`/`rank`는 논문이 Orin
  Nano/Qwen3-0.6B에 특화된 값을 공개하지 않으므로 시작점으로서 AGX 스윕 스크립트
  (`tab-4.sh`/`fig-10.sh`)에서 복사한 값을 기본값으로 사용한다.
- **`scripts/jtop_logger.py`** — `engine/.venv`가 아니라 jetson-stats 자체의 venv 하에서
  실행되어야 한다(`jtop`을 임포트 가능한 모듈로 제공하는 곳이 거기뿐이다). 한 프로세스 안에서
  1Hz로 동작하는 세 개의 샘플링 루프(이 jetson-stats 버전에서 jtop의 *클라이언트*는 1.0초
  미만에서 불안정하다 — 한 번 샘플링하고 멈춰버린다): `jtop`을 통한 CPU/GPU/RAM/SWAP/전력;
  `/proc/diskstats`에서 직접 읽는 디스크 읽기/쓰기 IOPS/처리량/큐 깊이/`%util`(jtop 자체
  API는 디스크 *용량*만 노출하고 I/O는 노출하지 않는다); 그리고 `sudo tegrastats`를 통한 EMC
  대역폭 %(jtop 자체의 `EMC` 통계값은 이 보드에서 잘못되어 있다 — 아래 알려진 한계 참고).
  `<run>.jtop.csv` + `<run>.diskio.csv`를 기록한다.
- **`scripts/analyze_nano_results.py`** — `eval_nano.sh`/ShadowKV/vLLM 로그와 jtop+diskio
  CSV들을 하나의 비교 결과로 파싱한다: 메서드/배치별 디코드 전용 처리량, 엔진 자체의 `Swap:`
  로그 라인으로부터 얻는 레이어당 디스크 비용, prefill/decode 구간별 리소스 사용량, 디스크 I/O
  ablation 차트. `.venv/bin/python scripts/analyze_nano_results.py`로 실행(`pandas`/
  `matplotlib` 필요, 이미 `engine/.venv`에 있음); 기본적으로 CSV와 PNG를 `RESULTS/nano/`에
  기록한다.

## 사용법

```bash
cd engine

# 1. 평소대로 환경 변수를 설정하되, eMMC 관련 변수는 제외. `.env_nano`(이 체크아웃에만 있는,
#    Git에 커밋되지 않은 로컬 파일)에 이 머신용 값이 이미 채워져 있으니 직접 export하는 대신 이걸
#    source할 것:
source .env_nano
# .env_nano 내용:
#   export NVME_DEV_NAME='nvme0n1p1'
#   export NVME_OFFLOAD_DIR='/home/jetson/Downloads/KVSWAP-CODE/data/nvme_offload'
#   export MODEL_PATH_BASE_HF='/home/jetson/Downloads/KVSWAP-CODE/data/model_weights_hf'
#   export MODEL_PATH_BASE='/home/jetson/Downloads/KVSWAP-CODE/data/model_weights'
#   export EVAL_LOG_DIR='/home/jetson/Downloads/KVSWAP-CODE/data/kvswap_logs'
#   export EVAL_USER='test0'
#   export INTERACTIVE_PROMPT=0

# 2. 가중치 다운로드, venv + NVMe 마운트 + fp16 np-weights 설정
bash ./scripts/download_models_nano.sh
bash ./scripts/setup_nano.sh

# 3. 정상 동작 확인: 어댑터가 필요 없는 풀-KV 베이스라인
bash ./scripts/eval_nano.sh flexgen

# 4. Qwen3-0.6B/1.7B용 어댑터는 이미 $MODEL_PATH_BASE/local_adapters에 포함되어 있음 — 아직
#    없는 모델/비율에 대해서만 필요:
bash ./scripts/prepare_adapter_nano.sh          # InfiniGen* 베이스라인도 필요하면 --with-infinigen 추가

# 5. KVSwap 자체 실행
bash ./scripts/eval_nano.sh kvswap

# 6. main.py 외부의 베이스라인들
bash ./scripts/run_vllm_nano.sh                     # vLLM, 오프로딩 없음
bash ./scripts/run_shadowkv_nano.sh                 # ShadowKV(먼저 CUDA 확장을 빌드할 것, 위 참고)

# 7. 한 묶음의 실행이 완료되면 로그를 비교:
.venv/bin/python scripts/analyze_nano_results.py
```

`eval_nano.sh`는 위치 인자를 받는다: `<mode> [total_len] ["batch_list"] [model]`, 예:

```bash
bash ./scripts/eval_nano.sh kvswap 16384 "1 2 4 8"
RATIO=0.25 bash ./scripts/eval_nano.sh kvswap 32768 "1 4"   # 예산이 빠듯한 어댑터
bash ./scripts/eval_nano.sh infinigen_ru_gp 16384 "1 2"     # InfiniGen* + 재사용 버퍼 + 그룹 I/O
```

로그는 `$EVAL_LOG_DIR/$EVAL_USER/logs/nano/<model>/` 아래에 `(mode, batch, context)` 조합당
파일 하나씩 저장된다; 재실행 시 이미 `Throughput Total:` 라인이 있는 로그는 건너뛴다 —
`scripts/eval.sh`와 동일한 규칙이다. jetson-stats가 설치되어 있다면 각 로그마다 `.jtop.csv` +
`.diskio.csv` 쌍도 함께 생성된다(위 `scripts/jtop_logger.py` 참고).

## 알려진 한계 / 확인해야 할 사항

이 스크립트들은 실제 Orin Nano에서 실행된 바 있지만, 이 보드의 8GB **통합(unified)** 메모리
(CPU, GPU, 페이지 캐시가 하나의 풀을 공유)로 인해 결과를 신뢰하거나 무거운 작업을 실행하기
전에 알아둘 만한 몇 가지가 있다:

- **전원 모드 / 클록.** `nano_common.sh`의 검사들은 확인한 값을 출력할 뿐 실행을 막지 않는다;
  `eval_nano.sh`는 각 배치 전에 별도로 `sudo jetson_clocks`를 직접 실행한다(권고 사항 —
  비밀번호 없는 sudo가 없으면 조용히 건너뛰며, 이 경우 클록이 낮은 유휴 상태에 머물러 처리량
  수치가 대표값보다 낮고 노이즈가 심해질 수 있다). 실행 결과의 `.jtop.csv`를 확인할 것 —
  `gpu_freq_khz`가 낮은 쪽으로 흘러내리지 않고 최댓값 근처에 고정되어 있으면 그 실행에서 클록이
  실제로 유지되었음을 확인할 수 있다. `jetson_clocks`만으로 충분하지 않다면 자신의
  Nano/캐리어 보드/JetPack 버전에 맞는 올바른 최대 성능 `nvpmodel` 모드 id를 직접 확인할 것
  (`sudo nvpmodel -q --verbose`, `sudo nvpmodel -m <id>`).
- **메모리 여유.** 모델, KV 예측기, (그리고 `kvswap`/`infinigen_ru*`의 경우) 재사용 버퍼가 모두
  상주하게 되면 8GB는 금방 소진된다 — `eval_nano.sh`가 배치 사이에 페이지 캐시를 비우는 것도
  이 때문이다. 배치 8은 이 보드의 모든 디스크 오프로딩 모드에서 빠듯하다; `flexgen`(오프로딩
  전혀 없음)이 배치가 커질수록 가장 먼저 메모리가 바닥나는데, 그렇게 된다면 이는 버그가 아니라
  KVSwap이 필요한 이유를 보여주는 예상된 동작이다. 실행 결과의 `.jtop.csv`의
  `ram_free_kb`/`swap_used_kb` 열을 보면 특정 설정이 한계에 얼마나 가깝게 동작하는지 알 수
  있다.
- **`io_uring`의 잠금 메모리(locked-memory) 한도.** 각 `DiskIO` 인스턴스는 고정(pinned)
  스테이징 버퍼를 커널에 등록하며(`ulimit -l`), `--reuse_budget`이 클수록 더 많은 버퍼가
  필요하다. 이 한도를 초과하면 `io_uring_queue_init_sqpoll`이 백그라운드 워커 스레드 내부에서
  `BlockingIOError: [Errno 11]`로 실패하는데, 이 실패가 현재 메인 프로세스로 전파되지 않기 때문에
  오류를 내는 대신 실행이 (CPU 사용률 거의 0, 디스크 I/O 0인 채로) 무한정 멈춰버린다. 처리량
  라인이 나타나지 않고 실행이 멈춘 것처럼 보인다면, 그냥 느린 것으로 단정하기 전에 로그에서 정확히
  이 현상이 있는지 확인할 것; `ulimit -l`을 올리거나 `--reuse_budget`/배치 크기를 낮추는 것이
  두 가지 해결 레버다.
- **`prepare_adapter.py`의 비율(ratio) 포맷팅 특이사항.** `--ratios`를 `float()`로 파싱하기
  때문에 비율 `1`은 디렉터리 접미사 `_1.0`이 되며, `engine/data/adapters/`에 이미 배포된
  어댑터들의 `_1`과는 다르다(`main.py`는 `--lr_proj_path`를 글로빙 없이 정확한 문자열
  연결로 로드하므로, 접미사가 맞지 않으면 나중에 조용히 실패한다). `prepare_adapter_nano.sh`는
  이를 우회하는 심볼릭 링크(`normalize_ratio_dirs`)를 만들며, 배포된 `local_adapters/*_mh_1`
  어댑터들에도 이미 동일한 심볼릭 링크가 적용되어 있다 — 하지만 `quality/src/prepare_adapter.py`를
  직접 호출하거나 새 비율을 수동으로 추가한다면 이 점을 주의할 것.
- **NVMe 여유 공간 최솟값**(`setup_nano.sh`의 `NVME_MIN_FREE_GB`, 기본값 20)과
  **`MAX_ALLOC_KV_SIZE`**(`eval_nano.sh`의 기본값, 디스크상 KV 파일당 1GiB)는 Qwen3-0.6B/1.7B,
  배치 ≤ 8, 컨텍스트 ≤ 32K를 기준으로 한 보수적인 추정치다 — 둘 다 디스크 공간 관련 값이지
  (NVMe는 여유 공간이 충분하다) RAM 관련 값이 아니다, 하지만 더 큰 배치/컨텍스트에서
  `create_kv_file`의 `total_bytes <= MAX_ALLOC_KV_SIZE` assertion에 걸린다면 값을 올릴 것.
- **jtop 자체의 `EMC` 필드는 이 보드에서 잘못된 값을 반환한다 — `jtop_logger.py`는 대신
  `sudo tegrastats`에서 EMC%를 가져온다.** jetson-stats의 `read_emc()`(`jtop/core/memory.py`,
  7.1.5와 7.2.0 버전 모두에서 확인됨)는 원시 `/sys/kernel/debug/bpmp/debug/actmon/mc_all_avg_activity`
  카운터에 대해 `*100`도 없고 단위 환산도 없이 `utilization // emc['cur']`를 계산한다
  (`utilization`은 스케일링되지 않은 값이고 `emc['cur']`는 kHz 단위다), 그래서 현실적인 부하
  수준에서도 항상 `0`으로 내림 처리된다 — 직접 측정(부하와 무관하게 모든 샘플에서 jtop의 `EMC`
  통계값이 `0`으로 읽힘)과 동일한 부하에 대해 실제로 변화하는 `EMC_FREQ%`를 보고하는
  `sudo tegrastats`/Jetson Power GUI와의 교차 확인 양쪽 모두로 검증되었다. 이는 한 줄로 고칠
  수 있는 문제도 아니다: Orin/T234에서 그 activity 카운터는 별도의 Cortex-R5 코프로세서에서
  동작하는 BPMP 펌웨어(클로즈드 소스 블롭)가 생성하는 것이지 리눅스 커널 드라이버가 아니기
  때문이다(NVIDIA의 공개 R36.5 `kernel_src.tbz2`로 확인했으며, 여기서
  `drivers/firmware/tegra/bpmp-debugfs.c`는 activity 카운터 계산 로직이 전혀 없는 범용
  debugfs-to-BPMP-MRQ 패스스루일 뿐이다), 따라서 올바른 변환식은 공개된 소스코드로부터
  재구성할 수 없다. `tegrastats`는 이미 이 값을 올바르게 계산하므로, `jtop_logger.py`의
  `emc_loop`는 `sudo tegrastats --interval <ms>`를 셸아웃으로 실행하고(이 장치는 비밀번호 없는
  sudo가 설정되어 있음) 거기서 `EMC_FREQ`를 파싱하여, `emc_pct` 열에서 jtop 자체의(여전히
  버그가 있는) 값을 덮어쓴다; `emc_freq_khz`는 단순한 sysfs 직접 읽기라 이미 정확했으므로
  jtop에서 그대로 가져온다. 여기서 사용하는 다른 모든 jtop 필드(CPU/GPU/RAM/전력)는
  `tegrastats`와 한 줄씩 교차 확인하여 일치함을 확인했으므로, 이 우회 처리가 필요한 것은
  `EMC`뿐이다.
  jetson-stats의 클라이언트도 여기서는 1.0초보다 빠르게 안정적으로 샘플링할 수 없다;
  `jtop(interval=<1.0)` 요청은 정확히 한 번 샘플을 전달한 뒤 멈춰버리므로, `jtop_logger.py`는
  더 세밀한 간격 대신 1.0초로 고정되어 있다.
