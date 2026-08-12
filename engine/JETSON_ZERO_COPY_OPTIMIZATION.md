# Jetson Zero-Copy Optimization Notes

## 목적과 범위

이 문서는 Qwen3-MoE M1의 resident expert weight를 Jetson 통합 메모리에서
중복 적재하지 않는 실험 방향을 기록한다. 기존 `experiment/orin-nano`와
KVSwap 실행 경로는 baseline으로 보존한다. 구현은
`feature/qwen3-moe-m1`의 opt-in backend로만 추가하며, 검증 전에는 기본값으로
활성화하지 않는다.

여기서 zero-copy는 NVMe 데이터를 연산 없이 사용하는 것이 아니라, CPU와 iGPU가
동일한 pinned/registered allocation을 공유하여 별도 CUDA resident copy를 만들지
않는 것을 뜻한다.

## Orin 지원 현황

Orin Nano 실장치(CC 8.7, CUDA 12.6, L4T R36.5.2)에서 확인한 attribute는
다음과 같다.

```text
integrated=1
canMapHostMemory=1
hostRegisterSupported=1
managedMemory=1
pageableMemoryAccess=0
concurrentManagedAccess=0
```

따라서 `cudaHostAlloc`/`cudaMallocHost`, `cudaHostRegister`,
`cudaHostGetDevicePointer`는 후보가 될 수 있다. 일반 `mmap`과 pageable CPU
tensor는 등록 없이 GPU가 직접 접근할 수 없다. Managed memory는 사용할 수 있지만
Orin에서 concurrent managed access와 `cudaMemPrefetchAsync`를 기대할 수 없으며,
coherency 처리의 지연 변동도 고려해야 한다. Sysmem full coherency는 Thor부터
지원된다. 자세한 제약은 NVIDIA의
[CUDA for Tegra](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html)와
[CUDA Runtime Memory API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html)를 따른다.

GPUDirect Storage는 사용하지 않는다. 설치된 `gdscheck`에서 Orin GPU와 NVMe가
`Unsupported`로 판정됐고 cuFile은 compatibility mode만 제공했다. 이는 direct
NVMe-to-GPU 경로가 아니라 CPU staging fallback이다.

## 제안 구현

`mapped_host_experimental` backend는 C++ CUDA extension에서 mapped pinned
allocation과 device alias를 만들고, PyTorch `CUDAPluggableAllocator` 및 scoped
`MemPool`을 통해 expert bank에만 적용한다. 기존 `resident_device` backend는
그대로 유지한다. Python `torch.frombuffer`로 CUDA pointer를 감싸거나 전역 CUDA
allocator를 교체하지 않는다.

Pinned/registered memory는 Orin GPU에서 uncached이므로 hot expert weight의 반복
GEMM이 기존 CUDA device allocation보다 느릴 수 있다. 기능 지원만으로 채택하지
않고 반드시 측정한다.

## 적용 Gate

1. tiny fixture에서 BF16 tensor lifetime, alignment, pointer alias와 해제를 검증한다.
2. router ID, expert output 및 greedy token이 `resident_device`와 일치해야 한다.
3. 반복 GEMM latency, prefill/decode throughput, EMC bandwidth, RSS 및 Torch 밖의
   allocation을 함께 측정한다.
4. OOM 또는 CUDA 오류가 발생해도 다음 실행에서 기본 backend로 복구되어야 한다.
5. 성능이나 peak memory가 개선되지 않으면 실험 backend로만 남기거나 제거한다.

대형 Qwen3-30B-A3B로 검증하지 않는다. tiny fixture를 통과한 뒤 AGX Orin에서
한 layer 또는 제한된 expert bank로 확대하며, 전체 model load는 별도 승인을 받는다.
