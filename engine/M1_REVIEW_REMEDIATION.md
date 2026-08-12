# M1 Review Remediation Plan

## 목표

독립 리뷰에서 발견된 dense runtime 회귀와 불완전한 OOM gate를 먼저 수정하고,
Qwen3-MoE M1의 검증 근거와 M2 provider 경계를 정리한다. 기존
`experiment/orin-nano` baseline은 변경하지 않으며 수정은 현재 feature branch의
후속 커밋으로 남긴다.

## 작업 순서

1. Qwen 계열 q/k norm 적재 조건을 복구하고 Qwen2, Llama3, dense Qwen3,
   Qwen3-MoE의 attention weight contract를 정적으로 검사한다.
2. Qwen3-MoE config의 필수 필드, head 관계, expert/top-k, sparse schedule 및
   RMSNorm epsilon을 allocation 전에 `ValueError`로 검증한다.
3. 사용자 승인용 weight limit과 실제 capacity gate를 분리한다. Capacity gate는
   resident weights, GPU KV/activation, 최대 checkpoint staging tensor, workspace,
   명시적 OS headroom을 `MemAvailable` 및 CUDA allocator 상한과 비교한다.
4. router/post-norm fixed weight와 expert-bank provider 생성 책임을 분리한다.
   M2는 provider factory를 demand loader로 교체하되 async/cache/storage format은
   M2에서 정의한다.
5. top-k, batch/decode shape, duplicate/unused expert 및 반복 실행 matrix를 보강하고
   수동 검증과 자동 검증을 문서에서 구분한다.

## 다른 MoE 모델을 위한 원칙

공통 경계는 config validation, fixed layer weights, routed expert provider 및 memory
accounting으로 제한한다. Qwen3.5의 recurrent/hybrid state, Maple의 물리 표현,
Gemma의 병렬 dense+routed FFN과 layer별 attention schema는 discovery 후 별도
adapter에서 추가한다. 미래 필드를 Qwen3 contract에 미리 넣지 않는다.

완료 조건은 기존 dense Qwen 회귀, tiny CPU/CUDA fixture, metadata-only 30B
preflight가 모두 통과하고, 부족한 memory가 shard open 및 CUDA 초기화 전에
거부되는 것이다.
