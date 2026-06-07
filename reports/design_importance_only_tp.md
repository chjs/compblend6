# Design note — Importance-only selection as a TP-native primitive for compressed-KV blending

> Contribution 재구성: 원래 "Gated HKVD"는 실험에서 무너졌다([[gate-ratio-sweep-results]]). 그 자리에
> **"압축 KV blending의 재계산 선택 신호는 런타임 deviation(HKVD)이 아니라 *사전계산된 query-agnostic
> importance*가 옳다"**를 둔다. 이 주장은 (a) 품질(압축·대형 모델에서 우월), (b) 메커니즘(HKVD가 압축에서
> 자해 신호로 뒤집힘), (c) **시스템(TP 서빙에 본질적으로 부합)** 세 축이 한 방향으로 정렬된다.

## 1. 핵심 주장
- **HKVD = 런타임 신호.** deviation `‖K_fresh − K_cached‖`을 서빙 시점에 계산하려면 (i) check_layer까지
  **dense fresh forward**를 모든 토큰에 돌리고, (ii) head를 가로지른 per-token 집계를 위해 **cross-rank
  all-reduce**, (iii) 모든 TP rank가 동일 top-k에 **동기화 합의**해야 한다.
- **importance = 정적 신호.** KVzip importance는 **압축 시점에 오프라인으로 한 번**, query-agnostic하게 계산돼
  청크 메타데이터로 저장된다. per-token 점수를 오프라인에서 full(unsharded) 집계해 두면, 서빙 시 **어느
  토큰을 재계산할지가 청크의 정적 속성**이 된다 → 런타임 forward 0, cross-rank 통신 0.

## 2. HKVD vs importance-only 비교

| 항목 | HKVD (deviation) | importance-only |
|---|---|---|
| 선택 신호 계산 시점 | **서빙 런타임** | **압축 시 오프라인 1회** (query-agnostic) |
| 선택을 위한 forward | check_layer까지 **dense**(전 토큰) | **없음** (사전계산) → sparse(ρS) from layer 0 가능 |
| TP cross-rank 통신(선택) | per-token deviation **all-reduce** + top-k **동기화 barrier** | **0** (정적 집합, rank마다 동일 by construction) |
| 서빙 latency | dense 초기 forward가 critical path | 초기 dense forward 제거 → 더 낮음 |
| 품질(압축 regime) | 압축에서 **자해**(아래 (c)) | 대형 모델서 우월(+0.016~0.044★), 소형(3B) 역전 |
| TP 친화성 | 낮음(런타임 결합·통신·동기화) | **높음**(embarrassingly parallel) |

(c) **HKVD-failure 메커니즘**([[hkvd-failure-mechanism]], cross-model 보편 [[crossmodel-qwen-results]]):
압축이 오르면 diff_k가 "reuse error"에서 "recompute damage"로 의미가 뒤집혀(corr −1→+0.8~0.87) HKVD가
*가장 망가질 토큰*을 고른다. → 런타임 deviation 신호 자체가 압축에서 신뢰 불가. importance는 이 함정이 없다.

## 3. TP 통신량 정량 (분석적 추정, Megatron-style: layer당 all-reduce 2회 on [S, hidden])

`HKVD 추가 통신 ≈ 2·(check_layer+1)·(1−ρ)·S·hidden·bytes·[2(p−1)/p]` (초기 dense층의 all-reduce 초과분),
`importance-only 선택 통신 = 0`.

| 모델 / 서빙 설정 | HKVD 추가 all-reduce/요청 | importance-only |
|---|---|---|
| Llama-3.1-8B (S=4096, TP=4, ρ=0.15, cl=1) | **~171 MB** + 동기화 barrier | **0 MB** |
| Qwen2.5-14B (S=8192, TP=4) | **~428 MB** | **0 MB** |
| Llama-3.1-70B (S=8192, TP=8) | **~799 MB** | **0 MB** |

추가로 HKVD는 **초기 (check_layer+1)개 층을 전 토큰에 dense**로 돌린다(critical-path 연산); importance-only는
**layer 0부터 sparse(ρS≈15%)** 로 바로 진행 → 통신뿐 아니라 연산·latency도 절감. (수치는 bf16, ring all-reduce
가정의 order-of-magnitude 추정 — 측정값 아님. check_layer·S·TP에 비례.)

## 4. regime 정합 (narrative 강점)
TP는 **대형 모델 + 압축 서빙**에서 의미 있다. 그런데 importance-only가 품질로 이기는 영역도 정확히 그곳
(8B +0.016★, 14B +0.044★; 3B는 역전). **TP가 필요한 regime = importance-only가 우월한 regime** → 품질과
시스템 당위가 동시에 성립. (소형 모델은 TP가 불필요하므로 3B 역전과 충돌 없음.)

## 5. 정직한 한계
- HKVD를 버리면 **무압축/저압축에서의 deviation 우위는 포기**(무압축선 HKVD 최적). 단 TP+압축 서빙은 그
  regime이 아니므로 옳은 트레이드오프 — 보편 주장 아닌 **regime 한정**으로 명시.
- importance 우위는 **모델/스케일 의존적**(3B 역전). "≥8B"로 조건화.
- importance 메타데이터 저장/전송 오버헤드 = KV의 ~0.4%(per-(L,head) fp16; per-token 스칼라면 더 작음) — 무시 가능.
- 위 통신 수치는 분석적 추정. 실제 vLLM/TP 측정으로 확정 필요.

## 6. 한 줄
**importance-only는 marginal F1 개선이 아니라, 분산·압축 서빙을 위한 *올바른 선택 primitive*다 — 사전계산·
query-agnostic·0-통신이라 TP에 본질적으로 부합하고, HKVD가 압축에서 자해 신호로 뒤집히는 것과 대조된다.**
