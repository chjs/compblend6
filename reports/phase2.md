# Phase 2 — Gate-ratio 전체 격자 (acceptance gate)

> `importance_gated_blending.md` §5 Phase 2. 핵심 질문: **recompute별로, 내부(interior) gate ratio 중
> 양 끝점(only_hkvd, importance_only)을 동시에 능가하는 값이 있는가.** 있으면 게이트 이득, 없으면(단조/노이즈)
> 게이트 무이득으로 정직하게 보고하고 멈춘다.

## 설정
- 모델·데이터: Llama-3.1-8B-Instruct, MuSiQue, **N=150**, full-decode F1(첫 토큰 평가 아님).
- 격자: kvzip[1,0.9,…,0.1] × recompute[1,0.2,0.15,0.1,0.05] × gate[1,…,0.1]. **분모 = retained set**(코드 검증: `recompute_k=int(total_seq×ratio)`, evict 토큰은 시퀀스에 부재 → 이미 retained 기준, 교정 불필요).
- 방법 3종: only_hkvd / importance_only(=max-over-heads) / gated_all_hkvd(2단계, max-over-heads). importance 집계: §3대로 **max-over-(layer,head)**.
- gate 제약: g ≥ rr, 끝점 g=1.0·g=rr 항상 포함. recompute=1.0은 oracle 천장으로만(no gate sweep).
- **끝점 항등식 수치 검증 통과**: gated@g=1.0 ≡ only_hkvd, gated@g=rr ≡ importance_only (per-question F1 완전 일치) → 구현 정합성 확인.
- 데이터: `results/gate_sweep/gate_sweep_n150.json`. CI = paired bootstrap 95%(질문 단위).

## 결과 1 — 고정 gate=0.5 (post-hoc 선택 없음), 전 셀 pooled
| 비교 | Δ(F1) | 95% CI | 판정 |
|---|---|---|---|
| gate=0.5 − only_hkvd(g=1) | **+0.004** | [−0.001, +0.009] | **ns** |
| gate=0.5 − importance_only(g=rr) | **−0.005** | [−0.012, +0.001] | **ns** |

→ 임의 고정 내부 gate는 **두 끝점 모두와 통계적으로 구별되지 않음.** 게이트는 추가 이득이 없다.

## 결과 2 — 내부 gate가 "양 끝점을 동시에 능가"하는가 (acceptance gate 직답)
- 수치상 best-interior가 양 끝점을 모두 넘은 셀: **23/40** (작은 양의 마진, best-of-~8 gate 선택).
- 그 중 **통계적으로 유의(★)한 셀: 2/40.**
- best-of-8 다중비교에서 5% 우연 오탐 기대치 ≈ 40×(1−0.95^8) 환산 시 **2건은 노이즈 수준**.
- 셀별 best-interior Δ는 ~8개 gate의 max라 **상향 편향**(winner's curse) → 위 2★도 과대평가.

→ **acceptance gate 답: 아니오.** 내부 gate가 양 끝점을 robust하게 동시 능가하는 regime은 사실상 없다.

## 결과 3 — 진짜 구조는 "regime 의존적 끝점" (게이트 아님)
| 압축(kv) | only_hkvd | importance_only | 더 나은 끝점 |
|---|---|---|---|
| 1.0 | **0.288** | 0.246 | HKVD |
| 0.9 | **0.251** | 0.239 | HKVD |
| 0.8 | 0.227 | 0.228 | ≈ |
| 0.7–0.1 | 0.217→0.162 | **0.238→0.202** | **importance** |

→ 저압축(kv≥0.9)에선 only_hkvd, 중·고압축(kv≤0.7)에선 importance_only가 더 낫다. **최적 전략은 regime에 맞는 끝점 선택**이며, 둘을 섞는 내부 gate는 더 나은 끝점을 robust하게 넘지 못한다. (고압축 셀에서 best-interior가 importance_only 끝점보다 **유의하게 나쁜** 셀도 존재 — kv=0.1, rr∈{0.2,0.15}.)

## 판정 (§5 Phase 2 acceptance gate)
> "아이디어 1(gated)이 바닐라 HKVD와 importance_only를 동일 budget에서 유의하게 능가하는가?"

**아니오.** 고정 gate=0.5는 두 끝점과 ns, 내부 최적점은 노이즈 수준(2/40, 다중비교 inflated), 진짜 신호는 게이트가 아니라 **regime 의존적 끝점**이다.

→ 지시 규칙대로 **정직하게 멈춘다. Phase 3(확장)로 진행하지 않는다.** (게이트가 gate를 통과하지 못함.)

## Phase 1과의 일관성
Phase 1에서 importance⊥deviation(Spearman≈−0.11)이었고 "중요 AND 많이 틀림" 사분면이 희소함을 예고(§4.1). Phase 2는 이를 F1에서 확증: 두 신호를 게이트로 결합해도 더 나은 단일 신호(끝점)를 못 넘는다. **"Gated HKVD가 HKVD를 이긴다"는 논문 헤드라인은 이 token-prune regime에서 지지되지 않는다.** 방어 가능한 주장은 더 좁다 — *importance와 deviation은 직교하며, 어느 쪽이 최적인지는 압축 regime이 결정한다.*

> 천장 참고(oracle = full-prefill 후 압축, recompute=1.0): kv=0.9→0.292, 0.5→0.247, 0.3→0.138, 0.1→0.085. 고압축일수록 천장 자체가 낮아 recompute 선택의 여지가 작다.
