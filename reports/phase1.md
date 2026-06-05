# Phase 1 — Importance↔Deviation 상관 측정 (최우선 gate)

> `importance_gated_blending.md` §4.1의 선행 gate. 게이트 가설의 이론적 토대를 검증한다.
> **이 gate 통과 전에는 어떤 방법도 "효과 있음"으로 결론짓지 않는다.**

## 측정 대상
fused 입력의 각 토큰에 대해 `(importance_j, deviation_j)`:
- **deviation D** = `‖K_full − K_chunk‖²` (per-(layer,head,token), CacheBlend HKVD 신호).
- **importance** = KVzip reconstruction max-attention. operative = `imp4`(per-chunk, blend 시 사용), 진단용 `imp3`(full-context).
- head 집계: §3 규칙대로 max-over-heads(rank-max) 기준.

데이터: `results/imp_vs_reuse/n150_raw.json` (MuSiQue, Llama-3.1-8B, N=150),
보조: `results/final_design/attn_vs_hkvd_n150.json` (attention-mass).

## 결과

| 지표 | 값 | 의미 |
|---|---|---|
| Spearman(D, imp4) 레이어평균 | **−0.113** | 약한 음의 상관 (거의 직교) |
| Spearman(D, imp3) 레이어평균 | −0.056 | 거의 0 |
| top-15% 집합 겹침(D ∩ imp4) | **0.102** | random(0.15) 수준 이하 |
| HKVD 선택 토큰의 importance 백분위 | **0.493** | ≈ random(0.5) |
| HKVD 픽 중 importance 중앙값 이하 비율 | **52%** | 예산 절반이 low-importance |
| attention-mass 포착(recompute 15%): HKVD / importance / random | **0.168 / 0.261 / 0.150** | HKVD≈random, importance 1.56× |

(레이어별 곡선·산점도: `results/imp_vs_reuse/viz_*`, `results/final_design/{hkvd_vs_importance,attention_received}.png`)

## 판정 (§4.1 기준)

- **강한 양의 상관 아님.** → "게이트가 바닐라 HKVD와 같은 집합을 고른다"는 무이득 시나리오는 **해당 없음**.
- 부호는 **약한 음/직교**. §4.1은 이 경우 "게이트 이득 가능성 있으나, *중요 AND 많이 틀림* 사분면이 희소해 게이트 설계가 까다롭다"고 예고. 즉 **이득의 상한이 작을 수 있음**을 시사.
- 그럼에도 **gate는 통과**(강한 양의 상관이 아니므로) → Phase 2(전체 격자) 진행 자격 충족.

## Phase 2로 넘기는 핵심 질문
직교성은 확인됐다. 남은 질문은 **"importance 게이트가 실제 full-decode F1에서 바닐라 HKVD를 (그리고 importance-only를) 동일 budget에서 능가하는가, 그리고 gate 축에 내부 최적점이 존재하는가"** 이며, 이는 Phase 2의 gate-ratio 스윕으로만 답할 수 있다.

> 주의: 위 상관은 KVzip importance(reconstruction query)와 deviation의 관계를 측정한 것이지, 실제 decode-query attention과 정확히 같지 않다(§2 근사 한계). Phase 2의 F1이 최종 판정.
