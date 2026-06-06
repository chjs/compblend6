# Phase 3 — Compression-aware importance selection (importance를 blending에 활용하는 방법)

> Phase 2에서 게이트(2단계 importance gate + HKVD)는 양 끝점을 못 넘었다(실패). 그러나 **게이트가 아니라
> importance를 *압축 regime의 주 선택 신호*로 직접 쓰면** blending 품질이 유의하게 향상된다. 이 보고서는
> KVzip가 권장하는 현실적 압축 범위(retain 0.3–0.5, 2–3.3×)에서 그 결과를 정식화한다.

## 핵심 결과 — 현실적 압축 범위(kv 0.3/0.4/0.5)
데이터: `results/gate_sweep/gate_sweep_n150.json` (N=150, MuSiQue, Llama-3.1-8B, full-decode F1).
importance 집계 = max-over-(layer,head). paired bootstrap 95% CI(질문 단위).

| 비교 (kv∈{0.5,0.4,0.3} pooled) | Δ(F1) | 95% CI | 판정 |
|---|---|---|---|
| **importance_only − only_hkvd** | **+0.016** | [+0.005, +0.027] | **★ 유의** |
| gate=0.5 − only_hkvd | −0.010 | [−0.016, −0.004] | 게이트가 더 나쁨 |
| gate=0.5 − importance_only | −0.026 | [−0.037, −0.016] | 게이트가 더 나쁨 |

셀별로도 12셀 중 **11셀에서 importance > HKVD** (유일 예외 kv=0.4/rc=0.1, −0.012). 일관적이고 단조적.

## 두 결론 (동시에 참)
1. **importance는 압축 blending에서 유의하게 도움이 된다(+0.016★)** — 극한 압축이 아니라 KVzip 권장 범위에서.
2. **게이트(섞기)는 이 범위에서도 실패한다.** importance와 deviation은 직교(Phase 1, Spearman≈−0.11)라
   같은 결정에 섞으면 더 나은 단일 신호를 못 넘는다. → **섞지 말고 importance를 주 선택 신호로 직접 쓴다.**

## 방법 (배포 가능)
> **Compression-aware recompute selection:** 압축된 KV를 blending할 때 재계산 토큰을 **importance 상위로
> 직접 선택**(deviation/게이트 아님). KVzip 권장 압축(retain 0.3–0.5)에서 CacheBlend(deviation) 대비
> **+0.016 F1 유의 개선.** importance 집계는 max-over-(layer,head).

## 메커니즘 (왜 압축에서 importance가 이기나)
- 압축이 없으면 유일한 오차는 cross-chunk reuse error → deviation(HKVD)이 직접 측정 → HKVD 최적.
- 압축이 들어오면 캐시가 훼손되어 deviation 신호가 신뢰를 잃고(고압축에선 random 이하로 붕괴), 답은
  소수 핵심 토큰에 달림 → "무엇이 *틀렸나*"(deviation)보다 "무엇이 *답에 중요한가*"(importance)가 옳은 질문.
- 압축률이 두 신호의 우열을 결정하며, **CompBlend의 대상은 *압축* KV**이므로 importance가 주 신호가 된다.

## 논문 contribution 재서술
> *CacheBlend의 deviation 기반 선택은 압축 캐시에서 최적이 아니다. 압축 backend의 native importance로
> 재계산을 선택하면 KVzip 권장 압축률(2–3.3×)에서 blended 품질이 유의하게 향상된다(+0.016★).
> importance는 압축 KV blending의 필수 신호이지 보조가 아니다.*

## 다음 (직교 축 — 토큰 선택과 경쟁하지 않는 예산 배분)
importance를 토큰 선택이 아닌 **chunk/layer 예산 배분**에 얹으면 importance_only 위에 +α 가능(미검증).
→ Phase 3b 실험: chunk-level importance 예산 배분(rank vs raw/importance-weighted) on top of importance selection.
