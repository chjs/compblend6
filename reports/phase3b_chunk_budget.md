# Phase 3b — 직교 축 실험: chunk-level importance 예산 배분 (null 결과)

> Phase 3에서 "압축 blending엔 importance로 직접 재계산 선택"이 HKVD를 유의하게 이김(+0.016★)을 확인했다.
> 여기서는 importance를 **토큰 선택이 아닌 chunk 예산 배분**(설계문서 idea 5)에 *추가로* 얹으면 +α가 있는지
> 검증한다. importance와 deviation을 경쟁시키지 않는 직교 축이므로, 성공하면 importance_only 위에 곱해진다.

## 설정
- 현실적 band: kvzip retain {0.5, 0.4, 0.3} × recompute {0.2, 0.15, 0.1, 0.05}, N=150, full-decode F1.
- 비교 축 = `COMPBLEND_CHUNK_NORM` (importance 집계는 둘 다 max-over-(layer,head), 토큰 선택은 둘 다 importance):
  - **rank** = 청크별 rank-정규화 후 concat → 청크당 ~균등 재계산 예산.
  - **none** = raw importance concat → raw importance 높은 청크가 global top-k 예산을 더 차지 = **importance-가중 청크 예산**.
- 데이터: `results/gate_sweep/{gate_sweep_n150.json(rank), chunk_none_n150.json(none)}`. 같은 150문항 → paired bootstrap.

## 정합성 검증
- only_hkvd(deviation 기반, importance 미사용)는 rank·none 두 런에서 **per-question F1 완전 동일** → 두 런이 비교 가능하고 chunk_norm이 deviation 경로엔 영향 없음을 확인. ✓

## 결과
| 비교 (band pooled) | Δ(F1) | 95% CI | 판정 |
|---|---|---|---|
| **importance_only: none − rank** (청크 예산 재배분 효과) | **+0.004** | [−0.004, +0.013] | **ns (무이득)** |
| importance_only(none) − only_hkvd | +0.020 | [+0.010, +0.031] | ★ (헤드라인 유지) |

셀별 none−rank: 대부분 ±0.01 내 작은 진동, kv=0.3에서 일부 음수. **일관된 이득 없음.**

## 판정
**chunk-level importance 예산 재배분은 importance_only 선택 위에 유의한 이득을 주지 않는다(+0.004 ns).**
원인(예상대로): KVzip importance는 reconstruction max-attention(≈[0,1])이라 **청크 간 스케일이 비슷**해, raw(none)와 rank가 청크 예산을 거의 똑같이 나눈다 → 재배분 여지가 작음.

→ **배포 방법은 그대로: importance_only 직접 선택**(chunk_norm은 rank/none 무관, 동등). 청크 예산 축은 닫힌다.

## 정직한 한계 / 미검증
- **layer/head 예산 배분(idea 3)** 은 미검증. fusor가 모든 layer에 *단일 토큰 집합*을 재사용하는 구조라(레이어별 비균등 예산은 비자명한 구조 변경 필요), 이번 범위에서 제외. 향후 과제.
- 더 *명시적인* chunk 예산(예: aggregate importance 하위 청크 통째 skip)도 미검증. 다만 none이 이미 importance-가중 배분의 약한 형태인데 null이라, 효과 기대치는 낮음.

## 종합 (Part 1 + Part 2)
- **Part 1(핵심, 양성):** 압축 blending에서 importance 직접 선택이 HKVD를 +0.016★ 유의 능가 — importance 활용 방법 확정.
- **Part 2(직교 축, null):** chunk 예산 재배분은 추가 이득 없음 — importance_only가 이미 충분.
→ "importance를 blending에 반드시 활용" 임무는 **Part 1의 compression-aware importance selection**으로 충족된다.
