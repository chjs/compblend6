# Phase 5 — Cross-model (Llama-3.1-8B / Qwen2.5-3B / Qwen2.5-14B)

> 우리 두 핵심 발견이 모델에 보편적인지 검증. MuSiQue, A100 80GB, full-decode F1.
> 실험: ① HKVD-failure 진단(N=40, KV cache ratio 1.0→0.1), ② 현실적 band importance vs HKVD(N=150, kv 0.5/0.4/0.3).
> Qwen2.5는 우리 스택(fusor qwen2 RoPE + KVzip "Replace qwen2.5 attention") 호환 확인 후 실행.

## 결과 1 — HKVD-failure 메커니즘은 **보편적** ✓
check_layer 기준, diff_k의 의미가 압축에서 뒤집히는가:

| 모델 | A benefit @1.0 / @0.3 | B corr(diff_k,reuse) @1.0 / @0.3 | C corr(diff_k,damage) @1.0 / @0.3 |
|---|---|---|---|
| Llama-3.1-8B | +145 / **−366** | 1.00 / **0.14** | −1.00 / **+0.87** |
| Qwen2.5-3B | +10 / **−113** | 1.00 / **0.11** | −1.00 / **+0.86** |
| Qwen2.5-14B | +3 / **−20** | 1.00 / **0.12** | −1.00 / **+0.80** |

→ **세 모델 모두 동일**: 무압축에선 diff_k=reuse error(B=1.0, C=−1.0), 압축되면 B가 ~0.12로 붕괴하고 **C가 −1.0→+0.8~0.87로 부호 반전**. 즉 "HKVD의 diff_k가 *재사용 오차*에서 *재계산 손상*으로 의미가 뒤집혀 가장 망가질 토큰을 고른다"는 메커니즘은 **모델·스케일에 무관(scale-free B/C)**. (A의 절대 크기만 모델마다 다름 — K norm/차원 차이, 예상된 것.)

## 결과 2 — importance vs HKVD(F1)는 **모델 의존적** ⚠️
현실적 band(kv 0.5/0.4/0.3) pooled, importance_only − only_hkvd:

| 모델 | Δ(imp − hkvd) | 95% CI | 판정 |
|---|---|---|---|
| Llama-3.1-8B | **+0.016** | [+0.005, +0.027] | ★ importance 승 |
| Qwen2.5-14B | **+0.044** | [+0.029, +0.059] | ★ importance 승 (더 크게) |
| Qwen2.5-3B | **−0.025** | [−0.038, −0.011] | ★ **HKVD 승 (역전!)** |

→ **"importance가 HKVD를 이긴다"는 F1 결과는 보편적이지 않다.** Llama-8B·Qwen-14B에선 importance가 이기지만(14B는 +0.044로 더 크게), **Qwen-3B에선 HKVD가 이긴다(−0.025).** 우리가 이전에 본 "승자는 모델 의존적"([[final-design-results]])과 일치.

스케일 경향(주의: 점 3개, 일화적): Qwen 3B(−0.025) → 14B(+0.044), Llama-8B(+0.016 중간) → **모델이 클수록 importance가 유리**해 보임. importance(=reconstruction attention) 신호가 큰 모델에서 더 신뢰성 있을 가능성 — 가설 수준, 추가 검증 필요.

## 종합
- **메커니즘(왜 HKVD가 압축에서 무너지나)은 보편적**: diff_k 의미 반전(C: −1→+0.8~0.87)이 Llama·Qwen·3B·14B 모두에서 동일. → Phase 4 발견은 모델 특이가 아니다. ✓
- **그 메커니즘이 importance 우위로 이어지는지는 모델 의존적**: 큰 모델(8B/14B)은 importance 승, 작은 모델(3B)은 HKVD 승. → "compression-aware importance selection"은 **모델/스케일 조건부 권장**으로 정직하게 한정해야 함.

## 정직한 단서
- A의 절대 크기는 pre-RoPE 위치 repack 성분이 섞이고 모델별 K 스케일이 달라 직접 비교 부적합. **B 붕괴·C 반전은 scale-free(순위 상관)라 cross-model 비교가 유효.**
- Qwen2.5는 표준 Instruct(1M 변형 아님), N: hkvd=40 / blend=150. 스케일 경향은 점 3개라 일화적.
- 데이터: results/gate_sweep/{hkvd,blend}_Qwen2.5-{3B,14B}-Instruct.json, hkvd_failure.json, gate_sweep_n150.json. 그림: crossmodel.png.
