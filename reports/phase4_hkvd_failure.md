# Phase 4 — 왜 HKVD는 압축률이 오르면 무너지나 (KV-공간 ground-truth 진단)

> 추론이 아니라 **KV-공간에서 직접 측정**해 메커니즘을 규명. 토큰별로 세 가지 pre-RoPE key를 비교:
> `K_full`(전체 무압축 prefill=정답) / `K_cached`(청크 단독=재사용값) / `K_recomp`(token-pruned 문맥 재계산).

## 측정 정의 (head*dim 제곱합, HKVD reduce와 동일)
- `reuse_error  = ‖K_cached − K_full‖²` (재사용이 얼마나 틀렸나)
- `recomp_error = ‖K_recomp − K_full‖²` (재계산 후 얼마나 틀렸나)
- `deviation    = ‖K_recomp − K_cached‖²` (HKVD 선택 점수)
- `Δ_benefit    = reuse_error − recomp_error` (>0 ⇒ 재계산이 정답에 가까워짐 = 도움)

데이터: MuSiQue, Llama-3.1-8B, N=40, check_layer(=1). 측정 A/B/C를 kv 1.0→0.1 스윕.

## 결과 (check_layer)
| kv | A: Δ_benefit (HKVD / random) | HKVD 재계산 도움 비율 | B: corr(dev, reuse_err) | C: corr(dev, recompute DAMAGE) |
|---|---|---|---|---|
| 1.0 | **+145 / +29** | 1.00 | **+1.00** | **−1.00** |
| 0.8 | −88 / −7 | 0.16 | +0.55 | +0.15 |
| 0.6 | −247 / −58 | 0.05 | +0.29 | +0.65 |
| 0.4 | −334 / −109 | 0.05 | +0.16 | +0.84 |
| 0.3 | −366 / −132 | 0.06 | +0.14 | +0.87 |
| 0.2 | −354 / −148 | 0.07 | +0.15 | +0.88 |
| 0.1 | −184 / −141 | 0.12 | +0.18 | +0.84 |

(layer 16·24에서도 동일한 부호 흐름; check_layer에서 flip이 가장 극단적 — HKVD가 *선택하는 바로 그 레이어*에서 신호가 가장 망가짐.)

## 메커니즘 (세 측정이 합쳐 설명)

**무압축(kv=1.0) — HKVD 최적:**
- B=+1.00: **deviation = reuse_error** (재계산이 곧 reuse 오차를 측정).
- A=+145, 도움비율 1.00: 재계산이 *모든* 토큰을 정답에 가깝게, HKVD 선택 토큰은 random(+29)보다 훨씬 더(+145) 이득.
- C=−1.00: high-deviation = 가장 도움되는 토큰. → HKVD가 최선의 토큰을 고름. ✓ CacheBlend 정석.

**압축이 들어오면 — 세 가지가 동시에 무너짐:**
1. **재계산 자체가 해롭다.** Δ_benefit이 모든 셀렉터에서 음수(random도 음수) → token-pruned(누더기) 문맥에서 재계산한 key가 cached보다 정답에서 *더 멀어진다* = depleted-sequence 병리 입증.
2. **deviation이 reuse_error 추적을 상실.** B: 1.00 → 0.14.
3. **deviation이 "재계산 손상"을 추적하기 시작.** C: −1.00 → **+0.88**. 즉 **high-deviation = 재계산하면 가장 망가지는 토큰.**
→ 결과: HKVD-선택 Δ_benefit이 random보다 **더 음수**(−334 vs −109 at kv=0.4). **HKVD가 가장 해로운 토큰을 정확히 골라 재계산 = 자해.**

## 결론 (한 줄)
> **HKVD의 deviation은 압축률이 오르면 의미가 *뒤집힌다*: "재사용 오차"(고쳐야 할 것)에서 "재계산 손상"(건드리면 안 되는 것)으로.** check_layer에서 corr이 −1.00→+0.88로 부호 반전. 그래서 HKVD는 압축에서 단지 무력한 게 아니라 *능동적으로 최악의 토큰을 골라 자해*한다. 이것이 importance가 압축 blending에서 HKVD를 이기는 근본 이유다 — importance는 "답에 중요한 토큰"을 고르지, "재계산하면 망가질 토큰"을 고르지 않는다.

## 정직한 단서
- pre-RoPE key는 check_layer에서 약간의 위치 의존성이 있고, token-prune은 위치를 repack한다 → A의 절대 크기엔 위치 이동 성분이 섞임. 그러나 **B의 붕괴(1.00→0.14)와 C의 부호 반전(−1.00→+0.88)은 위치 confound와 무관한 robust 신호**이며, 실제 blend도 위치가 이동하므로 측정은 실제 동작을 반영한다.
- N=40이지만 토큰 수가 많아(질문당 수천) 상관 추정은 안정적.
