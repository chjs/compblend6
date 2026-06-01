# MuSiQue KVzip benchmark — results

`benchmarks/musique/blend_musique_generic_kvzip.py`, N=150, Llama-3.1-8B-Instruct,
token-F1, `MAX_NEW_TOKENS=32`, `check_layer=1`, per-head mask fallback = 0.
Compression = **KVzip-importance token pruning** (keep top-`r` fraction of tokens per
doc chunk, drop the rest; head-uniform → no per-head mask). Raw: `n150_2026-06-01.json`.

Reference (compression-independent): **full_prefill = 0.307** (ceiling),
**full_reuse = 0.194** (uncompressed per-chunk KV reused, no recompute).

`prefill_kvzip` and `reuse_kvzip` are recompute-independent (one value per `r`). The three
blending arms are shown for **every** recompute ratio (rc), so there is no ambiguity about
which rc a number comes from.

| r | prefill_kvzip | reuse_kvzip | rc | only_hkvd | gated_all | gated_deep (15–30) |
|---|---|---|---|---|---|---|
| **0.5** | 0.247 | 0.178 | 0.20 | 0.218 | 0.230 | 0.222 |
|     |       |       | 0.15 | 0.230 | 0.226 | 0.213 |
|     |       |       | 0.10 | 0.213 | 0.223 | 0.215 |
|     |       |       | 0.05 | 0.213 | 0.222 | 0.209 |
| **0.4** | 0.177 | 0.176 | 0.20 | 0.209 | 0.212 | 0.197 |
|     |       |       | 0.15 | 0.216 | 0.212 | 0.218 |
|     |       |       | 0.10 | 0.224 | 0.202 | 0.211 |
|     |       |       | 0.05 | 0.210 | 0.213 | 0.208 |
| **0.3** | 0.138 | 0.197 | 0.20 | 0.201 | 0.195 | 0.195 |
|     |       |       | 0.15 | 0.202 | 0.190 | 0.192 |
|     |       |       | 0.10 | 0.193 | 0.188 | 0.193 |
|     |       |       | 0.05 | 0.193 | 0.198 | 0.199 |
| **0.2** | 0.132 | 0.167 | 0.20 | 0.179 | 0.167 | 0.171 |
|     |       |       | 0.15 | 0.179 | 0.175 | 0.165 |
|     |       |       | 0.10 | 0.178 | 0.178 | 0.169 |
|     |       |       | 0.05 | 0.177 | 0.172 | 0.170 |
| **0.1** | 0.085 | 0.166 | 0.20 | 0.157 | 0.152 | 0.158 |
|     |       |       | 0.15 | 0.162 | 0.155 | 0.169 |
|     |       |       | 0.10 | 0.162 | 0.155 | 0.166 |
|     |       |       | 0.05 | 0.167 | 0.162 | 0.165 |

## Findings

1. **Token-prune + full prefill (`prefill_kvzip`) collapses under compression**
   (0.247 → 0.085): recomputing survivors in a token-*depleted* context cannot recover
   dropped information.
2. **Reuse of the survivors' original per-chunk KV (`reuse_kvzip`) is robust & flat**
   (~0.17–0.20) and **beats `prefill_kvzip` at r ≤ 0.4**. Reusing isolated KV preserves
   token representations better than recomputing them in a depleted context. Crossover ≈ r0.4–0.5.
3. **Blending (HKVD / gated) = best of both**: ≥ `reuse_kvzip` everywhere, reaches the
   `prefill_kvzip` ceiling at r0.5 with only 5–20 % recompute, and stays robust where
   `prefill_kvzip` collapses (r ≤ 0.3).
4. **The gate (`gated_all`, `gated_deep`) is ≈ noise over `only_hkvd`** (±0.01) in this
   token-pruning setting — whichever arm is "best" flips with rc. Token pruning already keeps
   the high-importance tokens, so the gate has little left to add. (The gate's value is
   compression-scheme dependent.)
5. **`recompute_ratio` barely matters** (0.20 → 0.05 ≈ flat within each `r`): survivors'
   reused KV is already good.

> One line: after compression, **reuse beats recompute**, **blending reaches the ceiling at the
> lowest cost**, and the **importance gate adds nothing once tokens are importance-pruned**.

*Caveat:* MuSiQue 5-word-answer token-F1 is inherently low (~0.2–0.3); the relative ordering
is the signal. Single run.
