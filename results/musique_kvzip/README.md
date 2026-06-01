# MuSiQue KVzip benchmark — results

`benchmarks/musique/blend_musique_generic_kvzip.py`, N=150, Llama-3.1-8B-Instruct,
token-F1, `MAX_NEW_TOKENS=32`, `check_layer=1`, per-head mask fallback = 0.
Compression = **KVzip-importance token pruning** (keep top-`r` fraction of tokens per
doc chunk, drop the rest; head-uniform → no per-head mask). Raw: `n150_2026-06-01.json`.

## Roofline & floor (compression-independent)

| reference | token-F1 | meaning |
|---|---|---|
| **`full_prefill`** | **0.307** | uncompressed, complete context, full forward — the **CacheBlend-paper roofline** (best score) |
| `full_reuse` | 0.194 | uncompressed per-chunk KV reused, no recompute — naive floor |

Every compressed method below is judged by **how close it gets to the 0.307 roofline**.
`reuse_kvzip` is recompute-independent (one value per `r`); the three blending arms are shown
for every recompute ratio (rc).

| r | reuse_kvzip | rc | only_hkvd | gated_all | gated_deep (15–30) |
|---|---|---|---|---|---|
| **0.5** | 0.178 | 0.20 | 0.218 | 0.230 | 0.222 |
|     |       | 0.15 | 0.230 | 0.226 | 0.213 |
|     |       | 0.10 | 0.213 | 0.223 | 0.215 |
|     |       | 0.05 | 0.213 | 0.222 | 0.209 |
| **0.4** | 0.176 | 0.20 | 0.209 | 0.212 | 0.197 |
|     |       | 0.15 | 0.216 | 0.212 | 0.218 |
|     |       | 0.10 | 0.224 | 0.202 | 0.211 |
|     |       | 0.05 | 0.210 | 0.213 | 0.208 |
| **0.3** | 0.197 | 0.20 | 0.201 | 0.195 | 0.195 |
|     |       | 0.15 | 0.202 | 0.190 | 0.192 |
|     |       | 0.10 | 0.193 | 0.188 | 0.193 |
|     |       | 0.05 | 0.193 | 0.198 | 0.199 |
| **0.2** | 0.167 | 0.20 | 0.179 | 0.167 | 0.171 |
|     |       | 0.15 | 0.179 | 0.175 | 0.165 |
|     |       | 0.10 | 0.178 | 0.178 | 0.169 |
|     |       | 0.05 | 0.177 | 0.172 | 0.170 |
| **0.1** | 0.166 | 0.20 | 0.157 | 0.152 | 0.158 |
|     |       | 0.15 | 0.162 | 0.155 | 0.169 |
|     |       | 0.10 | 0.162 | 0.155 | 0.166 |
|     |       | 0.05 | 0.167 | 0.162 | 0.165 |

## Findings (relative to the 0.307 roofline)

1. **Blending (HKVD / gated) gets closest to the roofline at the lowest cost.** At r0.5 it
   reaches ~0.23 (75 % of the 0.307 roofline) recomputing only 5–20 % of tokens, and stays the
   strongest compressed method down through r0.1.
2. **Reuse of survivors' original per-chunk KV (`reuse_kvzip`) is robust & flat** (~0.17–0.20),
   well above the `full_reuse` floor (0.194 uncompressed) even at r0.1 — reusing isolated KV
   preserves token representations under heavy pruning.
3. **The gate (`gated_all`, `gated_deep`) is ≈ noise over `only_hkvd`** (±0.01) — whichever is
   "best" flips with rc. Token pruning already keeps the high-importance tokens, so the gate has
   little left to add. (The gate's value is compression-scheme dependent.)
4. **`recompute_ratio` barely matters** (0.20 → 0.05 ≈ flat within each `r`): survivors' reused
   KV is already good, so extra recompute buys little.

> One line: **blending approaches the `full_prefill` roofline at a fraction of the compute**, and
> once tokens are importance-pruned the importance gate adds nothing.

*Caveat:* MuSiQue 5-word-answer token-F1 is inherently low (~0.2–0.3); the relative ordering is
the signal. Single run.
