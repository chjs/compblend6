# Does KVzip importance predict where chunk-KV reuse breaks? (MuSiQue, N=150)

`benchmarks/musique/importance_vs_reuse.py`, Llama-3.1-8B. Per fused token, four aligned objects:

| | object | type |
|---|---|---|
| 1 | full-prefill pre-RoPE K (whole prompt) | KV |
| 2 | per-chunk isolated pre-RoPE K (recombined) | KV |
| 3 | importance_full (whole-prompt KVzip) | score |
| 4 | importance_chunk (per-chunk KVzip) | score |

**D(t) = ‖K1 − K2‖** = reuse-error / "needs-recompute" — the signal HKVD targets. We ask whether the
importance scores (3, 4) predict D. (Compressed at ratio=1.0 for clean K/scores; the 50% operating
point is applied as a top-50% analysis threshold — KVzip's `prune(0.5)` zero-fills evicted slots,
which would corrupt the comparison.) Data: `n150_global_2026-06-02.json`, plots in `viz_global/`.

## Finding 1 — importance ≈ uncorrelated with reuse-need (robust to aggregation)

layer-mean Spearman(D, ·), across head/layer aggregations:

| aggregation | sp(D, imp3 full) | sp(D, imp4 per-chunk) |
|---|---|---|
| mean (raw) | −0.056 | −0.113 |
| rank-mean (head) | −0.033 | −0.066 |
| rank-max (head) | +0.004 | −0.036 |
| **GLOBAL rank-max (layer+head)** | **+0.060** | **−0.007** |

MAX aggregation only removes the mean's negative-bias artifact; it never reaches a useful positive.
top-k overlap stays ≈ random.

## Finding 2 — the agreement is BOUNDARY-only

Splitting tokens into chunk-first ("boundary") vs interior (global rank-max scores):

| | sp(D, imp4) | sp(D, imp3) |
|---|---|---|
| **boundary (chunk-first token)** | **+0.534** | −0.211 |
| interior | −0.016 | +0.031 |

`imp4`'s first-token spike (per-chunk sink artifact, z≈+10 at chunk-local pos 0) **coincides** with
D's boundary spike (z≈+3.5) → it is a real boundary detector, and `imp3` (full-context) does NOT have
it. But boundary tokens are only ~0.2 % of tokens (1.3 % of the top-15 %-D set, ~6× enriched but tiny
in absolute terms) → diluted to ~0 in the global correlation.

## Takeaway

Importance (reconstruction value) and reuse-need (reposition error) are **different signals**; their
only real overlap — chunk boundaries — is already found by HKVD directly (it computes D). So the
importance gate is **redundant, not additive**. This is the mechanistic root of "gate ≈ noise"; the
downstream F1 confirms it (see `../musique_kvzip/`). Operative importance is `imp4` (per-chunk,
cached, free at blend time); `imp3` needs a full-context prefill we're trying to avoid → diagnostic only.

*Caveat:* single run; relative ordering is the signal.
