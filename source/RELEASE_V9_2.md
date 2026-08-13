# LegalQA v9.2 release checklist

## Fixed findings

1. Agentic Qwen is gated by route, retrieval confidence and legal cues.
2. Tool evidence is candidate-specific and never mutates seed evidence.
3. Selector tuning includes an independent Agentic source prior.
4. Non-fatal tool/search errors only drop that question's Agentic candidate;
   they are not cached and do not disable later questions.
5. `agentic_bm25_top_k` lives under `retrieval` and is honored by both backends.
6. The 12-wheel offline Kaggle wheelhouse is included.
7. Repository documentation describes only files actually present.
8. Repeated queries are blocked; tool hits pass citation/temporal/relation reranking.

## Compatibility

- Existing QLoRA split/full adapters remain compatible; no QLoRA retraining is
  required for these inference/retrieval fixes.
- Old selector weights are intentionally rejected. Run calibration again and
  use schema `selector-weights-v3-agentic-evidence`.
- Old recovery/candidate caches are intentionally separated by the v7 run
  fingerprint and the `agentic-v3` candidate-cache namespace.

## Verification

- `python -m pytest -q`: 57 passed.
- `python -m compileall -q src scripts kaggle`: passed.
- All YAML/TOML and all notebook JSON/code cells parse and compile.
- Wheelhouse contains all 12 pinned Hugging Face dependencies required by the
  Kaggle notebooks.
