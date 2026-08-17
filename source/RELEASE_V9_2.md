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
9. Duplicate exported adapter directories are collapsed only when PEFT config
   and adapter-weight SHA-256 match; distinct checkpoints remain a hard error.
10. Public inference enforces `qlora_scope=full`; grouped-dev calibration
    enforces `qlora_scope=split`, preventing accidental leakage/undertraining.
11. Selector split IDs, answers, counts and train/dev disjointness are verified
    before any expensive model inference begins.
12. Identical source, dense-index, recovery and selector-weight aliases are
    collapsed deterministically; conflicting content remains a hard error.
13. Offline wheel discovery pins the exact 12 approved package versions.
14. QLoRA resume copies the newest valid checkpoint and never ignores an
    attached checkpoint merely because the working directory has metadata.
15. Selector tuning falls back to baseline weights when primary-METEOR holdout
    regresses; malformed, NaN or unsafe selector overrides are rejected.
16. Final public inference fails fast when the full adapter, tuned selector, or
    prebuilt dense/train-memory indexes are missing or structurally corrupt.
17. Public inference pauses safely at 11.5 hours, snapshots candidate caches,
    and exits successfully so Kaggle can persist the recovery ZIP.
18. Runtime deadlines are excluded from the semantic run fingerprint, allowing
    an identical next session to restore the checkpoint.
19. Recovery extraction rejects traversal, duplicate targets, malformed state,
    and unexpectedly large archives before writing files.

## Compatibility

- Existing QLoRA split/full adapters remain compatible; no QLoRA retraining is
  required for these inference/retrieval fixes.
- Old selector weights are intentionally rejected. Run calibration again and
  use schema `selector-weights-v3-agentic-evidence`.
- Old recovery/candidate caches are intentionally separated by the v7 run
  fingerprint and the `agentic-v3` candidate-cache namespace.

## Verification

- `python -m pytest -q`: 71 passed.
- `python -m compileall -q src scripts kaggle`: passed.
- All YAML/TOML and all notebook JSON/code cells parse and compile.
- Wheelhouse contains all 12 pinned Hugging Face dependencies required by the
  Kaggle notebooks.
