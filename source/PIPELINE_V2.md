# LegalQA v2: global hybrid retrieval + legal relation graph

This version replaces the public baseline that scored 0.3093. It uses only the
official corpus and the four approved runtime checkpoints: Qwen, Vietnamese
Embedding, Vietnamese Reranker, and ViLegalBERT. BARTpho is disabled and is not
counted or loaded.

## Runtime pipeline

```text
question
  -> global chunk BM25 + exact global dense search
  -> typed citation boosts + RRF
  -> cross-encoder reranking
  -> retrieval-confidence Simple/Complex router
     -> Simple: lock top document + adjacent Article/Clause expansion
     -> Complex: bounded 1-3 hop expansion over REPLACES/REPEALS/
                 AMENDS/GUIDES/REFERENCES edges
  -> temporal ranking + adjacent-window merge + deduplication
  -> evidence-first Qwen + extractive fallback
  -> confidence-aware, fluency-aware selector
  -> submission.zip
```

SQLite schema v4 contains a global chunk FTS index, document aliases, relation
edges, and structural chunk metadata. A stale v2 database is rebuilt
automatically; use a new working directory when possible.

## Recommended Kaggle execution

The first global dense index contains roughly 215k float16 vectors. It is
restartable inside one session and reusable across submissions. For predictable
Kaggle limits, use two sessions:

1. In notebook cell 2 set `DENSE_INDEX_ONLY = True` and run all cells. Save a
   Kaggle Version, then create a Dataset from
   `/kaggle/working/legalqa_dense_index`.
2. Attach that Dataset, set `DENSE_INDEX_ONLY = False`, and rerun. The notebook
   auto-detects a completed `global-dense-v1` manifest and does not re-embed the
   corpus.

Qwen cache and full retrieval diagnostics remain in
`legalqa_recovery_latest.zip`. A v2 recovery ZIP cannot be resumed by v3 because
the retrieval semantics and candidate cache schema changed.

## Diagnostics written for every question

- global BM25 top 40;
- global dense top 40;
- fused top 48;
- reranked top 24;
- router route, confidence, features, and reason;
- every graph iteration, relation type, direction, and new document;
- final merged evidence;
- Qwen generated-token count, EOS status, token limit, and truncation flag;
- every selector feature and selected source.

## Selector calibration with official train data

Prepare a grouped holdout:

```bash
python scripts/prepare_selector_dev.py \
  --train /path/to/train.json \
  --output /path/to/selector_split \
  --dev-ratio 0.15
```

Run the normal pipeline on `selector_dev_questions.json`, then evaluate candidate
oracle and tune selector weights:

```bash
python kaggle/run_inference_only.py \
  --questions-file /path/to/selector_split/selector_dev_questions.json \
  --input-root /kaggle/input \
  --working /kaggle/working/legalqa_selector_dev \
  --submission-output /kaggle/working/selector_dev_predictions.zip \
  --dense-index-input /path/to/completed/legalqa_dense_index

python scripts/evaluate_selector.py \
  --trace /kaggle/working/legalqa_selector_dev/predictions.trace.jsonl \
  --references /path/to/selector_split/selector_dev.json \
  --config configs/kaggle_inference_t4.yaml \
  --output /path/to/selector_report.json \
  --weights-output /path/to/tuned_selection.yaml \
  --trials 2000
```

Only adopt tuned weights when the untouched 20% holdout improves. The local
METEOR implementation is an exact-token diagnostic; Codabench is authoritative.

## Important controls

- `--dense-index-input`: completed index attached as a Kaggle Dataset.
- `--dense-index-only`: build/reuse the index and stop before public inference.
- `--force-dense-index`: remove and rebuild only an index under
  `/kaggle/working`; it refuses to delete an attached input Dataset.
- `--resume-bundle`: resume v3 retrieval/Qwen candidates.
- `--checkpoint-every`: recovery ZIP cadence during Qwen generation.

Do not set an unbounded agent loop. Graph expansion is capped at three iterations
by default (hard maximum four), uses a visited-document set, and stops when no new
legal relation exists.
