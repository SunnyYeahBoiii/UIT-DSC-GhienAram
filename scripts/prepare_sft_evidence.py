from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from legalqa.core.config import load_config  # noqa: E402
from legalqa.core.data import load_qa, write_jsonl  # noqa: E402
from legalqa.storage.sqlite_retrieval import retrieve_sqlite_batch  # noqa: E402


def build_sft_evidence(config: dict, database_path, questions, output_path) -> dict:
    """Sinh context cho SFT bằng đúng pipeline RAG thật (BM25+dense+reranker+graph routing)
    dùng ở lúc inference thật (retrieve_sqlite_batch, backend SQLite Kaggle), thay vì
    weak-label answer-to-corpus alignment.
    """
    batch = retrieve_sqlite_batch(config, database_path, questions, cache_path=None, reuse_cache=False)
    rows = []
    empty = 0
    for record in questions:
        evidence = batch.get(record.question_id, [])
        if not evidence:
            empty += 1
        rows.append(
            {
                "question_id": record.question_id,
                "question": record.question,
                "evidence": [
                    {
                        "chunk": asdict(item.chunk),
                        "score": item.score,
                        "bm25_score": item.bm25_score,
                        "dense_score": item.dense_score,
                        "reranker_score": item.reranker_score,
                        "stage": item.stage,
                        "relation_type": item.relation_type,
                        "relation_path": item.relation_path,
                    }
                    for item in evidence
                ],
            }
        )
    write_jsonl(output_path, rows)
    return {"rows": len(rows), "empty": empty, "output": str(Path(output_path).resolve())}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sinh context cho SFT bằng đúng pipeline RAG thật của backend SQLite Kaggle "
        "(retrieve_sqlite_batch: BM25+dense+reranker+graph routing), không dùng weak-label alignment."
    )
    parser.add_argument("--config", default=str(PROJECT / "configs" / "kaggle_inference_t4.yaml"))
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--database", required=True, help="SQLite corpus (build bằng `legalqa build-sqlite`)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    records = load_qa(Path(args.train_file), require_answers=False)
    if args.max_samples:
        records = records[: args.max_samples]

    report = build_sft_evidence(config, args.database, records, args.output)
    print(
        f"[prepare-sft-evidence] xong: {report['rows']:,} câu, rỗng={report['empty']:,} "
        f"(nếu >0 nên kiểm tra lại corpus/dense-index trước khi train), output={report['output']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
