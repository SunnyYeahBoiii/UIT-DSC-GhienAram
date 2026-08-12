from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .budget import count_system_budget
from .config import ensure_artifacts, load_config
from .data import load_qa, prepare_warmup, read_json, write_json
from .index import build_index
from .metrics import evaluate_predictions
from .pipeline import run_pipeline
from .pipeline import generate_from_evidence
from .sqlite_retrieval import retrieve_sqlite_batch
from .sqlite_store import build_sqlite_store, prepare_sqlite_weak_pairs
from .submission import create_submission, normalize_predictions, validate_submission
from .training_data import prepare_retrieval_pairs


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UIT DSC 2026 LegalQA — offline 3.88B pipeline")
    parser.add_argument("--config", default="configs/p0.yaml", help="Đường dẫn YAML cấu hình")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("prepare-warmup", help="Chia train/dev và tạo corpus train-only chống leakage")

    build = subparsers.add_parser("build-index", help="Chunk corpus và tạo BM25/dense index")
    build.add_argument("--corpus", default=None, help="Ghi đè paths.corpus")
    build.add_argument("--sparse-only", action="store_true", help="Chỉ tạo BM25, không nạp embedding")

    run = subparsers.add_parser("run", help="Sinh predictions từ file câu hỏi")
    run.add_argument("--questions", required=True)
    run.add_argument("--output", default=None)

    evaluate = subparsers.add_parser("evaluate", help="Tính METEOR probe và ROUGE-L local")
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--references", required=True)
    evaluate.add_argument("--output", default=None)

    submit = subparsers.add_parser("make-submission", help="Tạo ZIP đúng schema của BTC")
    submit.add_argument("--questions", required=True)
    submit.add_argument("--predictions", required=True)
    submit.add_argument("--output", default="submission.zip")

    validate = subparsers.add_parser("validate-submission", help="Kiểm tra ZIP và toàn bộ question_id")
    validate.add_argument("--zip", required=True)
    validate.add_argument("--questions", default=None)

    subparsers.add_parser("count-budget", help="Đếm chính xác tham số đăng ký của toàn hệ thống")
    pairs = subparsers.add_parser("prepare-pairs", help="Tự sinh positive/hard-negative từ dữ liệu chính thức")
    pairs.add_argument("--train", default=None)
    sqlite_build = subparsers.add_parser("build-sqlite", help="Tạo SQLite FTS5 index cho corpus lớn/Kaggle")
    sqlite_build.add_argument("--corpus", required=True)
    sqlite_build.add_argument("--database", required=True)
    sqlite_build.add_argument("--force", action="store_true")
    sqlite_pairs = subparsers.add_parser("prepare-sqlite-pairs", help="Tạo weak labels từ SQLite corpus")
    sqlite_pairs.add_argument("--train", required=True)
    sqlite_pairs.add_argument("--database", required=True)
    sqlite_pairs.add_argument("--output", required=True)
    sqlite_pairs.add_argument("--top-k", type=int, default=40)
    sqlite_pairs.add_argument("--limit", type=int, default=None)
    sqlite_run = subparsers.add_parser("run-sqlite", help="Retrieval SQLite + model generation")
    sqlite_run.add_argument("--questions", required=True)
    sqlite_run.add_argument("--database", required=True)
    sqlite_run.add_argument("--output", required=True)
    sqlite_run.add_argument("--cache", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    ensure_artifacts(config)
    if args.command == "prepare-warmup":
        _print(prepare_warmup(config))
    elif args.command == "build-index":
        _print(build_index(config, args.corpus, args.sparse_only))
    elif args.command == "run":
        _print(run_pipeline(config, args.questions, args.output))
    elif args.command == "evaluate":
        predictions = normalize_predictions(read_json(args.predictions))
        references = normalize_predictions(read_json(args.references))
        report = evaluate_predictions(predictions, references)
        if args.output:
            write_json(args.output, report)
        _print({key: value for key, value in report.items() if key != "per_question"})
    elif args.command == "make-submission":
        _print(create_submission(args.questions, args.predictions, args.output))
    elif args.command == "validate-submission":
        _print(validate_submission(args.zip, args.questions))
    elif args.command == "count-budget":
        _print(count_system_budget(config))
    elif args.command == "prepare-pairs":
        _print(prepare_retrieval_pairs(config, args.train))
    elif args.command == "build-sqlite":
        _print(build_sqlite_store(config, args.corpus, args.database, args.force))
    elif args.command == "prepare-sqlite-pairs":
        _print(prepare_sqlite_weak_pairs(args.train, args.database, args.output, top_k=args.top_k, limit=args.limit))
    elif args.command == "run-sqlite":
        questions = load_qa(args.questions, require_answers=False)
        evidence = retrieve_sqlite_batch(config, args.database, questions, args.cache)
        _print(generate_from_evidence(config, questions, evidence, args.output))
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
