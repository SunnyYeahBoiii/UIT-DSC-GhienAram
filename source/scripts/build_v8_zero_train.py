from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from legalqa.submission import validate_submission  # noqa: E402
from legalqa.zero_train import (  # noqa: E402
    KNNThresholds,
    answer_hash,
    conservative_postprocess,
    conservative_reselect,
    high_confidence_train_knn,
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_submission(path: Path) -> dict[str, str]:
    with zipfile.ZipFile(path, "r") as archive:
        if archive.namelist() != ["submission.json"]:
            raise ValueError(f"Baseline ZIP không hợp lệ: {archive.namelist()}")
        raw = json.loads(archive.read("submission.json").decode("utf-8-sig"))
    result: dict[str, str] = {}
    for question_id, value in raw.items():
        if not isinstance(value, dict) or not isinstance(value.get("answer"), str):
            raise ValueError(f"Baseline {question_id} không có answer dạng chuỗi.")
        result[str(question_id)] = value["answer"]
    return result


def read_trace(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            question_id = str(row.get("question_id", ""))
            if not question_id or question_id in result:
                raise ValueError(f"Trace ID không hợp lệ dòng {line_number}: {question_id}")
            result[question_id] = row
    return result


def write_submission(payload: dict[str, dict[str, str]], target: Path) -> None:
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.writestr("submission.json", encoded)
    os.replace(temporary, target)


def main() -> None:
    parser = argparse.ArgumentParser(description="V8.1 zero-train: reselect + kNN + post-processing.")
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--public", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--knn-combined", type=float, default=.94)
    parser.add_argument("--knn-word", type=float, default=.95)
    parser.add_argument("--knn-character", type=float, default=.91)
    parser.add_argument("--knn-margin", type=float, default=.20)
    args = parser.parse_args()

    baseline = read_submission(args.baseline)
    trace = read_trace(args.trace)
    train_payload, public_payload = read_json(args.train), read_json(args.public)
    expected_ids = {str(question_id) for question_id in public_payload}
    for label, actual in (("baseline", set(baseline)), ("trace", set(trace))):
        if actual != expected_ids:
            raise ValueError(
                f"ID {label} sai: thiếu={sorted(expected_ids - actual)[:10]}, "
                f"thừa={sorted(actual - expected_ids)[:10]}"
            )

    # The published baseline and the trace must describe the exact same v8 run.
    mismatch: list[str] = []
    for question_id in expected_ids:
        selected = str(trace[question_id].get("selected", ""))
        selected_candidates = [
            candidate for candidate in trace[question_id].get("candidates", [])
            if candidate.get("source") == selected
        ]
        if len(selected_candidates) != 1 or selected_candidates[0].get("answer") != baseline[question_id]:
            mismatch.append(question_id)
    if mismatch:
        raise ValueError(f"Baseline không khớp trace v8 ở {len(mismatch)} ID: {mismatch[:10]}")

    matches, knn_report = high_confidence_train_knn(
        train_payload,
        public_payload,
        KNNThresholds(
            combined=args.knn_combined,
            word=args.knn_word,
            character=args.knn_character,
            margin=args.knn_margin,
        ),
    )

    counters: Counter[str] = Counter()
    audit: list[dict[str, Any]] = []
    output_payload: dict[str, dict[str, str]] = {}
    for question_id in public_payload:
        question_id = str(question_id)
        row = trace[question_id]
        baseline_answer = baseline[question_id]
        answer, source, reselection = conservative_reselect(row)
        if reselection["changed"]:
            counters["candidate_reselected"] += 1
        selected_candidate = next(
            candidate for candidate in row["candidates"] if candidate.get("source") == source
        )

        match = matches.get(question_id)
        if match is not None:
            answer = match.answer
            source = "train_knn"
            counters["train_knn_override"] += 1
            postprocess = {
                "capitalized": False,
                "duplicate_sentences_removed": 0,
                "truncated_tail_removed": False,
            }
        else:
            truncated = source == "qwen3.5-2b" and bool(selected_candidate.get("details", {}).get("truncated"))
            answer, postprocess = conservative_postprocess(answer, truncated=truncated)
            counters["capitalized"] += int(postprocess["capitalized"])
            counters["duplicate_sentences_removed"] += int(postprocess["duplicate_sentences_removed"])
            counters["answers_deduplicated"] += int(postprocess["duplicate_sentences_removed"] > 0)
            counters["truncated_tail_removed"] += int(postprocess["truncated_tail_removed"])

        if not answer.strip():
            raise ValueError(f"Answer rỗng sau xử lý: {question_id}")
        changed = answer != baseline_answer
        counters["answers_changed"] += int(changed)
        output_payload[question_id] = {"answer": answer}
        audit.append(
            {
                "question_id": question_id,
                "question": public_payload[question_id]["question"],
                "changed": changed,
                "baseline_source": row.get("selected"),
                "final_source": source,
                "baseline_words": len(baseline_answer.split()),
                "final_words": len(answer.split()),
                "baseline_sha256": answer_hash(baseline_answer),
                "final_sha256": answer_hash(answer),
                "reselection": reselection,
                "postprocess": postprocess,
                "knn": (
                    {
                        "train_id": match.train_id,
                        "combined_score": match.combined_score,
                        "word_score": match.word_score,
                        "character_score": match.character_score,
                        "margin": match.margin,
                    }
                    if match is not None else None
                ),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "submission.json"
    predictions_path.write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    zip_path = args.output_dir / "submission_v8_1_zero_train.zip"
    write_submission(output_payload, zip_path)
    validation = validate_submission(zip_path, args.public)
    audit_path = args.output_dir / "v8_1_zero_train_audit.jsonl"
    audit_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in audit),
        encoding="utf-8",
    )
    report = {
        "mode": "v8.1-zero-train-safe",
        "baseline": str(args.baseline.resolve()),
        "questions": len(output_payload),
        "actions": dict(sorted(counters.items())),
        "knn": knn_report,
        "validation": validation,
        "outputs": {
            "submission_json": str(predictions_path.resolve()),
            "submission_zip": str(zip_path.resolve()),
            "audit": str(audit_path.resolve()),
        },
    }
    report_path = args.output_dir / "v8_1_zero_train_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
