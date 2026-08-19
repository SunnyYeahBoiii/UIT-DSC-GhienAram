from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


def _load_answers(path: Path) -> dict[str, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {qid: (item["answer"] if isinstance(item, dict) else str(item)) for qid, item in raw.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chấm điểm local đúng công thức Scoring-Program-Task-LegalQA/scoring.py của BTC "
        "(rouge_scorer RougeL use_stemmer=False + nltk meteor_score có WordNet, không tokenize riêng), "
        "để kiểm tra trước khi nộp submission.zip lên Codabench. Không đụng tới pipeline train."
    )
    parser.add_argument("--predictions", required=True, help='submission json {qid: {"answer": ...}} hoặc {qid: "..."}')
    parser.add_argument("--reference", required=True, help='dev/reference json cùng dạng, ví dụ dev_sft.json')
    parser.add_argument("--output", default=None, help="ghi scores.json giống định dạng output của BTC")
    args = parser.parse_args()

    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer

    predictions = _load_answers(Path(args.predictions))
    reference = _load_answers(Path(args.reference))
    if set(predictions) != set(reference):
        missing = sorted(set(reference) - set(predictions))[:10]
        extra = sorted(set(predictions) - set(reference))[:10]
        raise ValueError(f"ID không khớp giữa predictions và reference. Thiếu={missing}; thừa={extra}")

    ids = sorted(reference)
    rouge_scoring = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rouge_result = float(
        np.mean(
            [rouge_scoring.score(str(reference[qid]), str(predictions[qid]))["rougeL"].fmeasure for qid in ids]
        )
    )
    meteor_result = float(
        np.mean(
            [meteor_score([str(reference[qid]).split()], str(predictions[qid]).split()) for qid in ids]
        )
    )
    scores = {"rouge": rouge_result, "meteor": meteor_result}
    print(json.dumps(scores, ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(scores) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
