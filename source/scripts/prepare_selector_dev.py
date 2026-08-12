from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from legalqa.data import load_qa, qa_as_object, write_json  # noqa: E402
from legalqa.text import question_type, words  # noqa: E402


def template_key(question: str) -> str:
    normalized = re.sub(r"\d+(?:[./-]\d+)*", "<num>", question.casefold())
    tokens = words(normalized, remove_stopwords=True)
    return f"{question_type(question)}::{' '.join(tokens[:18])}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a grouped, leakage-resistant selector dev split from official train.json"
    )
    parser.add_argument("--train", required=True)
    parser.add_argument("--output", required=True)
    size = parser.add_mutually_exclusive_group()
    size.add_argument("--dev-size", type=int, default=400)
    size.add_argument("--dev-ratio", type=float, default=None)
    parser.add_argument("--min-dev-size", type=int, default=None)
    parser.add_argument("--max-dev-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()
    if args.dev_ratio is not None and not 0.05 <= args.dev_ratio <= 0.4:
        raise ValueError("--dev-ratio phải nằm trong [0.05, 0.4].")
    if args.dev_ratio is None and not 50 <= args.dev_size <= 2000:
        raise ValueError("--dev-size phải nằm trong [50, 2000].")
    if args.min_dev_size is not None and args.min_dev_size < 1:
        raise ValueError("--min-dev-size phải >= 1.")
    if args.max_dev_size is not None and args.max_dev_size < 1:
        raise ValueError("--max-dev-size phải >= 1.")
    if (
        args.min_dev_size is not None
        and args.max_dev_size is not None
        and args.min_dev_size > args.max_dev_size
    ):
        raise ValueError("--min-dev-size không được lớn hơn --max-dev-size.")

    records = load_qa(args.train, require_answers=True)
    groups: dict[str, list] = {}
    for record in records:
        groups.setdefault(template_key(record.question), []).append(record)
    ranked_groups = sorted(
        groups.items(),
        key=lambda item: hashlib.sha256(
            f"{args.seed}:{item[0]}".encode("utf-8")
        ).hexdigest(),
    )
    target = (
        round(len(records) * args.dev_ratio)
        if args.dev_ratio is not None
        else min(args.dev_size, len(records) - 1)
    )
    dev = []
    train = []
    maximum = args.max_dev_size or (len(records) - 1)
    for _, group in ranked_groups:
        if len(dev) < target and len(dev) + len(group) <= maximum:
            dev.extend(group)
        else:
            train.extend(group)
    if not train or not dev:
        raise RuntimeError("Không tạo được train/dev không rỗng.")
    if args.min_dev_size is not None and len(dev) < args.min_dev_size:
        raise RuntimeError(
            f"Grouped dev chỉ có {len(dev)} câu, thấp hơn --min-dev-size={args.min_dev_size}; "
            "các template group không thể bị tách."
        )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "selector_train.json", qa_as_object(train))
    write_json(output / "selector_dev.json", qa_as_object(dev))
    write_json(output / "selector_dev_questions.json", qa_as_object(dev, include_answers=False))
    write_json(
        output / "selector_dev_references.json",
        {record.question_id: {"answer": record.answer} for record in dev},
    )
    manifest = {
        "source": str(Path(args.train).resolve()),
        "seed": args.seed,
        "requested_dev_size": args.dev_size if args.dev_ratio is None else None,
        "requested_dev_ratio": args.dev_ratio,
        "requested_min_dev_size": args.min_dev_size,
        "requested_max_dev_size": args.max_dev_size,
        "grouping": "question_type + first 18 normalized content tokens; numbers replaced",
        "total": len(records),
        "groups": len(groups),
        "train": len(train),
        "dev": len(dev),
        "dev_ratio_actual": len(dev) / len(records),
        "overlap_ids": len({item.question_id for item in train} & {item.question_id for item in dev}),
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
