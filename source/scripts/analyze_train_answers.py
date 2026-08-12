from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from legalqa.answer_style import build_answer_style_profile  # noqa: E402
from legalqa.data import load_qa, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze reference answer length/style using only official train.json"
    )
    parser.add_argument("--train", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    profile = build_answer_style_profile(load_qa(args.train, require_answers=True))
    write_json(args.output, profile)
    print(json.dumps(profile, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
