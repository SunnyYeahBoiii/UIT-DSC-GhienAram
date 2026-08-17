from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import yaml


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from legalqa.data import load_qa, read_json, write_json  # noqa: E402
from legalqa.metrics import meteor_exact, rouge_l_f1  # noqa: E402
from legalqa.selector_schema import SELECTION_WEIGHTS_SCHEMA  # noqa: E402


POSITIVE = (
    "evidence_support", "query_coverage", "citation_precision", "numeric_precision",
    "legal_semantic", "length_prior", "fluency",
)
NEGATIVE = ("unsupported", "duplicate", "fragment_ratio")
def template_key(question: str) -> str:
    """Group near-duplicate legal question templates without using answers."""
    normalized = re.sub(r"\d+(?:[./-]\d+)*", "<num>", question.casefold())
    tokens = re.findall(r"[\w<>]+", normalized, flags=re.UNICODE)
    return " ".join(tokens[:18])


def grouped_tune_holdout(
    ids: list[str], trace: dict[str, dict[str, Any]], seed: int
) -> tuple[list[str], list[str]]:
    """Make a deterministic 80/20 split while keeping templates together."""
    groups: dict[str, list[str]] = defaultdict(list)
    for question_id in ids:
        groups[template_key(str(trace[question_id].get("question", "")))].append(question_id)
    rng = random.Random(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    target = max(1, round(len(ids) * 0.20))
    holdout: list[str] = []
    for key in keys:
        if len(holdout) >= target and holdout:
            break
        holdout.extend(groups[key])
    holdout_set = set(holdout)
    tune = [question_id for question_id in ids if question_id not in holdout_set]
    if not tune:
        largest = max(groups.values(), key=len)
        tune = list(largest)
        holdout_set.difference_update(largest)
    return sorted(tune), sorted(holdout_set)


def load_trace(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Trace JSON lỗi ở dòng {line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Trace dòng {line_number} không phải JSON Object.")
            question_id = str(row.get("question_id", ""))
            if not question_id:
                raise ValueError(f"Trace thiếu question_id ở dòng {line_number}.")
            if question_id in result:
                raise ValueError(f"Trace lặp ID {question_id} ở dòng {line_number}.")
            candidates = row.get("candidates")
            if not isinstance(candidates, list) or not candidates:
                raise ValueError(f"Trace {question_id} thiếu candidates không rỗng.")
            sources = []
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    raise ValueError(f"Trace {question_id} có candidate không phải Object.")
                source = candidate.get("source")
                answer = candidate.get("answer")
                if not isinstance(source, str) or not source:
                    raise ValueError(f"Trace {question_id} có candidate thiếu source.")
                if not isinstance(answer, str) or not answer.strip():
                    raise ValueError(f"Trace {question_id}/{source} có answer rỗng/sai kiểu.")
                sources.append(source)
            if len(sources) != len(set(sources)):
                raise ValueError(f"Trace {question_id} lặp candidate source: {sources}")
            if row.get("selected") not in set(sources):
                raise ValueError(
                    f"Trace {question_id} selected={row.get('selected')!r} không có trong candidates."
                )
            result[question_id] = row
    return result


def metric(candidate: dict[str, Any], reference: str) -> tuple[float, float]:
    return meteor_exact(candidate["answer"], reference), rouge_l_f1(candidate["answer"], reference)


def candidate_score(candidate: dict[str, Any], weights: dict[str, Any]) -> float:
    details = candidate.get("details", {})
    source_priors = weights.get("source_priors", {})
    source_prior = source_priors.get(candidate["source"])
    if source_prior is None and candidate["source"] == "qwen3.5-2b-agentic":
        source_prior = source_priors.get("qwen3.5-2b", 0.0)
    trust = float(details.get("evidence_trust", 1.0))
    retrieval = float(details.get("retrieval_confidence", 1.0))
    unsupported = float(details.get("unsupported", 0.0))
    score = (
        float(weights["evidence_support"]) * float(details.get("evidence_support", 0.0)) * trust
        + float(weights["query_coverage"]) * float(details.get("query_coverage", 0.0))
        + float(weights["legal_citations"]) * float(details.get("citation_precision", 0.0))
        + float(weights["numeric_faithfulness"]) * float(details.get("numeric_precision", 0.0))
        + float(weights["legal_semantic"]) * float(details.get("legal_semantic", 0.0))
        + float(weights["length_prior"]) * float(details.get("length_prior", 0.0))
        + float(weights.get("fluency", 0.0)) * float(details.get("fluency", 0.0))
        + float(source_prior or 0.0)
        - float(weights["unsupported_penalty"]) * unsupported
        - float(weights["duplicate_penalty"]) * float(details.get("duplicate", 0.0))
        - float(weights.get("fragment_penalty", 0.0)) * float(details.get("fragment_ratio", 0.0))
        - float(weights.get("truncation_penalty", 0.0)) * float(bool(details.get("truncated", False)))
        - float(weights.get("retrieval_risk_penalty", 0.0))
        * (1.0 - retrieval)
        * (unsupported + float(candidate["source"].startswith("qwen")) * 0.15)
    )
    return score


def pick(
    row: dict[str, Any],
    weights: dict[str, Any],
    memory_threshold: float | None = None,
) -> dict[str, Any]:
    candidates = [
        candidate
        for candidate in row["candidates"]
        if not candidate["source"].startswith("bartpho")
        and not (
            memory_threshold is not None
            and candidate["source"] == "train-memory"
            and float(candidate.get("details", {}).get("memory_similarity", 0.0))
            < memory_threshold
        )
    ]
    if not candidates:
        raise RuntimeError("Không còn candidate hợp lệ sau bộ lọc BARTpho/train-memory.")
    ranked = sorted(candidates, key=lambda item: (-candidate_score(item, weights), item["source"]))
    qwen = next(
        (
            item for item in ranked
            if item["source"].startswith("qwen") and not item.get("details", {}).get("truncated", False)
        ),
        None,
    )
    margin = float(weights.get("qwen_preference_margin", 0.0))
    if qwen is not None and candidate_score(ranked[0], weights) - candidate_score(qwen, weights) <= margin:
        return qwen
    return ranked[0]


def aggregate(
    ids, trace, metrics, weights=None, memory_threshold: float | None = None
) -> dict[str, Any]:
    rows = []
    sources = Counter()
    for question_id in ids:
        row = trace[question_id]
        candidate = pick(row, weights, memory_threshold) if weights is not None else next(
            item for item in row["candidates"] if item["source"] == row["selected"]
        )
        score = metrics[(question_id, candidate["source"])]
        rows.append(score)
        sources[candidate["source"]] += 1
    return {
        "count": len(rows),
        "meteor_exact_probe": mean(item[0] for item in rows) if rows else 0.0,
        "rouge_l_f1": mean(item[1] for item in rows) if rows else 0.0,
        "sources": dict(sources),
    }


def better_primary_metric(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    """Accept tuning only when the METEOR-primary holdout does not regress."""
    meteor_delta = candidate["meteor_exact_probe"] - baseline["meteor_exact_probe"]
    if meteor_delta > 1e-12:
        return True
    if abs(meteor_delta) <= 1e-12:
        return candidate["rouge_l_f1"] > baseline["rouge_l_f1"] + 1e-12
    return False


def sampled_weights(base: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    candidate = copy.deepcopy(base)
    ranges = {
        "evidence_support": (0.04, 0.28), "query_coverage": (0.05, 0.25),
        "legal_citations": (0.03, 0.22), "numeric_faithfulness": (0.03, 0.20),
        "legal_semantic": (0.0, 0.08), "length_prior": (0.0, 0.10),
        "fluency": (0.08, 0.32), "unsupported_penalty": (0.08, 0.38),
        "duplicate_penalty": (0.02, 0.18), "fragment_penalty": (0.04, 0.28),
        "truncation_penalty": (0.08, 0.40),
        "retrieval_risk_penalty": (0.03, 0.20),
    }
    for key, (low, high) in ranges.items():
        candidate[key] = rng.uniform(low, high)
    candidate["qwen_preference_margin"] = rng.choice((0.0, 0.02, 0.04, 0.06, 0.08))
    candidate["source_priors"] = {
        "qwen3.5-2b": rng.uniform(-0.02, 0.14),
        "qwen3.5-2b-agentic": rng.uniform(-0.02, 0.14),
        "extractive": rng.uniform(-0.10, 0.04),
        "train-memory": rng.uniform(-0.08, 0.10),
    }
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate selected/candidate-oracle metrics and tune selector on official dev answers"
    )
    parser.add_argument("--trace", required=True)
    parser.add_argument("--references", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights-output", default=None)
    parser.add_argument("--trials", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()

    trace = load_trace(Path(args.trace))
    try:
        references = {
            item.question_id: item.answer or ""
            for item in load_qa(args.references, require_answers=True)
        }
    except ValueError:
        raw_references = read_json(args.references)
        references = {
            str(question_id): value["answer"]
            for question_id, value in raw_references.items()
            if isinstance(value, dict) and isinstance(value.get("answer"), str)
        }
    if set(trace) != set(references):
        raise ValueError(
            f"Trace/reference ID không khớp: trace={len(trace)}, references={len(references)}"
        )
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base_weights = config["selection"]
    base_memory_threshold = float(
        config.get("retrieval", {}).get("train_memory_candidate_threshold", 0.86)
    )
    ids = sorted(trace)
    metrics = {}
    oracle_rows = []
    oracle_sources = Counter()
    source_rows = defaultdict(list)
    for question_id in ids:
        scored = []
        for candidate in trace[question_id]["candidates"]:
            if candidate["source"].startswith("bartpho"):
                continue
            values = metric(candidate, references[question_id])
            metrics[(question_id, candidate["source"])] = values
            source_rows[candidate["source"]].append(values)
            scored.append((values, candidate["source"]))
        if not scored:
            raise RuntimeError(f"Trace {question_id} không có candidate hợp lệ sau khi loại BARTpho.")
        best, source = max(scored, key=lambda item: (item[0][0], item[0][1]))
        oracle_rows.append(best)
        oracle_sources[source] += 1

    # Tune on deterministic template groups; report untouched groups separately.
    rng = random.Random(args.seed)
    tune_ids, holdout_ids = grouped_tune_holdout(ids, trace, args.seed)
    if not tune_ids or not holdout_ids:
        raise RuntimeError(
            "Không tạo được tune/holdout template-group đều không rỗng; không xuất weights giả an toàn."
        )
    best_weights = copy.deepcopy(base_weights)
    best_report = aggregate(
        tune_ids, trace, metrics, best_weights, base_memory_threshold
    )
    best_objective = best_report["meteor_exact_probe"] + 0.10 * best_report["rouge_l_f1"]
    for _ in range(max(0, args.trials)):
        candidate_weights = sampled_weights(base_weights, rng)
        report = aggregate(
            tune_ids, trace, metrics, candidate_weights, base_memory_threshold
        )
        objective = report["meteor_exact_probe"] + 0.10 * report["rouge_l_f1"]
        if objective > best_objective:
            best_objective = objective
            best_weights = candidate_weights
            best_report = report

    threshold_candidates = sorted(
        {
            base_memory_threshold,
            *(
                round(value / 100, 2)
                for value in range(max(50, round(base_memory_threshold * 100)), 97, 2)
            ),
        }
    )
    threshold_reports = []
    best_memory_threshold = base_memory_threshold
    best_threshold_objective = -1.0
    for threshold in threshold_candidates:
        candidate_report = aggregate(
            tune_ids, trace, metrics, best_weights, threshold
        )
        objective = candidate_report["meteor_exact_probe"] + 0.10 * candidate_report["rouge_l_f1"]
        threshold_reports.append({"threshold": threshold, **candidate_report})
        if objective > best_threshold_objective:
            best_threshold_objective = objective
            best_memory_threshold = threshold
    best_report = aggregate(
        tune_ids, trace, metrics, best_weights, best_memory_threshold
    )
    selected_original = aggregate(ids, trace, metrics)
    base = aggregate(ids, trace, metrics, base_weights, base_memory_threshold)
    tuned_all = aggregate(ids, trace, metrics, best_weights, best_memory_threshold)
    tuned_holdout = aggregate(
        holdout_ids, trace, metrics, best_weights, best_memory_threshold
    )
    base_holdout = aggregate(
        holdout_ids, trace, metrics, base_weights, base_memory_threshold
    )
    accepted_on_holdout = better_primary_metric(tuned_holdout, base_holdout)
    recommended_weights = best_weights if accepted_on_holdout else base_weights
    recommended_memory_threshold = (
        best_memory_threshold if accepted_on_holdout else base_memory_threshold
    )
    recommended_all = aggregate(
        ids, trace, metrics, recommended_weights, recommended_memory_threshold
    )
    recommended_holdout = tuned_holdout if accepted_on_holdout else base_holdout
    report = {
        "count": len(ids),
        "selected_from_trace": selected_original,
        "rescored_base_weights": base,
        "candidate_oracle": {
            "meteor_exact_probe": mean(value[0] for value in oracle_rows),
            "rouge_l_f1": mean(value[1] for value in oracle_rows),
            "sources": dict(oracle_sources),
        },
        "per_source": {
            source: {
                "count": len(values),
                "meteor_exact_probe": mean(value[0] for value in values),
                "rouge_l_f1": mean(value[1] for value in values),
            }
            for source, values in source_rows.items()
        },
        "tuning": {
            "trials": args.trials,
            "tune_count": len(tune_ids),
            "holdout_count": len(holdout_ids),
            "split": "grouped by normalized question template; no template crosses tune/holdout",
            "tune": best_report,
            "base_holdout": base_holdout,
            "tuned_holdout": tuned_holdout,
            "accepted_on_holdout": accepted_on_holdout,
            "holdout": recommended_holdout,
            "all": recommended_all,
            "tuned_all": tuned_all,
            "best_on_tune_selection": best_weights,
            "recommended_selection": recommended_weights,
            "memory_candidate_threshold": {
                "base": base_memory_threshold,
                "best_on_tune": best_memory_threshold,
                "recommended": recommended_memory_threshold,
                "sweep": threshold_reports,
            },
        },
        "note": "METEOR probe uses exact Vietnamese token matching; Codabench remains authoritative.",
    }
    write_json(args.output, report)
    if args.weights_output:
        output = Path(args.weights_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            yaml.safe_dump(
                {
                    "schema": SELECTION_WEIGHTS_SCHEMA,
                    "selection": recommended_weights,
                    "retrieval": {
                        "train_memory_candidate_threshold": recommended_memory_threshold
                    },
                    "calibration": {
                        "accepted_on_holdout": accepted_on_holdout,
                        "base_holdout": base_holdout,
                        "tuned_holdout": tuned_holdout,
                    },
                },
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
