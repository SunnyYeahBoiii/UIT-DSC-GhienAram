from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .text import answer_length_bucket, normalize_text, question_type


@dataclass(frozen=True)
class QARecord:
    question_id: str
    question: str
    answer: str | None = None


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    title: str = ""
    heading: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def _reject_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Khóa JSON bị lặp: {key}")
        result[key] = value
    return result


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        return json.load(handle, object_pairs_hook=_reject_duplicate_object)


def write_json(path: str | Path, value: Any, indent: int = 2) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=indent)
        handle.write("\n")


def load_qa(path: str | Path, require_answers: bool = False) -> list[QARecord]:
    raw = read_json(path)
    records: list[QARecord] = []
    if isinstance(raw, dict):
        iterable = raw.items()
    elif isinstance(raw, list):
        iterable = ((str(item.get("question_id", item.get("id", ""))), item) for item in raw)
    else:
        raise ValueError("Dữ liệu câu hỏi phải là JSON Object hoặc Array.")

    seen: set[str] = set()
    for raw_id, item in iterable:
        question_id = str(raw_id)
        if not question_id or question_id in seen:
            raise ValueError(f"question_id rỗng hoặc bị lặp: {question_id!r}")
        if isinstance(item, str):
            question, answer = item, None
        elif isinstance(item, dict):
            question = item.get("question", item.get("query"))
            answer = item.get("answer")
        else:
            raise ValueError(f"Bản ghi {question_id} không phải Object/String.")
        if not isinstance(question, str) or not normalize_text(question):
            raise ValueError(f"Bản ghi {question_id} thiếu trường question dạng chuỗi.")
        if answer is not None and not isinstance(answer, str):
            raise ValueError(f"answer của {question_id} phải là chuỗi.")
        if require_answers and (answer is None or not normalize_text(answer)):
            raise ValueError(f"Bản ghi {question_id} thiếu answer dùng để huấn luyện/đánh giá.")
        seen.add(question_id)
        records.append(QARecord(question_id, normalize_text(question), normalize_text(answer) if answer else None))
    return records


def _first_value(item: dict[str, Any], fields: Iterable[str]) -> Any:
    for name in fields:
        if name in item and item[name] not in (None, ""):
            return item[name]
    return None


def iter_corpus(path: str | Path, config: dict[str, Any]):
    source = Path(path)
    if source.is_dir():
        def directory_items():
            for file_path in sorted(source.glob("*.json")):
                raw = read_json(file_path)
                if isinstance(raw, dict):
                    yield {"_source_file": file_path.name, **raw}
                elif isinstance(raw, list):
                    for item in raw:
                        if not isinstance(item, dict):
                            raise ValueError(f"{file_path}: phần tử corpus không phải Object.")
                        yield {"_source_file": file_path.name, **item}
                else:
                    raise ValueError(f"{file_path}: corpus JSON phải là Object hoặc Array.")
        raw_items = directory_items()
    elif source.suffix.lower() == ".jsonl":
        raw_items = []
        with source.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    try:
                        raw_items.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"JSONL lỗi ở dòng {line_number}: {exc}") from exc
    else:
        raw = read_json(source)
        if isinstance(raw, list):
            raw_items = raw
        elif isinstance(raw, dict):
            raw_items = []
            for key, value in raw.items():
                if isinstance(value, str):
                    raw_items.append({"id": key, "text": value})
                elif isinstance(value, dict):
                    raw_items.append({"_object_key": key, **value})
                else:
                    raise ValueError(f"Văn bản {key} không phải String/Object.")
        else:
            raise ValueError("Corpus phải là JSON Object, Array hoặc JSONL.")

    corpus_cfg = config["corpus"]
    seen: set[str] = set()
    for index, item in enumerate(raw_items):
        if not isinstance(item, dict):
            raise ValueError(f"Văn bản thứ {index} không phải Object.")
        doc_id = _first_value(item, corpus_cfg["id_fields"]) or item.get("_object_key") or str(index)
        text = _first_value(item, corpus_cfg["text_fields"])
        title = _first_value(item, corpus_cfg["title_fields"]) or ""
        doc_id = str(doc_id)
        if doc_id in seen:
            raise ValueError(f"doc_id bị lặp: {doc_id}")
        if not isinstance(text, str) or not normalize_text(text):
            if corpus_cfg.get("skip_empty", False):
                continue
            raise ValueError(f"Văn bản {doc_id} thiếu nội dung dạng chuỗi.")
        excluded = set(corpus_cfg["id_fields"] + corpus_cfg["text_fields"] + corpus_cfg["title_fields"])
        metadata = {
            key: value for key, value in item.items()
            if key not in excluded and key not in {"_object_key", "_source_file"}
        }
        if item.get("_source_file"):
            metadata["source_file"] = item["_source_file"]
        yield Document(doc_id, normalize_text(text), normalize_text(str(title)), metadata)
        seen.add(doc_id)


def load_corpus(path: str | Path, config: dict[str, Any]) -> list[Document]:
    return list(iter_corpus(path, config))


def deterministic_split(records: list[QARecord], dev_ratio: float, seed: int) -> tuple[list[QARecord], list[QARecord]]:
    """Stratify approximately by question type and answer length without labels."""
    strata: dict[str, list[QARecord]] = {}
    for item in records:
        key = f"{question_type(item.question)}::{answer_length_bucket(item.answer or '')}"
        strata.setdefault(key, []).append(item)
    train: list[QARecord] = []
    dev: list[QARecord] = []
    for key, group in sorted(strata.items()):
        ranked = sorted(
            group,
            key=lambda item: hashlib.sha256(f"{seed}:{key}:{item.question_id}".encode()).hexdigest(),
        )
        count = round(len(group) * dev_ratio)
        if len(group) >= 4:
            count = max(1, count)
        count = min(count, max(0, len(group) - 1))
        dev.extend(ranked[:count])
        train.extend(ranked[count:])
    return sorted(train, key=lambda item: item.question_id), sorted(dev, key=lambda item: item.question_id)


def qa_as_object(records: Iterable[QARecord], include_answers: bool = True) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in records:
        value: dict[str, Any] = {"question": item.question}
        if include_answers and item.answer is not None:
            value["answer"] = item.answer
        result[item.question_id] = value
    return result


def write_jsonl(path: str | Path, items: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_chunks(path: str | Path, chunks: Iterable[Chunk]) -> None:
    write_jsonl(path, (asdict(chunk) for chunk in chunks))


def load_chunks(path: str | Path) -> list[Chunk]:
    result: list[Chunk] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            try:
                result.append(Chunk(**item))
            except TypeError as exc:
                raise ValueError(f"Chunk JSONL lỗi ở dòng {line_number}: {exc}") from exc
    return result


def prepare_warmup(config: dict[str, Any]) -> dict[str, Any]:
    records = load_qa(config["paths"]["warmup"], require_answers=True)
    train, dev = deterministic_split(records, float(config["split"]["dev_ratio"]), int(config["seed"]))
    data_dir = Path(config["paths"]["artifacts"]) / "data"
    write_json(data_dir / "train.json", qa_as_object(train))
    write_json(data_dir / "dev.json", qa_as_object(dev))
    write_json(data_dir / "dev_questions.json", qa_as_object(dev, include_answers=False))
    write_json(data_dir / "dev_references.json", {item.question_id: {"answer": item.answer} for item in dev})
    # Leakage-safe warm-up memory: only training answers become pseudo-documents.
    write_jsonl(
        data_dir / "train_answer_corpus.jsonl",
        (
            {
                "document_id": f"warmup_train_{item.question_id}",
                "title": "Câu trả lời huấn luyện chính thức",
                "text": item.answer,
                "source_question_id": item.question_id,
            }
            for item in train
        ),
    )
    manifest = {
        "source": str(config["paths"]["warmup"]),
        "total": len(records),
        "train": len(train),
        "dev": len(dev),
        "dev_ratio_actual": len(dev) / len(records),
        "warning": (
            "train_answer_corpus chỉ dùng để smoke-test/cross-validation; "
            "không thay thế kho 8.500 văn bản của BTC."
        ),
    }
    write_json(data_dir / "manifest.json", manifest)
    return manifest
