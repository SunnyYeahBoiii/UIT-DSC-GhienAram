from __future__ import annotations

import io
import json
import os
import zipfile
from pathlib import Path
from typing import Any

from .data import load_qa, read_json


def normalize_predictions(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        raise ValueError("Predictions phải là JSON Object.")
    result: dict[str, str] = {}
    for raw_id, value in raw.items():
        question_id = str(raw_id)
        if isinstance(value, str):
            answer = value
        elif isinstance(value, dict) and set(value) >= {"answer"}:
            answer = value["answer"]
        else:
            raise ValueError(f"Prediction {question_id} phải là String hoặc Object có answer.")
        if not isinstance(answer, str):
            raise ValueError(f"answer của {question_id} không phải chuỗi.")
        result[question_id] = answer
    return result


def create_submission(questions_path: str | Path, predictions_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    questions = load_qa(questions_path, require_answers=False)
    predictions = normalize_predictions(read_json(predictions_path))
    expected = {item.question_id for item in questions}
    actual = set(predictions)
    if expected != actual:
        raise ValueError(
            f"ID không khớp: thiếu={sorted(expected - actual)[:10]}, thừa={sorted(actual - expected)[:10]}"
        )
    payload = {item.question_id: {"answer": predictions[item.question_id]} for item in questions}
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.writestr("submission.json", encoded)
    validate_submission(temporary, questions_path)
    os.replace(temporary, target)
    return validate_submission(target, questions_path)


def validate_submission(zip_path: str | Path, questions_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(zip_path)
    with zipfile.ZipFile(path, "r") as archive:
        names = archive.namelist()
        if names != ["submission.json"]:
            raise ValueError(f"ZIP phải chứa duy nhất submission.json; hiện có: {names}")
        raw_bytes = archive.read("submission.json")
    if raw_bytes.startswith(b"\xef\xbb\xbf"):
        raise ValueError("submission.json có UTF-8 BOM; hãy lưu UTF-8 không BOM.")
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("submission.json không phải UTF-8 hợp lệ.") from exc

    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"question_id xuất hiện nhiều lần: {key}")
            result[key] = value
        return result

    payload = json.load(io.StringIO(text), object_pairs_hook=reject_duplicates)
    if not isinstance(payload, dict):
        raise ValueError("submission.json phải là JSON Object.")
    for question_id, value in payload.items():
        if not isinstance(question_id, str):
            raise ValueError("question_id phải là chuỗi.")
        if not isinstance(value, dict) or set(value) != {"answer"}:
            raise ValueError(f"{question_id}: giá trị phải là Object chỉ có trường answer.")
        if not isinstance(value["answer"], str):
            raise ValueError(f"{question_id}: answer phải là chuỗi.")
    if questions_path:
        expected = {item.question_id for item in load_qa(questions_path)}
        actual = set(payload)
        if expected != actual:
            raise ValueError(
                f"ID không khớp: thiếu={sorted(expected - actual)[:10]}, thừa={sorted(actual - expected)[:10]}"
            )
    return {"valid": True, "questions": len(payload), "zip": str(path.resolve()), "files": ["submission.json"]}
