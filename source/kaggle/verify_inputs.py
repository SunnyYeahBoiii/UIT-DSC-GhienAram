from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def load(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--public", required=True)
    parser.add_argument("--corpus", required=True)
    args = parser.parse_args()
    train_path, public_path, corpus_path = Path(args.train), Path(args.public), Path(args.corpus)
    train, public = load(train_path), load(public_path)
    if not isinstance(train, dict) or len(train) != 7000:
        raise ValueError(f"train phải là Object 7.000 bản ghi; hiện tại={type(train).__name__}/{len(train)}")
    if not isinstance(public, dict) or len(public) != 1000:
        raise ValueError(f"public phải là Object 1.000 bản ghi; hiện tại={type(public).__name__}/{len(public)}")
    if any(not isinstance(value, dict) or not isinstance(value.get("question"), str) or not isinstance(value.get("answer"), str) for value in train.values()):
        raise ValueError("train có bản ghi không đúng {question:string, answer:string}.")
    if any(not isinstance(value, dict) or not isinstance(value.get("question"), str) or value.get("answer") is not None for value in public.values()):
        raise ValueError("public có bản ghi không đúng {question:string, answer:null}.")

    files = sorted(corpus_path.glob("context_*.json"))
    if len(files) != 8532:
        raise ValueError(f"Corpus phải có 8.532 file; hiện tại={len(files)}")
    aggregate = hashlib.sha256()
    ids, empty, schemas = set(), 0, {}
    for path in files:
        raw_bytes = path.read_bytes()
        aggregate.update(path.name.encode("utf-8") + b"\0" + hashlib.sha256(raw_bytes).digest())
        item = json.loads(raw_bytes.decode("utf-8-sig"))
        if not isinstance(item, dict) or "id" not in item or "passage" not in item:
            raise ValueError(f"Corpus schema không hợp lệ: {path}")
        doc_id = str(item["id"])
        if doc_id in ids:
            raise ValueError(f"Corpus id bị lặp: {doc_id}")
        ids.add(doc_id)
        if not isinstance(item["passage"], str) or not item["passage"].strip():
            empty += 1
        key = "|".join(sorted(item))
        schemas[key] = schemas.get(key, 0) + 1
    report = {
        "valid": True,
        "train": {"records": len(train), "bytes": train_path.stat().st_size, "sha256": sha256(train_path)},
        "public": {"records": len(public), "bytes": public_path.stat().st_size, "sha256": sha256(public_path)},
        "corpus": {
            "files": len(files), "nonempty": len(files) - empty, "empty_passage": empty,
            "bytes": sum(path.stat().st_size for path in files), "schemas": schemas,
            "aggregate_sha256": aggregate.hexdigest(),
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

