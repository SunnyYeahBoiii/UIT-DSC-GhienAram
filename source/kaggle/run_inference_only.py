from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import yaml


for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from legalqa.budget import count_system_budget  # noqa: E402
from legalqa.answer_style import build_answer_style_profile  # noqa: E402
from legalqa.config import load_config  # noqa: E402
from legalqa.data import load_qa, write_json  # noqa: E402
from legalqa.dense_store import DENSE_INDEX_SCHEMA, ensure_global_dense_index  # noqa: E402
from legalqa.fingerprint import (  # noqa: E402
    adapter_fingerprint,
    checkpoint_fingerprint,
    corpus_content_fingerprint,
    sha256_file,
)
from legalqa.models import (  # noqa: E402
    CrossEncoderReranker,
    DenseEncoder,
    LegalSemanticEncoder,
    QwenTextGenerator,
    unload_model,
)
from legalqa.pipeline import InferencePaused, generate_from_evidence  # noqa: E402
from legalqa.recovery import (  # noqa: E402
    RecoveryManager,
    recovery_input_state,
    restore_recovery_bundle,
)
from legalqa.selector_schema import SELECTION_WEIGHTS_SCHEMA  # noqa: E402
from legalqa.sqlite_retrieval import retrieve_sqlite_batch, sqlite_bm25_search_fn  # noqa: E402
from legalqa.sqlite_store import build_sqlite_store  # noqa: E402
from legalqa.sqlite_store import SQLiteLegalStore  # noqa: E402
from legalqa.submission import create_submission  # noqa: E402
from legalqa.train_memory import TRAIN_MEMORY_SCHEMA, ensure_train_memory_index  # noqa: E402
from run_end_to_end import (  # noqa: E402
    find_corpus,
    find_model,
    find_unique_file,
    require_environment,
    run_command,
    write_resolved_config,
)


CACHE_SCHEMA_VERSION = "inference-v7-agentic-error-isolation"
RUNTIME_MODEL_ROLES = ("generator", "embedding", "reranker", "verifier")
EPHEMERAL_RUNTIME_KEYS = {"max_runtime_hours", "runtime_started_unix", "stop_at_unix"}


def discover_recovery_inputs(
    root: str | Path,
    *,
    expected_fingerprint: str | None = None,
) -> list[Path]:
    """Find inference recovery as a ZIP or as Kaggle's extracted Dataset."""
    input_root = Path(root)
    candidates = set(input_root.rglob("legalqa_recovery_latest.zip"))
    candidates.update(state.parent for state in input_root.rglob("recovery_state.json"))
    if expected_fingerprint is not None:
        candidates = {
            candidate
            for candidate in candidates
            if str(recovery_input_state(candidate).get("run_fingerprint", ""))
            == expected_fingerprint
        }
    return sorted(candidates, key=lambda path: (len(path.parts), path.as_posix()))


def _safe_manifest_filename(value: object, label: str) -> str:
    name = str(value or "").strip()
    if not name or Path(name).name != name or "/" in name or "\\" in name:
        raise RuntimeError(f"{label} has an unsafe or empty filename: {value!r}")
    return name


def validate_dense_index_input(
    path: str | Path,
    *,
    corpus_identity: str | None = None,
    embedding_identity: str | None = None,
) -> dict:
    """Fail before model loading when an attached dense index is incomplete/corrupt."""
    root = Path(path)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dense index is missing manifest.json: {root}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        chunks = int(manifest.get("chunks", 0))
        dimension = int(manifest.get("dimension", 0))
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Dense index manifest is invalid: {manifest_path}: {exc}") from exc
    if manifest.get("status") != "complete" or manifest.get("schema") != DENSE_INDEX_SCHEMA:
        raise RuntimeError(
            f"Dense index is not a completed {DENSE_INDEX_SCHEMA} artifact: {manifest_path}"
        )
    if not isinstance(manifest.get("identity"), str) or not manifest["identity"].strip():
        raise RuntimeError(f"Dense index identity is missing: {manifest_path}")
    if manifest.get("dtype") not in (None, "float16"):
        raise RuntimeError(f"Dense index dtype must be float16: {manifest_path}")
    if chunks <= 0 or dimension <= 0:
        raise RuntimeError(f"Dense index has invalid shape chunks={chunks}, dimension={dimension}.")
    if corpus_identity and embedding_identity:
        expected_identity = hashlib.sha256(
            json.dumps(
                {
                    "schema": DENSE_INDEX_SCHEMA,
                    "corpus": corpus_identity,
                    "chunks": chunks,
                    "embedding": embedding_identity,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        if manifest["identity"] != expected_identity:
            raise RuntimeError(
                "Dense index belongs to another corpus or embedding checkpoint: "
                f"expected={expected_identity[:16]}, actual={manifest['identity'][:16]}."
            )
    vectors = root / _safe_manifest_filename(manifest.get("vectors"), "dense vectors")
    rowids = root / _safe_manifest_filename(manifest.get("rowids"), "dense rowids")
    expected_vectors = chunks * dimension * 2
    if not vectors.is_file() or vectors.stat().st_size != expected_vectors:
        raise RuntimeError(
            f"Dense vectors are missing/truncated: expected={expected_vectors}, "
            f"actual={vectors.stat().st_size if vectors.is_file() else '<missing>'}: {vectors}"
        )
    expected_rowids = chunks * 8
    if not rowids.is_file() or rowids.stat().st_size != expected_rowids:
        raise RuntimeError(
            f"Dense rowids are missing/truncated: expected={expected_rowids}, "
            f"actual={rowids.stat().st_size if rowids.is_file() else '<missing>'}: {rowids}"
        )
    return manifest


def validate_train_memory_input(
    path: str | Path,
    *,
    train_identity: str | None = None,
    embedding_identity: str | None = None,
) -> dict:
    """Validate the attached official-train kNN memory without loading vectors."""
    root = Path(path)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Train-memory index is missing manifest.json: {root}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        count = int(manifest.get("count", 0))
        dimension = int(manifest.get("dimension", 0))
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Train-memory manifest is invalid: {manifest_path}: {exc}") from exc
    if manifest.get("status") != "complete" or manifest.get("schema") != TRAIN_MEMORY_SCHEMA:
        raise RuntimeError(
            f"Train-memory is not a completed {TRAIN_MEMORY_SCHEMA} artifact: {manifest_path}"
        )
    if not isinstance(manifest.get("identity"), str) or not manifest["identity"].strip():
        raise RuntimeError(f"Train-memory identity is missing: {manifest_path}")
    if manifest.get("dtype") not in (None, "float16"):
        raise RuntimeError(f"Train-memory dtype must be float16: {manifest_path}")
    if count <= 0 or dimension <= 0:
        raise RuntimeError(f"Train-memory has invalid shape count={count}, dimension={dimension}.")
    if train_identity and embedding_identity:
        expected_identity = hashlib.sha256(
            json.dumps(
                {
                    "schema": TRAIN_MEMORY_SCHEMA,
                    "train": train_identity,
                    "embedding": embedding_identity,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        if manifest["identity"] != expected_identity:
            raise RuntimeError(
                "Train-memory belongs to another train set or embedding checkpoint: "
                f"expected={expected_identity[:16]}, actual={manifest['identity'][:16]}."
            )
    vectors = root / _safe_manifest_filename(manifest.get("vectors"), "train-memory vectors")
    records = root / _safe_manifest_filename(manifest.get("records"), "train-memory records")
    expected_vectors = count * dimension * 2
    if not vectors.is_file() or vectors.stat().st_size != expected_vectors:
        raise RuntimeError(
            f"Train-memory vectors are missing/truncated: expected={expected_vectors}, "
            f"actual={vectors.stat().st_size if vectors.is_file() else '<missing>'}: {vectors}"
        )
    if not records.is_file():
        raise RuntimeError(f"Train-memory records are missing: {records}")
    rows = 0
    try:
        with records.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict) or not all(
                    key in record for key in ("question_id", "question", "answer")
                ):
                    raise RuntimeError(
                        f"Train-memory record {line_number} has an invalid schema: {records}"
                    )
                rows += 1
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Train-memory records are invalid: {records}: {exc}") from exc
    if rows != count:
        raise RuntimeError(f"Train-memory records={rows} but manifest count={count}: {records}")
    return manifest


def _finite_number(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} phải là số hữu hạn, nhận {value!r}.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} phải là số hữu hạn, nhận {value!r}.")
    return number


def apply_selection_weights(config: dict, selection_payload: object) -> None:
    if not isinstance(selection_payload, dict):
        raise ValueError("--selection-weights phải chứa YAML Object.")
    if selection_payload.get("schema") != SELECTION_WEIGHTS_SCHEMA:
        raise ValueError(
            "Selector weights thuộc pipeline cũ hoặc thiếu schema. Hãy chạy lại "
            "notebook v9.2 selector calibration; expected "
            f"schema={SELECTION_WEIGHTS_SCHEMA!r}."
        )
    selection_override = selection_payload.get("selection")
    if not isinstance(selection_override, dict) or not selection_override:
        raise ValueError("--selection-weights phải chứa selection Object không rỗng.")
    unknown = sorted(set(selection_override) - set(config["selection"]))
    if unknown:
        raise ValueError(f"Selector weights chứa khóa không được hỗ trợ: {unknown}")
    validated = {}
    for key, value in selection_override.items():
        if key == "source_priors":
            if not isinstance(value, dict) or not value:
                raise ValueError("selection.source_priors phải là Object không rỗng.")
            allowed_sources = set(config["selection"].get("source_priors", {}))
            unknown_sources = sorted(set(map(str, value)) - allowed_sources)
            if unknown_sources:
                raise ValueError(
                    f"selection.source_priors chứa source không được hỗ trợ: {unknown_sources}"
                )
            override_priors = {
                str(source): _finite_number(prior, f"selection.source_priors.{source}")
                for source, prior in value.items()
            }
            validated[key] = {
                **config["selection"].get("source_priors", {}),
                **override_priors,
            }
        else:
            validated[key] = _finite_number(value, f"selection.{key}")
    config["selection"].update(validated)
    retrieval_override = selection_payload.get("retrieval")
    if retrieval_override is not None:
        if not isinstance(retrieval_override, dict):
            raise ValueError("retrieval trong tuning YAML phải là Object.")
        unknown_retrieval = sorted(
            set(retrieval_override) - {"train_memory_candidate_threshold"}
        )
        if unknown_retrieval:
            raise ValueError(
                f"Tuning YAML không được ghi đè retrieval runtime: {unknown_retrieval}"
            )
        if "train_memory_candidate_threshold" in retrieval_override:
            threshold = _finite_number(
                retrieval_override["train_memory_candidate_threshold"],
                "retrieval.train_memory_candidate_threshold",
            )
            if not 0.0 <= threshold <= 1.0:
                raise ValueError("train_memory_candidate_threshold phải nằm trong [0, 1].")
            config["retrieval"]["train_memory_candidate_threshold"] = threshold


def _completed_adapter_identity(path: Path) -> str:
    config_path = path / "adapter_config.json"
    weights_path = path / "adapter_model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(f"Adapter thiếu config/trọng số: {path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8"))
    digest.update(b"\0")
    digest.update(sha256_file(weights_path).encode("ascii"))
    return digest.hexdigest()


def _adapter_preference(path: Path) -> tuple[int, int, str]:
    canonical = 0 if path.name.startswith("qwen-official-lora-") else 1
    return canonical, len(path.parts), path.as_posix()


def _validate_generator_adapter(path: Path, expected_scope: str) -> Path:
    required = (
        path / "adapter_config.json",
        path / "adapter_model.safetensors",
        path / "training_complete.json",
        path / "qlora_scope.json",
    )
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        raise FileNotFoundError(f"Qwen adapter chưa hoàn tất hoặc thiếu file: {missing}")
    try:
        scope = json.loads((path / "qlora_scope.json").read_text(encoding="utf-8"))
        complete = json.loads(
            (path / "training_complete.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Metadata Qwen adapter không đọc được: {path}: {error}") from error
    actual_scope = scope.get("train_scope")
    if expected_scope != "any" and actual_scope != expected_scope:
        raise RuntimeError(
            f"Qwen adapter scope={actual_scope!r} không phù hợp expected_scope={expected_scope!r}: "
            f"{path}. Public inference phải dùng scope='full'; selector calibration dùng scope='split'."
        )
    if int(complete.get("global_step", 0)) <= 0:
        raise RuntimeError(f"Qwen adapter không có global_step hoàn tất: {path}")
    return path


def find_generator_adapter(
    root: Path,
    explicit: str | None = None,
    *,
    expected_scope: str = "full",
) -> Path | None:
    if expected_scope not in {"split", "full", "any"}:
        raise ValueError(f"expected_scope không hợp lệ: {expected_scope!r}")
    value = explicit or os.environ.get("LEGALQA_GENERATOR_ADAPTER_DIR")
    if value:
        return _validate_generator_adapter(Path(value), expected_scope)
    completed = sorted(
        {
            path.parent
            for path in root.rglob("training_complete.json")
            if not path.parent.name.startswith("checkpoint-")
            and (path.parent / "adapter_config.json").is_file()
        }
    )
    if not completed:
        return None
    matching = []
    rejected = []
    for path in completed:
        try:
            matching.append(_validate_generator_adapter(path, expected_scope))
        except RuntimeError as error:
            if "không phù hợp expected_scope" in str(error):
                rejected.append(str(error))
                continue
            raise
    if not matching:
        raise RuntimeError(
            f"Tìm thấy Qwen adapter nhưng không có adapter scope={expected_scope!r}: {rejected}"
        )
    identities: dict[str, list[Path]] = {}
    for path in matching:
        identities.setdefault(_completed_adapter_identity(path), []).append(path)
    if len(identities) > 1:
        report = {
            identity[:12]: [str(path) for path in paths]
            for identity, paths in identities.items()
        }
        raise RuntimeError(
            "Có nhiều Qwen LoRA adapter có trọng số KHÁC nhau. Chỉ attach một bản "
            f"hoặc dùng --generator-adapter/LEGALQA_GENERATOR_ADAPTER_DIR: {report}"
        )
    aliases = next(iter(identities.values()))
    selected = min(aliases, key=_adapter_preference)
    if len(aliases) > 1:
        print(
            f"[Adapter] phát hiện các alias cùng SHA-256; chọn {selected}; aliases={aliases}",
            flush=True,
        )
    return selected


def smoke_inference_models(config: dict) -> dict:
    """Exercise the exact four runtime wrappers before expensive retrieval/generation."""
    smoke_config = copy.deepcopy(config)
    smoke_config["generation"]["max_new_tokens"] = 2
    smoke_config["generation"]["max_context_chars"] = 1000
    report = {}

    instance = None
    try:
        instance = DenseEncoder(smoke_config)
        vectors = instance.encode(["Kiểm tra văn bản pháp luật."], batch_size=1)
        if vectors.shape[0] != 1:
            raise RuntimeError(f"Embedding smoke trả shape bất thường: {vectors.shape}")
        report["embedding"] = {"shape": list(vectors.shape)}
    finally:
        unload_model(instance, "embedding smoke")

    instance = None
    try:
        instance = CrossEncoderReranker(smoke_config)
        scores = instance.score("Quy định này áp dụng thế nào?", ["Điều 1. Phạm vi áp dụng."])
        if len(scores) != 1:
            raise RuntimeError(f"Reranker smoke trả {len(scores)} score.")
        report["reranker"] = {"scores": len(scores)}
    finally:
        unload_model(instance, "reranker smoke")

    instance = None
    try:
        instance = QwenTextGenerator(smoke_config)
        answer = instance.generate(
            "Bạn là trợ lý pháp luật.",
            "Dựa vào căn cứ: Điều 1. Phạm vi áp dụng. Hãy trả lời ngắn.",
        )
        report["generator"] = {"decoded_chars": len(answer)}
    finally:
        unload_model(instance, "Qwen smoke")

    instance = None
    try:
        instance = LegalSemanticEncoder(smoke_config)
        vectors = instance.encode(["Điều 1. Phạm vi áp dụng.", "Quy định về phạm vi áp dụng."])
        if vectors.shape[0] != 2:
            raise RuntimeError(f"ViLegalBERT smoke trả shape bất thường: {vectors.shape}")
        report["verifier"] = {"shape": list(vectors.shape)}
    finally:
        unload_model(instance, "ViLegalBERT smoke")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="UIT DSC 2026 LegalQA: full sequential inference, no fine-tuning"
    )
    parser.add_argument("--config", default=str(PROJECT / "configs" / "kaggle_inference_t4.yaml"))
    parser.add_argument("--input-root", default="/kaggle/input")
    parser.add_argument(
        "--questions-file",
        default=None,
        help="Optional question JSON for selector/dev evaluation; default is public-official.json",
    )
    parser.add_argument(
        "--train-file",
        default=None,
        help="Official train.json, or selector_train.json when evaluating grouped dev",
    )
    parser.add_argument(
        "--generator-adapter",
        default=None,
        help="Optional completed Qwen LoRA adapter directory",
    )
    parser.add_argument(
        "--require-generator-adapter",
        action="store_true",
        help="Fail instead of silently falling back to the base Qwen checkpoint",
    )
    parser.add_argument(
        "--expected-adapter-scope",
        choices=("split", "full", "any"),
        default="full",
        help="Guard against using the leakage-safe split adapter for public inference",
    )
    parser.add_argument(
        "--expected-generator-adapter-fingerprint",
        default=None,
        help=(
            "Optional SHA-256 identity guard for the completed QLoRA adapter. "
            "The run stops before retrieval if the attached adapter differs."
        ),
    )
    parser.add_argument(
        "--selection-weights",
        default=None,
        help="Optional YAML produced by evaluate_selector.py",
    )
    parser.add_argument(
        "--expected-run-fingerprint",
        default=None,
        help=(
            "Optional resolved-config identity guard. Use this for controlled "
            "ablations so any unexpected retrieval/generation/selector change fails fast."
        ),
    )
    parser.add_argument(
        "--require-selection-weights",
        action="store_true",
        help="Fail instead of silently using the untuned selector",
    )
    parser.add_argument("--working", default="/kaggle/working/legalqa_inference")
    parser.add_argument("--submission-output", default="/kaggle/working/submission.zip")
    parser.add_argument(
        "--recovery-output",
        default="/kaggle/working/legalqa_recovery_latest.zip",
        help="Atomic cross-session checkpoint ZIP written periodically",
    )
    parser.add_argument(
        "--resume-bundle",
        default=None,
        help=(
            "Recovery ZIP or the directory Kaggle extracted from it, attached "
            "from an earlier inference session"
        ),
    )
    parser.add_argument(
        "--auto-resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-detect one inference recovery ZIP/extracted directory under --input-root",
    )
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--embedding-batch-size", type=int, default=2)
    parser.add_argument("--reranker-batch-size", type=int, default=1)
    parser.add_argument("--verifier-batch-size", type=int, default=4)
    parser.add_argument("--reranker-max-length", type=int, default=512)
    parser.add_argument("--max-context-chars", type=int, default=18000)
    parser.add_argument("--max-new-tokens", type=int, default=900)
    parser.add_argument("--bm25-top-k", type=int, default=140)
    parser.add_argument("--dense-top-k", type=int, default=70)
    parser.add_argument("--fused-top-k", type=int, default=48)
    parser.add_argument("--rerank-top-k", type=int, default=24)
    parser.add_argument("--evidence-top-k", type=int, default=5)
    parser.add_argument(
        "--dense-index-dir",
        default="/kaggle/working/legalqa_dense_index",
        help="Writable/reusable global dense index directory",
    )
    parser.add_argument(
        "--dense-index-input",
        default=None,
        help="Optional completed dense index attached as a read-only Kaggle Dataset",
    )
    parser.add_argument(
        "--require-prebuilt-indexes",
        action="store_true",
        help="Require complete attached dense and train-memory indexes for final inference",
    )
    parser.add_argument("--force-dense-index", action="store_true")
    parser.add_argument(
        "--train-memory-dir",
        default="/kaggle/working/legalqa_dense_index/train_memory",
    )
    parser.add_argument("--train-memory-input", default=None)
    parser.add_argument("--force-train-memory", action="store_true")
    parser.add_argument(
        "--train-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--dense-index-only",
        action="store_true",
        help="Build/checkpoint the global dense index, then exit for a two-session Kaggle run",
    )
    parser.add_argument("--force-index", action="store_true")
    parser.add_argument(
        "--model-smoke",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load and execute all four runtime wrappers once before the long run",
    )
    parser.add_argument(
        "--reuse-retrieval",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--allow-unverified-budget", action="store_true")
    parser.add_argument(
        "--max-runtime-hours",
        type=float,
        default=0.0,
        help="Pause safely after this many hours (0 disables); public notebook passes 11.5",
    )
    args = parser.parse_args()

    positive = {
        "embedding_batch_size": args.embedding_batch_size,
        "reranker_batch_size": args.reranker_batch_size,
        "verifier_batch_size": args.verifier_batch_size,
        "reranker_max_length": args.reranker_max_length,
        "max_context_chars": args.max_context_chars,
        "max_new_tokens": args.max_new_tokens,
        "bm25_top_k": args.bm25_top_k,
        "dense_top_k": args.dense_top_k,
        "fused_top_k": args.fused_top_k,
        "rerank_top_k": args.rerank_top_k,
        "evidence_top_k": args.evidence_top_k,
    }
    invalid = {key: value for key, value in positive.items() if value < 1}
    if invalid:
        raise ValueError(f"Các tham số phải >= 1: {invalid}")
    if not args.device.startswith("cuda"):
        raise ValueError("Notebook inference-only yêu cầu device CUDA, ví dụ cuda:0.")
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every phải >= 1.")

    if not math.isfinite(args.max_runtime_hours) or args.max_runtime_hours < 0:
        raise ValueError("--max-runtime-hours must be finite and >= 0.")
    if args.force_dense_index and args.dense_index_input:
        raise ValueError("Do not combine --force-dense-index with a read-only --dense-index-input.")
    if args.force_train_memory and args.train_memory_input:
        raise ValueError("Do not combine --force-train-memory with --train-memory-input.")

    started = time.time()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    input_root = Path(args.input_root)
    working = Path(args.working)
    submission = Path(args.submission_output)
    working.mkdir(parents=True, exist_ok=True)
    submission.parent.mkdir(parents=True, exist_ok=True)
    # Never leave a stale file that could be mistaken for this run's output if
    # a fail-fast artifact/version check aborts before generation.
    submission.unlink(missing_ok=True)
    environment = require_environment()

    official_public_path = find_unique_file(input_root, "public-official.json")
    public_path = Path(args.questions_file) if args.questions_file else official_public_path
    if not public_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy --questions-file: {public_path}")
    data_root = official_public_path.parent
    official_train_path = data_root / "train.json"
    train_path = Path(args.train_file) if args.train_file else official_train_path
    if not train_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy official/selector train file: {train_path}")
    train_records = load_qa(train_path, require_answers=True)
    sibling_corpus = data_root / "selected-contexts"
    corpus_path = sibling_corpus if sibling_corpus.is_dir() else find_corpus(data_root)

    discovered = {role: find_model(input_root, role) for role in RUNTIME_MODEL_ROLES}
    generator_adapter = find_generator_adapter(
        input_root,
        args.generator_adapter,
        expected_scope=args.expected_adapter_scope,
    )
    if args.require_generator_adapter and generator_adapter is None:
        raise FileNotFoundError(
            "Final inference requires the completed full QLoRA adapter, but none was found. "
            "Attach the adapter Dataset or pass --generator-adapter."
        )
    if args.require_selection_weights and not args.selection_weights:
        raise FileNotFoundError(
            "Final inference requires tuned_selection.yaml, but --selection-weights was not set."
        )
    if args.selection_weights and not Path(args.selection_weights).is_file():
        raise FileNotFoundError(f"Selection weights do not exist: {args.selection_weights}")
    if args.require_prebuilt_indexes and not args.dense_index_input:
        raise FileNotFoundError(
            "Final inference requires an attached dense index; pass --dense-index-input."
        )
    missing = sorted(role for role in RUNTIME_MODEL_ROLES if discovered[role] is None)
    if missing:
        raise FileNotFoundError(
            f"Inference v9 cần 4 checkpoint local (không dùng BARTpho); đang thiếu: {missing}. "
            "Attach Dataset checkpoint đã được BTC duyệt."
        )

    preflight_path = working / "preflight.json"
    run_command(
        [
            sys.executable,
            str(PROJECT / "scripts" / "preflight_kaggle.py"),
            "--input-root",
            str(input_root),
            "--profile",
            "balanced",
            "--inference-only",
            "--output",
            str(preflight_path),
        ]
    )
    environment["preflight"] = json.loads(preflight_path.read_text(encoding="utf-8"))

    print("[Fingerprint] hashing public questions and official corpus", flush=True)
    input_identity = {
        "questions_sha256": sha256_file(public_path),
        "train_sha256": sha256_file(train_path),
        "corpus_sha256": corpus_content_fingerprint(corpus_path),
    }
    print("[Fingerprint] identifying four local checkpoints", flush=True)
    model_identity = {
        role: checkpoint_fingerprint(path)
        for role, path in discovered.items()
    }
    model_identity["generator_adapter"] = (
        adapter_fingerprint(generator_adapter) if generator_adapter else None
    )
    if (
        args.expected_generator_adapter_fingerprint
        and model_identity["generator_adapter"]
        != args.expected_generator_adapter_fingerprint
    ):
        raise RuntimeError(
            "Generator adapter fingerprint mismatch: "
            f"expected={args.expected_generator_adapter_fingerprint}, "
            f"actual={model_identity['generator_adapter']}"
        )

    config = load_config(args.config)
    config["paths"]["warmup"] = None
    config["paths"]["corpus"] = str(corpus_path)
    for role, path in discovered.items():
        config["paths"]["models"][role] = str(path)
    config["paths"]["models"]["bartpho"] = None
    config["paths"]["models"]["generator_adapter"] = (
        str(generator_adapter) if generator_adapter else None
    )
    style_profile = build_answer_style_profile(train_records)
    config["generation"]["answer_style_profile"] = style_profile
    config["retrieval"]["train_memory_enabled"] = bool(args.train_memory)
    if args.selection_weights:
        selection_payload = yaml.safe_load(
            Path(args.selection_weights).read_text(encoding="utf-8")
        )
        apply_selection_weights(config, selection_payload)
    config["runtime"].update(
        device=args.device,
        dtype="float16",
        batch_size=1,
        embedding_batch_size=args.embedding_batch_size,
        reranker_batch_size=args.reranker_batch_size,
        verifier_batch_size=args.verifier_batch_size,
        reranker_max_length=args.reranker_max_length,
        unload_after_stage=True,
        embedding_identity=model_identity["embedding"],
        corpus_identity=input_identity["corpus_sha256"],
        train_identity=input_identity["train_sha256"],
        train_path=str(train_path),
        max_runtime_hours=args.max_runtime_hours,
        runtime_started_unix=started,
        stop_at_unix=(
            started + args.max_runtime_hours * 3600.0
            if args.max_runtime_hours > 0
            else None
        ),
    )
    config["retrieval"].update(
        bm25_top_k=args.bm25_top_k,
        dense_top_k=args.dense_top_k,
        fused_top_k=args.fused_top_k,
        rerank_top_k=args.rerank_top_k,
        evidence_top_k=args.evidence_top_k,
    )
    config["generation"].update(
        qwen_enabled=True,
        bartpho_enabled=False,
        extractive_enabled=True,
        max_context_chars=args.max_context_chars,
        max_new_tokens=args.max_new_tokens,
        simple_max_new_tokens=min(
            args.max_new_tokens,
            int(config["generation"].get("simple_max_new_tokens", args.max_new_tokens)),
        ),
        complex_max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )
    runtime_fingerprint = {
        key: value
        for key, value in config["runtime"].items()
        if not key.endswith(("_path", "_dir", "_root", "_output"))
        and key not in EPHEMERAL_RUNTIME_KEYS
    }
    fingerprint_payload = {
        "cache_schema": CACHE_SCHEMA_VERSION,
        "inputs": input_identity,
        "models": model_identity,
        "runtime": runtime_fingerprint,
        "corpus": config["corpus"],
        "retrieval": config["retrieval"],
        "routing": config.get("routing", {}),
        "generation": config["generation"],
        "selection": config["selection"],
    }
    run_fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    if args.expected_run_fingerprint and run_fingerprint != args.expected_run_fingerprint:
        raise RuntimeError(
            "Run fingerprint mismatch; this is not the requested controlled ablation: "
            f"expected={args.expected_run_fingerprint}, actual={run_fingerprint}."
        )
    dense_index_dir = Path(args.dense_index_input or args.dense_index_dir)
    if args.dense_index_input:
        validate_dense_index_input(
            dense_index_dir,
            corpus_identity=input_identity["corpus_sha256"],
            embedding_identity=model_identity["embedding"],
        )
    if args.force_dense_index:
        resolved_dense = dense_index_dir.resolve()
        resolved_working = Path("/kaggle/working").resolve()
        try:
            resolved_dense.relative_to(resolved_working)
        except ValueError as exc:
            raise RuntimeError(
                "--force-dense-index chỉ được xóa index dưới /kaggle/working."
            ) from exc
        if resolved_dense.exists():
            shutil.rmtree(resolved_dense)
    nested_memory = dense_index_dir / "train_memory"
    train_memory_dir = Path(
        args.train_memory_input
        or (
            nested_memory
            if (
                args.dense_index_input
                and not args.train_file
                and (nested_memory / "manifest.json").is_file()
            )
            else args.train_memory_dir
        )
    )
    memory_validated = False
    if args.train_memory_input or (
        args.dense_index_input and (train_memory_dir / "manifest.json").is_file()
    ):
        validate_train_memory_input(
            train_memory_dir,
            train_identity=input_identity["train_sha256"],
            embedding_identity=model_identity["embedding"],
        )
        memory_validated = True
    if args.require_prebuilt_indexes and args.train_memory:
        if not (train_memory_dir / "manifest.json").is_file():
            raise FileNotFoundError(
                "Final inference requires a completed train-memory index. Attach it under "
                "<dense-index>/train_memory or pass --train-memory-input."
            )
        if not memory_validated:
            validate_train_memory_input(
                train_memory_dir,
                train_identity=input_identity["train_sha256"],
                embedding_identity=model_identity["embedding"],
            )
    if args.force_train_memory:
        resolved_memory = train_memory_dir.resolve()
        resolved_working = Path("/kaggle/working").resolve()
        try:
            resolved_memory.relative_to(resolved_working)
        except ValueError as exc:
            raise RuntimeError(
                "--force-train-memory chỉ được xóa index dưới /kaggle/working."
            ) from exc
        if resolved_memory.exists():
            shutil.rmtree(resolved_memory)
    config["runtime"]["dense_index_dir"] = str(dense_index_dir)
    config["runtime"]["train_memory_dir"] = str(train_memory_dir)
    config["paths"]["artifacts"] = str(working / "artifacts" / run_fingerprint)
    write_json(working / "answer_style_profile.json", style_profile)
    restored_recovery = None
    resume_input = args.resume_bundle
    if resume_input is None and args.auto_resume and not args.dense_index_only:
        discovered_recovery = discover_recovery_inputs(
            input_root,
            expected_fingerprint=run_fingerprint,
        )
        if len(discovered_recovery) > 1:
            raise RuntimeError(
                "Phát hiện nhiều inference recovery input; chỉ attach bản mới nhất "
                f"hoặc chỉ định --resume-bundle: {discovered_recovery}"
            )
        if discovered_recovery:
            resume_input = str(discovered_recovery[0])
            print(f"[Recovery] auto-detected: {resume_input}", flush=True)
    if resume_input:
        restored_recovery = restore_recovery_bundle(
            resume_input,
            working,
            expected_fingerprint=run_fingerprint,
        )
        write_json(preflight_path, environment["preflight"])
    config["runtime"].update(
        recovery_root=str(working),
        recovery_output=str(Path(args.recovery_output)),
        run_fingerprint=run_fingerprint,
        checkpoint_every=args.checkpoint_every,
    )
    resolved_config = working / "resolved_config.yaml"
    write_resolved_config(config, resolved_config)
    recovery = RecoveryManager.from_config(config)

    try:
        budget_report = count_system_budget(config)
        if not budget_report["strictly_below_4b"]:
            raise RuntimeError(f"Tổng tham số không còn dưới 4B: {budget_report}")
    except Exception as exc:
        budget_report = {"error": repr(exc), "must_verify_before_submission": True}
        if not args.allow_unverified_budget:
            raise RuntimeError("Không xác minh được parameter budget <4B; dừng trước inference.") from exc

    smoke_started = time.time()
    model_smoke_report = smoke_inference_models(config) if args.model_smoke else {"skipped": True}
    smoke_minutes = (time.time() - smoke_started) / 60
    database = working / "official_corpus.sqlite"
    index_started = time.time()
    index_report = build_sqlite_store(config, corpus_path, database, force=args.force_index)
    index_minutes = (time.time() - index_started) / 60

    if args.dense_index_only:
        encoder = None
        memory_report = {"enabled": False}
        try:
            encoder = DenseEncoder(config)
            with SQLiteLegalStore(database, readonly=True) as store:
                dense_report = ensure_global_dense_index(
                    config, store, encoder, config["runtime"]["dense_index_dir"]
                )
                if args.train_memory:
                    memory_report = ensure_train_memory_index(
                        config,
                        encoder,
                        train_path,
                        config["runtime"]["train_memory_dir"],
                    )
        finally:
            unload_model(encoder, "global embedding index-only")
        manifest = {
            "status": "complete",
            "mode": "global_dense_index_only",
            "training_performed": False,
            "run_fingerprint": run_fingerprint,
            "inputs": {
                "questions": str(public_path),
                "official_public": str(official_public_path),
                "train": str(train_path),
                "corpus": str(corpus_path),
                **input_identity,
            },
            "models": {key: str(value) if value else None for key, value in config["paths"]["models"].items()},
            "model_identity": model_identity,
            "index": index_report,
            "global_dense_index": dense_report,
            "train_memory_index": memory_report,
            "answer_style_profile": str((working / "answer_style_profile.json").resolve()),
            "budget": budget_report,
            "timing_minutes": {"total": (time.time() - started) / 60},
        }
        write_json(working / "run_manifest.json", manifest)
        print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
        print(
            "\nDENSE INDEX HOÀN TẤT. Save Version, tạo Kaggle Dataset từ "
            f"{config['runtime']['dense_index_dir']} (gồm train_memory), rồi attach vào phiên inference.",
            flush=True,
        )
        return

    questions = load_qa(public_path, require_answers=False)
    retrieval_cache = working / "cache" / run_fingerprint / "retrieval.v7.jsonl"
    retrieval_started = time.time()
    evidence = retrieve_sqlite_batch(
        config,
        database,
        questions,
        retrieval_cache,
        reuse_cache=args.reuse_retrieval and not args.force_index,
    )
    retrieval_minutes = (time.time() - retrieval_started) / 60
    if recovery:
        recovery.snapshot("retrieval", len(evidence), len(questions), force=True)

    predictions = working / "predictions.json"
    generation_started = time.time()
    try:
        if config["generation"].get("agentic_enabled", False):
            with sqlite_bm25_search_fn(config, database) as agentic_search_fn:
                generation_report = generate_from_evidence(
                    config, questions, evidence, predictions, agentic_search_fn
                )
        else:
            generation_report = generate_from_evidence(config, questions, evidence, predictions)
    except InferencePaused as pause:
        generation_minutes = (time.time() - generation_started) / 60
        manifest = {
            "status": "paused",
            "mode": "full_sequential_inference_only",
            "training_performed": False,
            "run_fingerprint": run_fingerprint,
            "restored_recovery": restored_recovery,
            "pause": {
                "stage": pause.stage,
                "completed": pause.completed,
                "total": pause.total,
                "reason": "runtime_budget",
            },
            "inputs": {
                "questions": str(public_path),
                "official_public": str(official_public_path),
                "train": str(train_path),
                "corpus": str(corpus_path),
                **input_identity,
            },
            "models": {
                key: str(value) if value else None
                for key, value in config["paths"]["models"].items()
            },
            "model_identity": model_identity,
            "model_smoke": model_smoke_report,
            "runtime": config["runtime"],
            "index": index_report,
            "global_dense_index": getattr(evidence, "index_report", {}),
            "train_memory_index": getattr(evidence, "memory_report", {}),
            "budget": budget_report,
            "timing_minutes": {
                "model_smoke": smoke_minutes,
                "index": index_minutes,
                "retrieval": retrieval_minutes,
                "generation_partial": generation_minutes,
                "total": (time.time() - started) / 60,
            },
        }
        write_json(working / "run_manifest.json", manifest)
        if recovery:
            recovery.snapshot(
                "paused", pause.completed, pause.total, force=True
            )
        print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
        print(
            "\nPAUSED SAFELY: save legalqa_recovery_latest.zip as a Kaggle Dataset, "
            "attach it to the next session, and resume. No submission was created yet.",
            flush=True,
        )
        return
    generation_minutes = (time.time() - generation_started) / 60
    submission_report = create_submission(public_path, predictions, submission)

    manifest = {
        "status": "complete",
        "mode": "full_sequential_inference_only",
        "training_performed": False,
        "run_fingerprint": run_fingerprint,
        "restored_recovery": restored_recovery,
        "environment": environment,
        "inputs": {
            "questions": str(public_path),
            "official_public": str(official_public_path),
            "train": str(train_path),
            "corpus": str(corpus_path),
            **input_identity,
        },
        "models": {key: str(value) if value else None for key, value in config["paths"]["models"].items()},
        "model_identity": model_identity,
        "model_smoke": model_smoke_report,
        "runtime": config["runtime"],
        "retrieval_config": config["retrieval"],
        "generation_config": config["generation"],
        "index": index_report,
        "global_dense_index": getattr(evidence, "index_report", {}),
        "train_memory_index": getattr(evidence, "memory_report", {}),
        "answer_style_profile": str((working / "answer_style_profile.json").resolve()),
        "budget": budget_report,
        "generation": generation_report,
        "submission": submission_report,
        "timing_minutes": {
            "model_smoke": smoke_minutes,
            "index": index_minutes,
            "retrieval": retrieval_minutes,
            "generation_and_selection": generation_minutes,
            "total": (time.time() - started) / 60,
        },
    }
    write_json(working / "run_manifest.json", manifest)
    if recovery:
        recovery.snapshot("complete", len(questions), len(questions), force=True)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    print(f"\nHOÀN TẤT INFERENCE-ONLY: {submission} ({submission.stat().st_size:,} bytes)", flush=True)


if __name__ == "__main__":
    main()
