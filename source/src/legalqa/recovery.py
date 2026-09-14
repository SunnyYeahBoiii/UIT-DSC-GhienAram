from __future__ import annotations

import hashlib
import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


RECOVERY_STATE = "recovery_state.json"


class RecoveryError(RuntimeError):
    """Raised when a recovery bundle is unsafe or belongs to another run."""


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _checkpoint_files(root: Path, fingerprint: str) -> list[Path]:
    """Return only small state that is expensive to reproduce.

    The SQLite corpus store and model checkpoints are deliberately excluded. They
    are deterministic inputs and would make each recovery ZIP several GB larger.
    """
    files: set[Path] = set()
    for name in (
        "resolved_config.yaml",
        "preflight.json",
        "predictions.json",
        "predictions.trace.jsonl",
        "run_manifest.json",
        "run_manifest.incomplete.json",
        RECOVERY_STATE,
    ):
        candidate = root / name
        if candidate.is_file():
            files.add(candidate)

    cache_root = root / "cache" / fingerprint
    if cache_root.is_dir():
        files.update(path for path in cache_root.rglob("*") if path.is_file())

    candidate_cache = root / "artifacts" / fingerprint / "candidate_cache"
    if candidate_cache.is_dir():
        files.update(path for path in candidate_cache.rglob("*") if path.is_file())
    return sorted(files)


def create_recovery_bundle(
    root: str | Path,
    output: str | Path,
    *,
    fingerprint: str,
    stage: str,
    completed: int,
    total: int,
) -> dict[str, Any]:
    root_path = Path(root).resolve()
    output_path = Path(output).resolve()
    state = {
        "format_version": 1,
        "run_fingerprint": fingerprint,
        "stage": stage,
        "completed": int(completed),
        "total": int(total),
    }
    _atomic_write_json(root_path / RECOVERY_STATE, state)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    files = _checkpoint_files(root_path, fingerprint)
    with zipfile.ZipFile(
        temporary,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for path in files:
            archive.write(path, path.relative_to(root_path).as_posix())
    os.replace(temporary, output_path)
    state.update(
        bundle=str(output_path),
        files=len(files),
        bytes=output_path.stat().st_size,
    )
    print(
        f"[Recovery] stage={stage} completed={completed:,}/{total:,} "
        f"files={len(files):,} zip={output_path} ({output_path.stat().st_size:,} bytes)",
        flush=True,
    )
    return state


def _read_bundle_state(bundle: Path) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(bundle) as archive:
            state = json.loads(archive.read(RECOVERY_STATE).decode("utf-8"))
            if not isinstance(state, dict):
                raise TypeError("recovery_state.json must be a JSON object")
            return state
    except (
        KeyError,
        OSError,
        TypeError,
        json.JSONDecodeError,
        UnicodeDecodeError,
        zipfile.BadZipFile,
    ) as exc:
        raise RecoveryError(f"Recovery ZIP không hợp lệ: {bundle}") from exc


def _read_directory_state(directory: Path) -> dict[str, Any]:
    state_path = directory / RECOVERY_STATE
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(state, dict):
            raise TypeError("recovery_state.json must be a JSON object")
        return state
    except (
        OSError,
        TypeError,
        json.JSONDecodeError,
        UnicodeDecodeError,
    ) as exc:
        raise RecoveryError(f"Thư mục recovery không hợp lệ: {directory}") from exc


def recovery_input_state(path: str | Path) -> dict[str, Any]:
    """Read metadata from either an atomic ZIP or Kaggle-extracted recovery."""
    recovery_path = Path(path).resolve()
    if not recovery_path.exists():
        raise FileNotFoundError(f"Không tìm thấy recovery input: {recovery_path}")
    if recovery_path.is_dir():
        return _read_directory_state(recovery_path)
    if recovery_path.is_file():
        return _read_bundle_state(recovery_path)
    raise RecoveryError(f"Recovery input không phải ZIP/thư mục: {recovery_path}")


def recovery_content_fingerprint(path: str | Path) -> str:
    """Path-independent identity for ZIP and extracted recovery aliases."""
    recovery_path = Path(path).resolve()
    state = recovery_input_state(recovery_path)
    fingerprint = str(state.get("run_fingerprint", ""))
    digest = hashlib.sha256()
    if recovery_path.is_dir():
        files = _checkpoint_files(recovery_path, fingerprint)
        if not files or recovery_path / RECOVERY_STATE not in files:
            raise RecoveryError(f"Thư mục recovery thiếu {RECOVERY_STATE}: {recovery_path}")
        for item in files:
            relative = item.relative_to(recovery_path).as_posix()
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            with item.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            digest.update(b"\0")
    else:
        try:
            with zipfile.ZipFile(recovery_path) as archive:
                files = sorted(
                    (info for info in archive.infolist() if not info.is_dir()),
                    key=lambda info: info.filename,
                )
                for info in files:
                    digest.update(info.filename.encode("utf-8"))
                    digest.update(b"\0")
                    with archive.open(info) as handle:
                        for block in iter(lambda: handle.read(1024 * 1024), b""):
                            digest.update(block)
                    digest.update(b"\0")
        except (OSError, zipfile.BadZipFile) as exc:
            raise RecoveryError(f"Recovery ZIP không hợp lệ: {recovery_path}") from exc
    return digest.hexdigest()


def _restore_recovery_directory(
    source_root: Path,
    target_root: Path,
    *,
    fingerprint: str,
) -> int:
    """Restore the allow-listed recovery payload Kaggle extracted from a ZIP."""
    files = _checkpoint_files(source_root, fingerprint)
    if not files or source_root / RECOVERY_STATE not in files:
        raise RecoveryError(f"Thư mục recovery thiếu {RECOVERY_STATE}: {source_root}")
    if len(files) > 100_000:
        raise RecoveryError(f"Recovery directory contains too many files: {len(files):,}")

    targets: set[str] = set()
    transfers: list[tuple[Path, Path]] = []
    total_bytes = 0
    for source in files:
        if source.is_symlink():
            raise RecoveryError(f"Recovery directory contains a symlink: {source}")
        resolved_source = source.resolve()
        try:
            relative = resolved_source.relative_to(source_root)
        except ValueError as exc:
            raise RecoveryError(f"Recovery file vượt source root: {source}") from exc
        target = (target_root / relative).resolve()
        try:
            target.relative_to(target_root)
        except ValueError as exc:
            raise RecoveryError(f"Đường dẫn vượt recovery root: {relative}") from exc
        target_key = str(target).casefold()
        if target_key in targets:
            raise RecoveryError(f"Duplicate recovery target: {relative}")
        targets.add(target_key)
        total_bytes += resolved_source.stat().st_size
        transfers.append((resolved_source, target))

    if total_bytes > 8 * 1024**3:
        raise RecoveryError(
            f"Recovery directory is unexpectedly large: {total_bytes:,} bytes"
        )

    restored = 0
    for resolved_source, target in transfers:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.restore.tmp")
        with resolved_source.open("rb") as source_handle, temporary.open("wb") as destination:
            while True:
                block = source_handle.read(1024 * 1024)
                if not block:
                    break
                destination.write(block)
        os.replace(temporary, target)
        restored += 1
    return restored


def restore_recovery_bundle(
    bundle: str | Path,
    root: str | Path,
    *,
    expected_fingerprint: str,
) -> dict[str, Any]:
    bundle_path = Path(bundle).resolve()
    root_path = Path(root).resolve()
    state = recovery_input_state(bundle_path)
    actual = str(state.get("run_fingerprint", ""))
    if actual != expected_fingerprint:
        raise RecoveryError(
            "Recovery input thuộc cấu hình/model khác: "
            f"expected={expected_fingerprint}, actual={actual or '<missing>'}."
        )

    root_path.mkdir(parents=True, exist_ok=True)
    if bundle_path.is_dir():
        restored = _restore_recovery_directory(
            bundle_path,
            root_path,
            fingerprint=actual,
        )
    else:
        restored = 0
        with zipfile.ZipFile(bundle_path) as archive:
            files = [info for info in archive.infolist() if not info.is_dir()]
            if len(files) > 100_000:
                raise RecoveryError(f"Recovery ZIP contains too many files: {len(files):,}")
            total_uncompressed = sum(int(info.file_size) for info in files)
            if total_uncompressed > 8 * 1024**3:
                raise RecoveryError(
                    f"Recovery ZIP is unexpectedly large when extracted: {total_uncompressed:,} bytes"
                )
            targets: set[str] = set()
            for info in files:
                if "\\" in info.filename or ":" in info.filename or "\x00" in info.filename:
                    raise RecoveryError(f"Unsafe path in recovery ZIP: {info.filename}")
                relative = PurePosixPath(info.filename)
                if relative.is_absolute() or ".." in relative.parts:
                    raise RecoveryError(f"Đường dẫn không an toàn trong recovery ZIP: {info.filename}")
                target = root_path.joinpath(*relative.parts).resolve()
                try:
                    target.relative_to(root_path)
                except ValueError as exc:
                    raise RecoveryError(f"Đường dẫn vượt recovery root: {info.filename}") from exc
                target_key = str(target).casefold()
                if target_key in targets:
                    raise RecoveryError(f"Duplicate recovery target: {info.filename}")
                targets.add(target_key)
            for info in files:
                relative = PurePosixPath(info.filename)
                target = root_path.joinpath(*relative.parts).resolve()
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(f".{target.name}.restore.tmp")
                with archive.open(info) as source, temporary.open("wb") as destination:
                    while True:
                        block = source.read(1024 * 1024)
                        if not block:
                            break
                        destination.write(block)
                os.replace(temporary, target)
                restored += 1
    print(
        f"[Recovery] restored {restored:,} files from {bundle_path}; "
        f"stage={state.get('stage')} completed={state.get('completed')}/{state.get('total')}",
        flush=True,
    )
    return {**state, "restored_files": restored, "source_bundle": str(bundle_path)}


@dataclass
class RecoveryManager:
    root: Path
    output: Path
    fingerprint: str
    every: int = 50
    _last_snapshot: tuple[str, int] | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "RecoveryManager | None":
        runtime = config.get("runtime", {})
        root = runtime.get("recovery_root")
        output = runtime.get("recovery_output")
        fingerprint = runtime.get("run_fingerprint")
        if not (root and output and fingerprint):
            return None
        every = max(1, int(runtime.get("checkpoint_every", 50)))
        return cls(Path(root), Path(output), str(fingerprint), every)

    def snapshot(self, stage: str, completed: int, total: int, *, force: bool = False) -> dict[str, Any] | None:
        marker = (stage, int(completed))
        if marker == self._last_snapshot:
            return None
        if not force and (completed <= 0 or completed % self.every != 0):
            return None
        report = create_recovery_bundle(
            self.root,
            self.output,
            fingerprint=self.fingerprint,
            stage=stage,
            completed=completed,
            total=total,
        )
        self._last_snapshot = marker
        return report
