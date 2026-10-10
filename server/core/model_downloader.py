"""Selected RK ASR + Matcha/Vocos artifacts; preserves OVS integrity checks."""
from __future__ import annotations
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
logger = logging.getLogger(__name__)
_MATCHA_MANIFEST_PATH = Path(__file__).with_name("matcha_artifacts.json")


def _load_matcha_manifest() -> dict:
    """Load and minimally validate the release-owned Matcha artifact lock."""
    try:
        manifest = json.loads(_MATCHA_MANIFEST_PATH.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Matcha artifact lock is unavailable: {_MATCHA_MANIFEST_PATH}: {exc}"
        ) from exc
    if manifest.get("model_id") != "matcha-icefall-zh-en":
        raise RuntimeError("Matcha artifact lock has the wrong model_id")
    return manifest

def _sha256_file(path: Path, bufsize: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as src:
        while chunk := src.read(bufsize):
            digest.update(chunk)
    return digest.hexdigest()

def _verify_locked_file(path: Path, lock: dict, *, label: str) -> None:
    """Fail closed unless ``path`` matches both published size and SHA256."""
    expected_sha = lock.get("sha256")
    expected_size = lock.get("size")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise RuntimeError(f"{label} has no valid published SHA256 lock")
    if not isinstance(expected_size, int) or expected_size <= 0:
        raise RuntimeError(f"{label} has no valid published size lock")
    try:
        actual_size = path.stat().st_size
    except OSError as exc:
        raise RuntimeError(f"{label} is missing: {path}") from exc
    if actual_size != expected_size:
        raise RuntimeError(
            f"{label} size mismatch: expected {expected_size}, got {actual_size}"
        )
    actual_sha = _sha256_file(path)
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"{label} SHA256 mismatch: expected {expected_sha}, got {actual_sha}"
        )

def _matcha_model_files_valid(model_path: Path) -> bool:
    """Return True only for the exact release-locked Matcha runtime inputs."""
    files = _load_matcha_manifest()["model_bundle"]["required_files"]
    try:
        for name, lock in files.items():
            _verify_locked_file(model_path / name, lock, label=f"Matcha {name}")
    except RuntimeError as exc:
        logger.warning("Matcha model cache is not release-valid: %s", exc)
        return False
    return True

def _detect_tar_mode(filename: str) -> str:
    """Return tar open mode based on filename extension."""
    if filename.endswith(".tar.bz2"):
        return "bz2"
    return "gz"

def _download_and_extract(
    url: str,
    dest_dir: str,
    *,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
) -> None:
    """Download a .tar.gz or .tar.bz2 from URL and extract to dest_dir.

    Uses curl (fast, with progress) if available, falls back to Python stdlib.
    """
    compress = _detect_tar_mode(url)

    # Release-locked artifacts must be downloaded in full before extraction.
    # The historical curl|tar path cannot hash the archive and can leave a
    # partially extracted model directory on transport failure.
    if expected_sha256 is not None or expected_size is not None:
        if expected_sha256 is None or expected_size is None:
            raise RuntimeError("verified archive downloads require SHA256 and size")
        suffix = ".tar.bz2" if compress == "bz2" else ".tar.gz"
        with tempfile.TemporaryDirectory(prefix="matcha_download_") as tmpdir:
            archive = Path(tmpdir) / f"artifact{suffix}"
            if shutil.which("curl"):
                subprocess.run(
                    ["curl", "-fSL", "--progress-bar", url, "-o", str(archive)],
                    check=True,
                )
            else:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "openvoicestream/1.0"}
                )
                with urllib.request.urlopen(req, timeout=600) as resp, archive.open("wb") as out:
                    shutil.copyfileobj(resp, out, length=1 << 20)
            _verify_locked_file(
                archive,
                {"sha256": expected_sha256, "size": expected_size},
                label="Matcha model bundle",
            )
            with tarfile.open(archive, f"r:{compress}") as tar:
                for member in tar.getmembers():
                    if member.name.startswith("/") or ".." in Path(member.name).parts:
                        raise RuntimeError(f"unsafe Matcha archive member: {member.name}")
                tar.extractall(path=dest_dir)
        return

    raise RuntimeError("archive size and SHA256 locks are mandatory")


def _ensure_rk_artifacts() -> None:
    """Verify or download RK model artifacts when an RK manifest is configured."""
    try:
        from server.core.rk_artifacts import ensure_rk_artifacts
        ensure_rk_artifacts()
    except Exception as exc:
        logger.error("RK artifact check/download failed: %s", exc)
        sys.exit(1)

def ensure_models(language_mode: str = "rk", model_dir: str = "/opt/tts/models",
                  qwen3_required_files=None) -> None:
    if language_mode != "rk":
        raise ValueError("SO-101 product requires LANGUAGE_MODE=rk")
    _ensure_rk_artifacts()
    if os.environ.get("RK_ENSURE_MATCHA_RESOURCES", "1").lower() in ("0", "false", "no"):
        return
    model_dir = os.environ.get("TTS_MODEL_DIR") or model_dir
    model_path = Path(model_dir) / "matcha-icefall-zh-en"
    if _matcha_model_files_valid(model_path):
        return
    if os.environ.get("OVS_AUTO_DOWNLOAD_ARTIFACTS", "1").lower() in ("0", "false", "no", "off"):
        raise RuntimeError("Matcha resources missing or invalid; downloads disabled")
    bundle = _load_matcha_manifest()["model_bundle"]
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    _download_and_extract(bundle["url"], model_dir, expected_sha256=bundle["sha256"], expected_size=bundle["size"])
    if not _matcha_model_files_valid(model_path):
        raise RuntimeError("Matcha archive failed required-file integrity checks")
