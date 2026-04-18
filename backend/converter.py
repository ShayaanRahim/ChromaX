"""
converter.py — Converts raw SCIEX .wiff / .wiff2 files to .mzML format.

Uses the ProteoWizard msconvert CLI under the hood.  The path to msconvert
must be configured in config.py (or overridden via the MSCONVERT_PATH
environment variable).

Public API
----------
convert_file(raw_path, output_dir) -> Path
    Convert a single .wiff file and return the .mzML output path.

convert_directory(raw_dir, output_dir) -> list[Path]
    Batch-convert every .wiff / .wiff2 file in a directory.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_msconvert_command(
    input_path: Path,
    output_dir: Path,
) -> list[str]:
    """
    Construct the msconvert CLI argument list for a single input file.

    Parameters
    ----------
    input_path:
        Absolute path to the .wiff / .wiff2 file.
    output_dir:
        Directory where the .mzML output will be written.

    Returns
    -------
    list[str]
        Ordered list of command tokens ready for ``subprocess.run``.
    """
    cmd: list[str] = [
        config.MSCONVERT_PATH,
        str(input_path),
        "--mzML",
        "-o",
        str(output_dir),
    ]

    if config.MSCONVERT_32BIT:
        cmd.append("--32")

    for extra_filter in config.MSCONVERT_EXTRA_FILTERS:
        cmd.extend(["--filter", extra_filter])

    return cmd


def _expected_mzml_path(raw_path: Path, output_dir: Path) -> Path:
    """
    Return the .mzML path that msconvert will produce for *raw_path*.

    MSConvert replaces the source extension with .mzML and writes the file
    into output_dir.

    Parameters
    ----------
    raw_path:
        Source .wiff / .wiff2 file.
    output_dir:
        msconvert output directory.

    Returns
    -------
    Path
        Expected absolute path of the converted .mzML file.
    """
    return output_dir / (raw_path.stem + ".mzML")


def _validate_msconvert_binary() -> None:
    """
    Confirm that the msconvert binary exists and is executable.

    Raises
    ------
    FileNotFoundError
        If the binary is not found at the configured path.
    PermissionError
        If the binary exists but is not executable.
    """
    binary = config.MSCONVERT_PATH

    # Allow the string "msconvert" (no path) to resolve via PATH.
    if os.sep not in binary and shutil.which(binary) is not None:
        return

    binary_path = Path(binary)
    if not binary_path.exists():
        raise FileNotFoundError(
            f"msconvert not found at '{binary}'. "
            "Install ProteoWizard and set MSCONVERT_PATH in config.py "
            "or the MSCONVERT_PATH environment variable."
        )
    if not os.access(binary_path, os.X_OK):
        raise PermissionError(
            f"msconvert binary at '{binary}' is not executable. "
            "Check file permissions."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def convert_file(
    raw_path: str | Path,
    output_dir: Optional[str | Path] = None,
) -> Path:
    """
    Convert a single .wiff or .wiff2 file to .mzML using msconvert.

    Conversion is idempotent: if the expected .mzML output already exists,
    the function returns immediately without re-running msconvert.

    Parameters
    ----------
    raw_path:
        Path to the source .wiff / .wiff2 file.
    output_dir:
        Directory where the .mzML file will be written.  Defaults to
        ``config.CONVERTED_DIR``.

    Returns
    -------
    Path
        Absolute path of the produced .mzML file.

    Raises
    ------
    FileNotFoundError
        If ``raw_path`` does not exist or msconvert is not found.
    RuntimeError
        If msconvert exits with a non-zero return code.
    """
    raw_path = Path(raw_path).resolve()
    output_dir = Path(output_dir or config.CONVERTED_DIR).resolve()

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    mzml_path = _expected_mzml_path(raw_path, output_dir)

    if mzml_path.exists():
        logger.info("Skipping conversion (already exists): %s", mzml_path.name)
        return mzml_path

    _validate_msconvert_binary()

    cmd = _build_msconvert_command(raw_path, output_dir)
    logger.info("Converting %s → %s", raw_path.name, mzml_path.name)
    logger.debug("msconvert command: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5-minute per-file hard limit
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"msconvert timed out after 300 s for file: {raw_path}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"Failed to launch msconvert for '{raw_path}': {exc}"
        ) from exc

    if result.returncode != 0:
        stderr_snippet = result.stderr[:500] if result.stderr else "(no stderr)"
        raise RuntimeError(
            f"msconvert failed for '{raw_path.name}' "
            f"(exit {result.returncode}): {stderr_snippet}"
        )

    if not mzml_path.exists():
        raise RuntimeError(
            f"msconvert reported success but expected output not found: "
            f"{mzml_path}"
        )

    logger.info("Conversion successful: %s", mzml_path.name)
    return mzml_path


def convert_directory(
    raw_dir: Optional[str | Path] = None,
    output_dir: Optional[str | Path] = None,
) -> list[Path]:
    """
    Batch-convert all .wiff and .wiff2 files in *raw_dir* to .mzML.

    Files that already have a corresponding .mzML in *output_dir* are
    skipped (idempotent).  A failed conversion is logged as an error and
    the pipeline continues with the remaining files.

    Parameters
    ----------
    raw_dir:
        Directory containing .wiff / .wiff2 files.  Defaults to
        ``config.RAW_DATA_DIR``.
    output_dir:
        Directory for .mzML outputs.  Defaults to ``config.CONVERTED_DIR``.

    Returns
    -------
    list[Path]
        Paths of all successfully converted (or already-existing) .mzML
        files.  Files that failed conversion are omitted.
    """
    raw_dir = Path(raw_dir or config.RAW_DATA_DIR).resolve()
    output_dir = Path(output_dir or config.CONVERTED_DIR).resolve()

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    raw_files = sorted(
        list(raw_dir.glob("*.wiff")) + list(raw_dir.glob("*.wiff2"))
    )

    if not raw_files:
        logger.warning("No .wiff / .wiff2 files found in %s", raw_dir)
        return []

    logger.info("Found %d raw file(s) in %s", len(raw_files), raw_dir)

    converted: list[Path] = []
    for raw_file in raw_files:
        try:
            mzml_path = convert_file(raw_file, output_dir)
            converted.append(mzml_path)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to convert '%s': %s — skipping.", raw_file.name, exc
            )

    logger.info(
        "Conversion complete: %d / %d file(s) succeeded.",
        len(converted),
        len(raw_files),
    )
    return converted
