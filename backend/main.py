"""
main.py — ChromaX pipeline orchestrator.

Run with:
    python main.py

The pipeline executes the following stages in order:
  1. Discover .wiff / .wiff2 files in RAW_DATA_DIR
  2. Convert each file to .mzML via MSConvert (converter.py)
  3. Parse each .mzML file into raw chromatogram dicts (parser.py)
  4. Analyse each chromatogram and produce a SampleResult (analyzer.py)
  5. Serialise all results to OUTPUT_PATH as a JSON array
  6. Print a human-readable summary to stdout

A single malformed file or chromatogram will be logged and skipped; the
pipeline never crashes the full batch.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import config
import converter
import parser
import analyzer

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger("chromax.main")


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


def _stage_convert(raw_dir: Path) -> list[Path]:
    """
    Convert all .wiff / .wiff2 files in *raw_dir* to .mzML.

    Also picks up any .mzML files already present in CONVERTED_DIR or
    RAW_DATA_DIR (e.g. files dropped in directly for testing without a
    .wiff source).  Returns a deduplicated, merged list.
    """
    logger.info("=== Stage 1 / 3 : Convert ===")
    try:
        mzml_paths = converter.convert_directory(raw_dir, config.CONVERTED_DIR)
    except FileNotFoundError as exc:
        logger.error("Raw data directory error: %s", exc)
        mzml_paths = []

    # Also pick up any .mzML files already present in converted/ or data/
    # (e.g. files dropped in directly for testing without a .wiff source)
    existing = list(Path(config.CONVERTED_DIR).glob("*.mzML")) + \
               list(Path(config.RAW_DATA_DIR).glob("*.mzML"))

    # Deduplicate by resolved path, preserve order
    seen: set[Path] = set()
    merged: list[Path] = []
    for p in mzml_paths + existing:
        resolved = p.resolve()
        if resolved not in seen:
            seen.add(resolved)
            merged.append(p)

    if len(merged) > len(mzml_paths):
        logger.info(
            "Found %d pre-converted .mzML file(s) — added to parse queue.",
            len(merged) - len(mzml_paths),
        )

    logger.info("%d .mzML file(s) ready for parsing.", len(merged))
    return merged


def _stage_parse(mzml_paths: list[Path]) -> list[dict]:
    """
    Parse all .mzML files into raw chromatogram dicts.

    Returns a flat list of chromatogram records across all files.
    """
    logger.info("=== Stage 2 / 3 : Parse ===")
    if not mzml_paths:
        logger.warning("No .mzML files to parse — skipping parse stage.")
        return []

    records = parser.parse_all(mzml_paths)
    logger.info("%d chromatogram record(s) extracted.", len(records))
    return records


def _stage_analyze(records: list[dict]) -> list:
    """
    Analyse each chromatogram record and return a list of SampleResults.

    Failures on individual records are logged and skipped.
    """
    logger.info("=== Stage 3 / 3 : Analyse ===")
    if not records:
        logger.warning("No chromatogram records to analyse.")
        return []

    results = []
    for record in records:
        sample_id     = record.get("sample_id", "?")
        compound_name = record.get("compound_name", "?")
        try:
            result = analyzer.analyze_sample(record)
            results.append(result)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Unhandled error analysing %s / %s: %s — skipping.",
                sample_id,
                compound_name,
                exc,
            )

    logger.info("%d SampleResult(s) produced.", len(results))
    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _write_results(results: list, output_path: Path) -> None:
    """
    Serialise a list of SampleResults to a JSON file.

    Parameters
    ----------
    results:
        List of SampleResult objects.
    output_path:
        Destination file path.  Parent directories are created as needed.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = [r.to_dict() for r in results]

    try:
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        logger.info("Results written to %s", output_path)
    except OSError as exc:
        logger.error("Failed to write results to %s: %s", output_path, exc)
        raise


def _print_summary(results: list) -> None:
    """
    Print a human-readable pipeline summary to stdout.

    Parameters
    ----------
    results:
        List of SampleResult objects.
    """
    total   = len(results)
    passed  = sum(1 for r in results if r.status == "pass")
    flagged = sum(1 for r in results if r.status == "flag")
    review  = sum(1 for r in results if r.status == "review")

    separator = "─" * 48

    print()
    print(separator)
    print("  ChromaX Pipeline — Results Summary")
    print(separator)
    print(f"  Total samples analysed : {total}")
    print(f"  Pass                   : {passed}")
    print(f"  Review (human queue)   : {review}")
    print(f"  Flag (auto-rejected)   : {flagged}")
    print(separator)

    if total > 0:
        pass_pct   = passed  / total * 100
        review_pct = review  / total * 100
        flag_pct   = flagged / total * 100
        print(
            f"  Distribution           : "
            f"{pass_pct:.1f}% pass / "
            f"{review_pct:.1f}% review / "
            f"{flag_pct:.1f}% flag"
        )
        print(separator)

    print(f"  Output                 : {config.OUTPUT_PATH}")
    print(separator)
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_pipeline() -> list:
    """
    Execute the full ChromaX pipeline from raw files to JSON output.

    Returns
    -------
    list[SampleResult]
        All results produced during this run.
    """
    raw_dir     = Path(config.RAW_DATA_DIR).resolve()
    output_path = Path(config.OUTPUT_PATH).resolve()

    logger.info("ChromaX pipeline starting.")
    logger.info("Raw data directory : %s", raw_dir)
    logger.info("Output path        : %s", output_path)

    mzml_paths = _stage_convert(raw_dir)
    records    = _stage_parse(mzml_paths)
    results    = _stage_analyze(records)

    _write_results(results, output_path)
    _print_summary(results)

    logger.info("ChromaX pipeline complete.")
    return results


if __name__ == "__main__":
    run_pipeline()
