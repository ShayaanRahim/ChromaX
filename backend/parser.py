"""
parser.py — Extracts chromatogram data from .mzML files.

Uses xml.etree.ElementTree to parse mzML files directly.  pymzml is
designed for mass spectra and does not reliably iterate <chromatogram>
elements in chromatogram-only files.

Public API
----------
parse_mzml(mzml_path) -> list[dict]
    Parse a single .mzML file and return all chromatogram records.

parse_all(mzml_paths) -> list[dict]
    Batch-parse a list of .mzML files.

Raw Chromatogram Dict Schema
----------------------------
{
    "sample_id":          str,
    "compound_name":      str,
    "ion_type":           "quantifier" | "qualifier" | "unknown",
    "rt":                 list[float],   # seconds
    "intensity":          list[float],
    "expected_rt":        float | None,  # from metadata, if available
    "expected_ion_ratio": float | None,  # from metadata, if available
    # Populated after pairing step:
    "qualifier_intensity": list[float] | None,
}
"""

from __future__ import annotations

import base64
import logging
import re
import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# mzML namespace — defined once, referenced everywhere.
NS = {"ms": "http://psi.hupo.org/ms/mzml"}

# CV accession numbers for binary array types.
_CV_TIME_ARRAY      = "MS:1000595"
_CV_INTENSITY_ARRAY = "MS:1000515"

# ---------------------------------------------------------------------------
# Constants — ID heuristics used when userParam metadata is absent
# ---------------------------------------------------------------------------

_CV_SAMPLE_ID_PATTERNS = [
    r"sample[_\s]?id[:\s=]+([^\s,;]+)",
    r"^([A-Z]{1,4}-?\d+)",              # common lab format e.g. PT-00142
]

_ION_TYPE_QUANTIFIER_KEYWORDS = {"quant", "quantifier", "quantitative", "q1", "mrm1"}
_ION_TYPE_QUALIFIER_KEYWORDS  = {"qual", "qualifier", "qualitative", "q3", "mrm2", "conf"}

_ION_TYPE_NORMALIZATION: dict[str, str] = {
    "quantitative": "quantifier",
    "qualitative":  "qualifier",
    "quantifier":   "quantifier",
    "qualifier":    "qualifier",
    "q1":           "quantifier",
    "q3":           "qualifier",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_sample_id(chrom_id: str) -> str:
    """
    Attempt to extract a sample ID from a chromatogram ID string.

    Many SCIEX exports embed the sample ID in the chromatogram title in
    formats like ``"PT-00142 Cortisol Q1"`` or ``"sampleId=PT-00142"``.
    Falls back to the full ``chrom_id`` if no pattern matches.

    Parameters
    ----------
    chrom_id:
        The raw chromatogram ID string from the mzML element.

    Returns
    -------
    str
        Best-effort sample ID string.
    """
    for pattern in _CV_SAMPLE_ID_PATTERNS:
        match = re.search(pattern, chrom_id, re.IGNORECASE)
        if match:
            return match.group(1)
    return chrom_id


def _extract_compound_name(chrom_id: str) -> str:
    """
    Extract a compound/analyte name from a chromatogram ID string.

    SCIEX MRM channel IDs typically follow the pattern
    ``"<SampleID> <CompoundName> <IonType>"``.  This function attempts to
    extract the middle token(s) after stripping known prefixes and suffixes.

    Parameters
    ----------
    chrom_id:
        The raw chromatogram ID string.

    Returns
    -------
    str
        Best-effort compound name, or ``"Unknown"`` if parsing fails.
    """
    # Strip leading sample-ID-like token (e.g. "PT-00142") if present.
    cleaned = re.sub(
        r"^[A-Z]{1,4}-?\d+\s+", "", chrom_id, flags=re.IGNORECASE
    ).strip()

    # Strip trailing ion-type tokens.
    all_ion_keywords = (
        _ION_TYPE_QUANTIFIER_KEYWORDS | _ION_TYPE_QUALIFIER_KEYWORDS
    )
    pattern = r"\s+(?:" + "|".join(re.escape(k) for k in all_ion_keywords) + r")$"
    cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()

    return cleaned if cleaned else "Unknown"


def _determine_ion_type(chrom_id: str) -> str:
    """
    Classify a chromatogram as ``"quantifier"``, ``"qualifier"``, or
    ``"unknown"`` based on keywords in its ID string.

    Parameters
    ----------
    chrom_id:
        The raw chromatogram ID string.

    Returns
    -------
    str
        One of ``"quantifier"``, ``"qualifier"``, or ``"unknown"``.
    """
    lower = chrom_id.lower()
    if any(kw in lower for kw in _ION_TYPE_QUANTIFIER_KEYWORDS):
        return "quantifier"
    if any(kw in lower for kw in _ION_TYPE_QUALIFIER_KEYWORDS):
        return "qualifier"
    return "unknown"


def _decode_array(b64_text: str) -> list[float]:
    """
    Decode a base64-encoded 32-bit float array from mzML binary data.

    Parameters
    ----------
    b64_text:
        Base64 string from a ``<binary>`` element.

    Returns
    -------
    list[float]
        Decoded float values.
    """
    raw = base64.b64decode(b64_text.strip())
    n = len(raw) // 4
    return list(struct.unpack(f"{n}f", raw))


def _parse_chromatogram_element(chrom_el: ET.Element) -> Optional[dict]:
    """
    Extract RT/intensity arrays and metadata from a single mzML
    ``<chromatogram>`` ElementTree element.

    Metadata is read from ``<userParam>`` child elements first; the
    chromatogram ``id`` attribute is used as a fallback for fields not
    present in userParams.

    Parameters
    ----------
    chrom_el:
        An ``xml.etree.ElementTree.Element`` representing one
        ``<chromatogram>`` node.

    Returns
    -------
    dict or None
        Populated chromatogram dict, or ``None`` if the element is empty
        or malformed.
    """
    chrom_id: str = chrom_el.get("id", "")

    # --- metadata from userParam elements -----------------------------------
    params: dict[str, str] = {
        up.get("name", ""): up.get("value", "")
        for up in chrom_el.findall("ms:userParam", NS)
    }

    # --- binary arrays ------------------------------------------------------
    rt_list: Optional[list[float]] = None
    intensity_list: Optional[list[float]] = None

    for bda in chrom_el.findall(".//ms:binaryDataArray", NS):
        cv_accessions = {
            cv.get("accession", "")
            for cv in bda.findall("ms:cvParam", NS)
        }
        binary_el = bda.find("ms:binary", NS)
        if binary_el is None or not binary_el.text:
            continue

        try:
            decoded = _decode_array(binary_el.text)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not decode binary array in chromatogram '%s': %s",
                chrom_id,
                exc,
            )
            continue

        if _CV_TIME_ARRAY in cv_accessions:
            rt_list = decoded
        elif _CV_INTENSITY_ARRAY in cv_accessions:
            intensity_list = decoded

    if rt_list is None or intensity_list is None:
        logger.warning(
            "Chromatogram '%s' — missing RT or intensity array, skipping.",
            chrom_id,
        )
        return None

    if len(rt_list) == 0 or len(intensity_list) == 0:
        logger.warning("Chromatogram '%s' is empty — skipping.", chrom_id)
        return None

    if len(rt_list) != len(intensity_list):
        logger.warning(
            "Chromatogram '%s': RT length (%d) ≠ intensity length (%d) — skipping.",
            chrom_id,
            len(rt_list),
            len(intensity_list),
        )
        return None

    # Normalise time to seconds: mzML spec stores time in minutes, so values
    # below 60 almost certainly need scaling.
    if max(rt_list) < 60:
        rt_list = [t * 60.0 for t in rt_list]

    # --- resolve metadata with fallback to ID heuristics -------------------
    sample_id     = params.get("sample_id")     or _extract_sample_id(chrom_id)
    compound_name = params.get("compound_name") or _extract_compound_name(chrom_id)
    raw_ion_type  = params.get("ion_type")      or _determine_ion_type(chrom_id)
    ion_type      = _ION_TYPE_NORMALIZATION.get(raw_ion_type.lower(), raw_ion_type)

    raw_expected_rt  = params.get("expected_retention_time")
    raw_expected_ir  = params.get("expected_ion_ratio")

    try:
        expected_rt: Optional[float] = float(raw_expected_rt) if raw_expected_rt else None
    except ValueError:
        expected_rt = None

    try:
        expected_ion_ratio: Optional[float] = float(raw_expected_ir) if raw_expected_ir else None
    except ValueError:
        expected_ion_ratio = None

    return {
        "sample_id":           sample_id,
        "compound_name":       compound_name,
        "ion_type":            ion_type,
        "rt":                  rt_list,
        "intensity":           intensity_list,
        "expected_rt":         expected_rt,
        "expected_ion_ratio":  expected_ion_ratio,
        "qualifier_intensity": None,   # populated by _pair_ion_traces()
        "_raw_id":             chrom_id,
    }


def _pair_ion_traces(chromatograms: list[dict]) -> list[dict]:
    """
    Match quantifier and qualifier traces for the same sample/compound pair.

    After pairing, the qualifier's intensity array is merged into the
    quantifier record as ``qualifier_intensity``.  Lone qualifier records
    (no matching quantifier) are retained as-is so they are not silently
    dropped.

    Parameters
    ----------
    chromatograms:
        All chromatogram dicts extracted from one mzML file.

    Returns
    -------
    list[dict]
        Paired chromatogram dicts (one per compound per sample, containing
        both quant and qual data where available).
    """
    # Index by (sample_id, compound_name) → list of records.
    index: dict[tuple[str, str], list[dict]] = {}
    for chrom in chromatograms:
        key = (chrom["sample_id"], chrom["compound_name"])
        index.setdefault(key, []).append(chrom)

    paired: list[dict] = []
    for (sample_id, compound_name), records in index.items():
        quantifiers = [r for r in records if r["ion_type"] == "quantifier"]
        qualifiers  = [r for r in records if r["ion_type"] == "qualifier"]
        unknowns    = [r for r in records if r["ion_type"] == "unknown"]

        # Prefer explicit quantifier; fall back to first unknown.
        primary = quantifiers[0] if quantifiers else (unknowns[0] if unknowns else None)

        if primary is None:
            # Only qualifier trace — keep it unmodified.
            paired.extend(qualifiers)
            continue

        if qualifiers:
            primary["qualifier_intensity"] = qualifiers[0]["intensity"]
            if len(qualifiers) > 1:
                logger.warning(
                    "Multiple qualifier traces for %s / %s — using first.",
                    sample_id,
                    compound_name,
                )

        paired.append(primary)

        # Emit any additional quantifier records without pairing.
        for extra in quantifiers[1:]:
            logger.warning(
                "Extra quantifier trace for %s / %s — emitting unpaired.",
                sample_id,
                compound_name,
            )
            paired.append(extra)

    return paired


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_mzml(mzml_path: str | Path) -> list[dict]:
    """
    Parse a single .mzML file and return all chromatogram records.

    Each record represents one compound × ion-type pair.  Where both a
    quantifier and qualifier trace exist for the same compound, they are
    merged into a single record (``qualifier_intensity`` is populated).

    Parameters
    ----------
    mzml_path:
        Path to the .mzML file.

    Returns
    -------
    list[dict]
        List of raw chromatogram dicts.  May be empty if the file contains
        no valid chromatograms.

    Raises
    ------
    FileNotFoundError
        If ``mzml_path`` does not exist.
    """
    mzml_path = Path(mzml_path).resolve()

    if not mzml_path.exists():
        raise FileNotFoundError(f"mzML file not found: {mzml_path}")

    logger.info("Parsing %s", mzml_path.name)

    raw_records: list[dict] = []

    try:
        tree = ET.parse(str(mzml_path))
        root = tree.getroot()
        chrom_elements = root.findall(".//ms:chromatogram", NS)
        logger.debug("Found %d <chromatogram> element(s).", len(chrom_elements))

        for chrom_el in chrom_elements:
            try:
                record = _parse_chromatogram_element(chrom_el)
                if record is not None:
                    raw_records.append(record)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to parse chromatogram '%s': %s — skipping.",
                    chrom_el.get("id", "?"),
                    exc,
                )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Failed to read mzML file '%s': %s", mzml_path.name, exc
        )
        return []

    logger.info(
        "Parsed %d chromatogram(s) from %s", len(raw_records), mzml_path.name
    )

    paired = _pair_ion_traces(raw_records)
    logger.info(
        "%d record(s) after ion-trace pairing for %s",
        len(paired),
        mzml_path.name,
    )

    return paired


def parse_all(mzml_paths: list[str | Path]) -> list[dict]:
    """
    Batch-parse a list of .mzML files.

    A failed parse on a single file is logged and skipped; the pipeline
    continues with the remaining files.

    Parameters
    ----------
    mzml_paths:
        Ordered list of .mzML file paths.

    Returns
    -------
    list[dict]
        Combined list of all chromatogram records from every file.
    """
    all_records: list[dict] = []
    for path in mzml_paths:
        try:
            records = parse_mzml(path)
            all_records.extend(records)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Skipping '%s' due to parse error: %s", Path(path).name, exc
            )

    logger.info("Total chromatograms parsed across all files: %d", len(all_records))
    return all_records
