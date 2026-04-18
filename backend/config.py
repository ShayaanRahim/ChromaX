"""
config.py — Centralized configuration for the ChromaX backend pipeline.

All tunable parameters live here. No magic numbers anywhere else in the
codebase — every module imports from this file.
"""

import os

# ---------------------------------------------------------------------------
# External tool paths
# ---------------------------------------------------------------------------

MSCONVERT_PATH: str = os.environ.get("MSCONVERT_PATH", "/usr/local/bin/msconvert")
"""Absolute path to the ProteoWizard msconvert binary."""

# ---------------------------------------------------------------------------
# Directory / file paths
# ---------------------------------------------------------------------------

RAW_DATA_DIR: str = os.path.join(os.path.dirname(__file__), "data")
"""Directory containing raw .wiff / .wiff2 files."""

CONVERTED_DIR: str = os.path.join(os.path.dirname(__file__), "converted")
"""Directory where converted .mzML files are written."""

OUTPUT_PATH: str = os.path.join(os.path.dirname(__file__), "output", "results.json")
"""Destination path for the final JSON results file."""

# ---------------------------------------------------------------------------
# MSConvert options
# ---------------------------------------------------------------------------

MSCONVERT_EXTRA_FILTERS: list[str] = []
"""
Additional --filter arguments passed to msconvert.
Example: ["msLevel 1"] to strip MS2 data for quantitative-only runs.
"""

MSCONVERT_32BIT: bool = True
"""Use 32-bit encoding for output mzML (smaller files, sufficient precision)."""

# ---------------------------------------------------------------------------
# Peak detection thresholds
# ---------------------------------------------------------------------------

PEAK_PROMINENCE_MIN: float = 1000.0
"""
Minimum signal prominence (intensity units) for a local maximum to be
considered a real chromatographic peak rather than noise.
"""

EXPECTED_RETENTION_TIME_WINDOW: float = 0.5
"""
Half-width (seconds) of the expected retention time window.
Peaks detected within ±EXPECTED_RETENTION_TIME_WINDOW of the expected RT
are considered the primary peak.
"""

# ---------------------------------------------------------------------------
# Baseline drift
# ---------------------------------------------------------------------------

BASELINE_DRIFT_THRESHOLD: float = 0.05
"""
Maximum allowable slope of a linear fit to the pre- and post-peak baseline
regions, expressed as a fraction of the peak apex intensity per second.
Exceeding this value triggers the "baseline_drift" flag.
"""

BASELINE_REGION_FRACTION: float = 0.15
"""
Fraction of the total chromatogram length used to define the pre-peak and
post-peak baseline sampling regions (e.g. 0.15 = first/last 15% of points).
"""

# ---------------------------------------------------------------------------
# Ion ratio
# ---------------------------------------------------------------------------

ION_RATIO_TOLERANCE: float = 0.20
"""
Maximum allowed fractional deviation between the observed and expected
qualifier/quantifier ion ratio (±20 %).  Exceeding this triggers
"ion_ratio_deviation".
"""

# ---------------------------------------------------------------------------
# Ghost peak detection
# ---------------------------------------------------------------------------

GHOST_PEAK_WINDOW: float = 5.0
"""
Retention time offset (seconds) beyond the expected RT window outside of
which a prominent peak is classified as a "ghost_peak".
"""

GHOST_PEAK_PROMINENCE_MIN: float = 500.0
"""
Minimum prominence for an out-of-window peak to be classified as a ghost
peak.  Set lower than PEAK_PROMINENCE_MIN to catch smaller contaminants.
"""

# ---------------------------------------------------------------------------
# Shoulder detection
# ---------------------------------------------------------------------------

SHOULDER_DETECTION_SENSITIVITY: float = 0.3
"""
Threshold for the normalised second-derivative magnitude used to identify
inflection points on peak flanks.  Lower values → more sensitive (more
false positives); higher values → less sensitive.
"""

SAVGOL_WINDOW_LENGTH: int = 11
"""
Window length for the Savitzky-Golay smoothing filter applied before
second-derivative shoulder detection.  Must be odd and ≥ polyorder + 2.
"""

SAVGOL_POLYORDER: int = 3
"""Polynomial order for the Savitzky-Golay smoothing filter."""

# ---------------------------------------------------------------------------
# Split-peak detection
# ---------------------------------------------------------------------------

SPLIT_PEAK_MIN_DISTANCE: int = 5
"""
Minimum number of data points that must separate two peaks for them to be
counted as distinct (passed to scipy.signal.find_peaks as `distance`).
"""

# ---------------------------------------------------------------------------
# Scoring / status thresholds
# ---------------------------------------------------------------------------

ALGORITHM_CONFIDENCE_REVIEW_THRESHOLD: float = 0.75
"""
Confidence scores at or above this value → status "pass".
Scores below this but above FLAG threshold → status "review".
"""

ALGORITHM_CONFIDENCE_FLAG_THRESHOLD: float = 0.40
"""Confidence scores strictly below this value → status "flag"."""

# Severity weights used by score_sample().  Higher weight = larger penalty.
FLAG_WEIGHTS: dict[str, float] = {
    "split_peak":          0.35,
    "ion_ratio_deviation": 0.30,
    "shoulder_peak":       0.20,
    "ghost_peak":          0.15,
    "baseline_drift":      0.10,
}
"""
Per-flag confidence penalties.  The score starts at 1.0 and each detected
flag subtracts its weight, floored at 0.0.
"""
