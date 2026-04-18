"""
analyzer.py — Peak quality analysis and sample scoring.

Each public check function accepts RT and intensity arrays and returns a
(detected: bool, reason: str) tuple.  The orchestrator ``analyze_sample``
runs all checks and returns a fully populated SampleResult.

Individual check functions are deliberately small and single-purpose so they
can be unit-tested in isolation.

Public API
----------
check_split_peak(rt, intensity)         -> (bool, str)
check_shoulder(rt, intensity)           -> (bool, str)
check_baseline_drift(rt, intensity)     -> (bool, str)
check_ghost_peak(rt, intensity, expected_rt) -> (bool, str)
check_ion_ratio(ion_ratio, expected_ion_ratio) -> (bool, str)
score_sample(flag_reasons)              -> float
analyze_sample(chromatogram_dict)       -> SampleResult
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import linregress

import config
from models import SampleResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------


def _arrays_from_dict(
    chrom: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract and validate RT and intensity arrays from a chromatogram dict.

    Parameters
    ----------
    chrom:
        Raw chromatogram dict produced by parser.py.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (rt, intensity) as float64 arrays of equal length.

    Raises
    ------
    ValueError
        If arrays are missing, mis-matched in length, or too short for
        meaningful analysis.
    """
    rt = np.array(chrom.get("rt", []), dtype=np.float64)
    intensity = np.array(chrom.get("intensity", []), dtype=np.float64)

    if rt.size == 0 or intensity.size == 0:
        raise ValueError("Chromatogram contains empty RT or intensity array.")
    if rt.size != intensity.size:
        raise ValueError(
            f"RT length ({rt.size}) ≠ intensity length ({intensity.size})."
        )
    if rt.size < 10:
        raise ValueError(
            f"Chromatogram too short ({rt.size} points) for reliable analysis."
        )

    return rt, intensity


def _find_primary_peak(
    rt: np.ndarray,
    intensity: np.ndarray,
    expected_rt: Optional[float],
) -> tuple[int, float, float, float]:
    """
    Locate the primary (highest-prominence) peak within the expected RT window.

    Parameters
    ----------
    rt:
        Retention time array (seconds).
    intensity:
        Intensity array.
    expected_rt:
        Centre of the expected retention-time window (seconds).  If ``None``,
        the global maximum is used as the primary peak.

    Returns
    -------
    tuple[int, float, float, float]
        (apex_index, apex_rt, peak_start_rt, peak_end_rt)
    """
    if expected_rt is not None:
        window_mask = np.abs(rt - expected_rt) <= config.EXPECTED_RETENTION_TIME_WINDOW
        window_indices = np.where(window_mask)[0]
    else:
        window_indices = np.arange(len(rt))

    if len(window_indices) == 0:
        # Fall back to global max when the expected window is empty.
        apex_index = int(np.argmax(intensity))
    else:
        local_apex_offset = int(
            np.argmax(intensity[window_indices])
        )
        apex_index = int(window_indices[local_apex_offset])

    apex_rt = float(rt[apex_index])

    # Estimate peak boundaries using the half-maximum crossing points.
    apex_height = intensity[apex_index]
    half_max = apex_height / 2.0

    # Walk left from apex to find start.
    left_idx = apex_index
    for i in range(apex_index, -1, -1):
        if intensity[i] < half_max:
            left_idx = i
            break

    # Walk right from apex to find end.
    right_idx = apex_index
    for i in range(apex_index, len(intensity)):
        if intensity[i] < half_max:
            right_idx = i
            break

    peak_start_rt = float(rt[left_idx])
    peak_end_rt   = float(rt[right_idx])

    return apex_index, apex_rt, peak_start_rt, peak_end_rt


def _compute_peak_area(
    rt: np.ndarray,
    intensity: np.ndarray,
    start_rt: float,
    end_rt: float,
) -> float:
    """
    Integrate peak area using the trapezoidal rule between start_rt and end_rt.

    Parameters
    ----------
    rt, intensity:
        Full chromatogram arrays.
    start_rt, end_rt:
        Integration limits (seconds).

    Returns
    -------
    float
        Integrated peak area.
    """
    mask = (rt >= start_rt) & (rt <= end_rt)
    if mask.sum() < 2:
        return 0.0
    return float(np.trapz(intensity[mask], rt[mask]))


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------


def check_split_peak(
    rt: np.ndarray,
    intensity: np.ndarray,
) -> tuple[bool, str]:
    """
    Detect multiple distinct peaks within the chromatogram.

    Uses ``scipy.signal.find_peaks`` with prominence and distance filtering.
    A split peak is flagged when more than one significant local maximum is
    found across the full retention time range.

    Parameters
    ----------
    rt:
        Retention time array (seconds).
    intensity:
        Intensity array.

    Returns
    -------
    tuple[bool, str]
        ``(True, "split_peak")`` if a split peak is detected, else
        ``(False, "")``.
    """
    peaks, properties = find_peaks(
        intensity,
        prominence=config.PEAK_PROMINENCE_MIN,
        distance=config.SPLIT_PEAK_MIN_DISTANCE,
    )

    if len(peaks) > 1:
        logger.debug(
            "split_peak: %d peaks found at RT=%s",
            len(peaks),
            [round(float(rt[p]), 2) for p in peaks],
        )
        return True, "split_peak"

    return False, ""


def check_shoulder(
    rt: np.ndarray,
    intensity: np.ndarray,
) -> tuple[bool, str]:
    """
    Detect a shoulder on the primary peak using second-derivative analysis.

    The signal is first smoothed with a Savitzky-Golay filter to reduce
    noise, then the second derivative is computed via two successive calls
    to ``numpy.gradient``.  An inflection point on the ascending or
    descending flank—indicated by a sign change in the second derivative
    above the sensitivity threshold—suggests a co-eluting compound.

    Parameters
    ----------
    rt:
        Retention time array (seconds).
    intensity:
        Intensity array.

    Returns
    -------
    tuple[bool, str]
        ``(True, "shoulder_peak")`` if a shoulder is detected, else
        ``(False, "")``.
    """
    window = min(config.SAVGOL_WINDOW_LENGTH, len(intensity) - 1)
    if window % 2 == 0:
        window -= 1
    if window < config.SAVGOL_POLYORDER + 2:
        return False, ""

    try:
        smoothed = savgol_filter(
            intensity,
            window_length=window,
            polyorder=config.SAVGOL_POLYORDER,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Savitzky-Golay filter failed: %s", exc)
        return False, ""

    d1 = np.gradient(smoothed, rt)
    d2 = np.gradient(d1, rt)

    # Normalise by peak apex to make the threshold scale-independent.
    apex_intensity = float(np.max(intensity))
    if apex_intensity == 0:
        return False, ""

    normalised_d2 = np.abs(d2) / apex_intensity

    # Find apex index for flank isolation.
    apex_idx = int(np.argmax(intensity))

    # Check ascending and descending flanks separately.
    ascending  = normalised_d2[:apex_idx]
    descending = normalised_d2[apex_idx:]

    threshold = config.SHOULDER_DETECTION_SENSITIVITY

    ascending_inflections  = np.sum(ascending  > threshold)
    descending_inflections = np.sum(descending > threshold)

    if ascending_inflections > 0 or descending_inflections > 0:
        logger.debug(
            "shoulder_peak: ascending inflections=%d, descending inflections=%d",
            ascending_inflections,
            descending_inflections,
        )
        return True, "shoulder_peak"

    return False, ""


def check_baseline_drift(
    rt: np.ndarray,
    intensity: np.ndarray,
) -> tuple[bool, str]:
    """
    Detect significant baseline drift in the pre- and post-peak regions.

    Fits a linear regression to the first and last ``BASELINE_REGION_FRACTION``
    of the chromatogram.  The slope is normalised by the peak apex intensity
    so the threshold is scale-independent.

    Parameters
    ----------
    rt:
        Retention time array (seconds).
    intensity:
        Intensity array.

    Returns
    -------
    tuple[bool, str]
        ``(True, "baseline_drift")`` if the baseline slope exceeds the
        configured threshold, else ``(False, "")``.
    """
    n = len(rt)
    region_size = max(3, int(n * config.BASELINE_REGION_FRACTION))

    pre_rt  = rt[:region_size]
    pre_int = intensity[:region_size]
    post_rt  = rt[-region_size:]
    post_int = intensity[-region_size:]

    combined_rt  = np.concatenate([pre_rt, post_rt])
    combined_int = np.concatenate([pre_int, post_int])

    if len(combined_rt) < 4:
        return False, ""

    try:
        slope, _, _, _, _ = linregress(combined_rt, combined_int)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Linear regression for baseline drift failed: %s", exc)
        return False, ""

    apex_intensity = float(np.max(intensity))
    if apex_intensity == 0:
        return False, ""

    normalised_slope = abs(slope) / apex_intensity

    if normalised_slope > config.BASELINE_DRIFT_THRESHOLD:
        logger.debug(
            "baseline_drift: normalised slope=%.4f > threshold=%.4f",
            normalised_slope,
            config.BASELINE_DRIFT_THRESHOLD,
        )
        return True, "baseline_drift"

    return False, ""


def check_ghost_peak(
    rt: np.ndarray,
    intensity: np.ndarray,
    expected_rt: Optional[float],
) -> tuple[bool, str]:
    """
    Detect peaks outside the expected retention time window (ghost peaks).

    Searches for prominent peaks in regions more than ``GHOST_PEAK_WINDOW``
    seconds away from ``expected_rt``.  If ``expected_rt`` is ``None``, the
    check is skipped.

    Parameters
    ----------
    rt:
        Retention time array (seconds).
    intensity:
        Intensity array.
    expected_rt:
        Expected compound retention time (seconds).  Pass ``None`` to skip.

    Returns
    -------
    tuple[bool, str]
        ``(True, "ghost_peak")`` if an out-of-window prominent peak is found,
        else ``(False, "")``.
    """
    if expected_rt is None:
        return False, ""

    outer_mask = np.abs(rt - expected_rt) > config.GHOST_PEAK_WINDOW
    outer_indices = np.where(outer_mask)[0]

    if len(outer_indices) < 3:
        return False, ""

    outer_intensity = np.zeros_like(intensity)
    outer_intensity[outer_indices] = intensity[outer_indices]

    ghost_peaks, _ = find_peaks(
        outer_intensity,
        prominence=config.GHOST_PEAK_PROMINENCE_MIN,
    )

    if len(ghost_peaks) > 0:
        ghost_rts = [round(float(rt[p]), 2) for p in ghost_peaks]
        logger.debug("ghost_peak: detected at RT=%s", ghost_rts)
        return True, "ghost_peak"

    return False, ""


def check_ion_ratio(
    ion_ratio: Optional[float],
    expected_ion_ratio: Optional[float],
) -> tuple[bool, str]:
    """
    Check whether the observed qualifier/quantifier ion ratio is within
    tolerance of the expected value.

    Parameters
    ----------
    ion_ratio:
        Observed qualifier/quantifier ratio.  Pass ``None`` to skip.
    expected_ion_ratio:
        Reference ratio for this compound.  Pass ``None`` to skip.

    Returns
    -------
    tuple[bool, str]
        ``(True, "ion_ratio_deviation")`` if the ratio deviation exceeds
        ``ION_RATIO_TOLERANCE``, else ``(False, "")``.
    """
    if ion_ratio is None or expected_ion_ratio is None:
        return False, ""

    if expected_ion_ratio == 0:
        logger.warning("Expected ion ratio is zero — skipping ratio check.")
        return False, ""

    deviation = abs(ion_ratio - expected_ion_ratio) / abs(expected_ion_ratio)

    if deviation > config.ION_RATIO_TOLERANCE:
        logger.debug(
            "ion_ratio_deviation: observed=%.4f expected=%.4f deviation=%.2f%%",
            ion_ratio,
            expected_ion_ratio,
            deviation * 100,
        )
        return True, "ion_ratio_deviation"

    return False, ""


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_sample(flag_reasons: list[str]) -> float:
    """
    Convert a list of flag reason strings into a confidence score in [0, 1].

    Each flag subtracts a severity-weighted penalty from a perfect score of
    1.0.  The penalty weights are defined in ``config.FLAG_WEIGHTS``.
    Unrecognised flag strings receive a default penalty of 0.10.

    Parameters
    ----------
    flag_reasons:
        List of flag code strings (may be empty).

    Returns
    -------
    float
        Confidence score clamped to [0.0, 1.0].
    """
    score = 1.0
    for reason in flag_reasons:
        penalty = config.FLAG_WEIGHTS.get(reason, 0.10)
        score -= penalty

    return max(0.0, min(1.0, score))


def _determine_status(score: float) -> str:
    """
    Map a numeric confidence score to a categorical status string.

    Parameters
    ----------
    score:
        Confidence score in [0.0, 1.0].

    Returns
    -------
    str
        One of ``"pass"``, ``"review"``, or ``"flag"``.
    """
    if score >= config.ALGORITHM_CONFIDENCE_REVIEW_THRESHOLD:
        return "pass"
    if score >= config.ALGORITHM_CONFIDENCE_FLAG_THRESHOLD:
        return "review"
    return "flag"


# ---------------------------------------------------------------------------
# Ion ratio computation
# ---------------------------------------------------------------------------


def _compute_ion_ratio(
    quant_intensity: np.ndarray,
    qual_intensity: np.ndarray,
    apex_index: int,
) -> Optional[float]:
    """
    Compute the qualifier/quantifier peak area ratio.

    Uses the apex neighbourhood (±5 points) rather than full peak
    integration to reduce the effect of baseline on both traces.

    Parameters
    ----------
    quant_intensity:
        Quantifier ion intensity array.
    qual_intensity:
        Qualifier ion intensity array (same length as quant_intensity).
    apex_index:
        Index of the quantifier peak apex.

    Returns
    -------
    float or None
        Ratio value, or ``None`` if the quantifier signal is zero.
    """
    half_win = 5
    lo = max(0, apex_index - half_win)
    hi = min(len(quant_intensity), apex_index + half_win + 1)

    quant_sum = float(np.sum(quant_intensity[lo:hi]))
    qual_sum  = float(np.sum(qual_intensity[lo:hi]))

    if quant_sum == 0:
        return None

    return qual_sum / quant_sum


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------


def analyze_sample(chromatogram_dict: dict) -> SampleResult:
    """
    Run all quality checks on a single chromatogram and return a SampleResult.

    This is the main entry point called by main.py for each parsed
    chromatogram record.

    Parameters
    ----------
    chromatogram_dict:
        A raw chromatogram dict as returned by ``parser.parse_mzml``.
        Expected keys: ``sample_id``, ``compound_name``, ``rt``,
        ``intensity``, ``expected_rt``, ``expected_ion_ratio``,
        ``qualifier_intensity``.

    Returns
    -------
    SampleResult
        Fully populated result object.
    """
    sample_id     = chromatogram_dict.get("sample_id", "Unknown")
    compound_name = chromatogram_dict.get("compound_name", "Unknown")
    expected_rt   = chromatogram_dict.get("expected_rt")
    expected_ion_ratio = chromatogram_dict.get("expected_ion_ratio")

    # ------------------------------------------------------------------
    # Extract arrays
    # ------------------------------------------------------------------

    try:
        rt, intensity = _arrays_from_dict(chromatogram_dict)
    except ValueError as exc:
        logger.warning(
            "Cannot analyse %s / %s: %s — returning low-confidence flag.",
            sample_id,
            compound_name,
            exc,
        )
        return SampleResult(
            sample_id=sample_id,
            compound_name=compound_name,
            status="flag",
            confidence_score=0.0,
            flag_reasons=["invalid_data"],
        )

    # ------------------------------------------------------------------
    # Locate primary peak
    # ------------------------------------------------------------------

    try:
        apex_index, apex_rt, peak_start, peak_end = _find_primary_peak(
            rt, intensity, expected_rt
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Peak detection failed for %s / %s: %s", sample_id, compound_name, exc
        )
        apex_index, apex_rt, peak_start, peak_end = 0, 0.0, 0.0, 0.0

    peak_area = _compute_peak_area(rt, intensity, peak_start, peak_end)

    # ------------------------------------------------------------------
    # Ion ratio
    # ------------------------------------------------------------------

    qual_intensity_raw = chromatogram_dict.get("qualifier_intensity")
    ion_ratio: Optional[float] = None

    if qual_intensity_raw is not None:
        qual_intensity = np.array(qual_intensity_raw, dtype=np.float64)
        if qual_intensity.size == intensity.size:
            ion_ratio = _compute_ion_ratio(intensity, qual_intensity, apex_index)
        else:
            logger.warning(
                "Qualifier intensity array length mismatch for %s / %s — skipping ratio.",
                sample_id,
                compound_name,
            )

    # ------------------------------------------------------------------
    # Run individual checks
    # ------------------------------------------------------------------

    flag_reasons: list[str] = []

    checks = [
        lambda: check_split_peak(rt, intensity),
        lambda: check_shoulder(rt, intensity),
        lambda: check_baseline_drift(rt, intensity),
        lambda: check_ghost_peak(rt, intensity, expected_rt),
        lambda: check_ion_ratio(ion_ratio, expected_ion_ratio),
    ]

    for check_fn in checks:
        try:
            detected, reason = check_fn()
            if detected:
                flag_reasons.append(reason)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Check raised an exception for %s / %s: %s",
                sample_id,
                compound_name,
                exc,
            )

    # ------------------------------------------------------------------
    # Score and classify
    # ------------------------------------------------------------------

    confidence = score_sample(flag_reasons)
    status     = _determine_status(confidence)

    logger.debug(
        "%s / %s → status=%s score=%.2f flags=%s",
        sample_id,
        compound_name,
        status,
        confidence,
        flag_reasons,
    )

    return SampleResult(
        sample_id=sample_id,
        compound_name=compound_name,
        status=status,
        confidence_score=round(confidence, 4),
        flag_reasons=flag_reasons,
        retention_time=round(apex_rt, 4),
        peak_area=round(peak_area, 2),
        ion_ratio=round(ion_ratio, 6) if ion_ratio is not None else None,
        expected_ion_ratio=expected_ion_ratio,
        chromatogram_rt=[round(v, 4) for v in rt.tolist()],
        chromatogram_intensity=[round(v, 2) for v in intensity.tolist()],
        peak_start=round(peak_start, 4),
        peak_end=round(peak_end, 4),
    )
