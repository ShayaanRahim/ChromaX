"""
models.py — Core data model for the ChromaX pipeline.

Every sample that flows through the pipeline is represented as a
SampleResult.  The dataclass is JSON-serialisable via dataclasses.asdict()
and is the single source of truth for what the frontend consumes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class SampleResult:
    """
    Represents the analysed result for a single compound within a single
    patient sample.

    Attributes
    ----------
    sample_id:
        Unique patient/sample identifier extracted from the raw file
        metadata (e.g. "PT-00142").
    compound_name:
        Name of the analyte being quantified (e.g. "Cortisol").
    status:
        Pipeline verdict — one of ``"pass"``, ``"flag"``, or ``"review"``.
        ``"review"`` items are forwarded to the human swipe queue.
    confidence_score:
        Algorithm confidence that the peak is clean, in [0.0, 1.0].
        Derived from ``score_sample()`` in analyzer.py.
    flag_reasons:
        List of string codes describing detected issues
        (e.g. ``["shoulder_peak", "baseline_drift"]``).
        Empty list when status is ``"pass"``.
    retention_time:
        Detected peak apex in seconds.
    peak_area:
        Integrated peak area (intensity × time units).
    ion_ratio:
        Observed qualifier/quantifier ion ratio.  ``None`` if only a
        single ion trace was available.
    expected_ion_ratio:
        Reference qualifier/quantifier ratio for this compound.
        ``None`` if the compound has no defined reference ratio.
    chromatogram_rt:
        Full retention-time array (seconds) for the quantifier trace —
        sent to the frontend for plotting.
    chromatogram_intensity:
        Full intensity array corresponding to ``chromatogram_rt``.
    peak_start:
        Left edge of the integrated peak region (seconds) — used by the
        frontend to shade the peak area.
    peak_end:
        Right edge of the integrated peak region (seconds).
    """

    sample_id: str
    compound_name: str
    status: str                          # "pass" | "flag" | "review"
    confidence_score: float              # 0.0 – 1.0
    flag_reasons: list[str] = field(default_factory=list)
    retention_time: float = 0.0
    peak_area: float = 0.0
    ion_ratio: Optional[float] = None
    expected_ion_ratio: Optional[float] = None
    chromatogram_rt: list[float] = field(default_factory=list)
    chromatogram_intensity: list[float] = field(default_factory=list)
    peak_start: float = 0.0
    peak_end: float = 0.0

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        valid_statuses = {"pass", "flag", "review"}
        if self.status not in valid_statuses:
            raise ValueError(
                f"SampleResult.status must be one of {valid_statuses!r}, "
                f"got {self.status!r}"
            )
        if not (0.0 <= self.confidence_score <= 1.0):
            raise ValueError(
                f"SampleResult.confidence_score must be in [0, 1], "
                f"got {self.confidence_score}"
            )

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a plain dict suitable for JSON serialisation."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Return a JSON string representation of this result."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> "SampleResult":
        """
        Reconstruct a SampleResult from a plain dict (e.g. loaded from JSON).

        Parameters
        ----------
        data:
            Dictionary with keys matching SampleResult fields.

        Returns
        -------
        SampleResult
        """
        return cls(
            sample_id=data["sample_id"],
            compound_name=data["compound_name"],
            status=data["status"],
            confidence_score=data["confidence_score"],
            flag_reasons=data.get("flag_reasons", []),
            retention_time=data.get("retention_time", 0.0),
            peak_area=data.get("peak_area", 0.0),
            ion_ratio=data.get("ion_ratio"),
            expected_ion_ratio=data.get("expected_ion_ratio"),
            chromatogram_rt=data.get("chromatogram_rt", []),
            chromatogram_intensity=data.get("chromatogram_intensity", []),
            peak_start=data.get("peak_start", 0.0),
            peak_end=data.get("peak_end", 0.0),
        )
