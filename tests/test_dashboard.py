from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from conjunction_screener.dashboard import write_html_dashboard
from conjunction_screener.propagator import PropagatedTrack
from conjunction_screener.screener import ClosestApproachSummary


def _primary_track(n: int = 5) -> PropagatedTrack:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    times = [start + timedelta(hours=i) for i in range(n)]
    return PropagatedTrack(
        norad_id=25544,
        name="ISS (ZARYA)",
        times=times,
        positions_km=np.zeros((n, 3)),
    )


def test_write_html_dashboard_clear_case(tmp_path: Path):
    primary = _primary_track()
    summaries = [
        ClosestApproachSummary(
            norad_id=2,
            name="SAFE OBJECT",
            miss_distance_km=80.0,
            time_of_closest_approach=primary.times[2],
            is_colocated=False,
            is_flagged=False,
        )
    ]

    out = write_html_dashboard(
        primary, summaries, threshold_km=25.0, window_hours=4.0,
        output_path=tmp_path / "dashboard.html",
    )

    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "ISS (ZARYA)" in content
    assert "CLEAR" in content
    assert "SAFE OBJECT" not in content  # not flagged, so not named in the table


def test_write_html_dashboard_flagged_case(tmp_path: Path):
    primary = _primary_track()
    summaries = [
        ClosestApproachSummary(
            norad_id=3,
            name="CLOSE OBJECT",
            miss_distance_km=1.5,
            time_of_closest_approach=primary.times[1],
            is_colocated=False,
            is_flagged=True,
        ),
        ClosestApproachSummary(
            norad_id=4,
            name="ISS (DUPLICATE MODULE)",
            miss_distance_km=0.0,
            time_of_closest_approach=primary.times[0],
            is_colocated=True,
            is_flagged=False,
        ),
    ]

    out = write_html_dashboard(
        primary, summaries, threshold_km=25.0, window_hours=4.0,
        output_path=tmp_path / "dashboard.html",
    )

    content = out.read_text(encoding="utf-8")
    assert "1 FLAGGED" in content
    assert "CLOSE OBJECT" in content
    # Co-located duplicate should be excluded from KPI counts and table
    assert "ISS (DUPLICATE MODULE)" not in content


def test_write_html_dashboard_no_objects_screened(tmp_path: Path):
    primary = _primary_track()
    out = write_html_dashboard(
        primary, [], threshold_km=5.0, window_hours=24.0,
        output_path=tmp_path / "dashboard.html",
    )
    content = out.read_text(encoding="utf-8")
    assert "n/a" in content  # closest-approach KPI has nothing to report
    assert "CLEAR" in content
