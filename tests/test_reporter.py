import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from conjunction_screener.propagator import PropagatedTrack
from conjunction_screener.reporter import (
    plot_miss_distances,
    write_cdm_csv_report,
    write_csv_report,
)
from conjunction_screener.screener import ConjunctionEvent


def _primary_track(n: int = 5) -> PropagatedTrack:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    times = [start + timedelta(seconds=60 * i) for i in range(n)]
    return PropagatedTrack(
        norad_id=1, name="PRIMARY", times=times, positions_km=np.zeros((n, 3))
    )


def _event(norad_id: int, name: str, miss_km: float, n: int = 5) -> ConjunctionEvent:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return ConjunctionEvent(
        norad_id=norad_id,
        name=name,
        time_of_closest_approach=start + timedelta(seconds=120),
        miss_distance_km=miss_km,
        distances_km=np.linspace(miss_km, miss_km + 10, n),
    )


def test_write_csv_report_with_events(tmp_path: Path):
    events = [_event(2, "SAT-A", 3.5), _event(3, "SAT-B", 1.2)]
    out = write_csv_report(events, tmp_path / "report.csv")

    assert out.exists()
    with out.open() as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert rows[0]["name"] == "SAT-A"
    assert float(rows[0]["miss_distance_km"]) == 3.5
    assert rows[1]["norad_id"] == "3"


def test_write_csv_report_empty_still_has_header(tmp_path: Path):
    out = write_csv_report([], tmp_path / "empty.csv")
    with out.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    assert header == ["norad_id", "name", "time_of_closest_approach", "miss_distance_km"]
    assert rows == []


def test_plot_miss_distances_creates_file(tmp_path: Path):
    primary = _primary_track()
    events = [_event(2, "SAT-A", 3.5)]

    out = plot_miss_distances(
        primary, events, tmp_path / "plot.png", threshold_km=5.0
    )

    assert out.exists()
    assert out.stat().st_size > 0


def test_write_cdm_csv_report_with_events(tmp_path: Path):
    from conjunction_screener.cdm import CDMEvent

    events = [
        CDMEvent(
            cdm_id="1",
            tca=datetime(2026, 8, 1, tzinfo=timezone.utc),
            miss_distance_km=0.437,
            collision_probability=2.5e-05,
            primary_name="ISS (ZARYA)",
            secondary_name="DEBRIS A",
            primary_norad_id=25544,
            secondary_norad_id=99999,
        )
    ]
    out = write_cdm_csv_report(events, tmp_path / "cdm_report.csv")

    assert out.exists()
    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["secondary_name"] == "DEBRIS A"
    assert rows[0]["primary_norad_id"] == "25544"


def test_write_cdm_csv_report_empty_still_has_header(tmp_path: Path):
    out = write_cdm_csv_report([], tmp_path / "empty_cdm.csv")
    with out.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    assert header == [
        "cdm_id",
        "tca",
        "miss_distance_km",
        "collision_probability",
        "primary_name",
        "primary_norad_id",
        "secondary_name",
        "secondary_norad_id",
    ]
    assert rows == []
