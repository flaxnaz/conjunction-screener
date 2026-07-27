"""Output: a CSV conjunction report and a miss-distance-vs-time plot."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe backend for CI / scripted runs
import matplotlib.pyplot as plt

from conjunction_screener.cdm import CDMEvent
from conjunction_screener.propagator import PropagatedTrack
from conjunction_screener.screener import ConjunctionEvent

CSV_FIELDS = ["norad_id", "name", "time_of_closest_approach", "miss_distance_km"]

CDM_CSV_FIELDS = [
    "cdm_id",
    "tca",
    "miss_distance_km",
    "collision_probability",
    "primary_name",
    "primary_norad_id",
    "secondary_name",
    "secondary_norad_id",
]


def write_cdm_csv_report(events: list[CDMEvent], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CDM_CSV_FIELDS)
        writer.writeheader()
        for e in events:
            writer.writerow(
                {
                    "cdm_id": e.cdm_id,
                    "tca": e.tca.isoformat(),
                    "miss_distance_km": f"{e.miss_distance_km:.3f}",
                    "collision_probability": (
                        f"{e.collision_probability:.3e}"
                        if e.collision_probability is not None
                        else ""
                    ),
                    "primary_name": e.primary_name,
                    "primary_norad_id": e.primary_norad_id or "",
                    "secondary_name": e.secondary_name,
                    "secondary_norad_id": e.secondary_norad_id or "",
                }
            )
    return output_path


def write_csv_report(events: list[ConjunctionEvent], output_path: str | Path) -> Path:
    """Write flagged conjunctions to a CSV file, closest approach first.

    Writes a header row even when `events` is empty, so downstream
    tooling always gets a well-formed file rather than a missing one.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for e in events:
            writer.writerow(
                {
                    "norad_id": e.norad_id,
                    "name": e.name,
                    "time_of_closest_approach": e.time_of_closest_approach.isoformat(),
                    "miss_distance_km": f"{e.miss_distance_km:.3f}",
                }
            )
    return output_path


def plot_miss_distances(
    primary: PropagatedTrack,
    events: list[ConjunctionEvent],
    output_path: str | Path,
    threshold_km: float | None = None,
) -> Path:
    """Plot miss distance vs. time for each flagged conjunction.

    Draws the configured threshold as a reference line when provided.
    Produces an (empty-axes) file even with no events, so a run with a
    clean sky still leaves a report artifact behind.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    for e in events:
        # matplotlib accepts a list[datetime] for the x-axis at runtime;
        # its stubs are stricter than the real signature, hence the ignore.
        ax.plot(primary.times, e.distances_km, label=f"{e.name} ({e.norad_id})")  # type: ignore[arg-type]

    if threshold_km is not None:
        ax.axhline(
            threshold_km, color="red", linestyle="--", linewidth=1, label="threshold"
        )

    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Distance (km)")
    ax.set_title(f"Miss distance vs. time — primary: {primary.name}")
    if events:
        ax.legend(fontsize="small")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
