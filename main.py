"""CLI entry point for the conjunction screener.

Example (explicit conjunctor list):
    python main.py --primary 25544 --conjunctors 48274 43013 44714 \
        --window-hours 24 --threshold-km 5 --out reports/

Example (auto-discover candidates near the primary's orbit):
    python main.py --primary 25544 --altitude-band-km 50 --threshold-km 10

Default primary (25544) is the ISS. If --conjunctors is omitted, candidate
conjunctors are found automatically from Space-Track based on orbital
regime (perigee/apogee proximity to the primary).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from conjunction_screener.auth import get_client
from conjunction_screener.dashboard import write_html_dashboard
from conjunction_screener.fetcher import (
    fetch_orbital_regime_catalog,
    fetch_primary_and_conjunctors,
)
from conjunction_screener.propagator import propagate_all, propagate_track
from conjunction_screener.reporter import plot_miss_distances, write_csv_report
from conjunction_screener.screener import (
    DEFAULT_THRESHOLD_KM,
    screen_conjunctions,
    summarize_closest_approaches,
)

ISS_NORAD_ID = 25544


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen for satellite conjunctions.")
    parser.add_argument(
        "--primary", type=int, default=ISS_NORAD_ID, help="Primary object NORAD ID."
    )
    parser.add_argument(
        "--conjunctors",
        type=int,
        nargs="*",
        default=None,
        help="Explicit NORAD IDs to screen against the primary. If omitted, "
        "candidates are found automatically from objects in a similar "
        "orbital regime (see --altitude-band-km).",
    )
    parser.add_argument(
        "--altitude-band-km",
        type=float,
        default=50.0,
        help="When --conjunctors is omitted, how close (km) another "
        "object's perigee/apogee must be to the primary's to be treated "
        "as a candidate conjunctor.",
    )
    parser.add_argument(
        "--catalog-limit",
        type=int,
        default=100,
        help="Max number of auto-discovered candidate conjunctors.",
    )
    parser.add_argument(
        "--window-hours", type=float, default=24.0, help="Propagation window (hours)."
    )
    parser.add_argument(
        "--timestep-seconds", type=float, default=60.0, help="Propagation timestep."
    )
    parser.add_argument(
        "--threshold-km",
        type=float,
        default=DEFAULT_THRESHOLD_KM,
        help="Miss-distance flagging threshold (km).",
    )
    parser.add_argument(
        "--out", type=str, default="reports", help="Output directory."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)

    client = get_client()

    if args.conjunctors is None:
        print(
            f"No --conjunctors given; auto-discovering candidates within "
            f"{args.altitude_band_km} km of primary's orbital regime..."
        )
        primary_tle, conjunctor_tles = fetch_orbital_regime_catalog(
            client, args.primary, args.altitude_band_km, args.catalog_limit
        )
        print(f"Found {len(conjunctor_tles)} candidate object(s).")
    else:
        primary_tle, conjunctor_tles = fetch_primary_and_conjunctors(
            client, args.primary, args.conjunctors
        )

    start_time = datetime.now(timezone.utc)

    primary_track = propagate_track(
        primary_tle, start_time, args.window_hours, args.timestep_seconds
    )
    conjunctor_tracks = propagate_all(
        conjunctor_tles, start_time, args.window_hours, args.timestep_seconds
    )

    events = screen_conjunctions(primary_track, conjunctor_tracks, args.threshold_km)
    summaries = summarize_closest_approaches(
        primary_track, conjunctor_tracks, args.threshold_km
    )

    csv_path = write_csv_report(events, out_dir / "conjunction_report.csv")
    plot_path = plot_miss_distances(
        primary_track, events, out_dir / "miss_distance.png", args.threshold_km
    )
    dashboard_path = write_html_dashboard(
        primary_track,
        summaries,
        args.threshold_km,
        args.window_hours,
        out_dir / "dashboard.html",
    )

    print(f"Primary: {primary_track.name} (NORAD {primary_track.norad_id})")
    print(f"Screened {len(conjunctor_tracks)} objects over {args.window_hours}h window.")
    print(f"Flagged {len(events)} conjunction(s) below {args.threshold_km} km.")
    for e in events:
        print(
            f"  {e.name} (NORAD {e.norad_id}): {e.miss_distance_km:.3f} km "
            f"at {e.time_of_closest_approach.isoformat()}"
        )
    print(f"Report written to: {csv_path}")
    print(f"Plot written to:   {plot_path}")
    print(f"Dashboard written to: {dashboard_path}")


if __name__ == "__main__":
    main()
