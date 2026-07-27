"""CLI entry point for pulling real operational CDMs from Space-Track."""

from __future__ import annotations

import argparse
from pathlib import Path

from conjunction_screener.auth import get_client
from conjunction_screener.cdm import fetch_recent_cdms
from conjunction_screener.reporter import write_cdm_csv_report

ISS_NORAD_ID = 25544


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pull real Space-Track CDMs.")
    parser.add_argument("--primary", type=int, default=ISS_NORAD_ID)
    parser.add_argument("--lookback-days", type=int, default=7)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--out", type=str, default="reports")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)

    client = get_client()
    events = fetch_recent_cdms(client, args.primary, args.lookback_days, args.limit)

    csv_path = write_cdm_csv_report(events, out_dir / "cdm_report.csv")

    print(f"Primary: NORAD {args.primary}")
    print(
        f"Found {len(events)} real CDM(s) issued in the last "
        f"{args.lookback_days} day(s)."
    )
    for e in sorted(events, key=lambda e: e.miss_distance_km)[:10]:
        pc = (
            f"{e.collision_probability:.2e}"
            if e.collision_probability is not None
            else "n/a"
        )
        print(
            f"  {e.secondary_name} vs {e.primary_name}: "
            f"{e.miss_distance_km:.3f} km at {e.tca.isoformat()} (Pc={pc})"
        )
    print(f"Report written to: {csv_path}")


if __name__ == "__main__":
    main()