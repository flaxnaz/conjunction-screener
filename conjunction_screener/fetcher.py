"""TLE retrieval from Space-Track.

Fetches the latest TLE (as a pair of GP element lines) for a primary
object plus a configurable catalog of potential conjunctors, and parses
them into a simple in-memory record.
"""

from __future__ import annotations

from dataclasses import dataclass

from spacetrack import SpaceTrackClient
import spacetrack.operators as op


@dataclass(frozen=True)
class TLERecord:
    """A single object's latest two-line element set."""

    norad_id: int
    name: str
    line1: str
    line2: str
    epoch: str


def fetch_latest_tles(
    client: SpaceTrackClient, norad_ids: list[int]
) -> list[TLERecord]:
    """Fetch the latest TLE for each NORAD catalog ID.

    Uses Space-Track's `tle_latest` class with `ordinal=1` (most recent
    element set per object). Objects with no current TLE on file are
    silently skipped rather than raising, since decayed or unlisted
    objects are common in an arbitrary conjunctor list.
    """
    if not norad_ids:
        return []

    rows = client.gp(
        norad_cat_id=op.inclusive_range(min(norad_ids), max(norad_ids))
        if len(norad_ids) > 1
        else norad_ids[0],
    )

    wanted = set(norad_ids)
    records: list[TLERecord] = []
    for row in rows:
        norad_id = int(row["NORAD_CAT_ID"])
        if norad_id not in wanted:
            continue
        records.append(
            TLERecord(
                norad_id=norad_id,
                name=row.get("OBJECT_NAME", f"OBJECT {norad_id}"),
                line1=row["TLE_LINE1"],
                line2=row["TLE_LINE2"],
                epoch=row.get("EPOCH", ""),
            )
        )
    return records

def fetch_orbital_regime_catalog(
    client: SpaceTrackClient,
    primary_norad_id: int,
    altitude_band_km: float = 50.0,
    limit: int = 100,
) -> tuple[TLERecord, list[TLERecord]]:
    """Find the primary object plus real candidate conjunctors nearby."""
    primary_rows = list(client.gp(norad_cat_id=primary_norad_id))
    if not primary_rows:
        raise ValueError(
            f"No TLE found on Space-Track for primary object {primary_norad_id}."
        )
    primary_row = primary_rows[0]
    periapsis = float(primary_row["PERIAPSIS"])
    apoapsis = float(primary_row["APOAPSIS"])
    primary = TLERecord(
        norad_id=int(primary_row["NORAD_CAT_ID"]),
        name=primary_row.get("OBJECT_NAME", f"OBJECT {primary_norad_id}"),
        line1=primary_row["TLE_LINE1"],
        line2=primary_row["TLE_LINE2"],
        epoch=primary_row.get("EPOCH", ""),
    )

    rows = client.gp(
        periapsis=op.inclusive_range(
            periapsis - altitude_band_km, periapsis + altitude_band_km
        ),
        apoapsis=op.inclusive_range(
            apoapsis - altitude_band_km, apoapsis + altitude_band_km
        ),
        decay_date=None,
        norad_cat_id=f"<>{primary_norad_id}",
        orderby="NORAD_CAT_ID",
        limit=limit,
    )

    conjunctors: list[TLERecord] = []
    for row in rows:
        conjunctors.append(
            TLERecord(
                norad_id=int(row["NORAD_CAT_ID"]),
                name=row.get("OBJECT_NAME", f"OBJECT {row['NORAD_CAT_ID']}"),
                line1=row["TLE_LINE1"],
                line2=row["TLE_LINE2"],
                epoch=row.get("EPOCH", ""),
            )
        )
    return primary, conjunctors

def fetch_primary_and_conjunctors(
    client: SpaceTrackClient,
    primary_norad_id: int,
    conjunctor_norad_ids: list[int],
) -> tuple[TLERecord, list[TLERecord]]:
    """Fetch the primary object's TLE plus its candidate conjunctors.

    Raises ValueError if the primary object's TLE could not be retrieved,
    since propagation cannot proceed without it.
    """
    all_ids = [primary_norad_id] + [
        n for n in conjunctor_norad_ids if n != primary_norad_id
    ]
    records = fetch_latest_tles(client, all_ids)
    by_id = {r.norad_id: r for r in records}

    if primary_norad_id not in by_id:
        raise ValueError(
            f"No TLE found on Space-Track for primary object "
            f"{primary_norad_id}."
        )

    primary = by_id[primary_norad_id]
    conjunctors = [
        by_id[n] for n in conjunctor_norad_ids if n in by_id and n != primary_norad_id
    ]
    return primary, conjunctors
