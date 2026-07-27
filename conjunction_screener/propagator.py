"""SGP4 propagation over a fixed window.

Propagates each TLE forward from its epoch (or a caller-supplied start
time) at a fixed timestep, returning ECI position vectors (km) at every
sample point. Kept separate from screening/reporting so it can be tested
and reused independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
from sgp4.api import Satrec, SGP4_ERRORS, jday

from conjunction_screener.fetcher import TLERecord


class PropagationError(RuntimeError):
    """Raised when SGP4 reports a propagation error for an object."""


@dataclass(frozen=True)
class PropagatedTrack:
    """Time-sampled ECI positions for one object over the window."""

    norad_id: int
    name: str
    times: list[datetime]
    positions_km: np.ndarray  # shape (N, 3)


def _to_julian(t: datetime) -> tuple[float, float]:
    t = t.astimezone(timezone.utc)
    return jday(
        t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond / 1e6
    )


def propagate_track(
    tle: TLERecord,
    start_time: datetime,
    window_hours: float = 24.0,
    timestep_seconds: float = 60.0,
) -> PropagatedTrack:
    """Propagate a single TLE across the window using SGP4.

    Raises PropagationError if SGP4 returns a non-zero error code at any
    sample point (e.g. decayed orbit, bad elements) rather than silently
    emitting garbage positions.
    """
    sat = Satrec.twoline2rv(tle.line1, tle.line2)

    n_steps = int(round((window_hours * 3600.0) / timestep_seconds)) + 1
    times = [start_time + timedelta(seconds=i * timestep_seconds) for i in range(n_steps)]
    positions = np.empty((n_steps, 3), dtype=float)

    for i, t in enumerate(times):
        jd, fr = _to_julian(t)
        error_code, r, _v = sat.sgp4(jd, fr)
        if error_code != 0:
            raise PropagationError(
                f"SGP4 error for {tle.name} (NORAD {tle.norad_id}) at "
                f"{t.isoformat()}: {SGP4_ERRORS[error_code]}"
            )
        positions[i, :] = r

    return PropagatedTrack(
        norad_id=tle.norad_id, name=tle.name, times=times, positions_km=positions
    )


def propagate_all(
    tles: list[TLERecord],
    start_time: datetime,
    window_hours: float = 24.0,
    timestep_seconds: float = 60.0,
    skip_failures: bool = True,
) -> list[PropagatedTrack]:
    """Propagate a list of TLEs, optionally skipping ones that fail.

    Conjunctor catalogs routinely include objects with stale or marginal
    elements; skipping failures (default) keeps a single bad object from
    aborting the whole screening run. The primary object should generally
    be propagated separately with skip_failures effectively False (i.e.
    call propagate_track directly and let it raise).
    """
    tracks: list[PropagatedTrack] = []
    for tle in tles:
        try:
            tracks.append(
                propagate_track(tle, start_time, window_hours, timestep_seconds)
            )
        except PropagationError:
            if not skip_failures:
                raise
            continue
    return tracks
