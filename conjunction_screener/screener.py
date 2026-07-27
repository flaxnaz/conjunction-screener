"""Miss-distance calculation and conjunction flagging.

For a primary track and a list of conjunctor tracks (already propagated
on the same time grid), computes the Euclidean distance at every
timestep and flags any pair whose minimum distance drops below a
threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from conjunction_screener.propagator import PropagatedTrack

DEFAULT_THRESHOLD_KM = 5.0

# Some catalog entries (e.g. individual ISS modules like ZVEZDA, UNITY,
# DESTINY, POISK) are permanently co-located with their parent object —
# they've never independently orbited anything, and Space-Track publishes
# essentially the same elements for all of them. Their distance to the
# primary stays flat near zero for the whole window, which is a duplicate
# catalog entry, not a conjunction. A genuine conjunction between two
# independently orbiting objects shows the distance dip and recover as
# their orbits cross; requiring some minimum spread between the closest
# and farthest separation over the window screens out the flat, co-located
# case without needing a hardcoded exclusion list.
DEFAULT_MIN_SEPARATION_VARIATION_KM = 0.5


@dataclass(frozen=True)
class ConjunctionEvent:
    """A flagged close approach between the primary and one conjunctor."""

    norad_id: int
    name: str
    time_of_closest_approach: datetime
    miss_distance_km: float
    distances_km: np.ndarray  # full distance-vs-time series for plotting


@dataclass(frozen=True)
class ClosestApproachSummary:
    """One-line summary of a track's closest approach to the primary.

    Lighter-weight than ConjunctionEvent (no full distance series) —
    built for every screened object, not just flagged ones, so a
    dashboard can show the full picture (including objects that stayed
    safely clear) rather than only the flagged subset.
    """

    norad_id: int
    name: str
    miss_distance_km: float
    time_of_closest_approach: datetime
    is_colocated: bool
    is_flagged: bool


def summarize_closest_approaches(
    primary: PropagatedTrack,
    conjunctors: list[PropagatedTrack],
    threshold_km: float = DEFAULT_THRESHOLD_KM,
    min_separation_variation_km: float = DEFAULT_MIN_SEPARATION_VARIATION_KM,
) -> list[ClosestApproachSummary]:
    """Summarize every screened object's closest approach to the primary.

    Unlike `screen_conjunctions`, this returns one entry per object
    regardless of whether it was flagged — including co-located catalog
    duplicates (marked `is_colocated=True` rather than dropped), so a
    caller building a full picture (e.g. a dashboard) can decide how to
    represent them instead of losing them silently.
    """
    summaries: list[ClosestApproachSummary] = []

    for track in conjunctors:
        if track.positions_km.shape != primary.positions_km.shape:
            continue
        distances = compute_distances(primary, track)
        min_idx = int(np.argmin(distances))
        min_dist = float(distances[min_idx])
        spread = float(np.max(distances) - np.min(distances))
        is_colocated = spread < min_separation_variation_km

        summaries.append(
            ClosestApproachSummary(
                norad_id=track.norad_id,
                name=track.name,
                miss_distance_km=min_dist,
                time_of_closest_approach=primary.times[min_idx],
                is_colocated=is_colocated,
                is_flagged=(min_dist <= threshold_km) and not is_colocated,
            )
        )

    return summaries


def compute_distances(
    primary: PropagatedTrack, conjunctor: PropagatedTrack
) -> np.ndarray:
    """Return the per-timestep distance (km) between two tracks.

    Requires both tracks to share the same time grid (same length, same
    start/timestep) — this is guaranteed when both were produced by
    `propagate_all`/`propagate_track` with matching start_time, window,
    and timestep arguments.
    """
    if primary.positions_km.shape != conjunctor.positions_km.shape:
        raise ValueError(
            f"Track length mismatch between primary ({primary.positions_km.shape}) "
            f"and {conjunctor.name} ({conjunctor.positions_km.shape}); "
            "propagate both on the same time grid."
        )
    diff = primary.positions_km - conjunctor.positions_km
    return np.linalg.norm(diff, axis=1)


def screen_conjunctions(
    primary: PropagatedTrack,
    conjunctors: list[PropagatedTrack],
    threshold_km: float = DEFAULT_THRESHOLD_KM,
    min_separation_variation_km: float = DEFAULT_MIN_SEPARATION_VARIATION_KM,
) -> list[ConjunctionEvent]:
    """Flag conjunctions below `threshold_km`, sorted by closest first.

    Objects whose track doesn't share the primary's time grid are
    skipped defensively (this shouldn't happen if both came from the
    same propagate_all call, but a length mismatch is cheap to guard).

    Objects that stay within `min_separation_variation_km` of the same
    distance for the entire window are skipped as co-located catalog
    duplicates rather than genuine conjunctions (see module docstring).
    """
    events: list[ConjunctionEvent] = []

    for track in conjunctors:
        if track.positions_km.shape != primary.positions_km.shape:
            continue
        distances = compute_distances(primary, track)
        min_idx = int(np.argmin(distances))
        min_dist = float(distances[min_idx])
        spread = float(np.max(distances) - np.min(distances))

        if min_dist <= threshold_km and spread >= min_separation_variation_km:
            events.append(
                ConjunctionEvent(
                    norad_id=track.norad_id,
                    name=track.name,
                    time_of_closest_approach=primary.times[min_idx],
                    miss_distance_km=min_dist,
                    distances_km=distances,
                )
            )

    events.sort(key=lambda e: e.miss_distance_km)
    return events
