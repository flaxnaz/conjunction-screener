"""Miss-distance calculation and conjunction flagging."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from conjunction_screener.propagator import PropagatedTrack

DEFAULT_THRESHOLD_KM = 5.0

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
    """Return the per-timestep distance (km) between two tracks."""
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

    Delegates the actual flagging decision (threshold + co-location
    check) to `summarize_closest_approaches` so there is one source of
    truth for what counts as a genuine conjunction. Only re-derives the
    full per-timestep distance series for objects that were actually
    flagged, since that's the one thing `ConjunctionEvent` needs that
    the lighter-weight summary doesn't carry.
    """
    summaries = summarize_closest_approaches(
        primary, conjunctors, threshold_km, min_separation_variation_km
    )
    flagged_ids = {s.norad_id for s in summaries if s.is_flagged}
    if not flagged_ids:
        return []

    tracks_by_id = {t.norad_id: t for t in conjunctors}
    events: list[ConjunctionEvent] = []
    for s in summaries:
        if s.norad_id not in flagged_ids:
            continue
        track = tracks_by_id[s.norad_id]
        distances = compute_distances(primary, track)
        events.append(
            ConjunctionEvent(
                norad_id=s.norad_id,
                name=s.name,
                time_of_closest_approach=s.time_of_closest_approach,
                miss_distance_km=s.miss_distance_km,
                distances_km=distances,
            )
        )

    events.sort(key=lambda e: e.miss_distance_km)
    return events