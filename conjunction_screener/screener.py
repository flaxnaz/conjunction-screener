"""Miss-distance calculation and conjunction flagging."""

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
# catalog entry, not a conjunction. A genuine conjunction shows the
# distance dip and recover as two independent orbits cross; requiring a
# minimum spread between closest and farthest separation over the window
# screens out the flat, co-located case.
DEFAULT_MIN_SEPARATION_VARIATION_KM = 0.5


@dataclass(frozen=True)
class ConjunctionEvent:
    norad_id: int
    name: str
    time_of_closest_approach: datetime
    miss_distance_km: float
    distances_km: np.ndarray


def compute_distances(
    primary: PropagatedTrack, conjunctor: PropagatedTrack
) -> np.ndarray:
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