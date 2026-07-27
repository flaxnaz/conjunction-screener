from datetime import datetime, timedelta, timezone

import numpy as np

from conjunction_screener.propagator import PropagatedTrack
from conjunction_screener.screener import compute_distances, screen_conjunctions


def _track(norad_id: int, name: str, positions: np.ndarray) -> PropagatedTrack:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    times = [start + timedelta(seconds=60 * i) for i in range(positions.shape[0])]
    return PropagatedTrack(
        norad_id=norad_id, name=name, times=times, positions_km=positions
    )


def test_compute_distances_simple_offset():
    n = 5
    primary_pos = np.zeros((n, 3))
    # constant 10 km offset along x
    conj_pos = np.zeros((n, 3))
    conj_pos[:, 0] = 10.0

    primary = _track(1, "PRIMARY", primary_pos)
    conjunctor = _track(2, "CONJ", conj_pos)

    distances = compute_distances(primary, conjunctor)
    assert distances.shape == (n,)
    assert np.allclose(distances, 10.0)


def test_compute_distances_shape_mismatch_raises():
    primary = _track(1, "PRIMARY", np.zeros((5, 3)))
    conjunctor = _track(2, "CONJ", np.zeros((4, 3)))
    try:
        compute_distances(primary, conjunctor)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_screen_conjunctions_flags_below_threshold():
    n = 10
    primary_pos = np.zeros((n, 3))
    primary = _track(1, "PRIMARY", primary_pos)

    # Object A: distance dips to 2 km at the midpoint -> should be flagged
    close_pos = np.full((n, 3), 50.0)
    close_pos[5] = [2.0, 0.0, 0.0]
    close = _track(2, "CLOSE", close_pos)

    # Object B: always far away -> should not be flagged
    far_pos = np.full((n, 3), 100.0)
    far = _track(3, "FAR", far_pos)

    events = screen_conjunctions(primary, [close, far], threshold_km=5.0)

    assert len(events) == 1
    assert events[0].norad_id == 2
    assert events[0].miss_distance_km == 2.0
    assert events[0].time_of_closest_approach == primary.times[5]


def test_screen_conjunctions_sorted_closest_first():
    n = 3
    primary = _track(1, "PRIMARY", np.zeros((n, 3)))

    obj_a = _track(2, "A", np.full((n, 3), 4.0))  # ~6.93 km
    obj_b = _track(3, "B", np.full((n, 3), 1.0))  # ~1.73 km

    events = screen_conjunctions(primary, [obj_a, obj_b], threshold_km=10.0)

    assert [e.norad_id for e in events] == [3, 2]


def test_screen_conjunctions_skips_mismatched_grid():
    primary = _track(1, "PRIMARY", np.zeros((5, 3)))
    mismatched = _track(2, "MISMATCH", np.zeros((4, 3)))

    events = screen_conjunctions(primary, [mismatched], threshold_km=100.0)
    assert events == []
