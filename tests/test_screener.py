from datetime import datetime, timedelta, timezone

import numpy as np

from conjunction_screener.propagator import PropagatedTrack
from conjunction_screener.screener import (
    compute_distances,
    screen_conjunctions,
    summarize_closest_approaches,
)


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

    # Both objects vary over time (not co-located), A stays farther than B
    obj_a = _track(2, "A", np.array([[4.0, 4.0, 4.0], [5.0, 5.0, 5.0], [6.0, 6.0, 6.0]]))
    obj_b = _track(3, "B", np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]))

    events = screen_conjunctions(primary, [obj_a, obj_b], threshold_km=10.0)

    assert [e.norad_id for e in events] == [3, 2]


def test_screen_conjunctions_skips_mismatched_grid():
    primary = _track(1, "PRIMARY", np.zeros((5, 3)))
    mismatched = _track(2, "MISMATCH", np.zeros((4, 3)))

    events = screen_conjunctions(primary, [mismatched], threshold_km=100.0)
    assert events == []


def test_screen_conjunctions_excludes_colocated_duplicate():
    """A catalog entry that stays at a constant near-zero distance the
    whole window (e.g. a duplicate NORAD ID for the same physical object,
    like an ISS module) should not be flagged as a genuine conjunction."""
    n = 10
    primary = _track(1, "PRIMARY", np.zeros((n, 3)))

    # Constant 0.1 km offset every timestep -> no variation -> co-located duplicate
    colocated_pos = np.full((n, 3), 0.1)
    colocated = _track(2, "ISS (DUPLICATE MODULE)", colocated_pos)

    # Genuine transient close approach: dips to 2 km, otherwise far away
    genuine_pos = np.full((n, 3), 50.0)
    genuine_pos[5] = [2.0, 0.0, 0.0]
    genuine = _track(3, "GENUINE", genuine_pos)

    events = screen_conjunctions(
        primary, [colocated, genuine], threshold_km=10.0
    )

    assert [e.norad_id for e in events] == [3]


def test_summarize_closest_approaches_covers_every_object():
    n = 10
    primary = _track(1, "PRIMARY", np.zeros((n, 3)))

    colocated_pos = np.full((n, 3), 0.1)
    colocated = _track(2, "ISS (DUPLICATE MODULE)", colocated_pos)

    genuine_pos = np.full((n, 3), 50.0)
    genuine_pos[5] = [2.0, 0.0, 0.0]
    genuine = _track(3, "GENUINE", genuine_pos)

    far_pos = np.array(
        [[100.0 + i, 100.0, 100.0] for i in range(n)], dtype=float
    )
    far = _track(4, "FAR", far_pos)

    summaries = summarize_closest_approaches(
        primary, [colocated, genuine, far], threshold_km=10.0
    )

    by_id = {s.norad_id: s for s in summaries}
    assert set(by_id) == {2, 3, 4}

    assert by_id[2].is_colocated is True
    assert by_id[2].is_flagged is False  # co-located, never a genuine flag

    assert by_id[3].is_colocated is False
    assert by_id[3].is_flagged is True
    assert by_id[3].miss_distance_km == 2.0

    assert by_id[4].is_colocated is False
    assert by_id[4].is_flagged is False
