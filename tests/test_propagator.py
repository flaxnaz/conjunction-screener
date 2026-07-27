from datetime import datetime, timezone

import numpy as np
import pytest

from conjunction_screener.fetcher import TLERecord
from conjunction_screener.propagator import (
    PropagationError,
    propagate_all,
    propagate_track,
)

# A real, well-known ISS TLE (fixed, not fetched live) used purely as a
# stable fixture for propagation tests.
ISS_TLE = TLERecord(
    norad_id=25544,
    name="ISS (ZARYA)",
    line1="1 25544U 98067A   24045.51782528  .00016717  00000-0  30197-3 0  9995",
    line2="2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.49560856437221",
    epoch="2024-02-14",
)

# Deliberately corrupted checksum/format to trigger an SGP4-level failure.
BAD_TLE = TLERecord(
    norad_id=99999,
    name="GARBAGE",
    line1="1 99999U 00000A   24045.51782528  .00016717  00000-0  99999-3 0  9990",
    line2="2 99999  51.6416 247.4627 9999999 130.5360 325.0288 15.49560856437221",
    epoch="2024-02-14",
)


def test_propagate_track_shape_and_grid():
    start = datetime(2024, 2, 15, tzinfo=timezone.utc)
    track = propagate_track(
        ISS_TLE, start, window_hours=2.0, timestep_seconds=60.0
    )

    expected_steps = int(round((2.0 * 3600.0) / 60.0)) + 1
    assert track.positions_km.shape == (expected_steps, 3)
    assert len(track.times) == expected_steps
    assert track.times[0] == start
    assert track.norad_id == 25544


def test_propagate_track_leo_altitude_sane():
    """Positions should sit at a plausible LEO radius (Earth radius ~6378 km
    plus ISS altitude ~400-420 km), as a sanity check on the propagation
    rather than an exact ephemeris match."""
    start = datetime(2024, 2, 15, tzinfo=timezone.utc)
    track = propagate_track(ISS_TLE, start, window_hours=1.0, timestep_seconds=300.0)

    radii = np.linalg.norm(track.positions_km, axis=1)
    assert np.all(radii > 6700.0)
    assert np.all(radii < 6900.0)


def test_propagate_track_raises_on_bad_elements():
    start = datetime(2024, 2, 15, tzinfo=timezone.utc)
    with pytest.raises(PropagationError):
        propagate_track(BAD_TLE, start, window_hours=24.0, timestep_seconds=60.0)


def test_propagate_all_skips_failures_by_default():
    start = datetime(2024, 2, 15, tzinfo=timezone.utc)
    tracks = propagate_all(
        [ISS_TLE, BAD_TLE], start, window_hours=1.0, timestep_seconds=300.0
    )
    assert len(tracks) == 1
    assert tracks[0].norad_id == 25544


def test_propagate_all_raises_when_skip_disabled():
    start = datetime(2024, 2, 15, tzinfo=timezone.utc)
    with pytest.raises(PropagationError):
        propagate_all(
            [BAD_TLE],
            start,
            window_hours=1.0,
            timestep_seconds=300.0,
            skip_failures=False,
        )
