from datetime import datetime

from conjunction_screener.cdm import _parse_row, fetch_recent_cdms


def test_parse_row_new_field_convention():
    """New Space-Track CSV format uses SAT1_/SAT2_ prefixes and meters."""
    row = {
        "CDM_ID": "123456",
        "TCA": "2026-08-01T12:00:00.000000",
        "MISS_DISTANCE": "437",
        "MISS_DISTANCE_UNIT": "m",
        "COLLISION_PROBABILITY": "2.5e-05",
        "SAT1_OBJECT_NAME": "ISS (ZARYA)",
        "SAT1_OBJECT_DESIGNATOR": "25544",
        "SAT2_OBJECT_NAME": "COSMOS 1408 DEB",
        "SAT2_OBJECT_DESIGNATOR": "49123",
    }
    event = _parse_row(row)

    assert event.cdm_id == "123456"
    assert event.tca == datetime.fromisoformat("2026-08-01T12:00:00.000000")
    assert abs(event.miss_distance_km - 0.437) < 1e-9
    assert event.collision_probability == 2.5e-05
    assert event.primary_name == "ISS (ZARYA)"
    assert event.primary_norad_id == 25544
    assert event.secondary_name == "COSMOS 1408 DEB"
    assert event.secondary_norad_id == 49123


def test_parse_row_legacy_field_convention():
    """Older Space-Track format uses SAT_1_/SAT_2_ with underscores."""
    row = {
        "CDM_ID": "654321",
        "TCA": "2026-08-02T00:00:00.000000",
        "MISS_DISTANCE": "1.2",
        "MISS_DISTANCE_UNIT": "km",
        "PC": "1.1e-04",
        "SAT_1_NAME": "ISS (ZARYA)",
        "SAT_1_ID": "25544",
        "SAT_2_NAME": "FENGYUN 1C DEB",
        "SAT_2_ID": "31234",
    }
    event = _parse_row(row)

    assert event.miss_distance_km == 1.2
    assert event.collision_probability == 1.1e-04
    assert event.primary_name == "ISS (ZARYA)"
    assert event.secondary_name == "FENGYUN 1C DEB"
    assert event.primary_norad_id == 25544
    assert event.secondary_norad_id == 31234


def test_parse_row_missing_optional_fields():
    """Missing probability/IDs shouldn't raise, just come back as None."""
    row = {
        "CDM_ID": "1",
        "TCA": "2026-08-01T00:00:00.000000",
        "MISS_DISTANCE": "500",
        "MISS_DISTANCE_UNIT": "m",
    }
    event = _parse_row(row)

    assert event.collision_probability is None
    assert event.primary_norad_id is None
    assert event.secondary_norad_id is None
    assert event.primary_name == "UNKNOWN"


class _FakeClient:
    """Minimal stand-in for SpaceTrackClient.cdm() to test the query path."""

    def __init__(self, rows):
        self.rows = rows
        self.last_kwargs = None

    def cdm(self, **kwargs):
        self.last_kwargs = kwargs
        return self.rows


def test_fetch_recent_cdms_passes_expected_query_and_parses_rows():
    fake_rows = [
        {
            "CDM_ID": "1",
            "TCA": "2026-08-01T00:00:00.000000",
            "MISS_DISTANCE": "100",
            "MISS_DISTANCE_UNIT": "m",
            "SAT1_OBJECT_NAME": "ISS (ZARYA)",
            "SAT1_OBJECT_DESIGNATOR": "25544",
            "SAT2_OBJECT_NAME": "DEBRIS A",
            "SAT2_OBJECT_DESIGNATOR": "99999",
        }
    ]
    client = _FakeClient(fake_rows)

    events = fetch_recent_cdms(client, norad_id=25544, lookback_days=7, limit=50)  # type: ignore[arg-type]

    assert len(events) == 1
    assert events[0].secondary_name == "DEBRIS A"
    assert client.last_kwargs["message_for"] == 25544
    assert client.last_kwargs["creation_date"] == ">now-7"
    assert client.last_kwargs["limit"] == 50