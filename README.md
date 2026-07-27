# conjunction-screener

A Python tool that pulls live TLE data from [Space-Track](https://www.space-track.org/),
propagates objects with SGP4, and flags close approaches between a primary
satellite and a catalog of potential conjunctors within a configurable time
window.

Tracking data on Space-Track refreshes roughly every 8-24 hours, while a
close-approach geometry can develop in minutes. This tool is a small,
transparent screening pass over that data: it doesn't replace an operational
conjunction assessment pipeline, but it demonstrates the same core steps
(propagation, miss-distance calculation, thresholding) end to end against
real catalog data.

## What it does

1. Authenticates with Space-Track and pulls the latest TLE for a primary
   object (default: ISS, NORAD 25544) plus a configurable list of
   conjunctor objects.
2. Propagates every object forward over a time window (default 24 hours) at
   a fixed timestep (default 60 s) using SGP4.
3. Computes the Euclidean distance between the primary and every conjunctor
   at each timestep.
4. Flags any pair whose minimum distance drops below a threshold (default
   5 km).
5. Writes a CSV report of flagged conjunctions and a miss-distance-vs-time
   plot for each one.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in your Space-Track credentials
```

A free Space-Track account is required: https://www.space-track.org/auth/createAccount

## Usage

```bash
python main.py --primary 25544 --conjunctors 48274 43013 44714 \
    --window-hours 24 --threshold-km 5 --out reports/
```

Output:

reports/conjunction_report.csv
reports/miss_distance.png


All flags are optional; running `python main.py` with no arguments screens
the ISS against a small default catalog over a 24-hour window. Omitting
`--conjunctors` entirely auto-discovers real candidates near the primary's
orbital regime instead (see `--altitude-band-km`).

## Real operational CDMs (`cdm_report.py`)

`main.py` computes its own miss-distance screening from public TLEs — a
useful demonstration, but not the actual operational product. Space-Track
also publishes real **Conjunction Data Messages (CDMs)**: the messages
18th Space Defense Squadron actually issues using full-fidelity
ephemerides and covariance data, on the same 8-24 hour cadence this
project is built around.

```bash
python cdm_report.py --primary 25544 --lookback-days 7
```

Pulls real CDMs issued for the given object in the last N days, prints
the closest approaches with their collision probability (Pc), and writes
a CSV to `reports/cdm_report.csv`. If your Space-Track account doesn't
have CDM access (`cdm` is a restricted class for some account tiers),
this will return an authorization error — the `main.py` SGP4 screening
path works regardless.

## Project layout

conjunction_screener/
├── auth.py # Space-Track authentication
├── fetcher.py # TLE retrieval
├── propagator.py # SGP4 propagation loop
├── screener.py # Miss distance calculation and flagging
├── cdm.py # Real Conjunction Data Message retrieval
└── reporter.py # CSV output and matplotlib plot
tests/
├── test_propagator.py
├── test_screener.py
├── test_reporter.py
└── test_cdm.py


## Testing

```bash
pytest -v
mypy conjunction_screener
```

CI runs the same on every push via GitHub Actions
(`.github/workflows/ci.yml`).

## Limitations

- Single-source TLE data (no covariance / uncertainty modelling), so this
  is a screening tool, not a conjunction assessment (CA) system — real
  CA relies on covariance-based probability of collision, not miss
  distance alone.
- SGP4 accuracy degrades over multi-day propagation windows; the default
  24-hour window keeps errors small relative to the flagging threshold.
- No de-duplication of TLE catalog entries for maneuvering or recently
  boosted objects.
- Auto-discovered candidates are matched on perigee/apogee altitude only,
  not orbital plane (inclination/RAAN). Two objects can share an
  altitude band while orbiting in completely different planes and never
  actually come close — so altitude-band matching alone is a coarse
  first pass, not a substitute for real plane-crossing geometry. In
  testing against 100 real LEO objects near the ISS's altitude, this
  correctly returned zero conjunctions rather than false positives, but
  it means a wider net (`--altitude-band-km`, `--threshold-km`) doesn't
  reliably surface close approaches on its own.
- `cdm_report.py` pulls real Space-Track Conjunction Data Messages, but
  the `cdm` class is a restricted "expandedspacedata" endpoint limited
  to satellite operators and approved government/commercial partners —
  most accounts, including a standard free account, will get a 401
  Unauthorized. It's included to show the intended integration against
  the real operational schema (and to handle both of Space-Track's
  historical CDM field-naming conventions), not as something a free
  account can run end-to-end.