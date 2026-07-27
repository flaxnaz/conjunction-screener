"""A self-contained HTML dashboard for a screening run.

Renders a single HTML file (no external JS dependency — the chart is
plain server-rendered SVG) summarizing a screening pass: how many
objects were checked, how many were flagged, and a radial "screening
sweep" plot where each object's position encodes real information —
angle is when in the window its closest approach happened, radius is
how close it got. The flagging threshold is drawn as its own ring so
"how close is close" is visible at a glance.
"""

from __future__ import annotations

import html
import math
from datetime import datetime, timezone
from pathlib import Path

from conjunction_screener.propagator import PropagatedTrack
from conjunction_screener.screener import ClosestApproachSummary

_SVG_SIZE = 560
_CENTER = _SVG_SIZE / 2
_MAX_RADIUS_PX = 220
_RING_FRACTIONS = (0.25, 0.5, 0.75, 1.0)


def _angle_for_time(t: datetime, start: datetime, window_hours: float) -> float:
    """Map a timestamp to degrees around the sweep (0 = top, clockwise)."""
    if window_hours <= 0:
        return -90.0
    elapsed_s = (t - start).total_seconds()
    frac = max(0.0, min(1.0, elapsed_s / (window_hours * 3600.0)))
    return frac * 360.0 - 90.0


def _point_on_circle(angle_deg: float, radius_px: float) -> tuple[float, float]:
    theta = math.radians(angle_deg)
    return (_CENTER + radius_px * math.cos(theta), _CENTER + radius_px * math.sin(theta))


def _build_svg(
    primary: PropagatedTrack,
    summaries: list[ClosestApproachSummary],
    threshold_km: float,
    window_hours: float,
) -> str:
    active = [s for s in summaries if not s.is_colocated]
    real_max = max([s.miss_distance_km for s in active], default=threshold_km)
    scale_max_km = max(threshold_km * 2.2, real_max * 1.1, 1.0)
    px_per_km = _MAX_RADIUS_PX / scale_max_km
    start_time = primary.times[0] if primary.times else datetime.min

    parts: list[str] = [
        f'<svg viewBox="0 0 {_SVG_SIZE} {_SVG_SIZE}" xmlns="http://www.w3.org/2000/svg">'
    ]

    # Concentric range rings (grid) with distance labels.
    for frac in _RING_FRACTIONS:
        r = _MAX_RADIUS_PX * frac
        parts.append(
            f'<circle cx="{_CENTER}" cy="{_CENTER}" r="{r:.1f}" '
            f'class="sweep-ring" />'
        )
        label_km = scale_max_km * frac
        parts.append(
            f'<text x="{_CENTER + 6}" y="{_CENTER - r + 12}" class="sweep-ring-label">'
            f"{label_km:.0f} km</text>"
        )

    # Time-axis tick marks around the circle (4 compass points), so the
    # angle encoding is actually readable rather than implicit.
    for frac in (0.0, 0.25, 0.5, 0.75):
        angle = frac * 360.0 - 90.0
        inner_x, inner_y = _point_on_circle(angle, _MAX_RADIUS_PX - 6)
        outer_x, outer_y = _point_on_circle(angle, _MAX_RADIUS_PX + 6)
        label_x, label_y = _point_on_circle(angle, _MAX_RADIUS_PX + 22)
        parts.append(
            f'<line x1="{inner_x:.1f}" y1="{inner_y:.1f}" '
            f'x2="{outer_x:.1f}" y2="{outer_y:.1f}" class="sweep-tick" />'
        )
        parts.append(
            f'<text x="{label_x:.1f}" y="{label_y:.1f}" class="sweep-tick-label" '
            f'text-anchor="middle">T+{frac * window_hours:.0f}h</text>'
        )

    # Flagging threshold ring, drawn distinctly.
    threshold_r = min(threshold_km * px_per_km, _MAX_RADIUS_PX)
    parts.append(
        f'<circle cx="{_CENTER}" cy="{_CENTER}" r="{threshold_r:.1f}" '
        f'class="sweep-threshold" />'
    )
    tx, ty = _point_on_circle(45.0, threshold_r)
    parts.append(
        f'<text x="{tx:.1f}" y="{ty:.1f}" class="sweep-threshold-label">'
        f"THRESHOLD {threshold_km:g} KM</text>"
    )

    # Object blips.
    for s in active:
        angle = _angle_for_time(s.time_of_closest_approach, start_time, window_hours)
        radius_px = min(s.miss_distance_km * px_per_km, _MAX_RADIUS_PX)
        x, y = _point_on_circle(angle, radius_px)
        css_class = "sweep-blip-flag" if s.is_flagged else "sweep-blip-clear"
        r = 6.5 if s.is_flagged else 3.5
        if s.is_flagged:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="12" class="sweep-halo" />')
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" class="{css_class}" />')
        if s.is_flagged:
            parts.append(
                f'<text x="{x + 10:.1f}" y="{y + 4:.1f}" class="sweep-blip-label">'
                f"{html.escape(s.name)}</text>"
            )

    # Primary marker at center: crosshair + dot.
    parts.append(
        f'<line x1="{_CENTER - 14}" y1="{_CENTER}" x2="{_CENTER + 14}" y2="{_CENTER}" '
        f'class="sweep-crosshair" />'
        f'<line x1="{_CENTER}" y1="{_CENTER - 14}" x2="{_CENTER}" y2="{_CENTER + 14}" '
        f'class="sweep-crosshair" />'
        f'<circle cx="{_CENTER}" cy="{_CENTER}" r="5" class="sweep-primary" />'
    )

    parts.append("</svg>")
    return "".join(parts)


def _status_block(
    flagged: list[ClosestApproachSummary], threshold_km: float, window_hours: float
) -> str:
    if not flagged:
        return (
            '<div class="status-pill status-clear">CLEAR</div>'
            f"<p class=\"status-text\">No object crossed the {threshold_km:g} km "
            f"flagging threshold across the {window_hours:g}-hour window.</p>"
        )

    rows = "".join(
        f"<tr><td>{html.escape(s.name)}</td><td>{s.norad_id}</td>"
        f"<td>{s.miss_distance_km:.3f}</td>"
        f'<td>{s.time_of_closest_approach.strftime("%Y-%m-%d %H:%M:%S")} UTC</td></tr>'
        for s in sorted(flagged, key=lambda s: s.miss_distance_km)
    )
    return (
        f'<div class="status-pill status-flag">{len(flagged)} FLAGGED</div>'
        '<table class="results-table"><thead><tr>'
        "<th>Object</th><th>NORAD ID</th><th>Miss distance (km)</th>"
        "<th>Time of closest approach</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def write_html_dashboard(
    primary: PropagatedTrack,
    summaries: list[ClosestApproachSummary],
    threshold_km: float,
    window_hours: float,
    output_path: str | Path,
    generated_at: datetime | None = None,
) -> Path:
    """Render a self-contained HTML dashboard for one screening run."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    active = [s for s in summaries if not s.is_colocated]
    flagged = [s for s in active if s.is_flagged]
    closest = min((s.miss_distance_km for s in active), default=float("nan"))
    generated_at = generated_at or datetime.now(timezone.utc)

    svg = _build_svg(primary, summaries, threshold_km, window_hours)
    status_html = _status_block(flagged, threshold_km, window_hours)
    closest_str = f"{closest:.2f}" if active else "n/a"

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Conjunction Screening — {html.escape(primary.name)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {{
    --void: #0a0e14;
    --panel: #121822;
    --panel-border: #1f2b38;
    --grid: #223042;
    --text-primary: #e7edf3;
    --text-dim: #6f8299;
    --track: #4fb2f0;
    --threshold: #f0b429;
    --flag: #ff5c5c;
    --clear: #3ddc97;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: radial-gradient(circle at 50% 0%, #0d1420 0%, var(--void) 60%);
    color: var(--text-primary);
    font-family: 'Inter', system-ui, sans-serif;
    padding: 48px 32px 64px;
  }}
  .wrap {{ max-width: 880px; margin: 0 auto; }}
  .eyebrow {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    letter-spacing: 0.18em;
    color: var(--track);
    text-transform: uppercase;
  }}
  h1 {{
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 34px;
    margin: 8px 0 4px;
    letter-spacing: -0.01em;
  }}
  .subtitle {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    color: var(--text-dim);
    margin: 0 0 32px;
  }}
  .kpi-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-bottom: 28px;
  }}
  .kpi-card {{
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: 14px;
    padding: 16px 18px;
  }}
  .kpi-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10.5px;
    letter-spacing: 0.12em;
    color: var(--text-dim);
    text-transform: uppercase;
  }}
  .kpi-value {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 26px;
    font-weight: 500;
    margin-top: 6px;
    color: var(--text-primary);
  }}
  .panel {{
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: 20px;
    padding: 28px;
    margin-bottom: 20px;
  }}
  .panel-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 500;
    font-size: 15px;
    color: var(--text-primary);
    margin: 0 0 4px;
  }}
  .panel-caption {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11.5px;
    color: var(--text-dim);
    margin: 0 0 18px;
  }}
  .sweep-wrap {{ display: flex; justify-content: center; }}
  svg {{ width: 100%; max-width: 480px; }}
  .sweep-ring {{ fill: none; stroke: #35485f; stroke-width: 1; opacity: 0.9; }}
  .sweep-ring-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9.5px;
    fill: var(--text-dim);
  }}
  .sweep-tick {{ stroke: #35485f; stroke-width: 1.4; }}
  .sweep-tick-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    fill: var(--text-dim);
  }}
  .sweep-threshold {{
    fill: none;
    stroke: var(--threshold);
    stroke-width: 1.6;
    stroke-dasharray: 5 4;
    opacity: 0.95;
  }}
  .sweep-threshold-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9.5px;
    fill: var(--threshold);
  }}
  .sweep-blip-clear {{ fill: var(--track); opacity: 0.75; }}
  .sweep-blip-flag {{ fill: var(--flag); }}
  .sweep-halo {{ fill: var(--flag); opacity: 0.18; }}
  .sweep-blip-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    fill: var(--flag);
  }}
  .sweep-crosshair {{ stroke: var(--text-dim); stroke-width: 1; }}
  .sweep-primary {{ fill: var(--track); }}
  .legend {{
    display: flex;
    gap: 22px;
    justify-content: center;
    margin-top: 16px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: var(--text-dim);
  }}
  .legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
  .dot {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; }}
  .status-pill {{
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.08em;
    padding: 6px 14px;
    border-radius: 999px;
    margin-bottom: 12px;
  }}
  .status-clear {{ background: rgba(61, 220, 151, 0.12); color: var(--clear); }}
  .status-flag {{ background: rgba(255, 92, 92, 0.12); color: var(--flag); }}
  .status-text {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    color: var(--text-dim);
    margin: 0;
  }}
  .results-table {{
    width: 100%;
    border-collapse: collapse;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12.5px;
    margin-top: 8px;
  }}
  .results-table th {{
    text-align: left;
    color: var(--text-dim);
    font-weight: 500;
    padding: 8px 10px;
    border-bottom: 1px solid var(--panel-border);
    text-transform: uppercase;
    font-size: 10.5px;
    letter-spacing: 0.06em;
  }}
  .results-table td {{
    padding: 9px 10px;
    border-bottom: 1px solid var(--panel-border);
    color: var(--text-primary);
  }}
  footer {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: var(--text-dim);
    margin-top: 24px;
    display: flex;
    justify-content: space-between;
  }}
</style>
</head>
<body>
<div class="wrap">
  <div class="eyebrow">Space Situational Awareness · Screening Pass</div>
  <h1>{html.escape(primary.name)}</h1>
  <p class="subtitle">NORAD {primary.norad_id} · {window_hours:g}h window · {threshold_km:g} km threshold · generated {generated_at.strftime("%Y-%m-%d %H:%M:%S")} UTC</p>

  <div class="kpi-grid">
    <div class="kpi-card">
      <div class="kpi-label">Objects screened</div>
      <div class="kpi-value">{len(active)}</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Flagged</div>
      <div class="kpi-value">{len(flagged)}</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Closest approach</div>
      <div class="kpi-value">{closest_str} km</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Window</div>
      <div class="kpi-value">{window_hours:g}h</div>
    </div>
  </div>

  <div class="panel">
    <p class="panel-title">Screening sweep</p>
    <p class="panel-caption">Angle = time of closest approach within the window · radius = miss distance</p>
    <div class="sweep-wrap">{svg}</div>
    <div class="legend">
      <span><span class="dot" style="background:var(--track)"></span>screened</span>
      <span><span class="dot" style="background:var(--flag)"></span>flagged</span>
      <span><span class="dot" style="background:var(--threshold)"></span>threshold ring</span>
    </div>
  </div>

  <div class="panel">
    <p class="panel-title">Result</p>
    {status_html}
  </div>

  <footer>
    <span>Data: Space-Track (TLE) · SGP4 propagation</span>
    <span>conjunction-screener</span>
  </footer>
</div>
</body>
</html>
"""
    output_path.write_text(html_doc, encoding="utf-8")
    return output_path