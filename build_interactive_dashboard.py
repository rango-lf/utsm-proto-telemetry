"""Build a self-contained interactive HTML dashboard for the afternoon run."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "Missing required package. Install dependencies with: pip install pandas numpy"
    ) from exc

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from analyze_strategy import build_laps
from utsm_telemetry import (
    FORWARD_AXIS_CHOICES,
    add_xy,
    compute_accel_candidate_scores,
    derive_motion_energy,
    merge_by_time,
    read_gpx,
    read_telemetry,
)
from utsm_telemetry.simulation import (
    build_full_run_distance,
    build_strategy_report,
    build_strategy_samples,
    build_strategy_segments,
    fit_empirical_energy_model,
    optimize_speed_profile,
)

DEFAULT_GPS = "Utsm-2.gpx"
DEFAULT_TELEMETRY = os.path.join(
    "telemetry_dumps",
    "telemetry_20260411_122713.csv",
)
DEFAULT_OUTPUT = os.path.join("outputs", "afternoon_interactive_dashboard.html")

METRICS = {
    "gpsAccel": {
        "label": "GPS accel magnitude",
        "unit": "m/s^2",
        "field": "gpsAccel",
        "source": "gps_longitudinal_accel_abs_m_s2",
        "color": "#2ca02c",
    },
    "imuAccel": {
        "label": "MPU dynamic acceleration",
        "unit": "m/s^2",
        "field": "imuAccel",
        "source": "imu_forward_dynamic_m_s2",
        "color": "#16a34a",
    },
    "speed": {
        "label": "Speed",
        "unit": "km/h",
        "field": "speed",
        "source": "speed_kph",
        "color": "#1f77b4",
    },
    "targetSpeed": {
        "label": "Strategy target speed",
        "unit": "km/h",
        "field": "targetSpeed",
        "source": "target_speed_kph",
        "color": "#f97316",
    },
    "current": {
        "label": "Current",
        "unit": "mA",
        "field": "current",
        "source": "current_mA",
        "color": "#d62728",
    },
    "power": {
        "label": "Power",
        "unit": "W",
        "field": "power",
        "source": "power_w",
        "color": "#9467bd",
    },
    "runEnergyJ": {
        "label": "Total energy",
        "unit": "J",
        "field": "runEnergyJ",
        "source": "cum_energy_j",
        "color": "#b45309",
        "map_selectable": False,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an interactive telemetry dashboard from a GPX and telemetry CSV."
    )
    parser.add_argument("--gps", default=DEFAULT_GPS)
    parser.add_argument("--telemetry", default=DEFAULT_TELEMETRY)
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT)
    parser.add_argument("--laps", type=int, default=3)
    parser.add_argument(
        "--split-method",
        choices=["points", "time", "line", "start"],
        default="start",
    )
    parser.add_argument("--lap-times", nargs="+", metavar="ELAPSED")
    parser.add_argument("--start-time")
    parser.add_argument("--time-offset-ms", type=float, default=0.0)
    parser.add_argument("--tolerance-sec", type=float, default=1.5)
    parser.add_argument(
        "--forward-axis",
        choices=FORWARD_AXIS_CHOICES,
        default="ax",
    )
    parser.add_argument("--accel-window", type=int, default=5)
    parser.add_argument("--accel-scale", type=float, default=1000.0)
    parser.add_argument("--imu-axis", choices=["ax", "ay", "az"], default="ax")
    parser.add_argument("--imu-axis-sign", type=int, choices=[-1, 1], default=1)
    parser.add_argument("--accel-bias-window-sec", type=float, default=30.0)
    parser.add_argument("--accel-smooth-window-sec", type=float, default=8.0)
    parser.add_argument("--strategy-segments", type=int, default=24)
    parser.add_argument("--strategy-speed-min-kph", type=float, default=8.0)
    parser.add_argument("--strategy-speed-max-kph", type=float, default=40.0)
    parser.add_argument("--strategy-max-delta-kph-per-segment", type=float, default=6.0)
    parser.add_argument("--strategy-speed-step-kph", type=float, default=1.0)
    parser.add_argument("--strategy-time-budget-sec", type=float)
    parser.add_argument("--strategy-lap-time-target-sec", type=float)
    return parser.parse_args()


def load_derived_run(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, str, float]:
    """Build one merged run plus the matching optimized strategy payload."""
    gps_df = read_gpx(args.gps)
    telem_df = read_telemetry(args.telemetry)
    gps_laps, telem_laps, _ = build_laps(gps_df, telem_df, args)

    rows = []
    for lap_num, (lap_gps, lap_telem) in enumerate(zip(gps_laps, telem_laps), start=1):
        if lap_gps.empty or lap_telem.empty:
            continue
        try:
            merged = merge_by_time(lap_telem, lap_gps, args.tolerance_sec)
        except ValueError as exc:
            print(f"Lap {lap_num}: skipping - {exc}")
            continue
        lap_start = lap_gps["time"].iloc[0]
        lap_end = lap_gps["time"].iloc[-1]
        merged = merged[
            (merged["time"] >= lap_start)
            & (merged["time"] <= lap_end)
        ].copy()
        if merged.empty:
            print(f"Lap {lap_num}: skipping - no samples inside lap time range")
            continue
        derived = derive_motion_energy(
            merged,
            forward_axis=args.forward_axis,
            accel_window=args.accel_window,
            accel_scale=args.accel_scale,
            imu_axis=args.imu_axis,
            imu_axis_sign=args.imu_axis_sign,
            accel_bias_window_sec=args.accel_bias_window_sec,
            accel_smooth_window_sec=args.accel_smooth_window_sec,
        )
        derived["lap"] = lap_num
        rows.append(derived)

    if not rows:
        raise ValueError("No lap data could be merged for dashboard generation.")

    df = pd.concat(rows, ignore_index=True).sort_values("time").reset_index(drop=True)
    df = add_xy(df)
    start_time = pd.to_datetime(df["time"].iloc[0])
    df["elapsed_s"] = (pd.to_datetime(df["time"]) - start_time).dt.total_seconds()
    df["cum_energy_j"] = pd.to_numeric(df["energy_j"], errors="coerce").fillna(0.0).cumsum()
    df["run_cumdist_m"] = pd.to_numeric(df["dist_m"], errors="coerce").fillna(0.0).cumsum()
    df = build_full_run_distance(df)

    baseline_time_s = float(pd.to_numeric(df["dt_s"], errors="coerce").fillna(0.0).sum())
    if args.strategy_time_budget_sec is not None:
        strategy_time_budget_s = float(args.strategy_time_budget_sec)
    elif args.strategy_lap_time_target_sec is not None:
        strategy_time_budget_s = float(args.strategy_lap_time_target_sec * args.laps)
    else:
        strategy_time_budget_s = baseline_time_s

    strategy_segments = build_strategy_segments(df, args.strategy_segments)
    strategy_model = fit_empirical_energy_model(df)
    strategy_profile = optimize_speed_profile(
        strategy_segments,
        strategy_model,
        time_budget_sec=strategy_time_budget_s,
        speed_min_kph=args.strategy_speed_min_kph,
        speed_max_kph=args.strategy_speed_max_kph,
        max_delta_kph_per_segment=args.strategy_max_delta_kph_per_segment,
        speed_step_kph=args.strategy_speed_step_kph,
    )
    aligned = build_strategy_samples(df, strategy_profile).reset_index(drop=True)
    strategy_report = build_strategy_report(df, strategy_profile, strategy_time_budget_s)
    df = df.reset_index(drop=True)
    df["segment"] = aligned["segment"]
    df["target_speed_kph"] = aligned["target_speed_kph"]
    df["strategy_action"] = aligned["strategy_action"]
    df["pred_power_w"] = aligned["pred_power_w"]
    df["pred_energy_j"] = aligned["pred_energy_j"]
    df["pred_cum_energy_j"] = aligned["pred_cum_energy_j"]
    return df, strategy_profile, strategy_report, strategy_time_budget_s


def finite_float(value: Any, digits: int = 3) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return round(result, digits)


def make_payload(
    df: pd.DataFrame,
    strategy_profile: pd.DataFrame,
    strategy_report: str,
    strategy_time_budget_s: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    samples = []
    for row in df.itertuples(index=False):
        samples.append({
            "t": finite_float(row.elapsed_s, 2),
            "lap": int(row.lap),
            "x": finite_float(row.x, 2),
            "y": finite_float(row.y, 2),
            "segment": int(row.segment),
            "current": finite_float(row.current_mA, 0),
            "speed": finite_float(row.speed_kph, 2),
            "targetSpeed": finite_float(row.target_speed_kph, 2),
            "gpsAccel": finite_float(row.gps_longitudinal_accel_abs_m_s2, 3),
            "imuAccel": finite_float(row.imu_forward_dynamic_m_s2, 3),
            "power": finite_float(row.power_w, 2),
            "runEnergyJ": finite_float(row.cum_energy_j, 2),
            "predRunEnergyJ": finite_float(row.pred_cum_energy_j, 2),
            "strategyAction": str(row.strategy_action),
        })

    metric_domains = {}
    for key, spec in METRICS.items():
        values = pd.to_numeric(df[spec["source"]], errors="coerce")
        if key == "speed":
            values = pd.concat(
                [values, pd.to_numeric(df["target_speed_kph"], errors="coerce")],
                ignore_index=True,
            )
        elif key == "runEnergyJ":
            values = pd.concat(
                [values, pd.to_numeric(df["pred_cum_energy_j"], errors="coerce")],
                ignore_index=True,
            )
        metric_domains[key] = domain(
            values,
            robust=key in ("gpsAccel", "imuAccel"),
            min_zero=key in ("gpsAccel", "runEnergyJ", "targetSpeed"),
        )

    x_domain = domain(df["x"])
    y_domain = domain(df["y"])
    lap_ranges = []
    for lap_num, group in df.groupby("lap"):
        lap_ranges.append({
            "lap": int(lap_num),
            "start": int(group.index.min()),
            "end": int(group.index.max()),
            "start_t": finite_float(group["elapsed_s"].iloc[0], 2),
            "end_t": finite_float(group["elapsed_s"].iloc[-1], 2),
        })

    accel_diagnostics = []
    for row in compute_accel_candidate_scores(df):
        accel_diagnostics.append({
            "candidate": row["candidate"],
            "gps_corr": finite_float(row["gps_corr"], 4),
            "current_corr": finite_float(row["current_corr"], 4),
            "power_corr": finite_float(row["power_corr"], 4),
        })

    baseline_energy_j = float(pd.to_numeric(df["energy_j"], errors="coerce").fillna(0.0).sum())
    predicted_energy_j = float(pd.to_numeric(strategy_profile["pred_energy_j"], errors="coerce").fillna(0.0).sum())
    delta_j = predicted_energy_j - baseline_energy_j
    delta_pct = (delta_j / baseline_energy_j * 100.0) if baseline_energy_j > 0 else 0.0
    strategy_segments = []
    for row in strategy_profile.itertuples(index=False):
        strategy_segments.append({
            "segment": int(row.segment),
            "dist_start_m": finite_float(row.dist_start_m, 2),
            "dist_end_m": finite_float(row.dist_end_m, 2),
            "target_speed_kph": finite_float(row.target_speed_kph, 2),
            "entry_speed_kph": finite_float(row.entry_speed_kph, 2),
            "speed_delta_kph": finite_float(row.speed_delta_kph, 2),
            "pred_energy_j": finite_float(row.pred_energy_j, 2),
            "action": str(row.action),
        })

    return {
        "meta": {
            "gps": args.gps,
            "telemetry": args.telemetry,
            "forward_axis": args.forward_axis,
            "sample_count": len(samples),
            "duration_s": finite_float(df["elapsed_s"].iloc[-1], 2),
            "accel": {
                "scale": args.accel_scale,
                "imu_axis": args.imu_axis,
                "imu_axis_sign": args.imu_axis_sign,
                "bias_window_s": args.accel_bias_window_sec,
                "smooth_window_s": args.accel_smooth_window_sec,
            },
        },
        "strategy": {
            "time_budget_s": finite_float(strategy_time_budget_s, 2),
            "baseline_energy_j": finite_float(baseline_energy_j, 2),
            "predicted_energy_j": finite_float(predicted_energy_j, 2),
            "delta_energy_j": finite_float(delta_j, 2),
            "delta_energy_pct": finite_float(delta_pct, 2),
            "segments": strategy_segments,
            "report": strategy_report,
        },
        "accel_diagnostics": accel_diagnostics,
        "samples": samples,
        "laps": lap_ranges,
        "domains": {
            "x": x_domain,
            "y": y_domain,
            "metrics": metric_domains,
        },
        "metrics": {
            key: {
                "label": spec["label"],
                "unit": spec["unit"],
                "field": spec["field"],
                "color": spec["color"],
                "map_selectable": spec.get("map_selectable", True),
            }
            for key, spec in METRICS.items()
        },
    }


def domain(
    values: pd.Series,
    pad_fraction: float = 0.06,
    robust: bool = False,
    min_zero: bool = False,
) -> list[float]:
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return [0.0, 1.0]
    if robust and len(finite) >= 20:
        lo = float(finite.quantile(0.02))
        hi = float(finite.quantile(0.98))
    else:
        lo = float(finite.min())
        hi = float(finite.max())
    if math.isclose(lo, hi):
        pad = max(abs(lo) * 0.05, 1.0)
        lower = 0.0 if min_zero else lo - pad
        return [round(lower, 3), round(hi + pad, 3)]
    pad = (hi - lo) * pad_fraction
    lower = 0.0 if min_zero else lo - pad
    return [round(lower, 3), round(hi + pad, 3)]


def build_html(payload: dict[str, Any]) -> str:
    data_json = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>UTSM Afternoon Telemetry Dashboard</title>
  <style>
    :root {{
      font-family: Arial, Helvetica, sans-serif;
      background: #f5f6f8;
      color: #111827;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; }}
    header {{
      position: sticky;
      top: 0;
      z-index: 20;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 12px 18px;
      background: rgba(255, 255, 255, 0.96);
      border-bottom: 1px solid #d7dce3;
    }}
    h1 {{
      margin: 0;
      font-size: 18px;
      line-height: 1.2;
    }}
    .meta {{
      font-size: 12px;
      color: #59636e;
      white-space: nowrap;
    }}
    main {{
      max-width: 1420px;
      margin: 0 auto;
      padding: 18px;
    }}
    .dashboard {{
      display: grid;
      grid-template-columns: minmax(520px, 1.08fr) minmax(480px, 0.92fr);
      gap: 16px;
      align-items: start;
    }}
    .panel {{
      background: #fff;
      border: 1px solid #d8dde5;
      border-radius: 8px;
      padding: 14px;
    }}
    .map-wrap {{
      display: grid;
      grid-template-columns: 1fr 56px;
      gap: 10px;
      align-items: stretch;
    }}
    svg {{
      display: block;
      width: 100%;
      height: auto;
      background: #fff;
    }}
    #trackSvg {{
      aspect-ratio: 1 / 1.08;
      border: 1px solid #d8dde5;
    }}
    .charts {{
      display: grid;
      grid-template-rows: repeat(6, minmax(100px, 1fr));
      gap: 10px;
    }}
    .chart {{
      border: 1px solid #d8dde5;
    }}
    .controls {{
      display: grid;
      grid-template-columns: auto minmax(240px, 1fr) auto auto auto auto auto;
      gap: 12px;
      align-items: center;
      margin-top: 14px;
      padding-top: 12px;
      border-top: 1px solid #e5e9ef;
    }}
    button, select, input[type="number"] {{
      height: 34px;
      border: 1px solid #b9c1cb;
      border-radius: 6px;
      background: #fff;
      color: #111827;
      font: inherit;
      padding: 0 10px;
    }}
    button {{ cursor: pointer; }}
    input[type="range"] {{ width: 100%; }}
    label {{
      font-size: 13px;
      color: #334155;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      white-space: nowrap;
    }}
    .readout {{
      display: grid;
      grid-template-columns: repeat(6, 1fr);
      gap: 8px;
      margin-top: 12px;
    }}
    .stat {{
      border: 1px solid #e3e7ed;
      border-radius: 6px;
      padding: 8px;
      background: #fafbfc;
    }}
    .stat span {{
      display: block;
      font-size: 11px;
      color: #64748b;
      margin-bottom: 3px;
    }}
    .stat strong {{ font-size: 16px; }}
    .legend {{
      border: 1px solid #d8dde5;
      display: grid;
      grid-template-rows: 1fr auto auto;
      align-items: stretch;
      overflow: hidden;
    }}
    .ramp {{
      background: linear-gradient(to top, #2d0a73, #7e03a8, #cc4778, #f89540, #f0f921);
    }}
    .legend-label {{
      font-size: 11px;
      color: #475569;
      text-align: center;
      padding: 4px 2px;
    }}
    .axis-label {{
      fill: #334155;
      font-size: 12px;
    }}
    .tick-label {{
      fill: #64748b;
      font-size: 10px;
    }}
    @media (max-width: 1050px) {{
      .dashboard {{ grid-template-columns: 1fr; }}
      .controls {{ grid-template-columns: 1fr 1fr; }}
      .readout {{ grid-template-columns: repeat(2, 1fr); }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>UTSM Afternoon Telemetry Dashboard</h1>
    <div class="meta" id="metaText"></div>
  </header>
  <main>
    <section class="dashboard">
      <div class="panel">
        <div class="map-wrap">
          <svg id="trackSvg" viewBox="0 0 720 780" role="img" aria-label="Track map">
            <g id="mapGrid"></g>
            <path id="fullTrack" fill="none" stroke="#d0d5dc" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"></path>
            <g id="strategyLayer"></g>
            <g id="trailLayer"></g>
            <g id="lapBoundaryLayer"></g>
            <circle id="lapStartMarker" r="7" fill="#3ecf59" stroke="#ffffff" stroke-width="2"></circle>
            <circle id="carMarker" r="8" fill="#ffffff" stroke="#111827" stroke-width="3"></circle>
            <text id="mapReadout" x="16" y="744" class="axis-label"></text>
          </svg>
          <div class="legend">
            <div class="ramp"></div>
            <div class="legend-label" id="legendMax"></div>
            <div class="legend-label" id="legendMin"></div>
          </div>
        </div>
        <div class="controls">
          <button id="playButton" type="button">Play</button>
          <input id="timeSlider" type="range" min="0" max="0" value="0" step="0.1">
          <label>Metric
            <select id="metricSelect"></select>
          </label>
          <label>
            <input id="showStrategy" type="checkbox" checked> Strategy
          </label>
          <label>Speed
            <select id="playSpeed">
              <option value="0.5">0.5x</option>
              <option value="1" selected>1x</option>
              <option value="2">2x</option>
              <option value="4">4x</option>
              <option value="8">8x</option>
            </select>
          </label>
          <label>
            <input id="showBoundaries" type="checkbox"> Boundaries
          </label>
          <span id="timeText" class="meta"></span>
        </div>
        <div class="readout">
          <div class="stat"><span>Lap</span><strong id="lapValue"></strong></div>
          <div class="stat"><span>Strategy</span><strong id="strategyValue"></strong></div>
          <div class="stat"><span>Speed</span><strong id="speedValue"></strong></div>
          <div class="stat"><span>Current</span><strong id="currentValue"></strong></div>
          <div class="stat"><span>GPS accel</span><strong id="gpsAccelValue"></strong></div>
          <div class="stat"><span>MPU accel</span><strong id="imuAccelValue"></strong></div>
        </div>
      </div>
      <div class="panel charts" id="charts"></div>
    </section>
  </main>
  <script>
    const DATA = {data_json};

    const W = 720;
    const H = 780;
    const PAD = 54;
    const CHART_W = 620;
    const CHART_H = 132;
    const CHART_PAD = {{ left: 58, right: 12, top: 14, bottom: 28 }};
    const samples = DATA.samples;
    const metricSpecs = DATA.metrics;
    const metricKeys = Object.keys(metricSpecs).filter(
      key => metricSpecs[key].map_selectable !== false
    );
    const chartKeys = ["current", "speed", "gpsAccel", "imuAccel", "power", "runEnergyJ"];
    const strategyChartFields = {{
      speed: {{ field: "targetSpeed", color: "#f97316" }},
      runEnergyJ: {{ field: "predRunEnergyJ", color: "#f97316" }}
    }};
    const state = {{
      index: 0,
      metric: "gpsAccel",
      playing: false,
      lastTick: null
    }};

    const el = {{
      metaText: document.getElementById("metaText"),
      trackSvg: document.getElementById("trackSvg"),
      fullTrack: document.getElementById("fullTrack"),
      strategyLayer: document.getElementById("strategyLayer"),
      trailLayer: document.getElementById("trailLayer"),
      lapBoundaryLayer: document.getElementById("lapBoundaryLayer"),
      lapStartMarker: document.getElementById("lapStartMarker"),
      carMarker: document.getElementById("carMarker"),
      mapReadout: document.getElementById("mapReadout"),
      timeSlider: document.getElementById("timeSlider"),
      metricSelect: document.getElementById("metricSelect"),
      showStrategy: document.getElementById("showStrategy"),
      playButton: document.getElementById("playButton"),
      playSpeed: document.getElementById("playSpeed"),
      showBoundaries: document.getElementById("showBoundaries"),
      timeText: document.getElementById("timeText"),
      lapValue: document.getElementById("lapValue"),
      strategyValue: document.getElementById("strategyValue"),
      speedValue: document.getElementById("speedValue"),
      currentValue: document.getElementById("currentValue"),
      gpsAccelValue: document.getElementById("gpsAccelValue"),
      imuAccelValue: document.getElementById("imuAccelValue"),
      legendMin: document.getElementById("legendMin"),
      legendMax: document.getElementById("legendMax"),
      charts: document.getElementById("charts")
    }};

    function clamp(v, lo, hi) {{
      return Math.max(lo, Math.min(hi, v));
    }}

    function scaleLinear(value, domain, range) {{
      const [d0, d1] = domain;
      const [r0, r1] = range;
      if (d0 === d1) return (r0 + r1) / 2;
      return r0 + ((value - d0) / (d1 - d0)) * (r1 - r0);
    }}

    function xMap(x) {{
      return scaleLinear(x, DATA.domains.x, [PAD, W - PAD]);
    }}

    function yMap(y) {{
      return scaleLinear(y, DATA.domains.y, [H - PAD, PAD]);
    }}

    function chartX(t) {{
      return scaleLinear(t, [0, DATA.meta.duration_s], [CHART_PAD.left, CHART_W - CHART_PAD.right]);
    }}

    function chartY(value, key) {{
      return scaleLinear(value, DATA.domains.metrics[key], [CHART_H - CHART_PAD.bottom, CHART_PAD.top]);
    }}

    function linePath(rows, xFn, yFn) {{
      let d = "";
      for (const row of rows) {{
        const x = xFn(row);
        const y = yFn(row);
        if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
        d += d ? `L${{x.toFixed(1)}} ${{y.toFixed(1)}}` : `M${{x.toFixed(1)}} ${{y.toFixed(1)}}`;
      }}
      return d;
    }}

    function nearestIndexByTime(t) {{
      let lo = 0;
      let hi = samples.length - 1;
      while (lo < hi) {{
        const mid = Math.floor((lo + hi) / 2);
        if (samples[mid].t < t) lo = mid + 1;
        else hi = mid;
      }}
      if (lo > 0 && Math.abs(samples[lo - 1].t - t) < Math.abs(samples[lo].t - t)) {{
        return lo - 1;
      }}
      return lo;
    }}

    function currentLapRange(index) {{
      const lap = samples[index].lap;
      return DATA.laps.find(item => item.lap === lap);
    }}

    function colorForMetric(value, key) {{
      const [lo, hi] = DATA.domains.metrics[key];
      const ratio = clamp((value - lo) / (hi - lo || 1), 0, 1);
      const stops = [
        [45, 10, 115],
        [126, 3, 168],
        [204, 71, 120],
        [248, 149, 64],
        [240, 249, 33]
      ];
      const scaled = ratio * (stops.length - 1);
      const idx = Math.min(Math.floor(scaled), stops.length - 2);
      const local = scaled - idx;
      const a = stops[idx];
      const b = stops[idx + 1];
      const rgb = a.map((v, i) => Math.round(v + (b[i] - v) * local));
      return `rgb(${{rgb[0]}},${{rgb[1]}},${{rgb[2]}})`;
    }}

    function drawMapGrid() {{
      const grid = document.getElementById("mapGrid");
      const lines = [];
      for (let i = 0; i <= 6; i++) {{
        const x = PAD + ((W - 2 * PAD) * i) / 6;
        const y = PAD + ((H - 2 * PAD) * i) / 6;
        lines.push(`<line x1="${{x}}" x2="${{x}}" y1="${{PAD}}" y2="${{H - PAD}}" stroke="#eef1f4" stroke-width="1"/>`);
        lines.push(`<line x1="${{PAD}}" x2="${{W - PAD}}" y1="${{y}}" y2="${{y}}" stroke="#eef1f4" stroke-width="1"/>`);
      }}
      grid.innerHTML = lines.join("");
    }}

    function drawFullTrack() {{
      el.fullTrack.setAttribute("d", linePath(samples, s => xMap(s.x), s => yMap(s.y)));
    }}

    function drawBoundaries() {{
      const visible = el.showBoundaries.checked;
      if (!visible) {{
        el.lapBoundaryLayer.innerHTML = "";
        return;
      }}
      el.lapBoundaryLayer.innerHTML = DATA.laps.map(lap => {{
        const s = samples[lap.start];
        return `<g><circle cx="${{xMap(s.x)}}" cy="${{yMap(s.y)}}" r="5" fill="#f59e0b" stroke="#fff" stroke-width="2"></circle><text x="${{xMap(s.x) + 8}}" y="${{yMap(s.y) - 8}}" class="tick-label">L${{lap.lap}}</text></g>`;
      }}).join("");
    }}

    function drawStrategyLayer(index) {{
      if (!el.showStrategy.checked) {{
        el.strategyLayer.innerHTML = "";
        return;
      }}
      const range = currentLapRange(index);
      let markup = "";
      for (let i = range.start + 1; i <= range.end; i++) {{
        const a = samples[i - 1];
        const b = samples[i];
        const color = colorForMetric(b.targetSpeed, "targetSpeed");
        markup += `<line x1="${{xMap(a.x).toFixed(1)}}" y1="${{yMap(a.y).toFixed(1)}}" x2="${{xMap(b.x).toFixed(1)}}" y2="${{yMap(b.y).toFixed(1)}}" stroke="${{color}}" stroke-width="3" stroke-dasharray="7 6" stroke-linecap="round" opacity="0.9"/>`;
      }}
      el.strategyLayer.innerHTML = markup;
    }}

    function drawTrail(index) {{
      const range = currentLapRange(index);
      const start = range.start;
      const end = index;
      if (end <= start) {{
        el.trailLayer.innerHTML = "";
        return;
      }}
      const key = state.metric;
      let markup = "";
      for (let i = start + 1; i <= end; i++) {{
        const a = samples[i - 1];
        const b = samples[i];
        const value = b[metricSpecs[key].field];
        const color = colorForMetric(value, key);
        markup += `<line x1="${{xMap(a.x).toFixed(1)}}" y1="${{yMap(a.y).toFixed(1)}}" x2="${{xMap(b.x).toFixed(1)}}" y2="${{yMap(b.y).toFixed(1)}}" stroke="${{color}}" stroke-width="5" stroke-linecap="round"/>`;
      }}
      el.trailLayer.innerHTML = markup;
    }}

    function makeChart(key) {{
      const spec = metricSpecs[key];
      const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      svg.setAttribute("class", "chart");
      svg.setAttribute("viewBox", `0 0 ${{CHART_W}} ${{CHART_H}}`);
      svg.dataset.key = key;
      svg.innerHTML = `
        <text x="8" y="18" class="axis-label">${{spec.label}} (${{spec.unit}})</text>
        <path class="full-line" fill="none" stroke="#d2d7de" stroke-width="2"></path>
        <path class="progress-line" fill="none" stroke="${{spec.color}}" stroke-width="2.4"></path>
        <path class="strategy-full-line" fill="none" stroke="#fdba74" stroke-width="1.8" stroke-dasharray="6 5"></path>
        <path class="strategy-progress-line" fill="none" stroke="#f97316" stroke-width="2.4" stroke-dasharray="6 5"></path>
        <line class="cursor-line" y1="${{CHART_PAD.top}}" y2="${{CHART_H - CHART_PAD.bottom}}" stroke="#111827" stroke-width="1"></line>
        <text class="tick-label min-label" x="${{CHART_PAD.left}}" y="${{CHART_H - 8}}"></text>
        <text class="tick-label max-label" x="${{CHART_PAD.left}}" y="11"></text>
      `;
      svg.querySelector(".full-line").setAttribute(
        "d",
        linePath(samples, s => chartX(s.t), s => chartY(s[spec.field], key))
      );
      const [lo, hi] = DATA.domains.metrics[key];
      svg.querySelector(".min-label").textContent = lo.toFixed(1);
      svg.querySelector(".max-label").textContent = hi.toFixed(1);
      const strategySpec = strategyChartFields[key];
      if (strategySpec) {{
        svg.querySelector(".strategy-full-line").setAttribute(
          "d",
          linePath(samples, s => chartX(s.t), s => chartY(s[strategySpec.field], key))
        );
      }} else {{
        svg.querySelector(".strategy-full-line").setAttribute("d", "");
        svg.querySelector(".strategy-progress-line").setAttribute("d", "");
      }}
      el.charts.appendChild(svg);
    }}

    function updateCharts(index) {{
      const row = samples[index];
      const lapRange = currentLapRange(index);
      const lapRows = samples.slice(lapRange.start, index + 1);
      document.querySelectorAll(".chart").forEach(svg => {{
        const key = svg.dataset.key;
        const spec = metricSpecs[key];
        const rows = key === "runEnergyJ" ? samples.slice(0, index + 1) : lapRows;
        svg.querySelector(".progress-line").setAttribute(
          "d",
          linePath(rows, s => chartX(s.t), s => chartY(s[spec.field], key))
        );
        const strategySpec = strategyChartFields[key];
        const strategyLine = svg.querySelector(".strategy-progress-line");
        const strategyFull = svg.querySelector(".strategy-full-line");
        if (strategySpec && el.showStrategy.checked) {{
          strategyLine.setAttribute(
            "d",
            linePath(rows, s => chartX(s.t), s => chartY(s[strategySpec.field], key))
          );
          strategyLine.style.display = "";
          strategyFull.style.display = "";
        }} else {{
          strategyLine.setAttribute("d", "");
          strategyLine.style.display = "none";
          strategyFull.style.display = strategySpec ? "none" : "";
        }}
        const x = chartX(row.t);
        const cursor = svg.querySelector(".cursor-line");
        cursor.setAttribute("x1", x);
        cursor.setAttribute("x2", x);
      }});
    }}

    function format(value, digits, suffix) {{
      if (!Number.isFinite(value)) return "-";
      return `${{value.toFixed(digits)}}${{suffix}}`;
    }}

    function update(index) {{
      state.index = clamp(index, 0, samples.length - 1);
      const row = samples[state.index];
      const range = currentLapRange(state.index);
      const lapStart = samples[range.start];
      el.timeSlider.value = row.t;
      el.carMarker.setAttribute("cx", xMap(row.x));
      el.carMarker.setAttribute("cy", yMap(row.y));
      el.lapStartMarker.setAttribute("cx", xMap(lapStart.x));
      el.lapStartMarker.setAttribute("cy", yMap(lapStart.y));
      drawStrategyLayer(state.index);
      drawTrail(state.index);
      updateCharts(state.index);
      drawBoundaries();
      el.timeText.textContent = `t=${{row.t.toFixed(1)}}s / ${{DATA.meta.duration_s.toFixed(1)}}s`;
      el.mapReadout.textContent = `t=${{row.t.toFixed(1)}}s  lap=${{row.lap}}  seg=${{row.segment}}  speed=${{row.speed.toFixed(1)}} km/h  target=${{row.targetSpeed.toFixed(1)}} km/h  action=${{row.strategyAction}}  current=${{row.current.toFixed(0)}} mA`;
      el.lapValue.textContent = String(row.lap);
      el.strategyValue.textContent = `${{row.strategyAction}} @ ${{row.targetSpeed.toFixed(1)}} km/h`;
      el.speedValue.textContent = format(row.speed, 1, " km/h");
      el.currentValue.textContent = format(row.current, 0, " mA");
      el.gpsAccelValue.textContent = format(row.gpsAccel, 2, " m/s^2");
      el.imuAccelValue.textContent = format(row.imuAccel, 2, " m/s^2");
      const domain = DATA.domains.metrics[state.metric];
      el.legendMin.textContent = `${{domain[0].toFixed(1)}}`;
      el.legendMax.textContent = `${{domain[1].toFixed(1)}}`;
    }}

    function animationTick(now) {{
      if (!state.playing) return;
      if (state.lastTick === null) state.lastTick = now;
      const delta = (now - state.lastTick) / 1000;
      state.lastTick = now;
      const speed = Number(el.playSpeed.value);
      const nextT = Math.min(DATA.meta.duration_s, samples[state.index].t + delta * speed * 12);
      update(nearestIndexByTime(nextT));
      if (nextT >= DATA.meta.duration_s) {{
        state.playing = false;
        el.playButton.textContent = "Play";
        state.lastTick = null;
        return;
      }}
      requestAnimationFrame(animationTick);
    }}

    function init() {{
      const accelMeta = DATA.meta.accel;
      el.metaText.textContent = `${{DATA.meta.sample_count}} samples | ${{DATA.meta.duration_s.toFixed(1)}}s | strategy ${{DATA.strategy.delta_energy_pct.toFixed(2)}}% energy | MPU ${{accelMeta.imu_axis}} x ${{accelMeta.imu_axis_sign}} | scale ${{accelMeta.scale}}`;
      el.timeSlider.max = DATA.meta.duration_s;
      for (const key of metricKeys) {{
        const option = document.createElement("option");
        option.value = key;
        option.textContent = metricSpecs[key].label;
        el.metricSelect.appendChild(option);
      }}
      el.metricSelect.value = state.metric;
      drawMapGrid();
      drawFullTrack();
      chartKeys.forEach(makeChart);
      el.timeSlider.addEventListener("input", event => update(nearestIndexByTime(Number(event.target.value))));
      el.metricSelect.addEventListener("change", event => {{
        state.metric = event.target.value;
        update(state.index);
      }});
      el.showStrategy.addEventListener("change", () => update(state.index));
      el.showBoundaries.addEventListener("change", () => drawBoundaries());
      el.playButton.addEventListener("click", () => {{
        state.playing = !state.playing;
        el.playButton.textContent = state.playing ? "Pause" : "Play";
        state.lastTick = null;
        if (state.playing) requestAnimationFrame(animationTick);
      }});
      update(0);
    }}

    init();
  </script>
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    df, strategy_profile, strategy_report, strategy_time_budget_s = load_derived_run(args)
    payload = make_payload(df, strategy_profile, strategy_report, strategy_time_budget_s, args)
    html = build_html(payload)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"Wrote interactive dashboard: {args.output}")
    print(f"Samples: {payload['meta']['sample_count']}")
    print(f"Duration: {payload['meta']['duration_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
