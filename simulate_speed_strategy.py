"""Empirical speed-profile optimizer for UTSM telemetry runs."""

from __future__ import annotations

import argparse
import os
import sys

try:
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
    build_full_run_distance,
    build_strategy_report,
    build_strategy_samples,
    build_strategy_segments,
    derive_motion_energy,
    fit_empirical_energy_model,
    merge_by_time,
    optimize_speed_profile,
    read_gpx,
    read_telemetry,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize a target speed profile from historical telemetry data."
    )
    parser.add_argument("gps", help="Path to the GPX track file")
    parser.add_argument("telemetry", help="Path to the telemetry CSV file")
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
    parser.add_argument("--segments", type=int, default=24)
    parser.add_argument("--time-budget-sec", type=float)
    parser.add_argument("--lap-time-target-sec", type=float)
    parser.add_argument("--speed-min-kph", type=float, default=8.0)
    parser.add_argument("--speed-max-kph", type=float, default=40.0)
    parser.add_argument("--max-delta-kph-per-segment", type=float, default=6.0)
    parser.add_argument("--speed-step-kph", type=float, default=1.0)
    parser.add_argument("--output-prefix", default="outputs/speed_strategy")
    return parser.parse_args()


def load_full_run(args: argparse.Namespace) -> pd.DataFrame:
    gps_df = read_gpx(args.gps)
    telem_df = read_telemetry(args.telemetry)
    gps_laps, telem_laps, _ = build_laps(gps_df, telem_df, args)

    rows = []
    distance_offset = 0.0
    energy_offset = 0.0
    for lap_num, (lap_gps, lap_telem) in enumerate(zip(gps_laps, telem_laps), start=1):
        if lap_gps.empty or lap_telem.empty:
            continue
        try:
            merged = merge_by_time(lap_telem, lap_gps, args.tolerance_sec)
        except ValueError as exc:
            print(f"Lap {lap_num}: skipping - {exc}")
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
        derived["run_cumdist_m"] = derived["cumdist_m"] + distance_offset
        derived["cum_energy_j"] = derived["cum_energy_j"] + energy_offset
        distance_offset = float(derived["run_cumdist_m"].iloc[-1])
        energy_offset = float(derived["cum_energy_j"].iloc[-1])
        rows.append(derived)

    if not rows:
        raise ValueError("No lap data could be merged for simulation.")
    return build_full_run_distance(
        pd.concat(rows, ignore_index=True).sort_values("time").reset_index(drop=True)
    )


def main() -> int:
    args = parse_args()
    out_dir = os.path.dirname(args.output_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Reading GPX: {args.gps}")
    print(f"Reading telemetry: {args.telemetry}")
    full_run = load_full_run(args)
    model = fit_empirical_energy_model(full_run)
    segments_df = build_strategy_segments(full_run, args.segments)

    baseline_time_s = float(pd.to_numeric(full_run["dt_s"], errors="coerce").fillna(0.0).sum())
    if args.time_budget_sec is not None:
        time_budget_sec = args.time_budget_sec
    elif args.lap_time_target_sec is not None and args.laps > 0:
        time_budget_sec = args.lap_time_target_sec * args.laps
    else:
        time_budget_sec = baseline_time_s

    profile_df = optimize_speed_profile(
        segments_df,
        model,
        time_budget_sec=time_budget_sec,
        speed_min_kph=args.speed_min_kph,
        speed_max_kph=args.speed_max_kph,
        max_delta_kph_per_segment=args.max_delta_kph_per_segment,
        speed_step_kph=args.speed_step_kph,
    )
    samples_df = build_strategy_samples(full_run, profile_df)
    report = build_strategy_report(full_run, profile_df, time_budget_sec)

    profile_csv = args.output_prefix + "_strategy_profile.csv"
    samples_csv = args.output_prefix + "_strategy_samples.csv"
    report_txt = args.output_prefix + "_strategy_report.txt"

    profile_df.to_csv(profile_csv, index=False)
    samples_df.to_csv(samples_csv, index=False)
    with open(report_txt, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")

    print(f"Wrote: {profile_csv}, {samples_csv}, {report_txt}")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
