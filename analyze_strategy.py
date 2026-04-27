"""Strategy and efficiency analysis for UTSM Shell Eco-Marathon telemetry runs.

Usage
-----
python analyze_strategy.py Utsm.gpx telemetry_20260411_112302.csv \\
    --laps 4 --segments 12 --split-method start --output-prefix outputs/run1_strategy

This script imports all heavy-lifting helpers from utsm_telemetry.core so
there is no duplicated alignment or lap-splitting code.
"""

from __future__ import annotations

import argparse
import os
import sys

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "Missing required package. Install dependencies with: pip install matplotlib pandas numpy"
    ) from exc

# ---------------------------------------------------------------------------
# Import shared helpers
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from utsm_telemetry import (
    FORWARD_AXIS_CHOICES,
    read_gpx,
    read_telemetry,
    align_telemetry,
    find_start_spike,
    find_nearest_gps_index,
    find_lap_boundaries_by_start_gate,
    split_gps_into_laps,
    merge_by_time,
    derive_motion_energy,
    parse_lap_time,
    parse_iso8601,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate lap, sector, and flat-speed efficiency reports from a UTSM run."
    )
    parser.add_argument("gps", help="Path to the GPX track file")
    parser.add_argument("telemetry", help="Path to the telemetry CSV file")
    parser.add_argument(
        "--laps", type=int, default=4,
        help="Number of laps expected in the run",
    )
    parser.add_argument(
        "--segments", type=int, default=12,
        help="Number of equal-distance sectors per lap",
    )
    parser.add_argument(
        "--split-method",
        choices=["points", "time", "line", "start"],
        default="start",
    )
    parser.add_argument(
        "--lap-times", nargs="+", metavar="ELAPSED",
        help="Elapsed MM:SS or H:MM:SS times for each lap start",
    )
    parser.add_argument(
        "--start-time",
        help="ISO 8601 timestamp to force-align telemetry start",
    )
    parser.add_argument(
        "--time-offset-ms", type=float, default=0.0,
        help="Millisecond offset added to telemetry timestamps after alignment",
    )
    parser.add_argument(
        "--tolerance-sec", type=float, default=1.5,
        help="Max seconds tolerance for GPS/telemetry time merge",
    )
    parser.add_argument(
        "--forward-axis",
        choices=FORWARD_AXIS_CHOICES,
        default="ax",
        help=(
            "Legacy IMU axis alias retained for older scripts."
        ),
    )
    parser.add_argument(
        "--accel-window",
        type=int,
        default=5,
        help="Legacy acceleration smoothing sample count retained for older scripts",
    )
    parser.add_argument(
        "--accel-scale",
        type=float,
        default=1000.0,
        help="Raw MPU accelerometer units per g. Afternoon data behaves like 1000.",
    )
    parser.add_argument(
        "--imu-axis",
        choices=["ax", "ay", "az"],
        default="ax",
        help="MPU axis used for the dashboard's selected dynamic acceleration channel.",
    )
    parser.add_argument(
        "--imu-axis-sign",
        type=int,
        choices=[-1, 1],
        default=1,
        help="Sign applied to the selected MPU dynamic acceleration channel.",
    )
    parser.add_argument(
        "--accel-bias-window-sec",
        type=float,
        default=30.0,
        help="Centered rolling median window used to estimate slow MPU gravity/bias.",
    )
    parser.add_argument(
        "--accel-smooth-window-sec",
        type=float,
        default=3.0,
        help="Centered rolling median window used to smooth acceleration channels.",
    )
    parser.add_argument(
        "--output-prefix", default="outputs/strategy",
        help="Prefix for output files (e.g. outputs/run1_strategy)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Build laps (handles all split modes)
# ---------------------------------------------------------------------------

def build_laps(
    gps_df: pd.DataFrame,
    telem_df: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], pd.DataFrame]:
    """Return (list_of_gps_lap_dfs, list_of_telem_lap_dfs, aligned_telem_df).

    Mirrors the lap-building logic in gps_current_heatmap.py but returns
    both sides of the merge so downstream functions can re-merge with
    full energy channels.
    """
    if args.lap_times:
        track_start = gps_df["time"].iloc[0]
        lap_timestamps = [parse_lap_time(t, track_start) for t in args.lap_times]
        if len(lap_timestamps) < 2:
            raise ValueError("--lap-times requires at least 2 timestamps.")
        spike_idx = find_start_spike(telem_df)
        spike_ms = float(telem_df.loc[spike_idx, "timestamp_ms"])
        telemetry_start = lap_timestamps[0] - pd.Timedelta(milliseconds=spike_ms)
        telem_df = align_telemetry(telem_df, gps_df, telemetry_start, args.time_offset_ms)

        gps_laps, telem_laps = [], []
        for i in range(len(lap_timestamps) - 1):
            ls, le = lap_timestamps[i], lap_timestamps[i + 1]
            gps_laps.append(
                gps_df[(gps_df["time"] >= ls) & (gps_df["time"] < le)]
                .copy().reset_index(drop=True)
            )
            telem_laps.append(
                telem_df[(telem_df["time"] >= ls) & (telem_df["time"] < le)]
                .copy().reset_index(drop=True)
            )
        return gps_laps, telem_laps, telem_df

    telem_df = align_telemetry(telem_df, gps_df, args.start_time, args.time_offset_ms)

    if args.split_method == "start":
        spike_idx = find_start_spike(telem_df)
        start_time = telem_df.loc[spike_idx, "time"]
        gps_start_idx = find_nearest_gps_index(gps_df, start_time)
        print(
            f"Start spike at telemetry index {spike_idx}, time {start_time}, "
            f"matching GPS index {gps_start_idx}."
        )
        gps_df = gps_df.loc[gps_start_idx:].reset_index(drop=True)
        boundaries = find_lap_boundaries_by_start_gate(gps_df, 0, args.laps)
        if len(boundaries) < args.laps + 1:
            print(
                f"Warning: only found {len(boundaries) - 1} complete laps "
                f"(wanted {args.laps}). Last segment will be appended."
            )
        gps_laps = []
        for i in range(min(len(boundaries) - 1, args.laps)):
            gps_laps.append(
                gps_df.iloc[boundaries[i]: boundaries[i + 1]].reset_index(drop=True)
            )
        if len(gps_laps) < args.laps and boundaries:
            gps_laps.append(gps_df.iloc[boundaries[-1]:].reset_index(drop=True))
    else:
        gps_laps = split_gps_into_laps(gps_df, args.laps, args.split_method)

    telem_laps = []
    for lap_gps in gps_laps:
        if lap_gps.empty:
            telem_laps.append(pd.DataFrame())
            continue
        ls = lap_gps["time"].iloc[0]
        le = lap_gps["time"].iloc[-1]
        telem_laps.append(
            telem_df[
                (telem_df["time"] >= ls - pd.Timedelta(seconds=args.tolerance_sec))
                & (telem_df["time"] <= le + pd.Timedelta(seconds=args.tolerance_sec))
            ].copy().reset_index(drop=True)
        )

    return gps_laps, telem_laps, telem_df


# ---------------------------------------------------------------------------
# Summary builders
# ---------------------------------------------------------------------------

def build_lap_summary(lap_merged: pd.DataFrame, lap_num: int) -> dict:
    """Build one-row lap summary dict from a merged+derived lap DataFrame."""
    dur = (lap_merged["time"].iloc[-1] - lap_merged["time"].iloc[0]).total_seconds()
    dist = float(lap_merged["cumdist_m"].iloc[-1]) if "cumdist_m" in lap_merged.columns else 0.0
    avg_speed = dist / dur if dur > 0 else 0.0
    total_energy = float(lap_merged["energy_wh"].sum())
    efficiency = (total_energy / (dist / 1000.0)) if dist > 0 else float("nan")

    elev_pos = float(lap_merged["elev_diff_m"].clip(lower=0).sum())
    elev_neg = float(lap_merged["elev_diff_m"].clip(upper=0).sum())

    return {
        "lap": lap_num,
        "duration_s": dur,
        "distance_m": dist,
        "avg_speed_kph": avg_speed * 3.6,
        "avg_current_mA": float(lap_merged["current_mA"].mean()),
        "max_current_mA": float(lap_merged["current_mA"].max()),
        "avg_power_w": float(lap_merged["power_w"].mean()),
        "total_energy_wh": total_energy,
        "efficiency_wh_per_km": efficiency,
        "avg_gps_accel_m_s2": float(
            lap_merged["gps_longitudinal_accel_m_s2"].mean()
        ),
        "max_gps_accel_m_s2": float(
            lap_merged["gps_longitudinal_accel_m_s2"].max()
        ),
        "min_gps_accel_m_s2": float(
            lap_merged["gps_longitudinal_accel_m_s2"].min()
        ),
        "avg_imu_dynamic_accel_m_s2": float(
            lap_merged["imu_forward_dynamic_m_s2"].mean()
        ),
        "max_imu_dynamic_accel_m_s2": float(
            lap_merged["imu_forward_dynamic_m_s2"].max()
        ),
        "min_imu_dynamic_accel_m_s2": float(
            lap_merged["imu_forward_dynamic_m_s2"].min()
        ),
        "max_jerk_m_s3": float(lap_merged["jerk_m_s3"].max()),
        "min_jerk_m_s3": float(lap_merged["jerk_m_s3"].min()),
        "elev_gain_m": elev_pos,
        "elev_loss_m": abs(elev_neg),
    }


def build_sector_summary(lap_merged: pd.DataFrame, lap_num: int, n_segments: int) -> list[dict]:
    """Split lap into equal-distance sectors and summarise each."""
    total_dist = float(lap_merged["cumdist_m"].iloc[-1])
    if total_dist == 0:
        return []

    sector_length = total_dist / n_segments
    rows = []
    for seg_idx in range(n_segments):
        d_lo = seg_idx * sector_length
        d_hi = (seg_idx + 1) * sector_length
        seg = lap_merged[
            (lap_merged["cumdist_m"] >= d_lo) & (lap_merged["cumdist_m"] < d_hi)
        ]
        if seg.empty:
            continue
        dur = float(seg["dt_s"].sum())
        dist = float(seg["dist_m"].sum())
        energy = float(seg["energy_wh"].sum())
        eff = (energy / (dist / 1000.0)) if dist > 0 else float("nan")
        avg_speed = (dist / dur * 3.6) if dur > 0 else 0.0
        rows.append({
            "lap": lap_num,
            "sector": seg_idx + 1,
            "duration_s": dur,
            "distance_m": dist,
            "avg_speed_kph": avg_speed,
            "avg_power_w": float(seg["power_w"].mean()),
            "avg_current_mA": float(seg["current_mA"].mean()),
            "max_current_mA": float(seg["current_mA"].max()),
            "avg_gps_accel_m_s2": float(
                seg["gps_longitudinal_accel_m_s2"].mean()
            ),
            "max_gps_accel_m_s2": float(
                seg["gps_longitudinal_accel_m_s2"].max()
            ),
            "min_gps_accel_m_s2": float(
                seg["gps_longitudinal_accel_m_s2"].min()
            ),
            "avg_imu_dynamic_accel_m_s2": float(
                seg["imu_forward_dynamic_m_s2"].mean()
            ),
            "max_imu_dynamic_accel_m_s2": float(
                seg["imu_forward_dynamic_m_s2"].max()
            ),
            "min_imu_dynamic_accel_m_s2": float(
                seg["imu_forward_dynamic_m_s2"].min()
            ),
            "max_jerk_m_s3": float(seg["jerk_m_s3"].max()),
            "min_jerk_m_s3": float(seg["jerk_m_s3"].min()),
            "energy_wh": energy,
            "efficiency_wh_per_km": eff,
            "avg_grade_pct": float(seg["grade_pct"].mean()),
            "peak_speed_kph": float(seg["speed_kph"].max()),
        })
    return rows


def build_speed_bins(all_laps_merged: list[pd.DataFrame], bin_size_kph: float = 5.0) -> list[dict]:
    """Pool all laps and build flat-section efficiency by 5 km/h speed bands."""
    if not all_laps_merged:
        return []

    combined = pd.concat(all_laps_merged, ignore_index=True)
    flat = combined[
        (combined["speed_kph"] >= 5)
        & (combined["speed_kph"] <= 70)
        & (combined["grade_pct"].abs() <= 1.0)
    ].copy()

    if flat.empty:
        return []

    flat["speed_bin"] = (flat["speed_kph"] // bin_size_kph) * bin_size_kph
    rows = []
    for bin_lo, group in flat.groupby("speed_bin"):
        dist_km = group["dist_m"].sum() / 1000.0
        energy = group["energy_wh"].sum()
        eff = energy / dist_km if dist_km > 0 else float("nan")
        rows.append({
            "speed_bin_lo_kph": float(bin_lo),
            "speed_bin_hi_kph": float(bin_lo + bin_size_kph),
            "sample_count": len(group),
            "total_dist_km": dist_km,
            "total_energy_wh": energy,
            "efficiency_wh_per_km": eff,
        })
    rows.sort(key=lambda r: r["speed_bin_lo_kph"])
    return rows


# ---------------------------------------------------------------------------
# Plain-English findings
# ---------------------------------------------------------------------------

def generate_findings(
    laps_df: pd.DataFrame,
    sectors_df: pd.DataFrame,
    speed_bins_df: pd.DataFrame,
) -> str:
    lines = ["=== Strategy Findings ===", ""]

    # Identify "full laps" as those within 90% of median lap distance
    median_dist = laps_df["distance_m"].median()
    full = laps_df[laps_df["distance_m"] >= 0.9 * median_dist]

    if full.empty:
        lines.append("Not enough full laps to derive findings.")
        return "\n".join(lines)

    best_eff = full.loc[full["efficiency_wh_per_km"].idxmin()]
    fastest = full.loc[full["duration_s"].idxmin()]

    lines.append(
        f"Most efficient full lap: Lap {int(best_eff['lap'])}  "
        f"({best_eff['efficiency_wh_per_km']:.2f} Wh/km, "
        f"{best_eff['duration_s']:.1f}s)"
    )
    lines.append(
        f"Fastest full lap:        Lap {int(fastest['lap'])}  "
        f"({fastest['duration_s']:.1f}s, "
        f"{fastest['efficiency_wh_per_km']:.2f} Wh/km)"
    )

    if int(best_eff["lap"]) == int(fastest["lap"]):
        lines.append("-> The fastest lap was also the most efficient.")
    else:
        lines.append(
            "-> The fastest and most efficient laps were different - "
            "there is likely a speed-vs-efficiency trade-off to explore."
        )

    lines.append("")

    # Speed bin findings
    if not speed_bins_df.empty:
        valid = speed_bins_df.dropna(subset=["efficiency_wh_per_km"])
        if not valid.empty:
            best_bin = valid.loc[valid["efficiency_wh_per_km"].idxmin()]
            worst_bin = valid.loc[valid["efficiency_wh_per_km"].idxmax()]
            lines.append(
                f"Flat-section best efficiency:  "
                f"{best_bin['speed_bin_lo_kph']:.0f}-{best_bin['speed_bin_hi_kph']:.0f} km/h  "
                f"({best_bin['efficiency_wh_per_km']:.2f} Wh/km)"
            )
            lines.append(
                f"Flat-section worst efficiency: "
                f"{worst_bin['speed_bin_lo_kph']:.0f}-{worst_bin['speed_bin_hi_kph']:.0f} km/h  "
                f"({worst_bin['efficiency_wh_per_km']:.2f} Wh/km)"
            )
            lines.append("")

    # Sector improvements between the last two full laps
    full_lap_nums = sorted(full["lap"].tolist())
    if len(full_lap_nums) >= 2:
        last_two = full_lap_nums[-2:]
        prev_lap, last_lap = last_two
        prev_sec = sectors_df[sectors_df["lap"] == prev_lap].set_index("sector")
        last_sec = sectors_df[sectors_df["lap"] == last_lap].set_index("sector")
        common = prev_sec.index.intersection(last_sec.index)
        if not common.empty:
            delta = last_sec.loc[common, "efficiency_wh_per_km"] - prev_sec.loc[common, "efficiency_wh_per_km"]
            if not delta.empty and not delta.isna().all():
                best_gain = delta.idxmin()
                worst_regress = delta.idxmax()
                lines.append(
                    f"Lap {prev_lap}->{last_lap}: biggest efficiency improvement in sector {best_gain}  "
                    f"({delta[best_gain]:+.2f} Wh/km)"
                )
                if delta[worst_regress] > 0:
                    lines.append(
                        f"Lap {prev_lap}->{last_lap}: biggest remaining regression in sector {worst_regress}  "
                        f"({delta[worst_regress]:+.2f} Wh/km)"
                    )

    lines.append("")
    lines.append("=== Lap Summary ===")
    lines.append(
        laps_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    # Ensure output directory exists
    out_dir = os.path.dirname(args.output_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Reading GPX: {args.gps}")
    gps_df = read_gpx(args.gps)
    print(f"Reading telemetry: {args.telemetry}")
    telem_df = read_telemetry(args.telemetry)

    print("Building laps...")
    gps_laps, telem_laps, telem_aligned = build_laps(gps_df, telem_df, args)

    lap_summaries = []
    sector_rows = []
    all_derived = []

    for idx, (lap_gps, lap_telem) in enumerate(zip(gps_laps, telem_laps), start=1):
        if lap_gps.empty or lap_telem.empty:
            print(f"Lap {idx}: skipping (no GPS or telemetry data)")
            continue

        try:
            merged = merge_by_time(lap_telem, lap_gps, args.tolerance_sec)
        except ValueError as exc:
            print(f"Lap {idx}: skipping - {exc}")
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
        all_derived.append(derived)

        lap_row = build_lap_summary(derived, idx)
        lap_summaries.append(lap_row)

        for row in build_sector_summary(derived, idx, args.segments):
            sector_rows.append(row)

        print(
            f"Lap {idx}: {lap_row['duration_s']:.1f}s, "
            f"{lap_row['distance_m']:.0f}m, "
            f"{lap_row['avg_speed_kph']:.1f}km/h, "
            f"{lap_row['efficiency_wh_per_km']:.2f}Wh/km"
        )

    if not lap_summaries:
        print("ERROR: No laps could be processed.")
        return 1

    laps_df = pd.DataFrame(lap_summaries)
    sectors_df = pd.DataFrame(sector_rows)
    speed_bins = build_speed_bins(all_derived)
    speed_bins_df = pd.DataFrame(speed_bins)

    # Write CSVs
    laps_csv = args.output_prefix + "_laps.csv"
    sectors_csv = args.output_prefix + "_sectors.csv"
    bins_csv = args.output_prefix + "_speed_bins.csv"
    report_txt = args.output_prefix + "_report.txt"

    laps_df.to_csv(laps_csv, index=False)
    sectors_df.to_csv(sectors_csv, index=False)
    speed_bins_df.to_csv(bins_csv, index=False)
    print(f"Wrote: {laps_csv}, {sectors_csv}, {bins_csv}")

    # Write plain-English report
    report = generate_findings(laps_df, sectors_df, speed_bins_df)
    with open(report_txt, "w") as fh:
        fh.write(report + "\n")
    print(f"Wrote: {report_txt}")
    print()
    print(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
