import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def load_heatmap_module():
    module_path = Path(__file__).with_name("gps_current_heatmap.py")
    spec = importlib.util.spec_from_file_location("gps_current_heatmap", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hm = load_heatmap_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze lap strategy, speed, grade, and energy efficiency from GPX + telemetry."
    )
    parser.add_argument("gps", help="Path to a GPX file with timestamps")
    parser.add_argument("telemetry", help="Path to a telemetry CSV dump")
    parser.add_argument("--laps", type=int, default=4, help="Number of laps to analyze")
    parser.add_argument(
        "--segments",
        type=int,
        default=12,
        help="Number of equal-distance sectors per lap for segment analysis",
    )
    parser.add_argument(
        "--split-method",
        choices=["start", "points", "time", "line"],
        default="start",
        help="Lap split mode. 'start' uses the first current spike and Y-crossing logic.",
    )
    parser.add_argument(
        "--tolerance-sec",
        type=float,
        default=1.5,
        help="Nearest-time merge tolerance between GPX and telemetry",
    )
    parser.add_argument(
        "--time-offset-ms",
        type=float,
        default=0.0,
        help="Manual timing offset applied to telemetry after alignment",
    )
    parser.add_argument(
        "--start-time",
        help="Manual telemetry start time in ISO 8601 format",
    )
    parser.add_argument(
        "--output-prefix",
        default="strategy_analysis",
        help="Prefix for generated CSV and text reports",
    )
    return parser.parse_args()


def merge_gps_to_telemetry(
    gps: pd.DataFrame, telemetry: pd.DataFrame, tolerance_sec: float
) -> pd.DataFrame:
    gps_sorted = gps.sort_values("time").reset_index(drop=True).copy()
    telem_sorted = telemetry.sort_values("time").reset_index(drop=True).copy()
    merged = pd.merge_asof(
        gps_sorted,
        telem_sorted,
        on="time",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=tolerance_sec),
    )
    merged = merged.dropna(subset=["current_mA", "voltage_mV"]).reset_index(drop=True)
    if merged.empty:
        raise ValueError("No GPS points matched telemetry within the requested tolerance.")
    return merged


def enrich_motion(df: pd.DataFrame) -> pd.DataFrame:
    df = hm.add_xy(df)
    df = df.copy()
    df["dt_s"] = df["time"].diff().dt.total_seconds().fillna(0.0).clip(lower=0.0)
    df["dist_m"] = np.sqrt(df["x"].diff().fillna(0.0) ** 2 + df["y"].diff().fillna(0.0) ** 2)
    df["elev_diff_m"] = df["elev"].diff().fillna(0.0)
    df["speed_m_s"] = df["dist_m"] / df["dt_s"].replace(0.0, np.nan)
    df["speed_kph"] = df["speed_m_s"] * 3.6
    df["grade_pct"] = 100.0 * df["elev_diff_m"] / df["dist_m"].replace(0.0, np.nan)
    df["power_w"] = (df["current_mA"].abs() * df["voltage_mV"]) / 1_000_000.0
    df["energy_wh"] = df["power_w"] * df["dt_s"] / 3600.0
    df["cumdist_m"] = df["dist_m"].cumsum()
    return df


def build_laps(args: argparse.Namespace) -> list[pd.DataFrame]:
    gps_df = hm.read_gpx(args.gps)
    telem_df = hm.read_telemetry(args.telemetry)
    telem_df = hm.align_telemetry(telem_df, gps_df, args.start_time, args.time_offset_ms)

    if args.split_method == "start":
        start_idx = hm.find_start_spike(telem_df)
        start_time = telem_df.loc[start_idx, "time"]
        gps_start_idx = hm.find_nearest_gps_index(gps_df, start_time)
        gps_df = gps_df.loc[gps_start_idx:].reset_index(drop=True)
        boundaries = hm.find_lap_boundaries_by_y_crossing(gps_df, 0, args.laps)
        laps = []
        for i in range(min(len(boundaries) - 1, args.laps)):
            laps.append(gps_df.iloc[boundaries[i] : boundaries[i + 1]].reset_index(drop=True))
        if len(laps) < args.laps and boundaries:
            laps.append(gps_df.iloc[boundaries[-1] :].reset_index(drop=True))
    else:
        laps = hm.split_gps_into_laps(gps_df, args.laps, args.split_method)

    merged_laps = []
    for idx, lap_gps in enumerate(laps, start=1):
        if lap_gps.empty:
            continue
        lap_telem = telem_df[
            (telem_df["time"] >= lap_gps["time"].iloc[0] - pd.Timedelta(seconds=args.tolerance_sec))
            & (telem_df["time"] <= lap_gps["time"].iloc[-1] + pd.Timedelta(seconds=args.tolerance_sec))
        ].copy()
        merged = merge_gps_to_telemetry(lap_gps, lap_telem, args.tolerance_sec)
        merged = enrich_motion(merged)
        merged["lap"] = idx
        merged_laps.append(merged)
    return merged_laps


def summarize_laps(laps: list[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for lap in laps:
        valid = lap[lap["dt_s"] > 0].copy()
        if valid.empty:
            continue
        duration_s = float(valid["dt_s"].sum())
        distance_m = float(valid["dist_m"].sum())
        energy_wh = float(valid["energy_wh"].sum())
        rows.append(
            {
                "lap": int(valid["lap"].iloc[0]),
                "duration_s": duration_s,
                "distance_m": distance_m,
                "avg_speed_kph": distance_m / duration_s * 3.6 if duration_s else np.nan,
                "avg_current_a": float(valid["current_mA"].mean() / 1000.0),
                "max_current_a": float(valid["current_mA"].max() / 1000.0),
                "avg_power_w": energy_wh * 3600.0 / duration_s if duration_s else np.nan,
                "energy_wh": energy_wh,
                "wh_per_km": energy_wh / (distance_m / 1000.0) if distance_m else np.nan,
                "elev_gain_m": float(valid.loc[valid["elev_diff_m"] > 0, "elev_diff_m"].sum()),
                "elev_loss_m": float(-valid.loc[valid["elev_diff_m"] < 0, "elev_diff_m"].sum()),
            }
        )
    return pd.DataFrame(rows)


def summarize_sectors(laps: list[pd.DataFrame], segments: int) -> pd.DataFrame:
    rows = []
    for lap in laps:
        valid = lap[lap["dt_s"] > 0].copy()
        if valid.empty:
            continue
        total_dist = float(valid["dist_m"].sum())
        valid["sector"] = np.minimum(
            (valid["cumdist_m"] / max(total_dist, 1.0) * segments).astype(int),
            segments - 1,
        ) + 1
        grouped = valid.groupby("sector", as_index=False)
        for sector_df in [grouped.get_group(sector) for sector in grouped.groups]:
            duration_s = float(sector_df["dt_s"].sum())
            distance_m = float(sector_df["dist_m"].sum())
            energy_wh = float(sector_df["energy_wh"].sum())
            rows.append(
                {
                    "lap": int(sector_df["lap"].iloc[0]),
                    "sector": int(sector_df["sector"].iloc[0]),
                    "duration_s": duration_s,
                    "distance_m": distance_m,
                    "avg_speed_kph": distance_m / duration_s * 3.6 if duration_s else np.nan,
                    "avg_power_w": energy_wh * 3600.0 / duration_s if duration_s else np.nan,
                    "avg_current_a": float(sector_df["current_mA"].mean() / 1000.0),
                    "max_current_a": float(sector_df["current_mA"].max() / 1000.0),
                    "energy_wh": energy_wh,
                    "wh_per_km": energy_wh / (distance_m / 1000.0) if distance_m else np.nan,
                    "avg_grade_pct": float(sector_df["grade_pct"].mean(skipna=True)),
                    "peak_speed_kph": float(sector_df["speed_kph"].max(skipna=True)),
                }
            )
    return pd.DataFrame(rows)


def summarize_speed_bins(laps: list[pd.DataFrame]) -> pd.DataFrame:
    all_points = pd.concat(laps, ignore_index=True)
    valid = all_points[(all_points["dt_s"] > 0) & (all_points["speed_kph"].between(5, 70))].copy()
    valid = valid[valid["grade_pct"].abs() <= 1.0]
    if valid.empty:
        return pd.DataFrame()
    valid["speed_bin"] = pd.cut(valid["speed_kph"], bins=np.arange(5, 75, 5), right=False)
    grouped = valid.groupby("speed_bin", observed=True, as_index=False)
    rows = []
    for _, df in grouped:
        distance_m = float(df["dist_m"].sum())
        energy_wh = float(df["energy_wh"].sum())
        if distance_m < 150:
            continue
        rows.append(
            {
                "speed_bin": str(df["speed_bin"].iloc[0]),
                "avg_speed_kph": float(df["speed_kph"].mean()),
                "distance_m": distance_m,
                "energy_wh": energy_wh,
                "wh_per_km": energy_wh / (distance_m / 1000.0),
                "avg_power_w": energy_wh * 3600.0 / float(df["dt_s"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("avg_speed_kph").reset_index(drop=True)


def build_findings(
    lap_summary: pd.DataFrame, sector_summary: pd.DataFrame, speed_bins: pd.DataFrame
) -> list[str]:
    findings = []
    if lap_summary.empty:
        return ["No valid laps were summarized."]

    full_laps = lap_summary[lap_summary["distance_m"] >= lap_summary["distance_m"].median() * 0.9].copy()
    if len(full_laps) >= 2:
        best_eff = full_laps.sort_values(["wh_per_km", "avg_speed_kph"], ascending=[True, False]).iloc[0]
        findings.append(
            f"Most efficient full lap: lap {int(best_eff['lap'])} at "
            f"{best_eff['wh_per_km']:.2f} Wh/km and {best_eff['avg_speed_kph']:.1f} km/h."
        )

        fastest = full_laps.sort_values("avg_speed_kph", ascending=False).iloc[0]
        findings.append(
            f"Fastest full lap: lap {int(fastest['lap'])} at {fastest['avg_speed_kph']:.1f} km/h "
            f"using {fastest['avg_current_a']:.2f} A average current."
        )

        if int(best_eff["lap"]) == int(fastest["lap"]):
            findings.append(
                "The same lap was both fastest and most energy-efficient, which usually points to smoother throttle application rather than simply pushing harder."
            )

    if not speed_bins.empty:
        best_bin = speed_bins.sort_values("wh_per_km").iloc[0]
        worst_bin = speed_bins.sort_values("wh_per_km", ascending=False).iloc[0]
        findings.append(
            f"On near-flat sections, the best observed efficiency band was {best_bin['avg_speed_kph']:.1f} km/h "
            f"at {best_bin['wh_per_km']:.2f} Wh/km."
        )
        findings.append(
            f"The least efficient observed flat-speed band was {worst_bin['avg_speed_kph']:.1f} km/h "
            f"at {worst_bin['wh_per_km']:.2f} Wh/km."
        )

    if not sector_summary.empty and len(full_laps) >= 2:
        compare_laps = full_laps.sort_values("lap")["lap"].tail(2).tolist()
        sector_pair = sector_summary[sector_summary["lap"].isin(compare_laps)].copy()
        if sector_pair["lap"].nunique() == 2:
            pivot = sector_pair.pivot(index="sector", columns="lap", values=["wh_per_km", "avg_speed_kph"])
            latest, prior = compare_laps[-1], compare_laps[-2]
            pivot[("delta", "eff_gain")] = pivot[("wh_per_km", latest)] - pivot[("wh_per_km", prior)]
            pivot[("delta", "speed_gain")] = (
                pivot[("avg_speed_kph", latest)] - pivot[("avg_speed_kph", prior)]
            )
            biggest_eff = pivot.sort_values(("delta", "eff_gain")).iloc[0]
            biggest_loss = pivot.sort_values(("delta", "eff_gain"), ascending=False).iloc[0]
            findings.append(
                f"Compared with lap {prior}, lap {latest} improved most in sector {int(biggest_eff.name)} "
                f"({biggest_eff[('delta', 'eff_gain')]:.2f} Wh/km change)."
            )
            findings.append(
                f"The main remaining efficiency leak was sector {int(biggest_loss.name)} on lap {latest}, "
                f"which moved {biggest_loss[('delta', 'eff_gain')]:.2f} Wh/km against lap {prior}."
            )

    return findings


def write_report(
    prefix: str,
    lap_summary: pd.DataFrame,
    sector_summary: pd.DataFrame,
    speed_bins: pd.DataFrame,
    findings: list[str],
) -> tuple[Path, Path, Path]:
    prefix_path = Path(prefix)
    lap_path = prefix_path.with_name(prefix_path.name + "_laps.csv")
    sector_path = prefix_path.with_name(prefix_path.name + "_sectors.csv")
    speed_path = prefix_path.with_name(prefix_path.name + "_speed_bins.csv")
    report_path = prefix_path.with_name(prefix_path.name + "_report.txt")

    lap_summary.to_csv(lap_path, index=False)
    sector_summary.to_csv(sector_path, index=False)
    speed_bins.to_csv(speed_path, index=False)

    lines = ["Strategy Analysis", "=================", ""]
    lines.append("Key Findings")
    for item in findings:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("Lap Summary")
    lines.append(lap_summary.to_string(index=False))
    if not speed_bins.empty:
        lines.append("")
        lines.append("Flat-Section Speed Bins")
        lines.append(speed_bins.to_string(index=False))
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path, lap_path, sector_path


def main() -> int:
    args = parse_args()
    laps = build_laps(args)
    if not laps:
        raise SystemExit("No laps could be built from the supplied data.")

    lap_summary = summarize_laps(laps)
    sector_summary = summarize_sectors(laps, args.segments)
    speed_bins = summarize_speed_bins(laps)
    findings = build_findings(lap_summary, sector_summary, speed_bins)
    report_path, lap_path, sector_path = write_report(
        args.output_prefix, lap_summary, sector_summary, speed_bins, findings
    )

    print("Key findings:")
    for item in findings:
        print(f"- {item}")
    print("")
    print("Lap summary:")
    print(lap_summary.to_string(index=False))
    print("")
    print(f"Saved report to: {report_path}")
    print(f"Saved lap summary to: {lap_path}")
    print(f"Saved sector summary to: {sector_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
