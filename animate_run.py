"""Animate a telemetry run with track color-coding and live graphs.

Usage
-----
python animate_run.py Utsm-2.gpx telemetry_dumps/telemetry_20260411_122713.csv \
    --laps 4 --split-method start --output outputs/accel_run.gif

The default output is a lightweight GIF preview.  Use a .html output path for
a browser-embedded animation, but expect it to take longer and produce a much
larger file.
"""

from __future__ import annotations

import argparse
import os
import sys

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.collections import LineCollection
except ImportError as exc:
    raise SystemExit(
        "Missing required package. Install dependencies with: pip install matplotlib pandas numpy"
    ) from exc

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from analyze_strategy import build_laps
from utsm_telemetry import (
    FORWARD_AXIS_CHOICES,
    add_xy,
    derive_motion_energy,
    merge_by_time,
    read_gpx,
    read_telemetry,
)

METRIC_CHOICES = (
    "current_mA",
    "power_w",
    "speed_kph",
    "accel_total_m_s2",
    "accel_longitudinal_smooth_m_s2",
    "jerk_m_s3",
)

PANEL_SPECS = (
    ("current_mA", "Current (mA)", "#d62728"),
    ("speed_kph", "Speed (km/h)", "#1f77b4"),
    ("accel_longitudinal_smooth_m_s2", "Longitudinal accel (m/s^2)", "#2ca02c"),
    ("power_w", "Power (W)", "#9467bd"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render an animated map and live telemetry graphs for a run."
    )
    parser.add_argument("gps", help="Path to the GPX track file")
    parser.add_argument("telemetry", help="Path to the telemetry CSV file")
    parser.add_argument("--laps", type=int, default=4)
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
        help="IMU axis treated as longitudinal acceleration.",
    )
    parser.add_argument(
        "--accel-window",
        type=int,
        default=5,
        help="Rolling median window for smoothed longitudinal acceleration.",
    )
    parser.add_argument(
        "--metric",
        choices=METRIC_CHOICES,
        default="accel_longitudinal_smooth_m_s2",
        help="Column used to color-code the track.",
    )
    parser.add_argument(
        "--trail-sec",
        type=float,
        default=30.0,
        help="Seconds of highlighted car trail to show behind the live marker.",
    )
    parser.add_argument(
        "--color-history-sec",
        type=float,
        default=45.0,
        help=(
            "Seconds of color-coded track to show behind the car. "
            "Use 0 to keep all driven color history."
        ),
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=25,
        help="Render every Nth telemetry sample to keep the animation lighter.",
    )
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=180,
        help="Cap total rendered frames after stride sampling. Lower is faster.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=90,
        help="Output DPI for GIF/video saves. Lower is faster and smaller.",
    )
    parser.add_argument(
        "--embed-limit-mb",
        type=float,
        default=100.0,
        help="Maximum embedded HTML animation size before Matplotlib drops frames.",
    )
    parser.add_argument(
        "--show-final-sample",
        action="store_true",
        help=(
            "Mark the final processed sample. This is hidden by default because "
            "the lap splitter can end on a different point along the start-line band."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        default="outputs/acceleration_animation.gif",
        help="Output animation path. .gif is fastest; .html is self-contained but larger.",
    )
    return parser.parse_args()


def load_derived_run(args: argparse.Namespace) -> pd.DataFrame:
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
        derived = derive_motion_energy(
            merged,
            forward_axis=args.forward_axis,
            accel_window=args.accel_window,
        )
        derived["lap"] = lap_num
        rows.append(derived)

    if not rows:
        raise ValueError("No lap data could be merged for animation.")

    df = pd.concat(rows, ignore_index=True).sort_values("time").reset_index(drop=True)
    df = add_xy(df)
    start_time = pd.to_datetime(df["time"].iloc[0])
    df["elapsed_s"] = (pd.to_datetime(df["time"]) - start_time).dt.total_seconds()
    return df


def _finite_limits(values: pd.Series) -> tuple[float, float]:
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return 0.0, 1.0
    lo = float(finite.quantile(0.02))
    hi = float(finite.quantile(0.98))
    if np.isclose(lo, hi):
        pad = max(abs(lo) * 0.05, 1.0)
        return lo - pad, hi + pad
    pad = (hi - lo) * 0.08
    return lo - pad, hi + pad


def _track_segments(df: pd.DataFrame) -> np.ndarray:
    coords = df[["x", "y"]].to_numpy()
    if len(coords) < 2:
        raise ValueError("Need at least two merged points to animate a track.")
    return np.stack([coords[:-1], coords[1:]], axis=1)


def build_animation(df: pd.DataFrame, args: argparse.Namespace) -> FuncAnimation:
    stride = max(args.stride, 1)
    frame_indices = np.arange(0, len(df), stride)
    if args.max_frames and len(frame_indices) > args.max_frames:
        frame_indices = np.linspace(
            frame_indices[0],
            frame_indices[-1],
            args.max_frames,
            dtype=int,
        )
    anim_df = df.iloc[frame_indices].copy().reset_index(drop=True)
    if len(anim_df) < 2:
        raise ValueError("Not enough samples after applying --stride.")

    fig = plt.figure(figsize=(12, 7.2))
    grid = fig.add_gridspec(4, 2, width_ratios=[1.35, 1.0], hspace=0.32, wspace=0.22)
    ax_map = fig.add_subplot(grid[:, 0])
    panel_axes = [fig.add_subplot(grid[i, 1]) for i in range(4)]

    color_values = pd.to_numeric(df[args.metric], errors="coerce")
    color_lo, color_hi = _finite_limits(color_values)
    ax_map.plot(
        df["x"],
        df["y"],
        color="#d0d0d0",
        linewidth=1.4,
        alpha=0.75,
        zorder=1,
    )
    line_collection = LineCollection(
        [],
        cmap="plasma",
        norm=plt.Normalize(color_lo, color_hi),
        linewidths=4.0,
        alpha=0.9,
        capstyle="round",
        zorder=3,
    )
    ax_map.add_collection(line_collection)
    fig.colorbar(line_collection, ax=ax_map, pad=0.015, label=args.metric)

    car_marker = ax_map.scatter([], [], s=110, c="white", edgecolors="black", linewidths=1.8, zorder=5)
    trail_line, = ax_map.plot([], [], color="black", linewidth=1.5, alpha=0.35, zorder=4)
    ax_map.scatter(df["x"].iloc[0], df["y"].iloc[0], color="limegreen", s=80, label="Start")
    if args.show_final_sample:
        ax_map.scatter(
            df["x"].iloc[-1],
            df["y"].iloc[-1],
            color="red",
            marker="X",
            s=80,
            label="Final sample",
        )
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.set_xlabel("X (m)")
    ax_map.set_ylabel("Y (m)")
    ax_map.set_title("Telemetry replay")
    ax_map.grid(True, linestyle="--", alpha=0.28)
    ax_map.legend(loc="upper right")

    x_pad = max((df["x"].max() - df["x"].min()) * 0.05, 5.0)
    y_pad = max((df["y"].max() - df["y"].min()) * 0.05, 5.0)
    ax_map.set_xlim(df["x"].min() - x_pad, df["x"].max() + x_pad)
    ax_map.set_ylim(df["y"].min() - y_pad, df["y"].max() + y_pad)

    progress_lines = []
    cursor_lines = []
    elapsed = df["elapsed_s"]
    for ax, (column, label, color) in zip(panel_axes, PANEL_SPECS):
        ax.plot(elapsed, df[column], color="#c8c8c8", linewidth=1.0)
        progress, = ax.plot([], [], color=color, linewidth=1.8)
        cursor = ax.axvline(0, color="black", linewidth=0.9, alpha=0.55)
        lo, hi = _finite_limits(df[column])
        ax.set_xlim(float(elapsed.min()), float(elapsed.max()))
        ax.set_ylim(lo, hi)
        ax.set_ylabel(label)
        ax.grid(True, linestyle="--", alpha=0.28)
        progress_lines.append((progress, column))
        cursor_lines.append(cursor)
    panel_axes[-1].set_xlabel("Elapsed time (s)")

    info_text = ax_map.text(
        0.02,
        0.02,
        "",
        transform=ax_map.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#888888"},
    )

    def update(frame_idx: int):
        idx = min(frame_idx, len(anim_df) - 1)
        row = anim_df.iloc[idx]
        t = float(row["elapsed_s"])
        car_marker.set_offsets([[row["x"], row["y"]]])

        trail = anim_df[
            (anim_df["elapsed_s"] >= t - args.trail_sec)
            & (anim_df["elapsed_s"] <= t)
        ]
        trail_line.set_data(trail["x"], trail["y"])

        upto = df[df["elapsed_s"] <= t]
        if args.color_history_sec > 0:
            color_track = upto[upto["elapsed_s"] >= t - args.color_history_sec]
        else:
            color_track = upto
        if len(color_track) >= 2:
            line_collection.set_segments(_track_segments(color_track))
            line_collection.set_array(
                pd.to_numeric(color_track[args.metric], errors="coerce")
                .iloc[:-1]
                .to_numpy()
            )
        else:
            line_collection.set_segments([])
            line_collection.set_array(np.array([]))

        for progress, column in progress_lines:
            progress.set_data(upto["elapsed_s"], upto[column])
        for cursor in cursor_lines:
            cursor.set_xdata([t, t])

        info_text.set_text(
            f"t={t:.1f}s  lap={int(row['lap'])}\n"
            f"speed={row['speed_kph']:.1f} km/h\n"
            f"current={row['current_mA']:.0f} mA\n"
            f"long accel={row['accel_longitudinal_smooth_m_s2']:.2f} m/s^2"
        )
        return [line_collection, car_marker, trail_line, info_text] + [
            item for pair in progress_lines for item in pair[:1]
        ] + cursor_lines

    return FuncAnimation(
        fig,
        update,
        frames=len(anim_df),
        interval=1000 / max(args.fps, 1),
        blit=False,
    )


def save_animation(
    animation: FuncAnimation,
    output_path: str,
    fps: int,
    embed_limit_mb: float,
    dpi: int,
) -> None:
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    ext = os.path.splitext(output_path)[1].lower()
    if ext == ".html":
        plt.rcParams["animation.embed_limit"] = max(float(embed_limit_mb), 20.0)
        html = animation.to_jshtml(fps=fps)
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write("<!doctype html><html><head><meta charset=\"utf-8\">")
            fh.write("<title>UTSM telemetry animation</title></head><body>")
            fh.write(html)
            fh.write("</body></html>")
    elif ext == ".gif":
        animation.save(output_path, writer=PillowWriter(fps=fps), dpi=dpi)
    else:
        animation.save(output_path, fps=fps, dpi=dpi)
    print(f"Saved animation to: {output_path}")


def main() -> int:
    args = parse_args()
    df = load_derived_run(args)
    animation = build_animation(df, args)
    save_animation(animation, args.output, args.fps, args.embed_limit_mb, args.dpi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
