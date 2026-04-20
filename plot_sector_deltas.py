"""Plot sector-by-sector efficiency deltas across laps.

This is the 'plotting script for sector deltas across laps'. Bro IDK if this works properly.

Usage
-----
python plot_sector_deltas.py outputs/run1_strategy_sectors.csv \\
    --metric efficiency_wh_per_km --output outputs/sector_deltas.png

The script reads the *_sectors.csv produced by analyze_strategy.py and draws
a bar chart for every consecutive lap pair, with one bar per sector showing
how much better or worse each sector became.  A separate line chart shows
the absolute metric value per sector so you can see which sectors still have
the most energy to recover.
"""

from __future__ import annotations

import argparse
import os
import sys

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "Missing required package. Install with: pip install matplotlib pandas numpy"
    ) from exc

VALID_METRICS = [
    "efficiency_wh_per_km",
    "avg_speed_kph",
    "avg_power_w",
    "avg_current_mA",
    "energy_wh",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot sector efficiency deltas across laps from a strategy sectors CSV."
    )
    parser.add_argument("sectors_csv", help="Path to the *_sectors.csv file from analyze_strategy.py")
    parser.add_argument(
        "--metric",
        choices=VALID_METRICS,
        default="efficiency_wh_per_km",
        help="Column to compare across laps",
    )
    parser.add_argument(
        "--output", "-o",
        default="outputs/sector_deltas.png",
        help="Output image path",
    )
    parser.add_argument(
        "--laps", nargs="+", type=int,
        help="Lap numbers to include (default: all laps found in the CSV)",
    )
    return parser.parse_args()


def load_and_validate(path: str, metric: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Sectors CSV not found: {path}")
    df = pd.read_csv(path)
    for col in ("lap", "sector", metric):
        if col not in df.columns:
            raise ValueError(
                f"Expected column '{col}' not found in {path}. "
                f"Available: {', '.join(df.columns)}"
            )
    return df


def plot_sector_deltas(df: pd.DataFrame, metric: str, output_path: str) -> None:
    laps = sorted(df["lap"].unique())
    sectors = sorted(df["sector"].unique())
    n_sectors = len(sectors)

    # Pivot: rows = lap, cols = sector
    pivot = df.pivot(index="lap", columns="sector", values=metric)

    fig, axes = plt.subplots(
        2, 1,
        figsize=(max(12, n_sectors * 0.8), 10),
        gridspec_kw={"hspace": 0.45},
    )

    # --- Top panel: absolute values per lap ---
    ax_abs = axes[0]
    x = np.arange(n_sectors)
    width = 0.8 / len(laps)
    for i, lap in enumerate(laps):
        if lap not in pivot.index:
            continue
        vals = [pivot.loc[lap, s] if s in pivot.columns else np.nan for s in sectors]
        ax_abs.bar(
            x + i * width - (len(laps) - 1) * width / 2,
            vals,
            width=width * 0.9,
            label=f"Lap {lap}",
        )
    ax_abs.set_xticks(x)
    ax_abs.set_xticklabels([f"S{s}" for s in sectors])
    ax_abs.set_ylabel(metric.replace("_", " "))
    ax_abs.set_title(f"Sector {metric.replace('_', ' ')} by lap")
    ax_abs.legend(loc="upper right", fontsize=8)
    ax_abs.grid(axis="y", linestyle="--", alpha=0.4)

    # Highlight the worst sector in the last lap
    last_lap = laps[-1]
    if last_lap in pivot.index:
        last_vals = pd.Series(
            {s: pivot.loc[last_lap, s] for s in sectors if s in pivot.columns}
        ).dropna()
        if not last_vals.empty:
            # For efficiency, lower is better; for speed, higher is better
            worst_sector = int(last_vals.idxmax()) if "efficiency" in metric else int(last_vals.idxmin())
            worst_x = x[sectors.index(worst_sector)]
            ax_abs.axvspan(worst_x - 0.45, worst_x + 0.45, alpha=0.12, color="red", label="Worst sector (last lap)")

    # --- Bottom panel: deltas between consecutive laps ---
    ax_delta = axes[1]
    pair_colors = plt.cm.tab10(np.linspace(0, 1, max(len(laps) - 1, 1)))

    for i, (prev_lap, next_lap) in enumerate(zip(laps[:-1], laps[1:])):
        if prev_lap not in pivot.index or next_lap not in pivot.index:
            continue
        deltas = []
        for s in sectors:
            if s in pivot.columns:
                prev_v = pivot.loc[prev_lap, s]
                next_v = pivot.loc[next_lap, s]
                deltas.append(next_v - prev_v if pd.notna(prev_v) and pd.notna(next_v) else np.nan)
            else:
                deltas.append(np.nan)

        bar_x = x + i * width - (len(laps) - 2) * width / 2
        bars = ax_delta.bar(bar_x, deltas, width=width * 0.9, color=pair_colors[i], label=f"Lap {prev_lap}→{next_lap}")

        # Colour bars: negative delta = improvement (green tint), positive = regression (red tint)
        for bar, d in zip(bars, deltas):
            if pd.isna(d):
                continue
            if "efficiency" in metric:
                bar.set_color("seagreen" if d < 0 else "tomato")
            else:
                bar.set_color("tomato" if d < 0 else "seagreen")
            bar.set_alpha(0.75)

    ax_delta.axhline(0, color="black", linewidth=0.8)
    ax_delta.set_xticks(x)
    ax_delta.set_xticklabels([f"S{s}" for s in sectors])
    ax_delta.set_ylabel(f"Δ {metric.replace('_', ' ')}")
    direction = "↓ better" if "efficiency" in metric else "↑ better"
    ax_delta.set_title(f"Sector delta between consecutive laps  ({direction})")
    ax_delta.legend(loc="upper right", fontsize=8)
    ax_delta.grid(axis="y", linestyle="--", alpha=0.4)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved sector delta chart to: {output_path}")


def main() -> int:
    args = parse_args()
    df = load_and_validate(args.sectors_csv, args.metric)

    if args.laps:
        df = df[df["lap"].isin(args.laps)]
        if df.empty:
            print(f"ERROR: No data for laps {args.laps}")
            return 1

    plot_sector_deltas(df, args.metric, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
