# UTSM Proto Telemetry, Analysis, and Simulation Project

These are some Python tools for capturing, analysing, and visualising SEMA telemetry data. The pipeline takes two inputs: a **GPX track file** from Strava/Garmin and a **telemetry CSV** from the car's on-board computer, and produces lap summaries, efficiency reports, strategy recommendations, and an interactive HTML dashboard.

**Primary Contributors:** Brayden Chan Carusone, Rango Lee-Fu

---

## Table of Contents

1. [A simplified overview](#a-simplified-overview)
2. [Setup](#setup)
3. [Your data files](#your-data-files)
4. [Typical workflow](#typical-workflow)
5. [Scripts reference](#scripts-reference)
   - [dumper.py — capture telemetry from the car](#dumperpy--capture-telemetry-from-the-car)
   - [analyze_strategy.py — lap and efficiency reports](#analyze_strategypy--lap-and-efficiency-reports)
   - [simulate_speed_strategy.py — speed profile optimiser](#simulate_speed_strategypy--speed-profile-optimiser)
   - [build_interactive_dashboard.py — HTML dashboard](#build_interactive_dashboardpy--html-dashboard)
   - [gps_current_heatmap.py — current heatmap images](#gps_current_heatmappy--current-heatmap-images)
   - [animate_run.py — animated replay](#animate_runpy--animated-replay)
   - [plot_sector_deltas.py — sector delta chart](#plot_sector_deltaspy--sector-delta-chart)
   - [build_animation_gallery.py — HTML animation gallery](#build_animation_gallerypy--html-animation-gallery)
6. [The utsm_telemetry package](#the-utsm_telemetry-package)
7. [Common flags explained](#common-flags-explained)
8. [Interpreting the outputs](#interpreting-the-outputs)
9. [Notes and known limits](#notes-and-known-limits)

---

## A simplified overview

```
On-board computer (serial)
        │
        ▼
   dumper.py  ──────────────────► telemetry_dumps/telemetry_YYYYMMDD_HHMMSS.csv
        
Strava / Garmin
        │
        ▼
   Utsm.gpx  (download manually)

        │                    │
        ▼                    ▼
  GPX track file      Telemetry CSV
        │                    │
        └──────────┬──────────┘
                   │
          utsm_telemetry/          ← shared helpers (don't run directly!)
          (aligns clocks, splits laps, derives speed/energy/acceleration)
                   │
         ┌─────────┼──────────────┐
         ▼         ▼              ▼
  analyze_    simulate_     build_interactive_
  strategy    speed_        dashboard.py
  .py         strategy.py   
         │         │              │
         ▼         ▼              ▼
  outputs/    outputs/        outputs/
  *_laps.csv  *_profile.csv   dashboard.html
  *_report    *_report
```

The `utsm_telemetry/` folder is a **shared library**. It contains all the reusable logic (GPS parsing, lap detection, sensor alignment, physics). The scripts at the root are the entry points you actually run.

---

## Setup

Requires Python 3.11+, Matplotlib, Pandas, and Numpy. Run these commands (line-by-line) once when you clone the repo:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
pip install matplotlib pandas numpy
```

On macOS/Linux, replace `Activate.ps1` with `source .venv/bin/activate`.

Verify everything works with the smoke tests (no pytest needed):

```powershell
python tests\test_smoke.py
```

---

## Your data files

### GPX track file

Download from Strava/Garmin/etc. after the run. This must contain latitude, longitude, elevation, and timestamps for every track point. The two April 11 runs are already included:

- `Utsm.gpx` — morning run
- `Utsm-2.gpx` — afternoon run

### Telemetry CSV

Captured by `dumper.py` during the run and saved to `telemetry_dumps/`. The CSV must have these columns:

| Column | Description |
|---|---|
| `timestamp_ms` | Milliseconds elapsed since capture start |
| `current_mA` | Motor current draw in milliamps |
| `voltage_mV` | Battery voltage in millivolts |
| `ax_x100` | IMU X-axis acceleration × 100 (milli-g units) |
| `ay_x100` | IMU Y-axis acceleration × 100 |
| `az_x100` | IMU Z-axis acceleration × 100 |

`amag_x100` (acceleration magnitude) is optional but used when present. A value of ~1000 means roughly 1 g.

The two April 11 telemetry files are already included in `telemetry_dumps/`.

---

## Typical workflow

After a test, you should run these four commands in order. The first two generate analysis reports, the third builds the interactive dashboard, then the fourth produces per-lap heatmap images.

```powershell
# 1. Lap reports and efficiency analysis
python analyze_strategy.py Utsm-2.gpx telemetry_dumps\telemetry_20260411_122713.csv --laps 4 --split-method start --output-prefix outputs\afternoon

# 2. Speed strategy optimisation
python simulate_speed_strategy.py Utsm-2.gpx telemetry_dumps\telemetry_20260411_122713.csv --laps 4 --split-method start --segments 24 --time-budget-sec 2100 --output-prefix outputs\afternoon_strategy

# 3. Interactive dashboard (covers both morning and afternoon runs)
python build_interactive_dashboard.py --laps 4 --output outputs\dashboard.html

# 4. Current heatmap images (one PNG per lap)
python gps_current_heatmap.py Utsm-2.gpx telemetry_dumps\telemetry_20260411_122713.csv --laps 4 --split-method start --output outputs\heatmap.png
```

Then open `outputs\dashboard.html` in any browser.

---

## Scripts reference

### `dumper.py` — capture telemetry from the car

Reads serial data from the on-board computer and saves it as a timestamped CSV in `telemetry_dumps/`.

---

### `analyze_strategy.py` — lap and efficiency reports

The main post-run analysis tool. Aligns the GPX track with the telemetry, splits into laps, and produces CSV summaries and a plain-text findings report.

```powershell
python analyze_strategy.py Utsm-2.gpx telemetry_dumps\telemetry_20260411_122713.csv --laps 4 --split-method start --output-prefix outputs\afternoon
```

**Outputs written to `outputs/`:**

| File | Contents |
|---|---|
| `*_laps.csv` | One row per lap: duration, distance, speed, current, power, Wh/km, elevation |
| `*_sectors.csv` | Each lap divided into equal-distance sectors with the same columns |
| `*_speed_bins.csv` | Efficiency grouped by speed band (flat road only) |
| `*_report.txt` | Plain-English findings: best/worst lap, efficiency trade-offs, sector regressions |

---

### `simulate_speed_strategy.py` — speed profile optimiser

Takes the historical run data and works out the most efficient `accelerate / hold / coast` strategy for each segment of the track, within a given time budget.

```powershell
python simulate_speed_strategy.py Utsm-2.gpx telemetry_dumps\telemetry_20260411_122713.csv --laps 4 --split-method start --segments 24 --time-budget-sec 2100 --fuse-current-ma 20000 --fuse-max-duration-sec 1.0 --output-prefix outputs\strategy
```

**Key flags:**

| Flag | What it does |
|---|---|
| `--segments` | How many equal-distance sections to divide the track into (more = finer strategy) |
| `--time-budget-sec` | Maximum allowed total run time in seconds |
| `--fuse-current-ma` | The fuse rating — strategies that exceed this for too long are rejected |
| `--fuse-max-duration-sec` | Maximum seconds allowed above the fuse threshold |
| `--vehicle-mass-kg` | Car + driver mass, used in physics calculations |
| `--rolling-resistance-coeff` | Measured from coastdown testing if available |

**Outputs:**

| File | Contents |
|---|---|
| `*_strategy_profile.csv` | Target speed and action (accelerate/hold/coast) for each segment |
| `*_strategy_samples.csv` | Detailed per-point predicted current, power, and energy |
| `*_strategy_report.txt` | Summary: predicted total energy, time, and comparison to baseline |

---

### `build_interactive_dashboard.py` — HTML dashboard

Builds a self-contained HTML file that lets you replay both runs, compare laps, and overlay the optimised strategy. This is the main tool for post-run review sessions.

```powershell
python build_interactive_dashboard.py --laps 4 --output outputs\dashboard.html
```

Open the output file in any browser.

**What's in the dashboard:**

- **Run switcher** — toggle between morning and afternoon datasets
- **Timeline scrubber** — drag to any point in the run; the track map and all charts update live
- **Track map** — shows the car's position, a colour-coded trail by metric (current, speed, acceleration, etc.), and the optimised strategy regions (accelerate = green, hold = blue, coast = grey)
- **Charts** — current, speed, GPS acceleration, IMU dynamic acceleration, power, and cumulative energy — all synchronised to the timeline
- **Prediction overlays** — dashed lines showing what the optimiser predicted vs. what actually happened
- **20 A fuse line** — visible threshold on the current chart
- **Live readout** — current segment, action label, target speed, predicted current and power

**Strategy toggles in the dashboard UI:**

- `Strategy` — show/hide the action-region colour overlay on the map
- `Labels` — show/hide per-segment labels
- The speed chart overlays actual speed against the optimiser's target speed

---

### `gps_current_heatmap.py` — current heatmap images

Produces one PNG image per lap showing where on the track current draw is high or low. Good for quickly spotting where energy is being used on track.

```powershell
python gps_current_heatmap.py Utsm-2.gpx telemetry_dumps\telemetry_20260411_122713.csv --laps 4 --split-method start --output outputs\heatmap.png
```

Use `--metric current` (default), `--metric accel`, or `--metric magnitude` to colour by a different channel.

---

### `animate_run.py` — animated replay

Produces an animated GIF or HTML replay of a single run with a moving marker and live chart. Slower and less interactive than the dashboard — use this when you need a shareable animation file rather than a live tool.

```powershell
python animate_run.py Utsm-2.gpx telemetry_dumps\telemetry_20260411_122713.csv --laps 4 --split-method start --output outputs\animation.html
```

---

### `plot_sector_deltas.py` — sector delta chart

Shows how efficiency changed sector-by-sector from one lap to the next. Useful for identifying where the car improved or regressed across laps.

```powershell
python plot_sector_deltas.py outputs\afternoon_sectors.csv --output outputs\sector_deltas.png
```

Requires a `*_sectors.csv` file produced by `analyze_strategy.py`.

> **Note:** This script is still being tested — double-check its output against the sector CSV if something looks off.

---

### `build_animation_gallery.py` — HTML animation gallery

Wraps a set of animation files into a scrollable HTML gallery page. Useful if you've generated animations for multiple runs and want a single page to browse them.

```powershell
python build_animation_gallery.py outputs\*.html --output outputs\gallery.html
```

---

## The `utsm_telemetry` package

This folder contains all the shared logic. You don't run these files directly — they're imported by the scripts above. Here's roughly what each module does:

**`core.py`** — everything to do with GPS and telemetry data itself:
- Reading GPX files and telemetry CSVs
- Aligning the two clocks so GPS and telemetry are in sync
- ~~Detecting where each lap starts and ends (using line-segment intersection at the start/finish line)~~
- ~~Computing Haversine distances, track curvature, speed, grade, and acceleration channels~~
- Merging GPS and telemetry rows by nearest timestamp

**`simulation.py`** — the strategy and physics layer:
- Fitting an empirical energy model from historical data
- The `accelerate / hold / coast` dynamic-programming optimiser
- Physics-based power estimation (rolling resistance, grade, acceleration, corner resistance)
- Building and evaluating strategy profiles

**`throttle.py`** — work in progress; intended for throttle-pattern analysis.

If you're writing a new analysis script, import from `utsm_telemetry` rather than copying logic out of `core.py`. See how `analyze_strategy.py` does it for an example.

---

## Common flags explained

These flags appear across multiple scripts and mean the same thing everywhere:

| Flag | Default | Meaning |
|---|---|---|
| `--laps` | 4 | How many racing laps to expect in the data |
| `--split-method` | `start` | How to detect lap boundaries. Use `start` — it finds the actual start/finish line crossing |
| `--lap-times` | — | Override automatic lap detection by providing elapsed times manually (e.g. `11:20 22:45 34:10 45:30`) in `MM:SS` format |
| `--start-time` | — | Force a specific ISO 8601 timestamp as the telemetry start if clock alignment fails |
| `--time-offset-ms` | 0 | Fine-tune clock alignment by nudging the telemetry by ±N milliseconds |
| `--tolerance-sec` | 1.5 | How far apart (in seconds) a telemetry row and GPS point can be and still be matched |
| `--forward-axis` | `ax` | Which IMU axis points forward on your car (`ax`, `ay`, `az`, or their negatives) |
| `--imu-axis-sign` | 1 | Flip the IMU axis sign if the car is mounted backwards (`-1`) |

---

## Interpreting the outputs

**Wh/km (efficiency)** — the primary performance metric. Lower is better. Shell Eco-Marathon scoring is based on equivalent fuel energy per 100 km, so minimising Wh/km directly maps to a better score.

**GPS acceleration vs. MPU dynamic acceleration** — these are two different things:
- *GPS acceleration* is derived from how the car's speed changes between GPS points. Smooth, low-noise, but low time resolution (~1 Hz).
- *MPU dynamic acceleration* is from the IMU sensor on the car. High time resolution but noisier; gravity and mounting bias are removed with a rolling median filter. Good for diagnostics (vibration, throttle pulses) rather than overall speed trend.

**Accelerate / Hold / Coast actions** — the three states the optimiser assigns to each track segment:
- *Accelerate* — motor on, speed increasing
- *Hold* — motor on just enough to maintain speed
- *Coast* — motor off, car rolling on momentum

**Fuse risk** — strategies that keep current above the fuse threshold (`--fuse-current-ma`) for longer than `--fuse-max-duration-sec` are automatically rejected by the optimiser.

---

## Notes and known limits

- **Energy figures are electrical, not mechanical.** Power = current × voltage at the battery terminals. Drivetrain losses (motor, gearbox) are estimated but not directly measured.
- **GPS speed accuracy depends on the GPX sample rate.** If the GPS logs at 1 Hz, speed resolution is coarser in tight corners.
- **IMU orientation must be set correctly.** If acceleration charts look inverted or wrong, try `--forward-axis neg_ax` (or another axis). The dashboard shows correlation scores for all six axis options to help diagnose this.
- **The morning run telemetry ends before lap 4 finishes.** The tools handle this gracefully — the fourth lap is filled with GPS-backed zero-telemetry samples so the strategy view stays complete.
- **Outputs are not committed to Git.** Regenerate them from the source data whenever needed. The `outputs/` folder is in `.gitignore`.
- **`plot_sector_deltas.py` and `throttle.py` are works in progress** and may not produce correct results in all cases.
