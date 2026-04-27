# UTSM Proto Telemetry

Python tools for reading car telemetry, aligning it with GPX track data, splitting laps, analyzing energy use, and replaying runs in an interactive dashboard.

The current analysis path is centered on:

1. `dumper.py` for serial telemetry capture.
2. `utsm_telemetry/` for shared parsing, alignment, lap detection, motion, energy, and acceleration helpers.
3. `analyze_strategy.py` for lap, sector, speed-bin, and strategy reports.
4. `build_interactive_dashboard.py` for the fast local HTML replay dashboard and strategy overlay.

Generated artifacts are reproducible and ignored by Git. Regenerate reports and dashboards into `outputs/` when needed.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Smoke tests can run without pytest:

```powershell
python tests\test_smoke.py
```

## Data Inputs

Telemetry CSVs must include:

- `timestamp_ms`
- `current_mA`
- `voltage_mV`
- `ax_x100`
- `ay_x100`
- `az_x100`

Some dumps also include `amag_x100`. Despite the column names, the MPU-6050 acceleration values in the 2026 afternoon data behave like milli-g units: about `1000` means `1 g`.

GPX files must include latitude, longitude, elevation, and timestamps. Speed is derived from GPX point-to-point movement on the GPX sampling clock, not by integrating noisy accelerometer data.

## Current Afternoon Demo

The canonical afternoon demo uses:

- `Utsm-2.gpx`
- `telemetry_dumps\telemetry_20260411_122713.csv`
- `--laps 3`
- `--split-method start`

The start split now uses a localized left-side start/finish gate. It ignores paddock movement before the real start and rejects the false right-side same-Y crossing.

## Interactive Dashboard

Build the dashboard:

```powershell
python build_interactive_dashboard.py --laps 3 --output outputs\afternoon_interactive_dashboard.html
```

Useful strategy knobs:

```powershell
python build_interactive_dashboard.py --laps 3 --strategy-segments 24 --strategy-speed-min-kph 8 --strategy-speed-max-kph 40 --strategy-max-delta-kph-per-segment 6 --output outputs\afternoon_interactive_dashboard.html
```

Open `outputs\afternoon_interactive_dashboard.html` in a browser. It is a self-contained HTML file with:

- one manual time slider
- play/pause replay
- full-course gray reference trace
- current-lap colored trail
- synchronized current, speed, GPS acceleration, MPU dynamic acceleration, power, and cumulative total-energy charts
- color selector for track overlay metrics
- strategy target-speed overlay on the map plus speed/energy comparison overlays
- optional lap-boundary debug markers

The strategy layer uses the same optimized profile as `simulate_speed_strategy.py`. In the dashboard:

- `Strategy` toggle shows or hides the optimized target-speed overlay
- `Strategy target speed` can be selected as the map coloring metric
- the speed chart overlays actual speed against optimized target speed
- the total-energy chart overlays actual cumulative joules against predicted cumulative joules
- the live readout shows current segment, target speed, and suggested action

Acceleration is split into two separate channels:

- `GPS acceleration`: derived from GPS speed changes, smoother and more physically interpretable for vehicle speed trend.
- `MPU dynamic acceleration`: MPU-6050 axis data scaled as milli-g, bias/gravity corrected with a rolling median, and kept as a diagnostic vibration/response channel.

The dashboard payload also includes MPU axis/sign diagnostic correlations for `ax`, `-ax`, `ay`, `-ay`, `az`, and `-az`.

The total-energy chart is cumulative run joules versus elapsed time. It spans the whole run and does not reset at lap boundaries.

## Strategy Analysis

Run the corrected afternoon analysis:

```powershell
python analyze_strategy.py Utsm-2.gpx telemetry_dumps\telemetry_20260411_122713.csv --laps 3 --split-method start --output-prefix outputs\afternoon_clean_demo
```

This writes:

- `PREFIX_laps.csv`
- `PREFIX_sectors.csv`
- `PREFIX_speed_bins.csv`
- `PREFIX_report.txt`

The analysis computes:

- lap duration and distance
- GPX-derived speed
- current, voltage, power, and integrated Wh
- Wh/km efficiency
- elevation gain/loss and grade
- GPS acceleration and MPU dynamic acceleration
- equal-distance sector summaries
- flat-road speed efficiency bins

## Speed Strategy Simulation

Run the first empirical optimizer:

```powershell
python simulate_speed_strategy.py Utsm-2.gpx telemetry_dumps\telemetry_20260411_122713.csv --laps 3 --split-method start --segments 24 --output-prefix outputs\speed_strategy
```

This writes:

- `PREFIX_strategy_profile.csv`
- `PREFIX_strategy_samples.csv`
- `PREFIX_strategy_report.txt`

The dashboard generator runs the same optimizer internally so the HTML stays in sync with the standalone strategy report.

The first version does not use a neural model. It fits an empirical power model from historical samples and then chooses a target speed profile by distance while staying within a time budget and per-segment speed-change limits.

For the corrected afternoon run, the current reference result is 3 complete laps. Lap 2 is the most efficient full lap, while lap 3 is the fastest full lap.

## Optional Animation Fallback

The interactive dashboard is the main visualization. The older animation scripts are kept as optional fallback/demo tools:

```powershell
python animate_run.py --help
python build_animation_gallery.py --help
```

Use them only when a pre-rendered GIF/HTML gallery is specifically needed. They are slower and less useful for data inspection than the interactive dashboard.

## Common Commands

Capture telemetry from the serial device:

```powershell
python dumper.py --port COM13
```

Generate legacy current heatmaps:

```powershell
python gps_current_heatmap.py Utsm.gpx telemetry_dumps\telemetry_20260411_112302.csv --laps 4 --split-method start --output outputs\current_heatmap.png
```

Run smoke tests and regenerate the afternoon demo:

```powershell
python tests\test_smoke.py
python build_interactive_dashboard.py --laps 3 --output outputs\afternoon_interactive_dashboard.html
python analyze_strategy.py Utsm-2.gpx telemetry_dumps\telemetry_20260411_122713.csv --laps 3 --split-method start --output-prefix outputs\afternoon_clean_demo
python simulate_speed_strategy.py Utsm-2.gpx telemetry_dumps\telemetry_20260411_122713.csv --laps 3 --split-method start --segments 24 --output-prefix outputs\speed_strategy
```

## Notes And Limits

- XY position is a local flat-earth approximation, which is fine for this track scale.
- Nearest-time merging assumes telemetry and GPX clocks can be aligned closely enough.
- Energy is electrical energy estimated from current and voltage; it is not drivetrain output energy.
- GPS acceleration is low bandwidth because it comes from GPX speed changes.
- MPU dynamic acceleration is useful for diagnostics, but sensor orientation and gravity compensation are still imperfect without gyro fusion or a known mounting calibration.
- Generated outputs, caches, and local scratch artifacts should stay out of Git.
