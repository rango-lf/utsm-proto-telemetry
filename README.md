# UTSM Telemetry Dumper

Small Python tooling for pulling telemetry off the car, aligning it with a GPX track, drawing lap heatmaps, and generating simple strategy/efficiency reports.

## What This Project Does

There are really three jobs in this repo:

1. `dumper.py` talks to the serial device and saves a telemetry dump to CSV.
2. `gps_current_heatmap.py` aligns telemetry to a timestamped GPX track and renders lap heatmaps.
3. `analyze_strategy.py` reuses the same alignment and lap-splitting logic to turn a run into lap, sector, and speed-efficiency summaries.

In practice, the workflow is:

1. Capture telemetry from the car.
2. Export a GPX track from Strava or another GPS source.
3. Align telemetry time to GPS time.
4. Split the run into laps.
5. Merge GPS motion with telemetry current/voltage.
6. Summarize energy use and speed by lap and by sector.

## Repo Layout

Current important files:

- `dumper.py`: serial dump utility
- `gps_current_heatmap.py`: GPX + telemetry alignment and heatmap generation
- `analyze_strategy.py`: strategy and efficiency analysis
- `Utsm.gpx`, `Utsm-2.gpx`: example GPX files
- `telemetry_*.csv`: raw telemetry dumps from the logger
- `*_strategy_report.txt`, `*_strategy_laps.csv`, `*_strategy_sectors.csv`, `*_strategy_speed_bins.csv`: generated analysis outputs
- `morning run/`, `afternoon run/`, `old maps/`: saved heatmap images from previous runs

## Data Expectations

### Telemetry CSV

The analysis scripts expect these columns:

- `timestamp_ms`
- `current_mA`
- `voltage_mV`
- `ax_x100`
- `ay_x100`
- `az_x100`

`timestamp_ms` is treated as elapsed milliseconds from the start of the telemetry session. The scripts later convert that into absolute timestamps by anchoring it to GPX time.

### GPX

The GPX file must contain:

- latitude
- longitude
- elevation
- timestamp for each track point

Without GPX timestamps, the alignment step cannot work.

## Setup

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## How The Project Works

### 1. Telemetry capture

`dumper.py` opens a serial connection, optionally sends a command to trigger a dump, and writes everything it receives to a timestamped file like `telemetry_20260411_112302.csv`.

Example:

```powershell
python dumper.py --port COM13 --baud 115200
```

Useful flags:

- `--idle-timeout`: stop after no new data arrives for a while
- `--binary`: save raw bytes instead of text
- `--command`: command sent to the device before reading

### 2. Heatmap generation

`gps_current_heatmap.py` does the heavy lifting for alignment and lap splitting:

- reads the GPX track into a dataframe
- reads telemetry and coerces numeric columns
- computes acceleration magnitude
- aligns telemetry timestamps onto GPS time
- merges each telemetry row to the nearest GPX point within a tolerance
- converts lat/lon to approximate local XY meters
- draws the track with color based on current or acceleration

Example:

```powershell
python gps_current_heatmap.py Utsm.gpx telemetry_20260411_112302.csv --laps 4 --split-method start
```

#### Lap splitting modes

`gps_current_heatmap.py` supports several ways to split laps:

- `start`: default and most race-aware; find the first big current spike, align that to the GPX trace, then detect lap boundaries using repeated crossings of the starting Y position
- `line`: scan for a good horizontal lap line and split on crossings
- `time`: divide the run into equal time windows
- `points`: divide the GPX points evenly

There is also a manual mode with `--lap-times` if you already know the lap boundaries from Strava or timing notes.

### 3. Strategy analysis

`analyze_strategy.py` is the new reporting layer. It imports helper functions from `gps_current_heatmap.py` instead of duplicating the alignment code.

High-level flow:

1. Read GPX and telemetry.
2. Align telemetry onto GPX time.
3. Build laps using the selected split method.
4. Merge each lap's GPS points with telemetry rows by nearest timestamp.
5. Derive motion and energy features.
6. Summarize the run at three levels.
7. Write CSV outputs plus a text report with plain-English findings.

Example:

```powershell
python analyze_strategy.py Utsm.gpx telemetry_20260411_112302.csv --laps 4 --segments 12 --split-method start --output-prefix afternoon_strategy
```

## What The New Strategy Code Computes

### Step 1: Merge GPS and telemetry

For each lap, the script performs a nearest-time join between GPX points and telemetry rows. If the timestamps are too far apart, the row is dropped. This is controlled by `--tolerance-sec`.

### Step 2: Derive motion and energy channels

After merging, the script computes:

- `dt_s`: time delta between points
- `dist_m`: point-to-point distance in meters
- `elev_diff_m`: elevation change
- `speed_m_s` and `speed_kph`
- `grade_pct`
- `power_w`: from current and voltage
- `energy_wh`: power integrated over each timestep
- `cumdist_m`: cumulative distance through the lap

That gives the strategy code enough information to reason about speed, climbing, and energy efficiency instead of just raw current.

### Step 3: Build lap summary

For each lap it calculates:

- duration
- distance
- average speed
- average and max current
- average power
- total energy in Wh
- efficiency in Wh/km
- elevation gain and loss

This becomes the `*_laps.csv` file.

### Step 4: Build sector summary

Each lap is split into equal-distance sectors, controlled by `--segments` (default `12`). For every sector it calculates:

- duration
- distance
- average speed
- average power
- average and max current
- energy
- Wh/km
- average grade
- peak speed

This becomes `*_sectors.csv`.

Equal-distance sectors are useful because they make lap-to-lap comparisons cleaner than equal-time sectors when one lap is faster than another.

### Step 5: Build flat-speed efficiency bins

The script also pools all laps together and looks only at samples where:

- speed is between 5 and 70 km/h
- absolute grade is at most 1%

Those points are grouped into 5 km/h bins, and the script calculates Wh/km for each band. This is trying to answer:

"On roughly flat sections, what cruising speed seems most efficient?"

This becomes `*_speed_bins.csv`.

### Step 6: Generate plain-English findings

The report text is not just a dump of tables. The code also creates short strategy takeaways:

- most efficient full lap
- fastest full lap
- whether the fastest and most efficient lap were the same lap
- best and worst flat-speed efficiency bands
- biggest sector improvement between the latest two full laps
- biggest remaining sector efficiency loss

The "full lap" logic intentionally ignores obviously short laps by keeping laps whose distance is at least about 90% of the median lap distance. That helps stop warm-up or partial laps from dominating the conclusions.

## Output Files

Running `analyze_strategy.py` writes:

- `PREFIX_report.txt`: readable summary and findings
- `PREFIX_laps.csv`: one row per lap
- `PREFIX_sectors.csv`: one row per sector per lap
- `PREFIX_speed_bins.csv`: pooled efficiency by flat-road speed band

Running `gps_current_heatmap.py` writes:

- one heatmap image, or
- one image per lap like `current_heatmap_lap1.png`

## Interpreting The Existing Afternoon Output

Your current `afternoon_strategy_report.txt` says:

- lap 4 was both the fastest and the most efficient full lap
- flat-section efficiency was best around 18 km/h and worst around 33 km/h
- sector 11 improved the most from lap 3 to lap 4
- sector 3 was the main efficiency regression still left in lap 4

That suggests the last lap was not just quicker because of more power. It was also materially cleaner in energy use, which usually means smoother pacing or fewer inefficient surges.

## Common Commands

Capture telemetry:

```powershell
python dumper.py --port COM13
```

Generate per-lap heatmaps:

```powershell
python gps_current_heatmap.py Utsm.gpx telemetry_20260411_112302.csv --laps 4 --split-method start --output current_heatmap.png
```

Generate strategy report:

```powershell
python analyze_strategy.py Utsm.gpx telemetry_20260411_112302.csv --laps 4 --segments 12 --split-method start --output-prefix run1_strategy
```

Use a manual timing offset if telemetry and GPX are slightly misaligned:

```powershell
python analyze_strategy.py Utsm.gpx telemetry_20260411_112302.csv --laps 4 --time-offset-ms 750 --output-prefix run1_strategy
```

## Limits And Assumptions

- XY position is a local flat-earth approximation, which is fine for short tracks but not a geodesic solution.
- Nearest-time merging assumes GPS and telemetry clocks are close after alignment.
- Energy is estimated from current and voltage only; it does not include drivetrain efficiency modeling.
- Sector analysis is distance-based, not corner-based.
- The default `start` lap detection assumes a strong launch current spike and a repeatable crossing of the start-line Y band.

## Suggested Next Cleanup

If you want to keep developing this, the next sensible cleanup would be:

1. move generated outputs into an `outputs/` folder
2. move reusable helpers into a `utsm_telemetry/` package
3. add a small sample dataset and one smoke test
4. add a plotting script for sector deltas across laps

For now, this README should make the current layout much easier to navigate without breaking the existing workflow.
