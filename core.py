"""UTSM Telemetry analysis package.

Shared helpers for GPX parsing, telemetry ingestion, lap splitting,
GPS/telemetry alignment, and energy-channel derivation.  Both
gps_current_heatmap.py and analyze_strategy.py import from here.
"""

from .core import (
    NAMESPACE,
    parse_iso8601,
    parse_lap_time,
    read_gpx,
    read_telemetry,
    add_xy,
    find_start_spike,
    find_nearest_gps_index,
    find_lap_boundaries_by_y_crossing,
    count_line_crossings,
    detect_lap_line,
    compute_distance,
    compute_lap_stats,
    split_gps_into_laps,
    align_telemetry,
    merge_by_time,
    derive_motion_energy,
)

__all__ = [
    "NAMESPACE",
    "parse_iso8601",
    "parse_lap_time",
    "read_gpx",
    "read_telemetry",
    "add_xy",
    "find_start_spike",
    "find_nearest_gps_index",
    "find_lap_boundaries_by_y_crossing",
    "count_line_crossings",
    "detect_lap_line",
    "compute_distance",
    "compute_lap_stats",
    "split_gps_into_laps",
    "align_telemetry",
    "merge_by_time",
    "derive_motion_energy",
]
