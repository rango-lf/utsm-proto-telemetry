"""Minimal smoke tests for the utsm_telemetry package.

Run with:  python -m pytest tests/ -v
Or simply: python tests/test_smoke.py
"""

import math
import os
import sys
import unittest
import io

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from utsm_telemetry.core import (
    add_xy,
    compute_distance,
    parse_iso8601,
    parse_lap_time,
    align_telemetry,
    add_gps_motion_features,
    derive_acceleration_features,
    derive_motion_energy,
    find_lap_boundaries_by_start_gate,
    merge_by_time,
    read_gpx,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gps(n: int = 10, lat0: float = 43.0, lon0: float = -79.0) -> pd.DataFrame:
    times = pd.date_range("2026-04-11T12:00:00Z", periods=n, freq="1s")
    return pd.DataFrame({
        "lat": lat0 + np.linspace(0, 0.001, n),
        "lon": lon0 + np.linspace(0, 0.001, n),
        "elev": np.linspace(100, 105, n),
        "time": times,
    })


def _make_telem(n: int = 10) -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp_ms": np.arange(n) * 1000,
        "current_mA": np.abs(np.random.randn(n) * 500 + 2000),
        "voltage_mV": np.full(n, 24000.0),
        "ax_x100": np.zeros(n),
        "ay_x100": np.zeros(n),
        "az_x100": np.full(n, 100.0),
    })


def _make_gps_from_xy(points: list[tuple[float, float]]) -> pd.DataFrame:
    lat0 = 43.0
    lon0 = -79.0
    meters_per_deg_lat = 110540.0
    meters_per_deg_lon = 111320.0 * math.cos(math.radians(lat0))
    times = pd.date_range("2026-04-11T12:00:00Z", periods=len(points), freq="1s")
    return pd.DataFrame({
        "lat": [lat0 + y / meters_per_deg_lat for x, y in points],
        "lon": [lon0 + x / meters_per_deg_lon for x, y in points],
        "elev": np.full(len(points), 100.0),
        "time": times,
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestParseISO8601(unittest.TestCase):
    def test_z_suffix(self):
        ts = parse_iso8601("2026-04-11T12:00:00Z")
        self.assertEqual(ts.year, 2026)
        self.assertEqual(ts.month, 4)

    def test_offset(self):
        ts = parse_iso8601("2026-04-11T12:00:00+00:00")
        self.assertEqual(ts.hour, 12)


class TestParseLapTime(unittest.TestCase):
    def test_mmss(self):
        start = pd.Timestamp("2026-04-11T12:00:00+00:00")
        result = parse_lap_time("02:30", start)
        self.assertEqual(result, start + pd.Timedelta(minutes=2, seconds=30))

    def test_hmmss(self):
        start = pd.Timestamp("2026-04-11T12:00:00+00:00")
        result = parse_lap_time("1:02:30", start)
        self.assertEqual(result, start + pd.Timedelta(hours=1, minutes=2, seconds=30))

    def test_bad_format(self):
        start = pd.Timestamp("2026-04-11T12:00:00+00:00")
        with self.assertRaises(ValueError):
            parse_lap_time("bad", start)


class TestAddXY(unittest.TestCase):
    def test_origin_at_zero(self):
        gps = _make_gps(5)
        xy = add_xy(gps)
        self.assertAlmostEqual(xy["x"].iloc[0], 0.0, places=5)
        self.assertAlmostEqual(xy["y"].iloc[0], 0.0, places=5)

    def test_monotone_increasing(self):
        gps = _make_gps(5)
        xy = add_xy(gps)
        self.assertTrue((xy["x"].diff().dropna() >= 0).all())
        self.assertTrue((xy["y"].diff().dropna() >= 0).all())


class TestComputeDistance(unittest.TestCase):
    def test_nonzero(self):
        gps = _make_gps(10)
        dist = compute_distance(gps)
        self.assertGreater(dist, 0)

    def test_single_point(self):
        gps = _make_gps(1)
        dist = compute_distance(gps)
        self.assertEqual(dist, 0.0)


class TestAlignTelemetry(unittest.TestCase):
    def test_creates_time_column(self):
        gps = _make_gps(10)
        telem = _make_telem(10)
        # Read telemetry without coercion (raw)
        from utsm_telemetry.core import read_telemetry as rt
        aligned = align_telemetry(telem, gps, None, 0.0)
        self.assertIn("time", aligned.columns)
        self.assertEqual(len(aligned), len(telem))

    def test_offset_applied(self):
        gps = _make_gps(10)
        telem = _make_telem(10)
        aligned_base = align_telemetry(telem, gps, None, 0.0)
        aligned_off = align_telemetry(telem, gps, None, 1000.0)
        delta = (aligned_off["time"].iloc[0] - aligned_base["time"].iloc[0]).total_seconds()
        self.assertAlmostEqual(delta, 1.0, places=3)


class TestMergeByTime(unittest.TestCase):
    def test_merge_produces_lat_lon(self):
        gps = _make_gps(20)
        telem = _make_telem(20)
        aligned = align_telemetry(telem, gps, None, 0.0)
        merged = merge_by_time(aligned, gps, tolerance_sec=2.0)
        self.assertIn("lat", merged.columns)
        self.assertIn("lon", merged.columns)
        self.assertIn("gps_speed_kph", merged.columns)
        self.assertGreater(len(merged), 0)


class TestStartGateLapDetection(unittest.TestCase):
    def test_rejects_paddock_and_right_side_crossings(self):
        points = [
            (0, 0),
            (0, 10),
            (-10, 0),
            (-10, 10),
            (-20, 0),
            (-25, 12),
            (-30, 0),
            (-35, 12),
            (300, 0),
            (300, 1000),
            (-500, 1000),
            (-500, 0),
            (-45, 0),
            (300, 0),
            (300, 1000),
            (-500, 1000),
            (-500, 0),
            (-45, 0),
            (300, 0),
            (300, 1000),
            (-500, 1000),
            (-500, 0),
            (-45, 0),
        ]
        gps = _make_gps_from_xy(points)

        boundaries = find_lap_boundaries_by_start_gate(
            gps,
            start_index=0,
            laps=3,
            min_gap_points=1,
            min_lap_distance_m=2500.0,
            pre_race_max_distance_m=100.0,
        )

        self.assertEqual(boundaries, [4, 12, 17, 22])

    def test_afternoon_run_has_three_left_side_laps(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        gpx_path = os.path.join(root, "Utsm-2.gpx")
        if not os.path.exists(gpx_path):
            self.skipTest("Afternoon GPX fixture is not present.")

        gps = read_gpx(gpx_path).loc[423:].reset_index(drop=True)
        boundaries = find_lap_boundaries_by_start_gate(gps, 0, laps=3)

        self.assertEqual(len(boundaries), 4)
        for actual, expected in zip(boundaries, [149, 795, 1280, 1763]):
            self.assertLessEqual(abs(actual - expected), 2)

        xy = add_xy(gps)
        anchor_x = float(xy.loc[boundaries[0], "x"])
        for boundary in boundaries:
            self.assertLessEqual(abs(float(xy.loc[boundary, "x"]) - anchor_x), 60.0)


class TestGPSMotionFeatures(unittest.TestCase):
    def test_gps_speed_uses_gps_timing(self):
        gps = _make_gps(6, lat0=43.0, lon0=-79.0)
        gps["lat"] = 43.0
        gps["lon"] = -79.0 + np.arange(6) * 0.00001
        with_speed = add_gps_motion_features(gps)
        self.assertIn("gps_speed_kph", with_speed.columns)
        self.assertGreater(with_speed["gps_speed_kph"].iloc[2], 0.0)

    def test_merged_speed_not_inflated_by_telemetry_frequency(self):
        gps = _make_gps(5)
        gps["lat"] = 43.0
        gps["lon"] = -79.0 + np.arange(5) * 0.00001
        telem = _make_telem(17)
        telem["timestamp_ms"] = np.arange(17) * 250
        aligned = align_telemetry(telem, gps, None, 0.0)
        merged = merge_by_time(aligned, gps, tolerance_sec=0.2)
        derived = derive_motion_energy(merged)
        gps_speed_max = add_gps_motion_features(gps)["gps_speed_kph"].max()
        self.assertLessEqual(derived["speed_kph"].max(), gps_speed_max + 1e-9)


class TestDeriveMotionEnergy(unittest.TestCase):
    def test_columns_present(self):
        gps = _make_gps(20)
        telem = _make_telem(20)
        aligned = align_telemetry(telem, gps, None, 0.0)
        merged = merge_by_time(aligned, gps, tolerance_sec=2.0)
        derived = derive_motion_energy(merged)
        for col in (
            "dt_s",
            "dist_m",
            "speed_kph",
            "power_w",
            "energy_wh",
            "cumdist_m",
            "accel_total_g",
            "accel_total_m_s2",
            "gps_longitudinal_accel_m_s2",
            "imu_ax_m_s2",
            "imu_total_g",
            "imu_forward_dynamic_m_s2",
            "accel_longitudinal_smooth_m_s2",
            "jerk_m_s3",
        ):
            self.assertIn(col, derived.columns, f"Missing column: {col}")

    def test_energy_nonnegative(self):
        gps = _make_gps(20)
        telem = _make_telem(20)
        aligned = align_telemetry(telem, gps, None, 0.0)
        merged = merge_by_time(aligned, gps, tolerance_sec=2.0)
        derived = derive_motion_energy(merged)
        self.assertTrue((derived["energy_wh"] >= 0).all())

    def test_cumdist_monotone(self):
        gps = _make_gps(20)
        telem = _make_telem(20)
        aligned = align_telemetry(telem, gps, None, 0.0)
        merged = merge_by_time(aligned, gps, tolerance_sec=2.0)
        derived = derive_motion_energy(merged)
        diffs = derived["cumdist_m"].diff().dropna()
        self.assertTrue((diffs >= 0).all())

    def test_gps_acceleration_from_speed_derivative(self):
        gps = _make_gps(4)
        telem = _make_telem(4)
        merged = align_telemetry(telem, gps, None, 0.0)
        merged["lat"] = gps["lat"]
        merged["lon"] = gps["lon"]
        merged["elev"] = gps["elev"]
        merged["gps_speed_m_s"] = [0.0, 1.0, 3.0, 6.0]
        derived = derive_motion_energy(
            merged,
            accel_smooth_window_sec=0.0,
        )
        self.assertIn("gps_longitudinal_accel_m_s2", derived.columns)
        self.assertAlmostEqual(derived["gps_longitudinal_accel_m_s2"].iloc[1], 1.0)
        self.assertAlmostEqual(derived["gps_longitudinal_accel_m_s2"].iloc[2], 2.0)


class TestAccelerationFeatures(unittest.TestCase):
    def test_mpu_scale_and_dynamic_columns(self):
        telem = pd.DataFrame({
            "timestamp_ms": [0, 1000, 2000],
            "current_mA": [1000, 1000, 1000],
            "voltage_mV": [24000, 24000, 24000],
            "ax_x100": [0, 1000, 2000],
            "ay_x100": [0, 0, 0],
            "az_x100": [1000, 1000, 1000],
            "amag_x100": [1000, 1414, 2236],
        })
        derived = derive_acceleration_features(
            telem,
            forward_axis="ax",
            accel_scale=1000.0,
            bias_window_s=0.0,
            smooth_window_s=0.0,
        )
        self.assertAlmostEqual(derived["accel_longitudinal_raw_g"].iloc[1], 1.0)
        self.assertAlmostEqual(
            derived["accel_longitudinal_m_s2"].iloc[1],
            9.80665,
            places=5,
        )
        self.assertAlmostEqual(derived["imu_total_g_reported"].iloc[0], 1.0)
        self.assertIn("imu_ax_dynamic_smooth_m_s2", derived.columns)

    def test_stationary_bias_removal_near_zero(self):
        telem = _make_telem(20)
        telem["ax_x100"] = 40
        telem["ay_x100"] = -120
        telem["az_x100"] = 1000
        derived = derive_acceleration_features(
            telem,
            imu_axis="ax",
            accel_scale=1000.0,
            bias_window_s=30.0,
            smooth_window_s=3.0,
        )
        self.assertLess(abs(float(derived["imu_forward_dynamic_m_s2"].median())), 1e-9)

    def test_negative_axis(self):
        telem = _make_telem(3)
        telem["ax_x100"] = [1000, 2000, 3000]
        derived = derive_acceleration_features(
            telem,
            forward_axis="neg_ax",
            accel_scale=1000.0,
            bias_window_s=0.0,
            smooth_window_s=0.0,
        )
        self.assertAlmostEqual(derived["accel_longitudinal_raw_g"].iloc[0], -1.0)
        self.assertAlmostEqual(derived["accel_longitudinal_raw_g"].iloc[2], -3.0)

    def test_raw_columns_are_not_mutated(self):
        telem = _make_telem(3)
        original = telem[["ax_x100", "ay_x100", "az_x100"]].copy()
        derived = derive_acceleration_features(telem)
        pd.testing.assert_frame_equal(
            original.reset_index(drop=True),
            derived[["ax_x100", "ay_x100", "az_x100"]].reset_index(drop=True),
        )

    def test_bad_axis(self):
        telem = _make_telem(3)
        with self.assertRaises(ValueError):
            derive_acceleration_features(telem, forward_axis="bad_axis")


if __name__ == "__main__":
    unittest.main()
