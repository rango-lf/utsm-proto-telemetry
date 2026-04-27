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
from argparse import Namespace
from build_interactive_dashboard import make_payload
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
from utsm_telemetry.simulation import (
    build_strategy_report,
    build_strategy_segments,
    build_strategy_samples,
    fit_empirical_energy_model,
    optimize_speed_profile,
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
            "energy_j",
            "cum_energy_j",
            "cumdist_m",
            "accel_total_g",
            "accel_total_m_s2",
            "gps_longitudinal_accel_m_s2",
            "gps_longitudinal_accel_abs_m_s2",
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
        self.assertTrue((derived["energy_j"] >= 0).all())

    def test_cumdist_monotone(self):
        gps = _make_gps(20)
        telem = _make_telem(20)
        aligned = align_telemetry(telem, gps, None, 0.0)
        merged = merge_by_time(aligned, gps, tolerance_sec=2.0)
        derived = derive_motion_energy(merged)
        diffs = derived["cumdist_m"].diff().dropna()
        self.assertTrue((diffs >= 0).all())

    def test_energy_j_matches_power_times_dt(self):
        gps = _make_gps(6)
        telem = _make_telem(6)
        aligned = align_telemetry(telem, gps, None, 0.0)
        merged = merge_by_time(aligned, gps, tolerance_sec=2.0)
        derived = derive_motion_energy(merged)
        expected = derived["power_w"] * derived["dt_s"]
        np.testing.assert_allclose(derived["energy_j"], expected)

    def test_cum_energy_j_monotone(self):
        gps = _make_gps(8)
        telem = _make_telem(8)
        aligned = align_telemetry(telem, gps, None, 0.0)
        merged = merge_by_time(aligned, gps, tolerance_sec=2.0)
        derived = derive_motion_energy(merged)
        diffs = derived["cum_energy_j"].diff().fillna(0.0)
        self.assertTrue((diffs >= -1e-12).all())

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
        self.assertIn("gps_longitudinal_accel_abs_m_s2", derived.columns)
        self.assertAlmostEqual(derived["gps_longitudinal_accel_m_s2"].iloc[1], 1.0)
        self.assertAlmostEqual(derived["gps_longitudinal_accel_m_s2"].iloc[2], 2.0)
        self.assertAlmostEqual(derived["gps_longitudinal_accel_abs_m_s2"].iloc[2], 2.0)

    def test_gps_acceleration_uses_gps_clock_when_available(self):
        gps = _make_gps(4)
        telem = _make_telem(7)
        aligned = align_telemetry(telem, gps, None, 0.0)
        merged = merge_by_time(aligned, gps, tolerance_sec=1.0)
        merged["gps_speed_m_s"] = merged["gps_time"].map({
            gps["time"].iloc[0]: 0.0,
            gps["time"].iloc[1]: 1.0,
            gps["time"].iloc[2]: 3.0,
            gps["time"].iloc[3]: 6.0,
        })
        derived = derive_motion_energy(
            merged,
            accel_smooth_window_sec=0.0,
        )
        by_gps_time = derived.groupby("gps_time")["gps_longitudinal_accel_m_s2"].first()
        self.assertAlmostEqual(by_gps_time.iloc[1], 1.0)
        self.assertAlmostEqual(by_gps_time.iloc[2], 2.0)
        self.assertAlmostEqual(by_gps_time.iloc[3], 3.0)


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


class TestSimulation(unittest.TestCase):
    def test_optimizer_prefers_sensible_flat_speed_profile(self):
        rows = []
        run_dist = 0.0
        run_energy = 0.0
        for i in range(24):
            speed = 12.0 + (i % 4) * 4.0
            dt_s = 1.0
            dist_m = speed / 3.6 * dt_s
            power_w = 20.0 + 2.0 * speed + 0.4 * (speed ** 2)
            energy_j = power_w * dt_s
            run_dist += dist_m
            run_energy += energy_j
            rows.append({
                "time": pd.Timestamp("2026-04-11T12:00:00Z") + pd.Timedelta(seconds=i),
                "dt_s": dt_s,
                "dist_m": dist_m,
                "run_cumdist_m": run_dist,
                "grade_pct": 0.0,
                "speed_kph": speed,
                "gps_longitudinal_accel_m_s2": 0.0,
                "power_w": power_w,
                "energy_j": energy_j,
                "cum_energy_j": run_energy,
            })
        df = pd.DataFrame(rows)
        model = fit_empirical_energy_model(df)
        segments = build_strategy_segments(df, segments=6)
        profile = optimize_speed_profile(
            segments,
            model,
            time_budget_sec=float(segments["baseline_time_s"].sum() * 1.2),
            speed_min_kph=10.0,
            speed_max_kph=30.0,
            max_delta_kph_per_segment=4.0,
            speed_step_kph=2.0,
        )
        self.assertTrue((profile["target_speed_kph"] >= 10.0).all())
        self.assertTrue((profile["target_speed_kph"] <= 30.0).all())
        self.assertTrue((profile["speed_delta_kph"].abs() <= 4.0 + 1e-9).all())
        self.assertLess(profile["pred_energy_j"].sum(), segments["baseline_energy_j"].sum())

    def test_strategy_samples_and_report(self):
        df = pd.DataFrame({
            "time": pd.date_range("2026-04-11T12:00:00Z", periods=4, freq="1s"),
            "dt_s": [1.0, 1.0, 1.0, 1.0],
            "dist_m": [5.0, 5.0, 5.0, 5.0],
            "run_cumdist_m": [5.0, 10.0, 15.0, 20.0],
            "grade_pct": [0.0, 0.0, 0.0, 0.0],
            "speed_kph": [18.0, 18.0, 20.0, 20.0],
            "gps_longitudinal_accel_m_s2": [0.0, 0.0, 0.1, 0.0],
            "power_w": [100.0, 100.0, 120.0, 120.0],
            "energy_j": [100.0, 100.0, 120.0, 120.0],
            "cum_energy_j": [100.0, 200.0, 320.0, 440.0],
        })
        profile = pd.DataFrame({
            "segment": [1, 2],
            "dist_start_m": [0.0, 10.0],
            "dist_end_m": [10.0, 20.0],
            "length_m": [10.0, 10.0],
            "target_speed_kph": [18.0, 16.0],
            "entry_speed_kph": [18.0, 18.0],
            "speed_delta_kph": [0.0, -2.0],
            "action": ["hold", "coast"],
            "pred_power_w": [100.0, 90.0],
            "segment_time_s": [2.0, 2.25],
            "pred_energy_j": [200.0, 202.5],
        })
        samples = build_strategy_samples(df, profile)
        report = build_strategy_report(df, profile, time_budget_sec=5.0)
        self.assertIn("pred_cum_energy_j", samples.columns)
        self.assertIn("Suggested acceleration segments:", report)

    def test_dashboard_payload_includes_strategy_overlay_fields(self):
        df = pd.DataFrame({
            "time": pd.date_range("2026-04-11T12:00:00Z", periods=4, freq="1s"),
            "lap": [1, 1, 1, 1],
            "elapsed_s": [0.0, 1.0, 2.0, 3.0],
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [0.0, 0.0, 0.0, 0.0],
            "segment": [1, 1, 2, 2],
            "current_mA": [100.0, 110.0, 120.0, 130.0],
            "speed_kph": [18.0, 20.0, 22.0, 24.0],
            "target_speed_kph": [19.0, 19.0, 21.0, 21.0],
            "gps_longitudinal_accel_abs_m_s2": [0.1, 0.2, 0.1, 0.2],
            "imu_forward_dynamic_m_s2": [0.05, 0.04, 0.06, 0.03],
            "power_w": [100.0, 110.0, 120.0, 130.0],
            "energy_j": [100.0, 110.0, 120.0, 130.0],
            "cum_energy_j": [100.0, 210.0, 330.0, 460.0],
            "pred_cum_energy_j": [95.0, 200.0, 310.0, 430.0],
            "strategy_action": ["hold", "hold", "accelerate", "accelerate"],
        })
        strategy_profile = pd.DataFrame({
            "segment": [1, 2],
            "dist_start_m": [0.0, 10.0],
            "dist_end_m": [10.0, 20.0],
            "target_speed_kph": [19.0, 21.0],
            "entry_speed_kph": [18.0, 19.0],
            "speed_delta_kph": [1.0, 2.0],
            "pred_energy_j": [200.0, 230.0],
            "action": ["accelerate", "accelerate"],
        })
        args = Namespace(
            gps="Utsm-2.gpx",
            telemetry="telemetry.csv",
            forward_axis="ax",
            accel_scale=1000.0,
            imu_axis="ax",
            imu_axis_sign=1,
            accel_bias_window_sec=30.0,
            accel_smooth_window_sec=8.0,
        )
        payload = make_payload(
            df,
            strategy_profile,
            "=== Speed Strategy Report ===",
            4.0,
            args,
        )
        self.assertIn("targetSpeed", payload["metrics"])
        self.assertIn("strategy", payload)
        self.assertEqual(payload["samples"][0]["targetSpeed"], 19.0)
        self.assertEqual(payload["samples"][0]["predRunEnergyJ"], 95.0)
        self.assertEqual(payload["samples"][2]["strategyAction"], "accelerate")
        self.assertEqual(len(payload["strategy"]["segments"]), 2)


if __name__ == "__main__":
    unittest.main()
