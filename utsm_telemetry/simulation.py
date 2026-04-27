from __future__ import annotations

import math

import numpy as np
import pandas as pd


def build_full_run_distance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("time").reset_index(drop=True)
    if "run_cumdist_m" not in df.columns:
        df["run_cumdist_m"] = pd.to_numeric(df["dist_m"], errors="coerce").fillna(0.0).cumsum()
    if "cum_energy_j" not in df.columns:
        df["cum_energy_j"] = pd.to_numeric(df["energy_j"], errors="coerce").fillna(0.0).cumsum()
    return df


def build_strategy_segments(df: pd.DataFrame, segments: int) -> pd.DataFrame:
    if segments < 2:
        raise ValueError("segments must be at least 2.")
    df = build_full_run_distance(df)
    total_dist = float(df["run_cumdist_m"].iloc[-1])
    if total_dist <= 0:
        raise ValueError("run_cumdist_m must be positive.")

    edges = np.linspace(0.0, total_dist, segments + 1)
    rows = []
    for idx in range(segments):
        lo = edges[idx]
        hi = edges[idx + 1]
        if idx == segments - 1:
            seg = df[(df["run_cumdist_m"] >= lo) & (df["run_cumdist_m"] <= hi)].copy()
        else:
            seg = df[(df["run_cumdist_m"] >= lo) & (df["run_cumdist_m"] < hi)].copy()
        if seg.empty:
            continue
        length_m = hi - lo
        speed = pd.to_numeric(seg["speed_kph"], errors="coerce").dropna()
        grade = pd.to_numeric(seg["grade_pct"], errors="coerce").dropna()
        power = pd.to_numeric(seg["power_w"], errors="coerce").dropna()
        energy_j = pd.to_numeric(seg["energy_j"], errors="coerce").fillna(0.0).sum()
        dt_s = pd.to_numeric(seg["dt_s"], errors="coerce").fillna(0.0).sum()
        rows.append({
            "segment": idx + 1,
            "dist_start_m": lo,
            "dist_end_m": hi,
            "length_m": length_m,
            "baseline_speed_kph": float(speed.median()) if not speed.empty else 0.0,
            "baseline_grade_pct": float(grade.mean()) if not grade.empty else 0.0,
            "baseline_power_w": float(power.mean()) if not power.empty else 0.0,
            "baseline_energy_j": float(energy_j),
            "baseline_time_s": float(dt_s) if dt_s > 0 else _segment_time_s(length_m, float(speed.median()) if not speed.empty else 0.0),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No strategy segments could be built.")
    return out


def fit_empirical_energy_model(df: pd.DataFrame, ridge: float = 1e-3) -> dict[str, float]:
    df = build_full_run_distance(df)
    speed_kph = pd.to_numeric(df["speed_kph"], errors="coerce")
    grade_pct = pd.to_numeric(df["grade_pct"], errors="coerce")
    gps_accel = pd.to_numeric(df["gps_longitudinal_accel_m_s2"], errors="coerce")
    power_w = pd.to_numeric(df["power_w"], errors="coerce")
    mask = speed_kph.notna() & grade_pct.notna() & gps_accel.notna() & power_w.notna()
    fit = pd.DataFrame({
        "speed_kph": speed_kph[mask],
        "grade_pct": grade_pct[mask],
        "gps_accel_m_s2": gps_accel[mask],
        "power_w": power_w[mask],
    })
    if len(fit) < 10:
        raise ValueError("Not enough samples to fit the empirical energy model.")

    speed = fit["speed_kph"].to_numpy(dtype=float)
    grade = fit["grade_pct"].to_numpy(dtype=float)
    accel = fit["gps_accel_m_s2"].to_numpy(dtype=float)
    uphill = np.clip(grade, 0.0, None)
    downhill = np.clip(-grade, 0.0, None)
    accel_pos = np.clip(accel, 0.0, None)
    design = np.column_stack([
        np.ones(len(fit)),
        speed,
        speed ** 2,
        uphill,
        downhill,
        accel_pos,
    ])
    target = fit["power_w"].to_numpy(dtype=float)
    penalty = ridge * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    coeffs = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    return {
        "intercept": float(coeffs[0]),
        "speed_kph": float(coeffs[1]),
        "speed_kph_sq": float(coeffs[2]),
        "uphill_grade_pct": float(coeffs[3]),
        "downhill_grade_pct": float(coeffs[4]),
        "accel_pos_m_s2": float(coeffs[5]),
        "ridge": float(ridge),
    }


def predict_power_w(
    model: dict[str, float],
    speed_kph: float,
    accel_pos_m_s2: float,
    grade_pct: float,
) -> float:
    uphill = max(grade_pct, 0.0)
    downhill = max(-grade_pct, 0.0)
    power = (
        model["intercept"]
        + model["speed_kph"] * speed_kph
        + model["speed_kph_sq"] * (speed_kph ** 2)
        + model["uphill_grade_pct"] * uphill
        + model["downhill_grade_pct"] * downhill
        + model["accel_pos_m_s2"] * max(accel_pos_m_s2, 0.0)
    )
    return max(float(power), 0.0)


def optimize_speed_profile(
    segments_df: pd.DataFrame,
    model: dict[str, float],
    time_budget_sec: float,
    speed_min_kph: float,
    speed_max_kph: float,
    max_delta_kph_per_segment: float,
    speed_step_kph: float = 1.0,
) -> pd.DataFrame:
    if time_budget_sec <= 0:
        raise ValueError("time_budget_sec must be positive.")
    if speed_min_kph <= 0 or speed_max_kph <= speed_min_kph:
        raise ValueError("speed bounds are invalid.")

    candidate_speeds = np.arange(speed_min_kph, speed_max_kph + speed_step_kph * 0.5, speed_step_kph)
    baseline_first_speed = float(segments_df["baseline_speed_kph"].iloc[0])
    initial_speed = float(np.clip(baseline_first_speed, speed_min_kph, speed_max_kph))

    def solve_for_lambda(lambda_time: float) -> pd.DataFrame:
        dp: list[dict[float, tuple[float, float, float | None]]] = []
        for idx, row in segments_df.iterrows():
            length_m = float(row["length_m"])
            grade_pct = float(row["baseline_grade_pct"])
            state_costs: dict[float, tuple[float, float, float | None]] = {}
            for speed in candidate_speeds:
                best: tuple[float, float, float | None] | None = None
                if idx == 0:
                    if abs(speed - initial_speed) > max_delta_kph_per_segment:
                        continue
                    accel_pos = _accel_from_speed_change(initial_speed, speed, length_m)
                    time_s = _segment_time_s(length_m, speed)
                    energy_j = predict_power_w(model, speed, accel_pos, grade_pct) * time_s
                    best = (energy_j + lambda_time * time_s, time_s, None)
                else:
                    for prev_speed, (prev_cost, prev_time, _) in dp[idx - 1].items():
                        if abs(speed - prev_speed) > max_delta_kph_per_segment:
                            continue
                        accel_pos = _accel_from_speed_change(prev_speed, speed, length_m)
                        time_s = _segment_time_s(length_m, speed)
                        energy_j = predict_power_w(model, speed, accel_pos, grade_pct) * time_s
                        total_cost = prev_cost + energy_j + lambda_time * time_s
                        total_time = prev_time + time_s
                        if best is None or total_cost < best[0]:
                            best = (total_cost, total_time, prev_speed)
                if best is not None:
                    state_costs[float(speed)] = best
            if not state_costs:
                raise ValueError("No feasible strategy found under the speed-step constraints.")
            dp.append(state_costs)

        final_speed, (final_cost, final_time, _) = min(
            dp[-1].items(),
            key=lambda item: item[1][0],
        )
        chosen: list[float] = [final_speed]
        for idx in range(len(dp) - 1, 0, -1):
            prev_speed = dp[idx][chosen[-1]][2]
            if prev_speed is None:
                break
            chosen.append(float(prev_speed))
        chosen.reverse()

        out = segments_df.copy().reset_index(drop=True)
        out["target_speed_kph"] = chosen
        out["segment_time_s"] = [
            _segment_time_s(length, speed)
            for length, speed in zip(out["length_m"], out["target_speed_kph"])
        ]
        prev_speeds = [initial_speed] + chosen[:-1]
        out["entry_speed_kph"] = prev_speeds
        out["accel_demand_m_s2"] = [
            _accel_from_speed_change(prev_speed, speed, length)
            for prev_speed, speed, length in zip(prev_speeds, chosen, out["length_m"])
        ]
        out["pred_power_w"] = [
            predict_power_w(model, speed, accel, grade)
            for speed, accel, grade in zip(
                out["target_speed_kph"],
                out["accel_demand_m_s2"],
                out["baseline_grade_pct"],
            )
        ]
        out["pred_energy_j"] = out["pred_power_w"] * out["segment_time_s"]
        out["cum_pred_energy_j"] = out["pred_energy_j"].cumsum()
        out["cum_pred_time_s"] = out["segment_time_s"].cumsum()
        out["speed_delta_kph"] = out["target_speed_kph"] - out["entry_speed_kph"]
        out["action"] = out["speed_delta_kph"].apply(_action_from_speed_delta)
        out.attrs["objective"] = final_cost
        out.attrs["total_time_s"] = float(out["segment_time_s"].sum())
        out.attrs["lambda_time"] = float(lambda_time)
        return out

    low = solve_for_lambda(0.0)
    if low.attrs["total_time_s"] <= time_budget_sec:
        return low

    hi_lambda = 1.0
    hi = solve_for_lambda(hi_lambda)
    while hi.attrs["total_time_s"] > time_budget_sec and hi_lambda < 1e6:
        hi_lambda *= 2.0
        hi = solve_for_lambda(hi_lambda)

    best = hi
    lo_lambda = 0.0
    for _ in range(18):
        mid_lambda = (lo_lambda + hi_lambda) / 2.0
        mid = solve_for_lambda(mid_lambda)
        if mid.attrs["total_time_s"] > time_budget_sec:
            lo_lambda = mid_lambda
        else:
            hi_lambda = mid_lambda
            best = mid
    return best


def build_strategy_samples(df: pd.DataFrame, profile_df: pd.DataFrame) -> pd.DataFrame:
    df = build_full_run_distance(df)
    samples = df.copy()
    segment_edges = profile_df["dist_end_m"].to_numpy(dtype=float)
    segment_index = np.searchsorted(segment_edges, samples["run_cumdist_m"].to_numpy(dtype=float), side="left")
    segment_index = np.clip(segment_index, 0, len(profile_df) - 1)
    mapped = profile_df.iloc[segment_index].reset_index(drop=True)
    samples = samples.reset_index(drop=True)
    samples["segment"] = mapped["segment"]
    samples["target_speed_kph"] = mapped["target_speed_kph"]
    samples["strategy_action"] = mapped["action"]
    samples["pred_power_w"] = mapped["pred_power_w"]
    samples["pred_energy_j"] = pd.to_numeric(samples["dt_s"], errors="coerce").fillna(0.0) * mapped["pred_power_w"]
    samples["pred_cum_energy_j"] = samples["pred_energy_j"].cumsum()
    return samples


def build_strategy_report(
    baseline_df: pd.DataFrame,
    profile_df: pd.DataFrame,
    time_budget_sec: float,
) -> str:
    baseline_df = build_full_run_distance(baseline_df)
    baseline_energy_j = float(pd.to_numeric(baseline_df["energy_j"], errors="coerce").fillna(0.0).sum())
    baseline_time_s = float(pd.to_numeric(baseline_df["dt_s"], errors="coerce").fillna(0.0).sum())
    pred_energy_j = float(profile_df["pred_energy_j"].sum())
    pred_time_s = float(profile_df["segment_time_s"].sum())
    delta_j = pred_energy_j - baseline_energy_j
    delta_pct = (delta_j / baseline_energy_j * 100.0) if baseline_energy_j > 0 else 0.0

    accel_rows = profile_df.nlargest(3, "speed_delta_kph")[["segment", "target_speed_kph", "speed_delta_kph"]]
    coast_rows = profile_df.nsmallest(3, "speed_delta_kph")[["segment", "target_speed_kph", "speed_delta_kph"]]

    lines = [
        "=== Speed Strategy Report ===",
        "",
        f"Time budget: {time_budget_sec:.1f}s",
        f"Baseline run: {baseline_time_s:.1f}s, {baseline_energy_j:.1f} J",
        f"Predicted strategy: {pred_time_s:.1f}s, {pred_energy_j:.1f} J",
        f"Predicted delta: {delta_j:+.1f} J ({delta_pct:+.2f}%)",
        "",
        "Suggested acceleration segments:",
    ]
    for row in accel_rows.itertuples(index=False):
        lines.append(
            f"Segment {int(row.segment)} -> target {row.target_speed_kph:.1f} km/h "
            f"(delta {row.speed_delta_kph:+.1f} km/h)"
        )
    lines.append("")
    lines.append("Suggested coast / bleed-off segments:")
    for row in coast_rows.itertuples(index=False):
        lines.append(
            f"Segment {int(row.segment)} -> target {row.target_speed_kph:.1f} km/h "
            f"(delta {row.speed_delta_kph:+.1f} km/h)"
        )
    return "\n".join(lines)


def _segment_time_s(length_m: float, speed_kph: float) -> float:
    speed_m_s = max(speed_kph / 3.6, 0.1)
    return float(length_m / speed_m_s)


def _accel_from_speed_change(prev_speed_kph: float, speed_kph: float, length_m: float) -> float:
    if length_m <= 0:
        return 0.0
    v0 = max(prev_speed_kph / 3.6, 0.0)
    v1 = max(speed_kph / 3.6, 0.0)
    accel = (v1 * v1 - v0 * v0) / (2.0 * length_m)
    return max(float(accel), 0.0)


def _action_from_speed_delta(delta_kph: float) -> str:
    if delta_kph > 0.5:
        return "accelerate"
    if delta_kph < -0.5:
        return "coast"
    return "hold"
