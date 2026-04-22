from __future__ import annotations

import os
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from configs.stackelberg_config import StackelbergConfig
from models.leader_problem import (
    evaluate_given_price,
    initialize_leader_price,
    solve_leader_problem,
)
from networks.plot_network_layout import plot_network_layout


def build_iteration_dataframe(history) -> pd.DataFrame:
    rows = []
    for item in history:
        rows.append(
            {
                "iteration": item["iteration"],
                "leader_objective": item["leader_metrics"]["leader_objective"],
                "peak_penalty": item["leader_metrics"]["peak_penalty"],
                "track_penalty": item["leader_metrics"]["track_penalty"],
                "price_magnitude_penalty": item["leader_metrics"]["price_magnitude_penalty"],
                "price_dynamic_penalty": item["leader_metrics"]["price_dynamic_penalty"],
                "security_penalty": item["leader_metrics"]["security_penalty"],
                "grid_capacity_penalty_raw": item["network_penalty"]["grid_capacity_penalty"],
                "voltage_penalty_raw": item["network_penalty"]["voltage_penalty"],
                "line_penalty_raw": item["network_penalty"]["line_penalty"],
                "trafo_penalty_raw": item["network_penalty"]["trafo_penalty"],
                "max_price_change": item["max_price_change"],
            }
        )
    return pd.DataFrame(rows)


def compute_comparison_metrics(
    baseline_result,
    best_result,
    cfg: StackelbergConfig,
) -> Dict[str, float]:
    before = np.asarray(
        baseline_result["pf_results"]["grid_import_from_ext_grid_kw"],
        dtype=float,
    )
    after = np.asarray(
        best_result["pf_results"]["grid_import_from_ext_grid_kw"],
        dtype=float,
    )

    lref = float(cfg.target_import_kw)

    peak_before = float(np.max(before))
    peak_after = float(np.max(after))

    valley_before = float(np.min(before))
    valley_after = float(np.min(after))

    std_before = float(np.std(before))
    std_after = float(np.std(after))

    peak_shaving_pct = 0.0
    if peak_before > 1e-9:
        peak_shaving_pct = 100.0 * (peak_before - peak_after) / peak_before

    valley_gap_before = max(lref - valley_before, 0.0)
    valley_gap_after = max(lref - valley_after, 0.0)
    valley_filling_pct = 0.0
    if valley_gap_before > 1e-9:
        valley_filling_pct = 100.0 * (valley_gap_before - valley_gap_after) / valley_gap_before

    fluctuation_reduction_pct = 0.0
    if std_before > 1e-9:
        fluctuation_reduction_pct = 100.0 * (std_before - std_after) / std_before

    before_energy_kwh = float(
        np.sum(np.asarray(baseline_result["aggregate"]["aggregate_grid_kw"], dtype=float)) * cfg.time_step_hours
    )
    after_energy_kwh = float(
        np.sum(np.asarray(best_result["aggregate"]["aggregate_grid_kw"], dtype=float)) * cfg.time_step_hours
    )

    energy_diff_kwh = after_energy_kwh - before_energy_kwh
    energy_diff_pct = 0.0
    if abs(before_energy_kwh) > 1e-9:
        energy_diff_pct = 100.0 * energy_diff_kwh / before_energy_kwh

    mse_before = float(np.mean((before - lref) ** 2))
    mse_after = float(np.mean((after - lref) ** 2))

    return {
        "lref_kw": lref,
        "peak_before_kw": peak_before,
        "peak_after_kw": peak_after,
        "peak_shaving_pct": peak_shaving_pct,
        "valley_before_kw": valley_before,
        "valley_after_kw": valley_after,
        "valley_filling_pct": valley_filling_pct,
        "std_before_kw": std_before,
        "std_after_kw": std_after,
        "fluctuation_reduction_pct": fluctuation_reduction_pct,
        "before_energy_kwh": before_energy_kwh,
        "after_energy_kwh": after_energy_kwh,
        "energy_diff_kwh": energy_diff_kwh,
        "energy_diff_pct": energy_diff_pct,
        "mse_before_to_lref": mse_before,
        "mse_after_to_lref": mse_after,
    }


def save_outputs(
    history,
    baseline_result,
    best_result,
    metrics,
    cfg: StackelbergConfig,
) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)

    iteration_df = build_iteration_dataframe(history)
    iteration_df.to_csv(
        os.path.join(cfg.output_dir, "stackelberg_iteration_history.csv"),
        index=False,
    )

    hourly_df = pd.DataFrame(
        {
            "hour": np.arange(cfg.horizon),
            "real_price_eur_per_kwh": np.asarray(baseline_result["price_signal"], dtype=float),
            "leader_price_eur_per_kwh": np.asarray(best_result["leader_price"], dtype=float),
            "baseline_pcc_import_kw": np.asarray(
                baseline_result["pf_results"]["grid_import_from_ext_grid_kw"], dtype=float
            ),
            "optimized_pcc_import_kw": np.asarray(
                best_result["pf_results"]["grid_import_from_ext_grid_kw"], dtype=float
            ),
            "baseline_aggregate_grid_kw": np.asarray(
                baseline_result["aggregate"]["aggregate_grid_kw"], dtype=float
            ),
            "optimized_aggregate_grid_kw": np.asarray(
                best_result["aggregate"]["aggregate_grid_kw"], dtype=float
            ),
            "baseline_aggregate_battery_kw": np.asarray(
                baseline_result["aggregate"]["aggregate_battery_kw"], dtype=float
            ),
            "optimized_aggregate_battery_kw": np.asarray(
                best_result["aggregate"]["aggregate_battery_kw"], dtype=float
            ),
        }
    )
    hourly_df.to_csv(
        os.path.join(cfg.output_dir, "stackelberg_hourly_summary.csv"),
        index=False,
    )

    pd.DataFrame([metrics]).to_csv(
        os.path.join(cfg.output_dir, "stackelberg_metrics_summary.csv"),
        index=False,
    )


def plot_main_result_figure(
    baseline_result,
    best_result,
    metrics,
    cfg: StackelbergConfig,
) -> None:
    t = np.arange(cfg.horizon)

    before = np.asarray(
        baseline_result["pf_results"]["grid_import_from_ext_grid_kw"],
        dtype=float,
    )
    after = np.asarray(
        best_result["pf_results"]["grid_import_from_ext_grid_kw"],
        dtype=float,
    )

    real_price = np.asarray(baseline_result["price_signal"], dtype=float)
    leader_price = np.asarray(best_result["leader_price"], dtype=float)

    lref = metrics["lref_kw"]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.5]},
    )

    ax1 = axes[0]
    ax1.plot(t, before, linestyle="--", linewidth=2.2, label="Baseline total import")
    ax1.plot(t, after, linewidth=2.8, label="Optimized total import")
    ax1.axhline(lref, linestyle=":", linewidth=2.0, label="Lref")
    ax1.set_ylabel("PCC Import (kW)")
    ax1.set_title("Stackelberg Result")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    info = (
        f"Peak shaving: {metrics['peak_shaving_pct']:.1f}%\n"
        f"Valley filling: {metrics['valley_filling_pct']:.1f}%\n"
        f"Fluctuation reduction: {metrics['fluctuation_reduction_pct']:.1f}%\n"
        f"Daily energy diff: {metrics['energy_diff_pct']:.2f}%"
    )
    ax1.text(
        0.02,
        0.98,
        info,
        transform=ax1.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        fontsize=10,
    )

    ax2 = axes[1]
    ax2.plot(t, real_price, linewidth=2.0, label="Real market price")
    ax2.plot(t, leader_price, linewidth=2.0, label="Leader price")
    ax2.set_xlabel("Time step")
    ax2.set_ylabel("Price (EUR/kWh)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    plt.tight_layout()

    if cfg.save_plots:
        os.makedirs(cfg.output_dir, exist_ok=True)
        fig.savefig(
            os.path.join(cfg.output_dir, "main_result_price_network.png"),
            dpi=250,
            bbox_inches="tight",
        )

    plt.show()


def print_run_summary(
    history,
    best_result,
    metrics,
    cfg: StackelbergConfig,
) -> None:
    lm = best_result["leader_metrics"]
    npen = best_result["network_penalty"]

    print("\n" + "=" * 70)
    print("Stackelberg simulation summary")
    print("=" * 70)
    print(f"Iterations used                 : {len(history)}")
    print(f"Best leader objective           : {lm['leader_objective']:.6f}")
    print(f"Peak penalty                    : {lm['peak_penalty']:.6f}")
    print(f"Track penalty                   : {lm['track_penalty']:.6f}")
    print(f"Price magnitude penalty         : {lm['price_magnitude_penalty']:.6f}")
    print(f"Price dynamic penalty           : {lm['price_dynamic_penalty']:.6f}")
    print(f"Security penalty                : {lm['security_penalty']:.6f}")
    print(f"Raw grid-capacity penalty       : {npen['grid_capacity_penalty']:.6f}")
    print(f"Lref (daily average baseline)   : {metrics['lref_kw']:.3f} kW")
    print(f"Peak shaving                    : {metrics['peak_shaving_pct']:.2f}%")
    print(f"Valley filling                  : {metrics['valley_filling_pct']:.2f}%")
    print(f"Fluctuation reduction           : {metrics['fluctuation_reduction_pct']:.2f}%")
    print(f"Daily net energy difference     : {metrics['energy_diff_pct']:.3f}%")
    print(f"MSE to Lref before              : {metrics['mse_before_to_lref']:.3f}")
    print(f"MSE to Lref after               : {metrics['mse_after_to_lref']:.3f}")
    print("=" * 70)


def run_stackelberg_simulation(cfg: StackelbergConfig) -> Dict[str, Any]:
    cfg.validate()

    net, coords, groups, node_data, device_df, bus_map = plot_network_layout(
        cfg=cfg,
        save_path=os.path.join(cfg.output_dir, "network_layout.png"),
        show_plot=False,
    )

    real_price_signal = np.asarray(node_data["leader_price_init"], dtype=float)

    # baseline阶段：只算 follower + powerflow + penalty + aggregate
    # 不算 leader objective，因为这时 Lref 还没定义
    baseline_result = evaluate_given_price(
        net=net,
        node_data=node_data,
        price_signal=real_price_signal,
        cfg=cfg,
        compute_leader_metrics=False,
    )

    baseline_import = np.asarray(
        baseline_result["pf_results"]["grid_import_from_ext_grid_kw"],
        dtype=float,
    )

    if cfg.lref_mode == "baseline_mean":
        cfg.target_import_kw = float(np.mean(baseline_import))
    elif cfg.lref_mode == "fixed":
        if cfg.lref_fixed_kw is None:
            raise ValueError("cfg.lref_fixed_kw must be provided when lref_mode='fixed'.")
        cfg.target_import_kw = float(cfg.lref_fixed_kw)
    else:
        raise ValueError(f"Unsupported lref_mode: {cfg.lref_mode}")

    if cfg.debug_mode:
        print(f"\nComputed Lref from baseline PCC import mean: {cfg.target_import_kw:.4f} kW")

    initial_price = initialize_leader_price(cfg, real_price_signal)

    leader_solution = solve_leader_problem(
        net=net,
        node_data=node_data,
        real_price_signal=real_price_signal,
        initial_price=initial_price,
        cfg=cfg,
    )

    history = leader_solution["history"]
    best_result = leader_solution["best_result"]

    metrics = compute_comparison_metrics(
        baseline_result=baseline_result,
        best_result=best_result,
        cfg=cfg,
    )

    save_outputs(
        history=history,
        baseline_result=baseline_result,
        best_result=best_result,
        metrics=metrics,
        cfg=cfg,
    )

    print_run_summary(
        history=history,
        best_result=best_result,
        metrics=metrics,
        cfg=cfg,
    )

    plot_main_result_figure(
        baseline_result=baseline_result,
        best_result=best_result,
        metrics=metrics,
        cfg=cfg,
    )

    return {
        "history": history,
        "best_result": best_result,
        "baseline_result": baseline_result,
        "metrics": metrics,
    }


if __name__ == "__main__":
    cfg = StackelbergConfig()
    run_stackelberg_simulation(cfg)