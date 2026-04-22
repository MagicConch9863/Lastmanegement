from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from configs.stackelberg_config import StackelbergConfig
from models.exact_stackelberg_18bus import solve_exact_stackelberg_18bus


def compute_exact_metrics(result: dict, time_step_hours: float) -> dict:
    optimized_pf = result["pf_results"]
    lref = float(result["lref_kw"])

    after = np.asarray(optimized_pf["grid_import_from_ext_grid_kw"], dtype=float)

    peak_after = float(np.max(after))
    valley_after = float(np.min(after))
    std_after = float(np.std(after))

    # 用 Lref 作为参考水平，不再显示 baseline 曲线
    valley_gap_after = max(lref - valley_after, 0.0)

    after_energy_kwh = float(np.sum(after) * time_step_hours)

    return {
        "lref_kw": lref,
        "peak_after_kw": peak_after,
        "valley_after_kw": valley_after,
        "std_after_kw": std_after,
        "valley_gap_after_kw": valley_gap_after,
        "after_energy_kwh": after_energy_kwh,
    }


def load_exact_results(output_dir: str = "outputs") -> tuple[pd.DataFrame, pd.DataFrame]:
    hourly_path = Path(output_dir) / "stackelberg_exact_hourly_summary.csv"
    summary_path = Path(output_dir) / "stackelberg_exact_summary.csv"

    if not hourly_path.exists():
        raise FileNotFoundError(f"Missing file: {hourly_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing file: {summary_path}")

    hourly_df = pd.read_csv(hourly_path)
    summary_df = pd.read_csv(summary_path)
    return hourly_df, summary_df


def plot_exact_stackelberg_dashboard(result: dict, output_dir: str = "outputs", show_plot: bool = True) -> None:
    hourly_df, summary_df = load_exact_results(output_dir)

    optimized_pf = result["pf_results"]
    after = np.asarray(optimized_pf["grid_import_from_ext_grid_kw"], dtype=float)

    if "real_price_eur_per_kwh" not in hourly_df.columns:
        raise KeyError("Column 'real_price_eur_per_kwh' not found in stackelberg_exact_hourly_summary.csv")
    if "leader_price_eur_per_kwh" not in hourly_df.columns:
        raise KeyError("Column 'leader_price_eur_per_kwh' not found in stackelberg_exact_hourly_summary.csv")

    real_price = hourly_df["real_price_eur_per_kwh"].to_numpy(dtype=float)
    leader_price = hourly_df["leader_price_eur_per_kwh"].to_numpy(dtype=float)

    agg_load = None
    if "L_aggregate_kw" in hourly_df.columns:
        agg_load = hourly_df["L_aggregate_kw"].to_numpy(dtype=float)

    eps_grid = hourly_df["eps_grid_kw"].to_numpy(dtype=float) if "eps_grid_kw" in hourly_df.columns else None
    pcc_import = (
        hourly_df["pcc_import_kw_ex_post"].to_numpy(dtype=float)
        if "pcc_import_kw_ex_post" in hourly_df.columns
        else after
    )
    pcc_export = (
        hourly_df["pcc_export_kw_ex_post"].to_numpy(dtype=float)
        if "pcc_export_kw_ex_post" in hourly_df.columns
        else None
    )

    T = len(after)
    t = np.arange(T)

    time_step_hours = float(getattr(result.get("cfg", None), "time_step_hours", 0.25))
    metrics = compute_exact_metrics(result=result, time_step_hours=time_step_hours)

    lref = metrics["lref_kw"]

    obj = float(summary_df.loc[0, "objective_value"]) if "objective_value" in summary_df.columns else None
    mip_gap = float(summary_df.loc[0, "mip_gap"]) if "mip_gap" in summary_df.columns else None
    status = summary_df.loc[0, "solver_status"] if "solver_status" in summary_df.columns else None
    solve_time = float(summary_df.loc[0, "solve_time_sec"]) if "solve_time_sec" in summary_df.columns else None

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    fig.suptitle("Exact Stackelberg Dashboard", fontsize=16)

    # 左上：只画 optimized import 和 Lref，不再画 baseline
    ax = axes[0, 0]
    ax.plot(t, after, linewidth=2.8, label="Optimized total import")
    ax.axhline(lref, linestyle=":", linewidth=2.0, label="Lref")
    ax.set_title("Optimized PCC Import")
    ax.set_ylabel("Power (kW)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    info = (
        f"Peak import: {metrics['peak_after_kw']:.3f} kW\n"
        f"Valley import: {metrics['valley_after_kw']:.3f} kW\n"
        f"Std of import: {metrics['std_after_kw']:.3f} kW\n"
        f"Daily import energy: {metrics['after_energy_kwh']:.3f} kWh"
    )
    ax.text(
        0.02,
        0.98,
        info,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        fontsize=10,
    )

    # 右上：价格
    ax = axes[0, 1]
    ax.plot(t, real_price[:T], linewidth=2.0, label="Real market price")
    ax.plot(t, leader_price[:T], linewidth=2.0, label="Leader price")
    ax.set_title("Price Trajectories")
    ax.set_ylabel("Price (EUR/kWh)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    summary_lines = []
    if obj is not None:
        summary_lines.append(f"Objective: {obj:.6f}")
    summary_lines.append(f"Lref: {lref:.3f} kW")
    if mip_gap is not None and not np.isnan(mip_gap):
        summary_lines.append(f"MIP gap: {100.0 * mip_gap:.3f}%")
    if status is not None:
        summary_lines.append(f"Solver status: {status}")
    if solve_time is not None:
        summary_lines.append(f"Solve time: {solve_time:.1f}s")

    ax.text(
        0.98,
        0.02,
        "\n".join(summary_lines),
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        fontsize=10,
    )

    # 左下：PCC import/export
    ax = axes[1, 0]
    ax.plot(t, pcc_import[:T], linewidth=2.2, label="PCC import (ex-post)")
    if pcc_export is not None:
        ax.plot(t, pcc_export[:T], linewidth=2.0, label="PCC export (ex-post)")
    ax.set_title("Ex-post PCC Exchange")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Power (kW)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # 右下：Aggregate load 和 eps_grid
    ax = axes[1, 1]
    if agg_load is not None:
        ax.plot(t, agg_load[:T], linewidth=2.2, label="Aggregate load L_t")
    ax.axhline(lref, linestyle="--", linewidth=1.8, label="Lref")
    if eps_grid is not None:
        ax.plot(t, eps_grid[:T], linewidth=1.8, label="Epsilon_grid")
    ax.set_title("Aggregate Load and Grid Slack")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Power (kW)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = Path(output_dir) / "exact_stackelberg_dashboard.png"
    fig.savefig(save_path, dpi=250, bbox_inches="tight")
    print(f"Saved: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def main():
    cfg = StackelbergConfig()

    cfg.gurobi_output_flag = 1
    cfg.exact_horizon = 5
    cfg.exact_stackelberg_time_limit_sec = 3600
    cfg.exact_stackelberg_mip_gap = 1e-4

    cfg.validate()

    result = solve_exact_stackelberg_18bus(cfg)
    result["cfg"] = cfg

    print("\n" + "=" * 70)
    print("Exact Stackelberg 18-bus summary")
    print("=" * 70)
    print(f"Objective value       : {result['objective_value']:.6f}")
    print(f"Lref                  : {result['lref_kw']:.6f} kW")
    print(f"Leader price length   : {len(result['leader_price'])}")
    print(f"Solver status         : {result['solver_status']}")
    print("=" * 70)

    plot_exact_stackelberg_dashboard(
        result=result,
        output_dir=cfg.output_dir,
        show_plot=True,
    )


if __name__ == "__main__":
    main()