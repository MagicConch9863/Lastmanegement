from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from configs.stackelberg_config import StackelbergConfig
from models.exact_stackelberg_18bus import solve_exact_stackelberg_18bus
from networks.plot_network_layout import plot_network_layout


def compute_exact_metrics(result: dict, time_step_hours: float) -> dict:
    optimized_pf = result["pf_results"]
    lref = float(result["lref_kw"])

    after = np.asarray(optimized_pf["grid_import_from_ext_grid_kw"], dtype=float)

    peak_after = float(np.max(after))
    valley_after = float(np.min(after))
    std_after = float(np.std(after))
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


def _safe_col(df: pd.DataFrame, name: str, default=None):
    return df[name].to_numpy(dtype=float) if name in df.columns else default


def _set_axis_with_margin(ax, data: np.ndarray, ratio: float = 0.12, min_margin: float = 1e-3):
    data = np.asarray(data, dtype=float)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return
    dmin = float(np.min(finite))
    dmax = float(np.max(finite))
    span = dmax - dmin
    margin = max(span * ratio, min_margin)
    if span < 1e-12:
        center = 0.5 * (dmax + dmin)
        ax.set_ylim(center - margin, center + margin)
    else:
        ax.set_ylim(dmin - margin, dmax + margin)


def _format_summary_box(summary_df: pd.DataFrame) -> str:
    lines = []
    if "objective_value" in summary_df.columns:
        lines.append(f"Objective: {float(summary_df.loc[0, 'objective_value']):.6f}")
    if "lref_kw" in summary_df.columns:
        lines.append(f"Lref: {float(summary_df.loc[0, 'lref_kw']):.3f} kW")
    if "mip_gap" in summary_df.columns:
        gap = float(summary_df.loc[0, "mip_gap"])
        if np.isfinite(gap):
            lines.append(f"MIP gap: {100.0 * gap:.3f}%")
    if "solver_status" in summary_df.columns:
        lines.append(f"Solver status: {summary_df.loc[0, 'solver_status']}")
    if "solve_time_sec" in summary_df.columns:
        lines.append(f"Solve time: {float(summary_df.loc[0, 'solve_time_sec']):.1f}s")
    return "\n".join(lines)


def plot_exact_stackelberg_dashboard_all(
    result: dict,
    output_dir: str = "outputs",
    show_plot: bool = True,
) -> None:
    hourly_df, summary_df = load_exact_results(output_dir)

    optimized_pf = result["pf_results"]
    after = np.asarray(optimized_pf["grid_import_from_ext_grid_kw"], dtype=float)

    if "t" in hourly_df.columns:
        t = hourly_df["t"].to_numpy(dtype=float)
    else:
        t = np.arange(len(hourly_df), dtype=float)

    real_price = _safe_col(hourly_df, "real_price_eur_per_kwh")
    leader_price = _safe_col(hourly_df, "leader_price_eur_per_kwh")
    aggregate_load = _safe_col(hourly_df, "L_aggregate_kw")
    eps_grid = _safe_col(hourly_df, "eps_grid_kw")
    pcc_import = _safe_col(hourly_df, "pcc_import_kw_ex_post")
    pcc_export = _safe_col(hourly_df, "pcc_export_kw_ex_post")

    if real_price is None or leader_price is None:
        raise KeyError("Columns 'real_price_eur_per_kwh' and 'leader_price_eur_per_kwh' are required.")
    if aggregate_load is None:
        raise KeyError("Column 'L_aggregate_kw' is required.")
    if pcc_import is None:
        raise KeyError("Column 'pcc_import_kw_ex_post' is required.")
    if eps_grid is None:
        eps_grid = np.zeros_like(aggregate_load)
    if pcc_export is None:
        pcc_export = np.zeros_like(pcc_import)

    lref = None
    if "lref_kw" in summary_df.columns:
        lref = float(summary_df.loc[0, "lref_kw"])

    time_step_hours = float(getattr(result.get("cfg", None), "time_step_hours", 0.25))
    metrics = compute_exact_metrics(result=result, time_step_hours=time_step_hours)

    fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
    fig.suptitle("Exact Stackelberg Dashboard", fontsize=18)

    # 1. Optimized PCC import
    ax = axes[0, 0]
    ax.plot(t, pcc_import, linewidth=2.5, label="Optimized PCC import")
    if lref is not None:
        ax.axhline(lref, linestyle=":", linewidth=2.0, label="Lref")
    _set_axis_with_margin(ax, pcc_import)
    ax.set_title("Optimized PCC Import")
    ax.set_ylabel("Power (kW)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    info_lines = []
    info_lines.append(f"Peak import: {np.max(pcc_import):.3f} kW")
    info_lines.append(f"Valley import: {np.min(pcc_import):.3f} kW")
    info_lines.append(f"Std of import: {np.std(pcc_import):.3f} kW")
    info_lines.append(f"Mean import: {np.mean(pcc_import):.3f} kW")
    ax.text(
        0.02,
        0.98,
        "\n".join(info_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        fontsize=10,
    )

    # 2. Price trajectories
    ax = axes[0, 1]
    ax.plot(t, real_price, linewidth=2.2, label="Real market price")
    ax.plot(t, leader_price, linewidth=2.2, label="Leader price")
    _set_axis_with_margin(ax, np.concatenate([real_price, leader_price]))
    ax.set_title("Price Trajectories")
    ax.set_ylabel("Price (EUR/kWh)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    summary_text = _format_summary_box(summary_df)
    if summary_text:
        ax.text(
            0.98,
            0.02,
            summary_text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
            fontsize=10,
        )

    # 3. Ex-post PCC exchange with dual axes
    ax_left = axes[1, 0]
    ax_right = ax_left.twinx()

    line1 = ax_left.plot(t, pcc_import, linewidth=2.4, label="PCC import (left axis)")
    line2 = ax_right.plot(t, pcc_export, linewidth=2.2, label="PCC export (right axis)")

    _set_axis_with_margin(ax_left, pcc_import, ratio=0.15, min_margin=0.02)
    _set_axis_with_margin(ax_right, pcc_export, ratio=0.25, min_margin=0.02)

    if np.max(np.abs(pcc_export)) < 1e-6:
        ax_right.set_ylim(-0.1, 0.1)

    ax_left.set_title("Ex-post PCC Exchange (Dual Axes)")
    ax_left.set_ylabel("Import Power (kW)")
    ax_right.set_ylabel("Export Power (kW)")
    ax_left.grid(True, alpha=0.3)

    lines = line1 + line2
    labels = [ln.get_label() for ln in lines]
    ax_left.legend(lines, labels, loc="best")

    imp_mean = float(np.mean(pcc_import))
    exp_mean = float(np.mean(pcc_export))
    ax_left.axhline(imp_mean, linestyle="--", linewidth=1.2, alpha=0.8)
    ax_right.axhline(exp_mean, linestyle="--", linewidth=1.2, alpha=0.8)

    # 4. Aggregate load and grid slack
    ax = axes[1, 1]
    ax.plot(t, aggregate_load, linewidth=2.4, label="Aggregate load $L_t$")
    if lref is not None:
        ax.axhline(lref, linestyle="--", linewidth=1.8, label="Lref")
    ax.plot(t, eps_grid, linewidth=2.0, label="Epsilon_grid")
    compare = [aggregate_load, eps_grid]
    if lref is not None:
        compare.append(np.array([lref]))
    _set_axis_with_margin(ax, np.concatenate(compare))
    ax.set_title("Aggregate Load and Grid Slack")
    ax.set_ylabel("Power (kW)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # 5. Price deviation from market
    ax = axes[2, 0]
    price_gap = leader_price - real_price
    ax.plot(t, price_gap, linewidth=2.2, label="Leader price - Real price")
    ax.axhline(0.0, linestyle="--", linewidth=1.3)
    _set_axis_with_margin(ax, price_gap, ratio=0.18, min_margin=0.005)
    ax.set_title("Price Deviation from Market")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Deviation (EUR/kWh)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # 6. Aggregate vs ex-post PCC import
    ax = axes[2, 1]
    ax.plot(t, aggregate_load, linewidth=2.3, label="Aggregate load $L_t$")
    ax.plot(t, pcc_import, linewidth=2.3, label="Ex-post PCC import")
    _set_axis_with_margin(ax, np.concatenate([aggregate_load, pcc_import]))
    ax.set_title("Aggregate Load vs Ex-post PCC Import")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Power (kW)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    plt.tight_layout(rect=[0, 0, 1, 0.965])

    save_path = Path(output_dir) / "exact_stackelberg_dashboard_all.png"
    fig.savefig(save_path, dpi=250, bbox_inches="tight")
    print(f"Saved: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def main():
    cfg = StackelbergConfig()

    # 只想测试绘图/小规模 exact 时，在这里改 bus 选择
    cfg.selected_residential_bus_names = [
        "Bus R1", "Bus R3", "Bus R6", "Bus R9", "Bus R12", "Bus R15"
    ]
    cfg.n_passive_load = 2
    cfg.n_pv_only = 2
    cfg.n_pv_battery = 2

    # exact 求解建议先小规模
    cfg.gurobi_output_flag = 1
    cfg.exact_horizon = 12
    cfg.exact_stackelberg_time_limit_sec = 18000
    cfg.exact_stackelberg_mip_gap = 1e-4

    # 若只想先看布局图，把这个开关改成 True
    preview_layout_only = False

    cfg.validate()

    if preview_layout_only:
        plot_network_layout(
            cfg=cfg,
            save_path=Path(cfg.output_dir) / "test_selected_bus_layout.png",
            show_plot=True,
        )
        return

    result = solve_exact_stackelberg_18bus(cfg)
    result["cfg"] = cfg

    print("\n" + "=" * 70)
    print("Exact Stackelberg 18-bus summary")
    print("=" * 70)
    print(f"Selected buses        : {cfg.selected_residential_bus_names}")
    print(f"Objective value       : {result['objective_value']:.6f}")
    print(f"Lref                  : {result['lref_kw']:.6f} kW")
    print(f"Leader price length   : {len(result['leader_price'])}")
    print(f"Solver status         : {result['solver_status']}")
    print("=" * 70)

    plot_exact_stackelberg_dashboard_all(
        result=result,
        output_dir=cfg.output_dir,
        show_plot=True,
    )


if __name__ == "__main__":
    main()