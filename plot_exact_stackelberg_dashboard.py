from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
    output_dir: str = "outputs",
    show_plot: bool = True,
) -> None:
    hourly_df, summary_df = load_exact_results(output_dir)

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
    baseline_import = _safe_col(hourly_df, "baseline_pcc_import_kw", default=None)

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

    fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
    fig.suptitle("Exact Stackelberg Dashboard", fontsize=18)

    # 1. Optimized PCC Import
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
    price_all = np.concatenate([real_price, leader_price])
    _set_axis_with_margin(ax, price_all)
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

    # 3. Ex-post PCC Exchange with dual y-axes
    ax_left = axes[1, 0]
    ax_right = ax_left.twinx()

    line1 = ax_left.plot(t, pcc_import, linewidth=2.8, label="PCC import (left axis)")
    line2 = ax_right.plot(t, pcc_export, linewidth=2.2, label="PCC export (right axis)")

    _set_axis_with_margin(ax_left, pcc_import, ratio=0.15, min_margin=0.02)
    _set_axis_with_margin(ax_right, pcc_export, ratio=0.25, min_margin=0.02)

    # 如果 export 基本为零，仍然人为给右轴一个小范围，避免线完全贴边
    if np.max(np.abs(pcc_export)) < 1e-6:
        ax_right.set_ylim(-0.1, 0.1)

    ax_left.set_title("Ex-post PCC Exchange (Dual Axes)")
    ax_left.set_ylabel("Import Power (kW)")
    ax_right.set_ylabel("Export Power (kW)")
    ax_left.grid(True, alpha=0.3)

    # 合并图例
    lines = line1 + line2
    labels = [ln.get_label() for ln in lines]
    ax_left.legend(lines, labels, loc="best")

    # 均值线，帮助视觉区分
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
    data_all = np.concatenate([aggregate_load, eps_grid, np.array([lref]) if lref is not None else np.array([])])
    _set_axis_with_margin(ax, data_all)
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
    compare_data = np.concatenate([aggregate_load, pcc_import])
    _set_axis_with_margin(ax, compare_data)
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
    plot_exact_stackelberg_dashboard_all(output_dir="outputs", show_plot=True)


if __name__ == "__main__":
    main()