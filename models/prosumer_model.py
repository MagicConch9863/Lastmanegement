from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List

import gurobipy as gp
from gurobipy import GRB
import numpy as np


@dataclass
class ProsumerResult:
    p_grid_kw: List[float]
    p_grid_import_kw: List[float]
    p_grid_export_kw: List[float]
    p_bat_kw: List[float]
    energy_kwh: List[float]
    eps_soc_kwh: List[float]
    eps_bat_kw: List[float]
    bill_cost: float
    battery_cost: float
    slack_cost: float
    objective_value: float


def _to_1d_array(x, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    return arr


def solve_prosumer_problem(
    load_kw,
    pv_kw,
    lambda_price_eur_per_kwh,
    battery_capacity_kwh,
    battery_pmax_kw,
    energy_init_kwh,
    energy_min_kwh,
    energy_max_kwh,
    dt_hours,
    battery_cycle_cost_eur_per_kwh2=0.01,
    soc_tracking_cost_eur_per_kwh2=0.01,
    soc_ref_kwh: float | None = None,
    soc_slack_penalty_eur_per_kwh2=1000.0,
    bat_slack_penalty_eur_per_kw2=1000.0,
    sell_price_ratio=1.0 / 3.0,
    solver_output_flag=0,
    enforce_terminal_soc=True,
):
    load = _to_1d_array(load_kw, "load_kw")
    pv = _to_1d_array(pv_kw, "pv_kw")
    price_buy = _to_1d_array(lambda_price_eur_per_kwh, "lambda_price_eur_per_kwh")

    T = len(load)
    if len(pv) != T or len(price_buy) != T:
        raise ValueError("load_kw, pv_kw, and lambda_price_eur_per_kwh must have the same length.")

    if dt_hours <= 0.0:
        raise ValueError("dt_hours must be positive.")
    if sell_price_ratio < 0.0:
        raise ValueError("sell_price_ratio must be nonnegative.")

    if soc_ref_kwh is None:
        soc_ref_kwh = energy_init_kwh

    price_sell = sell_price_ratio * price_buy

    model = gp.Model("prosumer_problem")
    model.Params.OutputFlag = solver_output_flag

    p_grid = model.addVars(T, lb=-GRB.INFINITY, name="p_grid_kw")
    l_grid_plus = model.addVars(T, lb=0.0, name="l_grid_plus_kw")
    l_grid_minus = model.addVars(T, lb=0.0, name="l_grid_minus_kw")

    p_bat = model.addVars(T, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="p_bat_kw")
    energy = model.addVars(T + 1, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="energy_kwh")

    eps_soc = model.addVars(T + 1, lb=0.0, name="eps_soc_kwh")
    eps_bat = model.addVars(T, lb=0.0, name="eps_bat_kw")

    model.addConstr(energy[0] == energy_init_kwh, name="energy_init")

    for t in range(T):
        model.addConstr(
            p_grid[t] == load[t] + p_bat[t] - pv[t],
            name=f"power_balance_{t}",
        )
        model.addConstr(
            p_grid[t] == l_grid_plus[t] - l_grid_minus[t],
            name=f"grid_split_{t}",
        )
        model.addConstr(
            energy[t + 1] == energy[t] + p_bat[t] * dt_hours,
            name=f"energy_dyn_{t}",
        )
        model.addConstr(
            p_bat[t] >= -battery_pmax_kw - eps_bat[t],
            name=f"pbat_lb_{t}",
        )
        model.addConstr(
            p_bat[t] <= battery_pmax_kw + eps_bat[t],
            name=f"pbat_ub_{t}",
        )

    for t in range(T + 1):
        model.addConstr(
            energy[t] >= energy_min_kwh - eps_soc[t],
            name=f"soc_lb_{t}",
        )
        model.addConstr(
            energy[t] <= energy_max_kwh + eps_soc[t],
            name=f"soc_ub_{t}",
        )

    if enforce_terminal_soc and battery_pmax_kw > 0.0:
        model.addConstr(energy[T] == energy_init_kwh, name="terminal_energy")

    bill_cost_expr = gp.quicksum(
        (price_buy[t] * l_grid_plus[t] - price_sell[t] * l_grid_minus[t]) * dt_hours
        for t in range(T)
    )

    battery_cost_expr = gp.quicksum(
        battery_cycle_cost_eur_per_kwh2 * (p_bat[t] * p_bat[t]) * dt_hours
        + soc_tracking_cost_eur_per_kwh2 * ((energy[t] - soc_ref_kwh) * (energy[t] - soc_ref_kwh))
        for t in range(T)
    )

    slack_cost_expr = gp.quicksum(
        soc_slack_penalty_eur_per_kwh2 * (eps_soc[t] * eps_soc[t])
        for t in range(T + 1)
    ) + gp.quicksum(
        bat_slack_penalty_eur_per_kw2 * (eps_bat[t] * eps_bat[t])
        for t in range(T)
    )

    objective = bill_cost_expr + battery_cost_expr + slack_cost_expr

    model.setObjective(objective, GRB.MINIMIZE)
    model.optimize()

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Prosumer optimization failed. Status={model.Status}")

    return ProsumerResult(
        p_grid_kw=[float(p_grid[t].X) for t in range(T)],
        p_grid_import_kw=[float(l_grid_plus[t].X) for t in range(T)],
        p_grid_export_kw=[float(l_grid_minus[t].X) for t in range(T)],
        p_bat_kw=[float(p_bat[t].X) for t in range(T)],
        energy_kwh=[float(energy[t].X) for t in range(T + 1)],
        eps_soc_kwh=[float(eps_soc[t].X) for t in range(T + 1)],
        eps_bat_kw=[float(eps_bat[t].X) for t in range(T)],
        bill_cost=float(bill_cost_expr.getValue()),
        battery_cost=float(battery_cost_expr.getValue()),
        slack_cost=float(slack_cost_expr.getValue()),
        objective_value=float(model.ObjVal),
    )


def solve_all_prosumers(node_data: Dict[str, Any], leader_price, cfg) -> Dict[int, ProsumerResult]:
    results: Dict[int, ProsumerResult] = {}

    for bus_id in node_data["bus_ids"]:
        has_battery = bool(node_data["has_battery"][bus_id])

        results[bus_id] = solve_prosumer_problem(
            load_kw=node_data["load_kw"][bus_id],
            pv_kw=node_data["pv_kw"][bus_id],
            lambda_price_eur_per_kwh=leader_price,
            battery_capacity_kwh=node_data["battery_capacity_kwh"][bus_id] if has_battery else 0.0,
            battery_pmax_kw=node_data["battery_pmax_kw"][bus_id] if has_battery else 0.0,
            energy_init_kwh=node_data["energy_init_kwh"][bus_id] if has_battery else 0.0,
            energy_min_kwh=cfg.soc_min_kwh if has_battery else 0.0,
            energy_max_kwh=cfg.soc_max_kwh if has_battery else 0.0,
            dt_hours=cfg.time_step_hours,
            battery_cycle_cost_eur_per_kwh2=cfg.battery_cycle_cost_eur_per_kwh2,
            soc_tracking_cost_eur_per_kwh2=cfg.soc_tracking_cost_eur_per_kwh2,
            soc_ref_kwh=cfg.soc_ref_kwh if has_battery else 0.0,
            soc_slack_penalty_eur_per_kwh2=cfg.soc_slack_penalty_eur_per_kwh2,
            bat_slack_penalty_eur_per_kw2=cfg.bat_slack_penalty_eur_per_kw2,
            sell_price_ratio=cfg.sell_price_ratio,
            solver_output_flag=cfg.gurobi_output_flag,
            enforce_terminal_soc=cfg.enforce_terminal_soc,
        )

    return results