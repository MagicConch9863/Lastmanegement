from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple

import os
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

from networks.assign_devices import assign_devices
from models.powerflow_interface import run_time_series_powerflow


@dataclass
class ExactProsumerResult:
    p_grid_kw: list[float]
    p_grid_import_kw: list[float]
    p_grid_export_kw: list[float]
    p_bat_kw: list[float]
    energy_kwh: list[float]
    eps_soc_kwh: list[float]
    eps_bat_kw: list[float]
    objective_value: float


def _cfg(cfg, name: str, default):
    return getattr(cfg, name, default)


def _time_horizon(cfg) -> int:
    exact_horizon = getattr(cfg, "exact_horizon", None)
    if exact_horizon is None:
        return int(cfg.horizon)
    return int(exact_horizon)


def _slice_profile(x, T: int) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if len(arr) < T:
        raise ValueError(f"Profile length {len(arr)} < T={T}")
    return arr[:T].copy()


def solve_followers_baseline_qp(node_data: Dict[str, Any], buy_price: np.ndarray, cfg) -> Dict[int, ExactProsumerResult]:
    T = len(buy_price)
    dt = float(cfg.time_step_hours)
    sell_ratio = float(_cfg(cfg, "sell_price_ratio", 1.0 / 3.0))
    sell_price = sell_ratio * buy_price

    c_deg = float(_cfg(cfg, "battery_cycle_cost_eur_per_kwh2", 0.01))
    c_soc = float(_cfg(cfg, "soc_tracking_cost_eur_per_kwh2", 0.01))
    rho_soc = float(_cfg(cfg, "soc_slack_penalty_eur_per_kwh2", 1000.0))
    rho_bat = float(_cfg(cfg, "bat_slack_penalty_eur_per_kw2", 1000.0))
    s_ref = float(_cfg(cfg, "soc_ref_kwh", cfg.soc_init_kwh))

    results: Dict[int, ExactProsumerResult] = {}

    for bus_id in node_data["bus_ids"]:
        load = _slice_profile(node_data["load_kw"][bus_id], T)
        pv = _slice_profile(node_data["pv_kw"][bus_id], T)
        has_bat = bool(node_data["has_battery"][bus_id])

        pmax = float(node_data["battery_pmax_kw"][bus_id]) if has_bat else 0.0
        s_init = float(node_data["energy_init_kwh"][bus_id]) if has_bat else 0.0
        s_min = float(cfg.soc_min_kwh) if has_bat else 0.0
        s_max = float(cfg.soc_max_kwh) if has_bat else 0.0

        m = gp.Model(f"baseline_bus_{bus_id}")
        m.Params.OutputFlag = 0

        l_plus = m.addVars(T, lb=0.0, name=f"l_plus_{bus_id}")
        l_minus = m.addVars(T, lb=0.0, name=f"l_minus_{bus_id}")
        p_bat = m.addVars(T, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"p_bat_{bus_id}")
        s = m.addVars(T + 1, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"s_{bus_id}")
        eps_soc = m.addVars(T + 1, lb=0.0, name=f"eps_soc_{bus_id}")
        eps_bat = m.addVars(T, lb=0.0, name=f"eps_bat_{bus_id}")

        m.addConstr(s[0] == s_init, name=f"init_soc_{bus_id}")

        if _cfg(cfg, "enforce_terminal_soc", True):
            m.addConstr(s[T] == s_init, name=f"terminal_soc_{bus_id}")

        for t in range(T):
            m.addConstr(
                l_plus[t] - l_minus[t] - p_bat[t] == load[t] - pv[t],
                name=f"power_balance_{bus_id}_{t}",
            )
            m.addConstr(
                s[t + 1] - s[t] - dt * p_bat[t] == 0.0,
                name=f"soc_dyn_{bus_id}_{t}",
            )
            m.addConstr(-p_bat[t] - eps_bat[t] <= pmax, name=f"bat_lb_{bus_id}_{t}")
            m.addConstr(p_bat[t] - eps_bat[t] <= pmax, name=f"bat_ub_{bus_id}_{t}")

        for t in range(T + 1):
            m.addConstr(-s[t] - eps_soc[t] <= -s_min, name=f"soc_lb_{bus_id}_{t}")
            m.addConstr(s[t] - eps_soc[t] <= s_max, name=f"soc_ub_{bus_id}_{t}")

        obj = gp.quicksum(
            dt * (buy_price[t] * l_plus[t] - sell_price[t] * l_minus[t])
            + c_deg * p_bat[t] * p_bat[t]
            + c_soc * (s[t] - s_ref) * (s[t] - s_ref)
            + rho_bat * eps_bat[t] * eps_bat[t]
            for t in range(T)
        ) + gp.quicksum(
            rho_soc * eps_soc[t] * eps_soc[t]
            for t in range(T + 1)
        )

        m.setObjective(obj, GRB.MINIMIZE)
        m.optimize()

        if m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Baseline follower QP failed for bus {bus_id}, status={m.Status}")

        results[bus_id] = ExactProsumerResult(
            p_grid_kw=[float(l_plus[t].X - l_minus[t].X) for t in range(T)],
            p_grid_import_kw=[float(l_plus[t].X) for t in range(T)],
            p_grid_export_kw=[float(l_minus[t].X) for t in range(T)],
            p_bat_kw=[float(p_bat[t].X) for t in range(T)],
            energy_kwh=[float(s[t].X) for t in range(T + 1)],
            eps_soc_kwh=[float(eps_soc[t].X) for t in range(T + 1)],
            eps_bat_kw=[float(eps_bat[t].X) for t in range(T)],
            objective_value=float(m.ObjVal),
        )

    return results


def compute_baseline_lref(net, node_data: Dict[str, Any], cfg) -> Tuple[float, Dict[int, ExactProsumerResult], dict]:
    T = _time_horizon(cfg)
    real_price = _slice_profile(node_data["leader_price_init"], T)

    follower_results = solve_followers_baseline_qp(
        node_data=node_data,
        buy_price=real_price,
        cfg=cfg,
    )

    pf_results = run_time_series_powerflow(
        net_base=net,
        selected_bus_ids=node_data["bus_ids"],
        follower_results=follower_results,
        horizon=T,
        cfg=cfg,
    )

    lref_mode = _cfg(cfg, "lref_source", "pcc_import_mean")

    if lref_mode == "pcc_import_mean":
        lref = float(np.mean(np.asarray(pf_results["grid_import_from_ext_grid_kw"], dtype=float)))
    elif lref_mode == "aggregate_load_mean":
        lt = np.zeros(T, dtype=float)
        for bus_id in node_data["bus_ids"]:
            lt += np.asarray(follower_results[bus_id].p_grid_kw, dtype=float)
        lref = float(np.mean(lt))
    else:
        raise ValueError(f"Unsupported lref_source: {lref_mode}")

    return lref, follower_results, pf_results


def solve_exact_stackelberg_18bus(cfg):
    cfg.validate()

    T = _time_horizon(cfg)
    dt = float(cfg.time_step_hours)

    rho_peak = float(_cfg(cfg, "weight_peak_penalty", 1.0))
    rho_track = float(_cfg(cfg, "weight_track_penalty", 1.0))
    rho_inc = float(_cfg(cfg, "weight_price_magnitude_penalty", 0.01))
    rho_dyn = float(_cfg(cfg, "weight_price_dynamic_penalty", 1.0))
    rho_mkt = float(_cfg(cfg, "weight_market_tracking_penalty", 10.0))

    sell_ratio = float(_cfg(cfg, "sell_price_ratio", 1.0 / 3.0))
    c_deg = float(_cfg(cfg, "battery_cycle_cost_eur_per_kwh2", 0.01))
    c_soc = float(_cfg(cfg, "soc_tracking_cost_eur_per_kwh2", 0.01))
    rho_soc = float(_cfg(cfg, "soc_slack_penalty_eur_per_kwh2", 1000.0))
    rho_bat = float(_cfg(cfg, "bat_slack_penalty_eur_per_kw2", 1000.0))
    s_ref = float(_cfg(cfg, "soc_ref_kwh", cfg.soc_init_kwh))

    lambda_min = float(cfg.price_min_eur_per_kwh)
    lambda_max = float(cfg.price_max_eur_per_kwh)

    lmin = float(_cfg(cfg, "grid_import_min_kw", 0.0))
    lmax = float(_cfg(cfg, "grid_import_max_kw", 1e6))

    M_dual = float(_cfg(cfg, "kkt_big_m_dual", 100.0))
    M_slack = float(_cfg(cfg, "kkt_big_m_slack", 100.0))
    time_limit = _cfg(cfg, "exact_stackelberg_time_limit_sec", None)
    mip_gap = float(_cfg(cfg, "exact_stackelberg_mip_gap", 1e-4))

    net, coords, groups, node_data, device_df, bus_map = assign_devices(cfg)
    real_price = _slice_profile(node_data["leader_price_init"], T)

    if cfg.target_import_kw is None:
        lref, baseline_followers, baseline_pf = compute_baseline_lref(net, node_data, cfg)
        cfg.target_import_kw = lref
    else:
        lref = float(cfg.target_import_kw)
        baseline_followers = None
        baseline_pf = None

    bus_ids = list(node_data["bus_ids"])

    m = gp.Model("exact_stackelberg_18bus")
    m.Params.OutputFlag = int(_cfg(cfg, "gurobi_output_flag", 0))
    m.Params.MIPGap = mip_gap
    if time_limit is not None:
        m.Params.TimeLimit = float(time_limit)

    lam = m.addVars(T, lb=lambda_min, ub=lambda_max, name="lambda_buy")
    eps_grid = m.addVars(T, lb=0.0, name="eps_grid")
    L = m.addVars(T, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="L")

    l_plus = {}
    l_minus = {}
    p_bat = {}
    s = {}
    eps_soc = {}
    eps_bat = {}

    eta_pb = {}
    eta_soc = {}
    eta_init = {}
    eta_term = {}

    mu_soc_lb = {}
    mu_soc_ub = {}
    mu_bat_lb = {}
    mu_bat_ub = {}
    mu_lplus = {}
    mu_lminus = {}
    mu_epssoc = {}
    mu_epsbat = {}

    z_soc_lb = {}
    z_soc_ub = {}
    z_bat_lb = {}
    z_bat_ub = {}
    z_lplus = {}
    z_lminus = {}
    z_epssoc = {}
    z_epsbat = {}

    slack_soc_lb = {}
    slack_soc_ub = {}
    slack_bat_lb = {}
    slack_bat_ub = {}
    slack_lplus = {}
    slack_lminus = {}
    slack_epssoc = {}
    slack_epsbat = {}

    for bus_id in bus_ids:
        has_bat = bool(node_data["has_battery"][bus_id])

        pmax = float(node_data["battery_pmax_kw"][bus_id]) if has_bat else 0.0
        s_init = float(node_data["energy_init_kwh"][bus_id]) if has_bat else 0.0
        s_min = float(cfg.soc_min_kwh) if has_bat else 0.0
        s_max = float(cfg.soc_max_kwh) if has_bat else 0.0

        load = _slice_profile(node_data["load_kw"][bus_id], T)
        pv = _slice_profile(node_data["pv_kw"][bus_id], T)

        l_plus[bus_id] = m.addVars(T, lb=0.0, name=f"l_plus_{bus_id}")
        l_minus[bus_id] = m.addVars(T, lb=0.0, name=f"l_minus_{bus_id}")
        p_bat[bus_id] = m.addVars(T, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"p_bat_{bus_id}")
        s[bus_id] = m.addVars(T + 1, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"s_{bus_id}")
        eps_soc[bus_id] = m.addVars(T + 1, lb=0.0, name=f"eps_soc_{bus_id}")
        eps_bat[bus_id] = m.addVars(T, lb=0.0, name=f"eps_bat_{bus_id}")

        eta_pb[bus_id] = m.addVars(T, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"eta_pb_{bus_id}")
        eta_soc[bus_id] = m.addVars(T, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"eta_soc_{bus_id}")
        eta_init[bus_id] = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"eta_init_{bus_id}")
        eta_term[bus_id] = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"eta_term_{bus_id}")

        mu_soc_lb[bus_id] = m.addVars(T + 1, lb=0.0, name=f"mu_soc_lb_{bus_id}")
        mu_soc_ub[bus_id] = m.addVars(T + 1, lb=0.0, name=f"mu_soc_ub_{bus_id}")
        mu_bat_lb[bus_id] = m.addVars(T, lb=0.0, name=f"mu_bat_lb_{bus_id}")
        mu_bat_ub[bus_id] = m.addVars(T, lb=0.0, name=f"mu_bat_ub_{bus_id}")
        mu_lplus[bus_id] = m.addVars(T, lb=0.0, name=f"mu_lplus_{bus_id}")
        mu_lminus[bus_id] = m.addVars(T, lb=0.0, name=f"mu_lminus_{bus_id}")
        mu_epssoc[bus_id] = m.addVars(T + 1, lb=0.0, name=f"mu_epssoc_{bus_id}")
        mu_epsbat[bus_id] = m.addVars(T, lb=0.0, name=f"mu_epsbat_{bus_id}")

        z_soc_lb[bus_id] = m.addVars(T + 1, vtype=GRB.BINARY, name=f"z_soc_lb_{bus_id}")
        z_soc_ub[bus_id] = m.addVars(T + 1, vtype=GRB.BINARY, name=f"z_soc_ub_{bus_id}")
        z_bat_lb[bus_id] = m.addVars(T, vtype=GRB.BINARY, name=f"z_bat_lb_{bus_id}")
        z_bat_ub[bus_id] = m.addVars(T, vtype=GRB.BINARY, name=f"z_bat_ub_{bus_id}")
        z_lplus[bus_id] = m.addVars(T, vtype=GRB.BINARY, name=f"z_lplus_{bus_id}")
        z_lminus[bus_id] = m.addVars(T, vtype=GRB.BINARY, name=f"z_lminus_{bus_id}")
        z_epssoc[bus_id] = m.addVars(T + 1, vtype=GRB.BINARY, name=f"z_epssoc_{bus_id}")
        z_epsbat[bus_id] = m.addVars(T, vtype=GRB.BINARY, name=f"z_epsbat_{bus_id}")

        m.addConstr(s[bus_id][0] == s_init, name=f"init_soc_{bus_id}")
        if _cfg(cfg, "enforce_terminal_soc", True):
            m.addConstr(s[bus_id][T] == s_init, name=f"terminal_soc_{bus_id}")

        for t in range(T):
            m.addConstr(
                l_plus[bus_id][t] - l_minus[bus_id][t] - p_bat[bus_id][t] == load[t] - pv[t],
                name=f"power_balance_{bus_id}_{t}",
            )
            m.addConstr(
                s[bus_id][t + 1] - s[bus_id][t] - dt * p_bat[bus_id][t] == 0.0,
                name=f"soc_dyn_{bus_id}_{t}",
            )

        for t in range(T + 1):
            slack_soc_lb[bus_id, t] = s[bus_id][t] + eps_soc[bus_id][t] - s_min
            slack_soc_ub[bus_id, t] = s_max - s[bus_id][t] + eps_soc[bus_id][t]
            slack_epssoc[bus_id, t] = eps_soc[bus_id][t]

            m.addConstr(slack_soc_lb[bus_id, t] >= 0.0, name=f"sl_soc_lb_ge0_{bus_id}_{t}")
            m.addConstr(slack_soc_ub[bus_id, t] >= 0.0, name=f"sl_soc_ub_ge0_{bus_id}_{t}")
            m.addConstr(slack_epssoc[bus_id, t] >= 0.0, name=f"sl_epssoc_ge0_{bus_id}_{t}")

            m.addConstr(mu_soc_lb[bus_id][t] <= M_dual * z_soc_lb[bus_id][t], name=f"comp_mu_soc_lb_{bus_id}_{t}")
            m.addConstr(slack_soc_lb[bus_id, t] <= M_slack * (1 - z_soc_lb[bus_id][t]), name=f"comp_sl_soc_lb_{bus_id}_{t}")

            m.addConstr(mu_soc_ub[bus_id][t] <= M_dual * z_soc_ub[bus_id][t], name=f"comp_mu_soc_ub_{bus_id}_{t}")
            m.addConstr(slack_soc_ub[bus_id, t] <= M_slack * (1 - z_soc_ub[bus_id][t]), name=f"comp_sl_soc_ub_{bus_id}_{t}")

            m.addConstr(mu_epssoc[bus_id][t] <= M_dual * z_epssoc[bus_id][t], name=f"comp_mu_epssoc_{bus_id}_{t}")
            m.addConstr(slack_epssoc[bus_id, t] <= M_slack * (1 - z_epssoc[bus_id][t]), name=f"comp_sl_epssoc_{bus_id}_{t}")

        for t in range(T):
            slack_bat_lb[bus_id, t] = pmax + eps_bat[bus_id][t] + p_bat[bus_id][t]
            slack_bat_ub[bus_id, t] = pmax + eps_bat[bus_id][t] - p_bat[bus_id][t]
            slack_lplus[bus_id, t] = l_plus[bus_id][t]
            slack_lminus[bus_id, t] = l_minus[bus_id][t]
            slack_epsbat[bus_id, t] = eps_bat[bus_id][t]

            m.addConstr(slack_bat_lb[bus_id, t] >= 0.0, name=f"sl_bat_lb_ge0_{bus_id}_{t}")
            m.addConstr(slack_bat_ub[bus_id, t] >= 0.0, name=f"sl_bat_ub_ge0_{bus_id}_{t}")
            m.addConstr(slack_lplus[bus_id, t] >= 0.0, name=f"sl_lplus_ge0_{bus_id}_{t}")
            m.addConstr(slack_lminus[bus_id, t] >= 0.0, name=f"sl_lminus_ge0_{bus_id}_{t}")
            m.addConstr(slack_epsbat[bus_id, t] >= 0.0, name=f"sl_epsbat_ge0_{bus_id}_{t}")

            m.addConstr(mu_bat_lb[bus_id][t] <= M_dual * z_bat_lb[bus_id][t], name=f"comp_mu_bat_lb_{bus_id}_{t}")
            m.addConstr(slack_bat_lb[bus_id, t] <= M_slack * (1 - z_bat_lb[bus_id][t]), name=f"comp_sl_bat_lb_{bus_id}_{t}")

            m.addConstr(mu_bat_ub[bus_id][t] <= M_dual * z_bat_ub[bus_id][t], name=f"comp_mu_bat_ub_{bus_id}_{t}")
            m.addConstr(slack_bat_ub[bus_id, t] <= M_slack * (1 - z_bat_ub[bus_id][t]), name=f"comp_sl_bat_ub_{bus_id}_{t}")

            m.addConstr(mu_lplus[bus_id][t] <= M_dual * z_lplus[bus_id][t], name=f"comp_mu_lplus_{bus_id}_{t}")
            m.addConstr(slack_lplus[bus_id, t] <= M_slack * (1 - z_lplus[bus_id][t]), name=f"comp_sl_lplus_{bus_id}_{t}")

            m.addConstr(mu_lminus[bus_id][t] <= M_dual * z_lminus[bus_id][t], name=f"comp_mu_lminus_{bus_id}_{t}")
            m.addConstr(slack_lminus[bus_id, t] <= M_slack * (1 - z_lminus[bus_id][t]), name=f"comp_sl_lminus_{bus_id}_{t}")

            m.addConstr(mu_epsbat[bus_id][t] <= M_dual * z_epsbat[bus_id][t], name=f"comp_mu_epsbat_{bus_id}_{t}")
            m.addConstr(slack_epsbat[bus_id, t] <= M_slack * (1 - z_epsbat[bus_id][t]), name=f"comp_sl_epsbat_{bus_id}_{t}")

        for t in range(T):
            m.addConstr(
                dt * lam[t] + eta_pb[bus_id][t] - mu_lplus[bus_id][t] == 0.0,
                name=f"st_lplus_{bus_id}_{t}",
            )
            m.addConstr(
                -dt * sell_ratio * lam[t] - eta_pb[bus_id][t] - mu_lminus[bus_id][t] == 0.0,
                name=f"st_lminus_{bus_id}_{t}",
            )
            m.addConstr(
                2.0 * c_deg * p_bat[bus_id][t]
                - eta_pb[bus_id][t]
                - dt * eta_soc[bus_id][t]
                + mu_bat_lb[bus_id][t]
                - mu_bat_ub[bus_id][t]
                == 0.0,
                name=f"st_pbat_{bus_id}_{t}",
            )
            m.addConstr(
                2.0 * rho_bat * eps_bat[bus_id][t]
                - mu_bat_lb[bus_id][t]
                - mu_bat_ub[bus_id][t]
                - mu_epsbat[bus_id][t]
                == 0.0,
                name=f"st_epsbat_{bus_id}_{t}",
            )

        for t in range(T + 1):
            expr = 2.0 * c_soc * (s[bus_id][t] - s_ref)

            if t >= 1:
                expr += eta_soc[bus_id][t - 1]
            if t <= T - 1:
                expr += -eta_soc[bus_id][t]

            if t == 0:
                expr += eta_init[bus_id]
            if t == T and _cfg(cfg, "enforce_terminal_soc", True):
                expr += eta_term[bus_id]

            expr += -mu_soc_lb[bus_id][t] + mu_soc_ub[bus_id][t]

            m.addConstr(expr == 0.0, name=f"st_soc_{bus_id}_{t}")

            m.addConstr(
                2.0 * rho_soc * eps_soc[bus_id][t]
                - mu_soc_lb[bus_id][t]
                - mu_soc_ub[bus_id][t]
                - mu_epssoc[bus_id][t]
                == 0.0,
                name=f"st_epssoc_{bus_id}_{t}",
            )

    for t in range(T):
        m.addConstr(
            L[t] == gp.quicksum(l_plus[bus_id][t] - l_minus[bus_id][t] for bus_id in bus_ids),
            name=f"aggregate_L_{t}",
        )
        m.addConstr(L[t] >= lmin - eps_grid[t], name=f"L_lower_{t}")
        m.addConstr(L[t] <= lmax + eps_grid[t], name=f"L_upper_{t}")

    obj = gp.quicksum(
        rho_peak * eps_grid[t] * eps_grid[t]
        + rho_track * (L[t] - lref) * (L[t] - lref)
        + rho_inc * lam[t] * lam[t]
        + rho_mkt * (lam[t] - real_price[t]) * (lam[t] - real_price[t])
        for t in range(T)
    )

    if T >= 2:
        obj += gp.quicksum(
            rho_dyn * (lam[t] - lam[t - 1]) * (lam[t] - lam[t - 1])
            for t in range(1, T)
        )

    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()

    if m.SolCount == 0:
        raise RuntimeError(f"Exact Stackelberg MIQP found no feasible solution, status={m.Status}")

    leader_price = np.array([float(lam[t].X) for t in range(T)], dtype=float)
    L_opt = np.array([float(L[t].X) for t in range(T)], dtype=float)
    eps_grid_opt = np.array([float(eps_grid[t].X) for t in range(T)], dtype=float)

    follower_results: Dict[int, ExactProsumerResult] = {}
    for bus_id in bus_ids:
        follower_results[bus_id] = ExactProsumerResult(
            p_grid_kw=[float(l_plus[bus_id][t].X - l_minus[bus_id][t].X) for t in range(T)],
            p_grid_import_kw=[float(l_plus[bus_id][t].X) for t in range(T)],
            p_grid_export_kw=[float(l_minus[bus_id][t].X) for t in range(T)],
            p_bat_kw=[float(p_bat[bus_id][t].X) for t in range(T)],
            energy_kwh=[float(s[bus_id][t].X) for t in range(T + 1)],
            eps_soc_kwh=[float(eps_soc[bus_id][t].X) for t in range(T + 1)],
            eps_bat_kw=[float(eps_bat[bus_id][t].X) for t in range(T)],
            objective_value=np.nan,
        )

    pf_results = run_time_series_powerflow(
        net_base=net,
        selected_bus_ids=bus_ids,
        follower_results=follower_results,
        horizon=T,
        cfg=cfg,
    )

    outputs_dir = _cfg(cfg, "output_dir", "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    pd.DataFrame(
        {
            "t": np.arange(T),
            "real_price_eur_per_kwh": real_price,
            "leader_price_eur_per_kwh": leader_price,
            "L_aggregate_kw": L_opt,
            "eps_grid_kw": eps_grid_opt,
            "pcc_import_kw_ex_post": np.asarray(pf_results["grid_import_from_ext_grid_kw"], dtype=float),
            "pcc_export_kw_ex_post": np.asarray(
                pf_results.get("grid_export_to_ext_grid_kw", np.zeros(T)),
                dtype=float,
            ) if "grid_export_to_ext_grid_kw" in pf_results else np.zeros(T, dtype=float),
        }
    ).to_csv(os.path.join(outputs_dir, "stackelberg_exact_hourly_summary.csv"), index=False)

    gap_value = np.nan
    try:
        gap_value = float(m.MIPGap)
    except Exception:
        pass

    pd.DataFrame(
        [
            {
                "objective_value": float(m.ObjVal),
                "lref_kw": float(lref),
                "solver_status": int(m.Status),
                "mip_gap": gap_value,
                "solve_time_sec": float(m.Runtime),
            }
        ]
    ).to_csv(os.path.join(outputs_dir, "stackelberg_exact_summary.csv"), index=False)

    return {
        "leader_price": leader_price,
        "real_price": real_price,
        "L_opt": L_opt,
        "eps_grid_opt": eps_grid_opt,
        "lref_kw": float(lref),
        "follower_results": follower_results,
        "baseline_followers": baseline_followers,
        "baseline_pf": baseline_pf,
        "pf_results": pf_results,
        "device_df": device_df,
        "objective_value": float(m.ObjVal),
        "solver_status": int(m.Status),
    }