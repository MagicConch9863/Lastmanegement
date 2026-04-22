from __future__ import annotations

import copy
import numpy as np
import pandapower as pp


def apply_time_step_net_injections(net_base, selected_bus_ids, follower_results, t):
    net_t = copy.deepcopy(net_base)

    for bus_id in selected_bus_ids:
        p_grid_kw = float(follower_results[bus_id].p_grid_kw[t])

        if p_grid_kw > 1e-9:
            pp.create_load(
                net_t,
                bus=bus_id,
                p_mw=p_grid_kw / 1000.0,
                q_mvar=0.0,
                name=f"load_bus_{bus_id}_t_{t}",
            )
        elif p_grid_kw < -1e-9:
            pp.create_sgen(
                net_t,
                bus=bus_id,
                p_mw=(-p_grid_kw) / 1000.0,
                q_mvar=0.0,
                name=f"sgen_bus_{bus_id}_t_{t}",
            )

    return net_t


def _safe_runpp(net_t):
    try:
        pp.runpp(net_t, init="auto", numba=False)
        return True
    except Exception as exc:
        print(f"[PowerFlow] runpp failed: {exc}")
        return False


def _extract_ext_grid_exchange_kw(net_t):
    if len(net_t.res_ext_grid) == 0:
        return 0.0
    return float(net_t.res_ext_grid["p_mw"].sum()) * 1000.0


def run_time_series_powerflow(net_base, selected_bus_ids, follower_results, horizon, cfg):
    pf_results = {
        "bus_vm_pu": [],
        "line_loading_percent": [],
        "trafo_loading_percent": [],
        "aggregate_net_load_kw": [],
        "grid_import_from_followers_kw": [],
        "grid_export_from_followers_kw": [],
        "grid_exchange_with_ext_grid_kw": [],
        "grid_import_from_ext_grid_kw": [],
        "grid_export_to_ext_grid_kw": [],
        "powerflow_success": [],
    }

    for t in range(horizon):
        net_t = apply_time_step_net_injections(net_base, selected_bus_ids, follower_results, t)

        agg_net_load_kw = float(
            sum(float(follower_results[bus_id].p_grid_kw[t]) for bus_id in selected_bus_ids)
        )
        agg_import_kw = float(
            sum(max(float(follower_results[bus_id].p_grid_import_kw[t]), 0.0) for bus_id in selected_bus_ids)
        )
        agg_export_kw = float(
            sum(max(float(follower_results[bus_id].p_grid_export_kw[t]), 0.0) for bus_id in selected_bus_ids)
        )

        success = _safe_runpp(net_t)

        if success:
            vm = net_t.res_bus.vm_pu.to_numpy(dtype=float)
            line_loading = (
                net_t.res_line.loading_percent.to_numpy(dtype=float)
                if len(net_t.line) > 0 else np.array([], dtype=float)
            )
            trafo_loading = (
                net_t.res_trafo.loading_percent.to_numpy(dtype=float)
                if len(net_t.trafo) > 0 else np.array([], dtype=float)
            )
            ext_grid_exchange_kw = _extract_ext_grid_exchange_kw(net_t)
        else:
            vm = np.array([], dtype=float)
            line_loading = np.array([], dtype=float)
            trafo_loading = np.array([], dtype=float)
            ext_grid_exchange_kw = agg_net_load_kw

        ext_grid_import_kw = max(ext_grid_exchange_kw, 0.0)
        ext_grid_export_kw = max(-ext_grid_exchange_kw, 0.0)

        pf_results["bus_vm_pu"].append(vm)
        pf_results["line_loading_percent"].append(line_loading)
        pf_results["trafo_loading_percent"].append(trafo_loading)
        pf_results["aggregate_net_load_kw"].append(agg_net_load_kw)
        pf_results["grid_import_from_followers_kw"].append(agg_import_kw)
        pf_results["grid_export_from_followers_kw"].append(agg_export_kw)
        pf_results["grid_exchange_with_ext_grid_kw"].append(float(ext_grid_exchange_kw))
        pf_results["grid_import_from_ext_grid_kw"].append(float(ext_grid_import_kw))
        pf_results["grid_export_to_ext_grid_kw"].append(float(ext_grid_export_kw))
        pf_results["powerflow_success"].append(bool(success))

        if getattr(cfg, "debug_mode", False):
            print(
                f"[PowerFlow] t={t:02d}, "
                f"L_t={agg_net_load_kw:.3f} kW, "
                f"followers_import={agg_import_kw:.3f} kW, "
                f"followers_export={agg_export_kw:.3f} kW, "
                f"ext_grid_exchange={ext_grid_exchange_kw:.3f} kW, "
                f"success={success}"
            )

    return pf_results


def compute_network_penalty(pf_results, cfg):
    voltage_penalty = 0.0
    line_penalty = 0.0
    trafo_penalty = 0.0
    grid_capacity_penalty = 0.0

    eps_grid_series = []

    aggregate_load_signal = np.asarray(pf_results["aggregate_net_load_kw"], dtype=float)

    grid_lmin = getattr(cfg, "grid_import_min_kw", None)
    grid_lmax = getattr(cfg, "grid_import_max_kw", None)

    for t in range(len(aggregate_load_signal)):
        vm = pf_results["bus_vm_pu"][t]
        line_loading = pf_results["line_loading_percent"][t]
        trafo_loading = pf_results["trafo_loading_percent"][t]
        lt_kw = float(aggregate_load_signal[t])

        eps_grid_upper = 0.0
        eps_grid_lower = 0.0

        if grid_lmax is not None:
            eps_grid_upper = max(0.0, lt_kw - float(grid_lmax))
        if grid_lmin is not None:
            eps_grid_lower = max(0.0, float(grid_lmin) - lt_kw)

        eps_grid_t = max(eps_grid_upper, eps_grid_lower)
        eps_grid_series.append(eps_grid_t)
        grid_capacity_penalty += eps_grid_t * eps_grid_t

        for val in vm:
            if val < cfg.voltage_min_pu:
                voltage_penalty += (cfg.voltage_min_pu - float(val)) ** 2
            elif val > cfg.voltage_max_pu:
                voltage_penalty += (float(val) - cfg.voltage_max_pu) ** 2

        for val in line_loading:
            if val > cfg.line_loading_limit_percent:
                line_penalty += ((float(val) - cfg.line_loading_limit_percent) / 100.0) ** 2

        for val in trafo_loading:
            if val > cfg.trafo_loading_limit_percent:
                trafo_penalty += ((float(val) - cfg.trafo_loading_limit_percent) / 100.0) ** 2

    security_penalty = (
        cfg.weight_voltage_penalty * voltage_penalty
        + cfg.weight_line_penalty * line_penalty
        + cfg.weight_trafo_penalty * trafo_penalty
    )

    return {
        "voltage_penalty": float(voltage_penalty),
        "line_penalty": float(line_penalty),
        "trafo_penalty": float(trafo_penalty),
        "grid_capacity_penalty": float(grid_capacity_penalty),
        "eps_grid_series": np.asarray(eps_grid_series, dtype=float),
        "security_penalty": float(security_penalty),
    }