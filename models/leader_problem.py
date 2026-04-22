from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from configs.stackelberg_config import StackelbergConfig
from models.prosumer_model import solve_all_prosumers
from models.powerflow_interface import run_time_series_powerflow, compute_network_penalty


def initialize_leader_price(cfg: StackelbergConfig, real_price_signal) -> np.ndarray:
    price = np.asarray(real_price_signal, dtype=float).copy()
    return np.clip(price, cfg.price_min_eur_per_kwh, cfg.price_max_eur_per_kwh)


def summarize_follower_results(follower_results) -> Dict[str, np.ndarray]:
    sample_bus = next(iter(follower_results.keys()))
    horizon = len(follower_results[sample_bus].p_grid_kw)

    total_grid_kw = np.zeros(horizon, dtype=float)
    total_bat_kw = np.zeros(horizon, dtype=float)
    total_import_kw = np.zeros(horizon, dtype=float)
    total_export_kw = np.zeros(horizon, dtype=float)

    for result in follower_results.values():
        total_grid_kw += np.asarray(result.p_grid_kw, dtype=float)
        total_bat_kw += np.asarray(result.p_bat_kw, dtype=float)
        total_import_kw += np.asarray(result.p_grid_import_kw, dtype=float)
        total_export_kw += np.asarray(result.p_grid_export_kw, dtype=float)

    return {
        "aggregate_grid_kw": total_grid_kw,
        "aggregate_battery_kw": total_bat_kw,
        "aggregate_import_kw": total_import_kw,
        "aggregate_export_kw": total_export_kw,
    }


def compute_leader_objective(
    leader_price,
    aggregate,
    pf_results,
    network_penalty,
    cfg: StackelbergConfig,
) -> Dict[str, float]:
    leader_price = np.asarray(leader_price, dtype=float)
    lt_signal = np.asarray(aggregate["aggregate_grid_kw"], dtype=float)

    lref = cfg.target_import_kw
    if lref is None:
        raise ValueError("cfg.target_import_kw must be set before leader optimization.")

    peak_penalty = cfg.weight_peak_penalty * float(network_penalty["grid_capacity_penalty"])
    track_penalty = cfg.weight_track_penalty * float(np.sum((lt_signal - lref) ** 2))
    price_magnitude_penalty = cfg.weight_price_magnitude_penalty * float(np.sum(leader_price ** 2))

    if len(leader_price) > 1:
        price_dynamic_penalty = cfg.weight_price_dynamic_penalty * float(np.sum(np.diff(leader_price) ** 2))
    else:
        price_dynamic_penalty = 0.0

    security_penalty = float(network_penalty["security_penalty"])

    total_objective = (
        peak_penalty
        + track_penalty
        + price_magnitude_penalty
        + price_dynamic_penalty
        + security_penalty
    )

    return {
        "peak_penalty": float(peak_penalty),
        "track_penalty": float(track_penalty),
        "price_magnitude_penalty": float(price_magnitude_penalty),
        "price_dynamic_penalty": float(price_dynamic_penalty),
        "security_penalty": float(security_penalty),
        "leader_objective": float(total_objective),
    }


def evaluate_given_price(
    net,
    node_data,
    price_signal,
    cfg,
    compute_leader_metrics: bool = True,
):
    follower_results = solve_all_prosumers(
        node_data=node_data,
        leader_price=price_signal,
        cfg=cfg,
    )

    pf_results = run_time_series_powerflow(
        net_base=net,
        selected_bus_ids=node_data["bus_ids"],
        follower_results=follower_results,
        horizon=cfg.horizon,
        cfg=cfg,
    )

    network_penalty = compute_network_penalty(
        pf_results=pf_results,
        cfg=cfg,
    )

    aggregate = summarize_follower_results(follower_results)

    result = {
        "price_signal": np.asarray(price_signal, dtype=float),
        "follower_results": follower_results,
        "pf_results": pf_results,
        "network_penalty": network_penalty,
        "aggregate": aggregate,
        "leader_metrics": None,
    }

    if compute_leader_metrics:
        leader_metrics = compute_leader_objective(
            leader_price=price_signal,
            aggregate=aggregate,
            pf_results=pf_results,
            network_penalty=network_penalty,
            cfg=cfg,
        )
        result["leader_metrics"] = leader_metrics

    return result


def _project_price(candidate, cfg: StackelbergConfig) -> np.ndarray:
    return np.clip(
        np.asarray(candidate, dtype=float),
        cfg.price_min_eur_per_kwh,
        cfg.price_max_eur_per_kwh,
    )


def generate_candidate_prices(
    current_price,
    real_price_signal,
    current_evaluation,
    cfg: StackelbergConfig,
    step_size: float,
) -> List[np.ndarray]:
    current_price = np.asarray(current_price, dtype=float)
    real_price_signal = np.asarray(real_price_signal, dtype=float)
    lt_signal = np.asarray(current_evaluation["aggregate"]["aggregate_grid_kw"], dtype=float)

    lref = float(cfg.target_import_kw)
    deviation_sign = np.sign(lt_signal - lref)

    candidates: List[np.ndarray] = []

    # 候选1：保持不变
    candidates.append(current_price.copy())

    # 候选2/3：整体上调/下调
    candidates.append(_project_price(current_price + step_size, cfg))
    candidates.append(_project_price(current_price - step_size, cfg))

    # 候选4：朝真实市场价格回拉
    candidates.append(
        _project_price(
            (1.0 - cfg.market_price_pull_weight) * current_price
            + cfg.market_price_pull_weight * real_price_signal,
            cfg,
        )
    )

    # 候选5/6：按 Lt-Lref 的符号方向调整
    candidates.append(_project_price(current_price + step_size * deviation_sign, cfg))
    candidates.append(_project_price(current_price - step_size * deviation_sign, cfg))

    # 候选7：平滑版
    if len(current_price) > 2:
        smooth_price = current_price.copy()
        smooth_price[1:-1] = (
            0.25 * current_price[:-2]
            + 0.50 * current_price[1:-1]
            + 0.25 * current_price[2:]
        )
        candidates.append(_project_price(smooth_price, cfg))

    # 去重
    unique_candidates: List[np.ndarray] = []
    seen = set()
    for cand in candidates:
        key = tuple(np.round(cand, 10))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(cand)

    return unique_candidates


def solve_leader_problem(
    net,
    node_data,
    real_price_signal,
    initial_price,
    cfg: StackelbergConfig,
):
    current_price = np.asarray(initial_price, dtype=float)

    current_eval = evaluate_given_price(
        net=net,
        node_data=node_data,
        price_signal=current_price,
        cfg=cfg,
        compute_leader_metrics=True,
    )

    history = []
    best_result = None
    best_objective = np.inf

    for k in range(cfg.max_stackelberg_iter):
        step_size = cfg.leader_price_search_step * (cfg.leader_price_step_decay ** k)

        candidates = generate_candidate_prices(
            current_price=current_price,
            real_price_signal=real_price_signal,
            current_evaluation=current_eval,
            cfg=cfg,
            step_size=step_size,
        )

        candidate_results = []
        for cand in candidates:
            eval_result = evaluate_given_price(
                net=net,
                node_data=node_data,
                price_signal=cand,
                cfg=cfg,
                compute_leader_metrics=True,
            )
            candidate_results.append(eval_result)

        chosen = min(
            candidate_results,
            key=lambda x: x["leader_metrics"]["leader_objective"],
        )

        max_price_change = float(
            np.max(np.abs(np.asarray(chosen["price_signal"], dtype=float) - current_price))
        )

        iteration_result = {
            "iteration": k,
            "leader_price": np.asarray(chosen["price_signal"], dtype=float).copy(),
            "real_price_signal": np.asarray(real_price_signal, dtype=float).copy(),
            "follower_results": chosen["follower_results"],
            "pf_results": chosen["pf_results"],
            "network_penalty": chosen["network_penalty"],
            "leader_metrics": chosen["leader_metrics"],
            "aggregate": chosen["aggregate"],
            "max_price_change": max_price_change,
        }
        history.append(iteration_result)

        current_objective = float(chosen["leader_metrics"]["leader_objective"])
        if current_objective < best_objective:
            best_objective = current_objective
            best_result = iteration_result

        if cfg.debug_mode:
            print(
                f"[LeaderSearch] iter={k:02d}, "
                f"obj={current_objective:.6f}, "
                f"peak={chosen['leader_metrics']['peak_penalty']:.6f}, "
                f"track={chosen['leader_metrics']['track_penalty']:.6f}, "
                f"inc={chosen['leader_metrics']['price_magnitude_penalty']:.6f}, "
                f"dyn={chosen['leader_metrics']['price_dynamic_penalty']:.6f}, "
                f"security={chosen['leader_metrics']['security_penalty']:.6f}, "
                f"max_price_change={max_price_change:.6f}"
            )

        current_price = np.asarray(chosen["price_signal"], dtype=float)
        current_eval = chosen

        if max_price_change < cfg.price_convergence_tol:
            if cfg.debug_mode:
                print(f"[LeaderSearch] Converged at iteration {k}.")
            break

    if best_result is None:
        raise RuntimeError("Leader optimization did not produce any valid result.")

    return {
        "history": history,
        "best_result": best_result,
    }