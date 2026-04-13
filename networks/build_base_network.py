from pathlib import Path
import numpy as np
import pandas as pd
import pandapower as pp

try:
    import simbench as sb
except ImportError:
    sb = None


def create_18bus_residential_network():
    net = pp.create_empty_network(name="Residential_18Bus")

    coords = {
        "Bus 0": (0, 1.2), "Bus R0": (0, 0.5),
        "Bus R1": (0, -0.5), "Bus R2": (0, -1.5), "Bus R3": (0, -2.5), "Bus R4": (0, -3.5),
        "Bus R5": (0, -4.5), "Bus R6": (0, -5.5), "Bus R7": (0, -6.5), "Bus R8": (0, -7.5),
        "Bus R9": (0, -8.5), "Bus R10": (0, -9.5),
        "Bus R11": (-1.8, -2.5), "Bus R12": (1.8, -3.5), "Bus R13": (3.6, -3.5),
        "Bus R14": (5.4, -3.5), "Bus R15": (5.4, -4.8), "Bus R16": (-1.8, -5.5),
        "Bus R17": (1.8, -8.5), "Bus R18": (-1.8, -9.5),
    }

    bus_map = {}
    for name in coords:
        vn = 20.0 if name == "Bus 0" else 0.4
        bus_map[name] = pp.create_bus(net, vn_kv=vn, name=name)

    pp.create_ext_grid(net, bus=bus_map["Bus 0"])
    pp.create_transformer(
        net,
        hv_bus=bus_map["Bus 0"],
        lv_bus=bus_map["Bus R0"],
        std_type="0.4 MVA 20/0.4 kV",
    )

    lines = [
        ("Bus R0", "Bus R1"), ("Bus R1", "Bus R2"), ("Bus R2", "Bus R3"), ("Bus R3", "Bus R4"),
        ("Bus R4", "Bus R5"), ("Bus R5", "Bus R6"), ("Bus R6", "Bus R7"), ("Bus R7", "Bus R8"),
        ("Bus R8", "Bus R9"), ("Bus R9", "Bus R10"), ("Bus R3", "Bus R11"), ("Bus R4", "Bus R12"),
        ("Bus R12", "Bus R13"), ("Bus R13", "Bus R14"), ("Bus R14", "Bus R15"),
        ("Bus R6", "Bus R16"), ("Bus R9", "Bus R17"), ("Bus R10", "Bus R18"),
    ]
    for u, v in lines:
        pp.create_line(
            net,
            from_bus=bus_map[u],
            to_bus=bus_map[v],
            length_km=0.035,
            std_type="NAYY 4x150 SE",
        )

    return net, coords, bus_map


def load_price_profile(cfg):
    df = pd.read_csv(Path(cfg.profile_csv_path))
    series = pd.to_numeric(df[cfg.price_col], errors="coerce").dropna().reset_index(drop=True)
    if len(series) < cfg.horizon:
        raise ValueError(f"price length {len(series)} < horizon {cfg.horizon}")
    price = series.iloc[:cfg.horizon].to_numpy(dtype=float)
    if cfg.price_unit_is_eur_per_mwh:
        price = price / 1000.0
    return price, df


def load_simbench_profiles_for_18_nodes(cfg):
    """
    Use 18 actual SimBench load profiles, not the whole feeder sum.
    PV profiles are drawn from actual SimBench sgen profiles.
    """
    if sb is None:
        raise ImportError("simbench is not installed")

    sb_code = getattr(cfg, "simbench_code", "1-LV-urban6--0-sw")
    sb_net = sb.get_simbench_net(sb_code)
    abs_values = sb.get_absolute_values(sb_net, profiles_instead_of_study_cases=True)

    load_df = abs_values[("load", "p_mw")].iloc[:cfg.horizon].copy()
    sgen_df = abs_values[("sgen", "p_mw")].iloc[:cfg.horizon].copy()

    if load_df.shape[1] < 18:
        raise RuntimeError(f"SimBench load columns {load_df.shape[1]} < 18")

    rng = np.random.default_rng(getattr(cfg, "random_seed", 42))

    load_cols = list(load_df.columns)
    rng.shuffle(load_cols)
    load_cols = load_cols[:18]

    sgen_cols = list(sgen_df.columns)
    if len(sgen_cols) == 0:
        pv_pool = [np.zeros(cfg.horizon, dtype=float) for _ in range(18)]
    else:
        rng.shuffle(sgen_cols)
        pv_pool = []
        for i in range(18):
            col = sgen_cols[i % len(sgen_cols)]
            pv_pool.append(np.maximum(sgen_df[col].to_numpy(dtype=float), 0.0) * 1000.0)

    load_pool = [load_df[col].to_numpy(dtype=float) * 1000.0 for col in load_cols]
    raw_total_demand = np.sum(np.column_stack(load_pool), axis=1)

    return load_pool, pv_pool, raw_total_demand


def build_base_network_and_data(cfg):
    cfg.validate()

    net, coords, bus_map = create_18bus_residential_network()
    price_signal, price_df = load_price_profile(cfg)
    load_pool, pv_pool, raw_total_demand = load_simbench_profiles_for_18_nodes(cfg)

    bus_names = [f"Bus R{i}" for i in range(1, 19)]
    bus_ids = [bus_map[name] for name in bus_names]

    node_data = {
        "bus_ids": bus_ids,
        "bus_names": bus_names,
        "coords": coords,
        "load_kw": {},
        "pv_kw": {},
        "raw_total_load_kw": raw_total_demand.copy(),
        "has_battery": {},
        "battery_capacity_kwh": {},
        "battery_pmax_kw": {},
        "energy_init_kwh": {},
        "leader_price_init": price_signal.copy(),
        "raw_price_df": price_df.copy(),
    }

    for bus_id, load_profile, pv_profile in zip(bus_ids, load_pool, pv_pool):
        node_data["load_kw"][bus_id] = load_profile.tolist()
        node_data["pv_kw"][bus_id] = pv_profile.tolist()
        node_data["has_battery"][bus_id] = False
        node_data["battery_capacity_kwh"][bus_id] = 0.0
        node_data["battery_pmax_kw"][bus_id] = 0.0
        node_data["energy_init_kwh"][bus_id] = 0.0

    if getattr(cfg, "debug_mode", False):
        print(f"SimBench source: {getattr(cfg, 'simbench_code', '1-LV-urban6--0-sw')}")
        print(f"18-node raw demand max: {np.max(raw_total_demand):.3f} kW")

    return net, coords, bus_map, node_data

def load_simbench_network(simbench_code: str):
    """
    加载原始的 SimBench 网络。
    """
    if sb is None:
        raise ImportError("simbench library is not installed or import failed.")
    return sb.get_simbench_net(simbench_code)