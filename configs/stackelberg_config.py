from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StackelbergConfig:
    simbench_code: str = "1-LV-urban6--0-sw"

    horizon: int = 96
    time_step_hours: float = 0.25

    profile_csv_path: str = "data/price_data.csv"
    price_col: str = "price_eur_per_mwh"
    price_unit_is_eur_per_mwh: bool = True

    n_passive_load: int = 6
    n_pv_only: int = 6
    n_pv_battery: int = 6
    random_seed: int = 42

    battery_capacity_kwh: float = 12.0
    battery_pmax_kw: float = 5.0
    soc_init_kwh: float = 6.0
    soc_min_kwh: float = 1.0
    soc_max_kwh: float = 12.0
    soc_ref_kwh: float = 6.0

    battery_cycle_cost_eur_per_kwh2: float = 0.01
    soc_tracking_cost_eur_per_kwh2: float = 0.01
    soc_slack_penalty_eur_per_kwh2: float = 1000.0
    bat_slack_penalty_eur_per_kw2: float = 1000.0
    enforce_terminal_soc: bool = True

    sell_price_ratio: float = 1.0 / 3.0

    price_min_eur_per_kwh: float = 0.01
    price_max_eur_per_kwh: float = 1.00

    lref_source: str = "pcc_import_mean"
    target_import_kw: float | None = None

    weight_peak_penalty: float = 1.0
    weight_track_penalty: float = 1.0
    weight_price_magnitude_penalty: float = 0.01
    weight_price_dynamic_penalty: float = 1.0

    # 新增：让 leader price 不要偏离真实市场价格太远
    weight_market_tracking_penalty: float = 10.0

    grid_import_min_kw: float = 0.0
    grid_import_max_kw: float = 120.0

    voltage_min_pu: float = 0.95
    voltage_max_pu: float = 1.05
    line_loading_limit_percent: float = 20
    trafo_loading_limit_percent: float = 20

    weight_voltage_penalty: float = 10.0
    weight_line_penalty: float = 10.0
    weight_trafo_penalty: float = 10.0

    exact_horizon: int | None = 6
    exact_stackelberg_time_limit_sec: float = 3600
    exact_stackelberg_mip_gap: float = 1e-4

    kkt_big_m_dual: float = 50
    kkt_big_m_slack: float = 50

    gurobi_output_flag: int = 1
    output_dir: str = "outputs"
    debug_mode: bool = True
    save_plots: bool = True

    def validate(self):
        if self.horizon <= 0:
            raise ValueError("horizon must be positive.")
        if self.time_step_hours <= 0:
            raise ValueError("time_step_hours must be positive.")

        total_nodes = self.n_passive_load + self.n_pv_only + self.n_pv_battery
        if total_nodes != 18:
            raise ValueError("The feeder must contain exactly 18 residential nodes.")

        if not (self.soc_min_kwh <= self.soc_init_kwh <= self.soc_max_kwh):
            raise ValueError("Invalid SOC bounds.")
        if not (self.soc_min_kwh <= self.soc_ref_kwh <= self.soc_max_kwh):
            raise ValueError("soc_ref_kwh must lie within [soc_min_kwh, soc_max_kwh].")

        if not (0.0 < self.price_min_eur_per_kwh <= self.price_max_eur_per_kwh):
            raise ValueError("Invalid price bounds.")

        if self.sell_price_ratio < 0.0:
            raise ValueError("sell_price_ratio must be nonnegative.")

        if self.grid_import_min_kw > self.grid_import_max_kw:
            raise ValueError("grid_import_min_kw must be <= grid_import_max_kw.")

        if self.kkt_big_m_dual <= 0.0 or self.kkt_big_m_slack <= 0.0:
            raise ValueError("Big-M constants must be positive.")

        if self.exact_horizon is not None and self.exact_horizon <= 0:
            raise ValueError("exact_horizon must be positive.")