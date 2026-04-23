from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StackelbergConfig:
    # =========================
    # Network / data
    # =========================
    simbench_code: str = "1-LV-urban6--0-sw"
    horizon: int = 96
    time_step_hours: float = 0.25

    profile_csv_path: str = "data/price_data.csv"
    price_col: str = "price_eur_per_mwh"
    price_unit_is_eur_per_mwh: bool = True

    random_seed: int = 42

    # =========================
    # Residential bus selection
    # If empty, all 18 buses are used.
    # Allowed names: Bus R1 ... Bus R18
    # =========================
    selected_residential_bus_names: list[str] = field(default_factory=list)

    # =========================
    # Device counts on the SELECTED buses
    # These three numbers must sum to the number of selected buses.
    # =========================
    n_passive_load: int = 6
    n_pv_only: int = 6
    n_pv_battery: int = 6

    # =========================
    # Battery
    # =========================
    battery_capacity_kwh: float = 12.0
    battery_pmax_kw: float = 5.0
    soc_init_kwh: float = 6.0
    soc_min_kwh: float = 1.0
    soc_max_kwh: float = 12.0

    # follower simple QP
    battery_cycle_cost_eur_per_kwh2: float = 0.01
    enforce_terminal_soc: bool = True

    # =========================
    # Price
    # =========================
    price_min_eur_per_kwh: float = 0.01
    price_max_eur_per_kwh: float = 1.00
    leader_step_size: float = 0.1
    price_damping: float = 0.25
    price_smoothing_weight: float = 1.0

    # =========================
    # Leader objective
    # =========================
    target_import_kw: float | None = None
    weight_wholesale_cost: float = 1.0
    weight_peak_penalty: float = 180.0
    weight_voltage_penalty: float = 10.0
    weight_line_penalty: float = 10.0
    weight_trafo_penalty: float = 10.0

    # =========================
    # Iteration
    # =========================
    max_stackelberg_iter: int = 20
    price_convergence_tol: float = 1e-3

    # =========================
    # Solver / Output
    # =========================
    gurobi_output_flag: int = 0
    output_dir: str = "outputs"
    save_plots: bool = True
    debug_mode: bool = True

    # =========================
    # Exact Stackelberg options
    # =========================
    exact_horizon: int | None = 6
    exact_stackelberg_time_limit_sec: float = 3600.0
    exact_stackelberg_mip_gap: float = 1e-4
    sell_price_ratio: float = 2.5 / 3.0
    soc_tracking_cost_eur_per_kwh2: float = 0.01
    soc_slack_penalty_eur_per_kwh2: float = 1000.0
    bat_slack_penalty_eur_per_kw2: float = 1000.0
    soc_ref_kwh: float = 6.0
    weight_track_penalty: float = 1.0
    weight_price_magnitude_penalty: float = 0.01
    weight_price_dynamic_penalty: float = 1.0
    weight_market_tracking_penalty: float = 10.0
    grid_import_min_kw: float = 0.0
    grid_import_max_kw: float = 120.0
    lref_source: str = "pcc_import_mean"
    kkt_big_m_dual: float = 100.0
    kkt_big_m_slack: float = 100.0

    def get_selected_residential_bus_names(self) -> list[str]:
        all_residential = [f"Bus R{i}" for i in range(1, 19)]
        if not self.selected_residential_bus_names:
            return all_residential
        return self.selected_residential_bus_names

    def validate(self):
        if self.horizon <= 0:
            raise ValueError("horizon must be positive.")

        if self.time_step_hours <= 0:
            raise ValueError("time_step_hours must be positive.")

        if not (self.soc_min_kwh <= self.soc_init_kwh <= self.soc_max_kwh):
            raise ValueError("Invalid SOC bounds.")

        if not (0.0 < self.price_min_eur_per_kwh <= self.price_max_eur_per_kwh):
            raise ValueError("Invalid price bounds.")

        selected_bus_names = self.get_selected_residential_bus_names()
        allowed = {f"Bus R{i}" for i in range(1, 19)}

        if len(selected_bus_names) == 0:
            raise ValueError("At least one residential bus must be selected.")

        if len(set(selected_bus_names)) != len(selected_bus_names):
            raise ValueError("selected_residential_bus_names contains duplicates.")

        invalid = [b for b in selected_bus_names if b not in allowed]
        if invalid:
            raise ValueError(
                f"Invalid selected_residential_bus_names: {invalid}. "
                f"Allowed names are Bus R1 ... Bus R18."
            )

        total_nodes = self.n_passive_load + self.n_pv_only + self.n_pv_battery
        if total_nodes != len(selected_bus_names):
            raise ValueError(
                "The number of selected buses must equal "
                "n_passive_load + n_pv_only + n_pv_battery. "
                f"Got selected={len(selected_bus_names)}, counts={total_nodes}."
            )