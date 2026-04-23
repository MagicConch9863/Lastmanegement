from configs.stackelberg_config import StackelbergConfig
from networks.plot_network_layout import plot_network_layout


def main():
    cfg = StackelbergConfig()

    # 先随便选几个 bus 看效果
    cfg.selected_residential_bus_names = [
        "Bus R1", "Bus R5", "Bus R10"
    ]

    # 这三个数必须加起来等于你选中的 bus 数量
    cfg.n_passive_load = 1
    cfg.n_pv_only = 1
    cfg.n_pv_battery = 1

    cfg.validate()

    plot_network_layout(
        cfg=cfg,
        save_path="outputs/test_selected_bus_layout.png",
        show_plot=True,
    )


if __name__ == "__main__":
    main()