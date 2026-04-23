import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from networks.assign_devices import assign_devices


def plot_network_layout(cfg, save_path=None, show_plot=False):
    net, coords, groups, node_data, device_df, bus_map = assign_devices(cfg)

    if save_path is None:
        save_path = os.path.join(cfg.output_dir, "network_layout.png")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 9), facecolor="#FAF9F6")

    for _, line in net.line.iterrows():
        p1 = coords[net.bus.name.at[line.from_bus]]
        p2 = coords[net.bus.name.at[line.to_bus]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="black", linewidth=1.2, zorder=1)

    style = {
        "upstream": {"c": "black", "m": "o", "s": 120, "lbl": "Upstream bus"},
        "passive": {"c": "#1f77b4", "m": "s", "s": 100, "lbl": "Selected passive"},
        "pv_only": {"c": "#2ca02c", "m": "^", "s": 140, "lbl": "Selected load + PV"},
        "active": {"c": "#d62728", "m": "o", "s": 120, "lbl": "Selected load + PV + battery"},
        "inactive_residential": {"c": "#BDBDBD", "m": "o", "s": 80, "lbl": "Unselected residential"},
    }

    for group_name, s in style.items():
        names = groups.get(group_name, [])
        if not names:
            continue
        x = [coords[n][0] for n in names]
        y = [coords[n][1] for n in names]
        ax.scatter(
            x,
            y,
            c=s["c"],
            marker=s["m"],
            s=s["s"],
            edgecolors="white",
            linewidths=0.5,
            zorder=5,
        )

    for name, (x, y) in coords.items():
        ha, offset = "left", 0.25
        if x < 0:
            ha, offset = "right", -0.25

        if x == 0 and name in ["Bus R3", "Bus R4", "Bus R9", "Bus R10"]:
            ax.text(x, y + 0.3, name, fontsize=9, ha="center")
        else:
            ax.text(x + offset, y, name, fontsize=9, ha=ha, va="center")

    legend_elements = [
        Line2D([0], [0], marker=s["m"], color="w", label=s["lbl"], markerfacecolor=s["c"], markersize=10)
        for s in style.values()
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=9,
        frameon=True,
        facecolor="white",
    )

    plt.title("Residential Feeder Device Layout", fontsize=16, fontweight="bold", pad=20)
    plt.axis("off")
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Success: plot saved to {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return net, coords, groups, node_data, device_df, bus_map