from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "exports" / ".mpl_cache"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import pandas as pd


def get_project_root() -> Path:
    return PROJECT_ROOT


def ensure_plots_dir(project_root: Path) -> Path:
    plots_dir = project_root / "exports" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "font.sans-serif": [
                "PingFang SC",
                "Arial Unicode MS",
                "Noto Sans CJK SC",
                "DejaVu Sans",
            ],
        }
    )

def _annotate_bars(ax: plt.Axes, values: list[float], fmt: str) -> None:
    for i, val in enumerate(values):
        ax.text(i, val, fmt.format(val), ha="center", va="bottom", fontsize=10)


def plot_summary_by_day(summary_path: Path, plots_dir: Path) -> Path:
    df = pd.read_csv(summary_path)
    df["promo_label"] = df["promo_day_type"].replace(
        {
            "monday_flat_4": "Monday: Flat $4 (Fruit Tea)",
            "wednesday_bogo_50": "Wednesday: BOGO 50% (Fruit Tea)",
        }
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    values = df["total_net_sales"].tolist()
    title = "Happy Hour Net Sales by Promo Day"
    if "estimated_revenue" in df.columns and sum(values) == 0:
        values = df["estimated_revenue"].tolist()
        title = "Happy Hour Estimated Revenue (Qty x Price) by Promo Day"
    ax.bar(df["promo_label"], values, color="#2A9D8F")
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Total Revenue")
    fig.subplots_adjust(bottom=0.26)
    fig.text(
        0.5,
        0.02,
        "Happy Hour window: Mon/Wed 2–5pm (PDT), 2025-10-20 to 2025-12-31",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    ax.tick_params(axis="x", rotation=15, labelsize=9)
    ax.yaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}"))
    _annotate_bars(ax, values, "${:,.0f}")
    output = plots_dir / "hh_net_sales_by_day.png"
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def plot_total_items_by_day(summary_path: Path, plots_dir: Path) -> Path:
    df = pd.read_csv(summary_path)
    df["promo_label"] = df["promo_day_type"].replace(
        {
            "monday_flat_4": "Monday: Flat $4 (Fruit Tea)",
            "wednesday_bogo_50": "Wednesday: BOGO 50% (Fruit Tea)",
        }
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(df["promo_label"], df["total_items"], color="#457B9D")
    ax.set_title("Happy Hour Total Items by Promo Day")
    ax.set_xlabel("")
    ax.set_ylabel("Total Items")
    fig.subplots_adjust(bottom=0.26)
    fig.text(
        0.5,
        0.02,
        "Happy Hour window: Mon/Wed 2–5pm (PDT), 2025-10-20 to 2025-12-31",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    ax.tick_params(axis="x", rotation=15, labelsize=9)
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    _annotate_bars(ax, df["total_items"].tolist(), "{:,.0f}")
    output = plots_dir / "hh_total_items_by_day.png"
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def plot_fruit_tea_summary(summary_path: Path, plots_dir: Path) -> Path:
    df = pd.read_csv(summary_path)
    df["promo_label"] = df["promo_day_type"].replace(
        {
            "monday_flat_4": "Monday: Flat $4 (Fruit Tea)",
            "wednesday_bogo_50": "Wednesday: BOGO 50% (Fruit Tea)",
        }
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(df["promo_label"], df["fruit_tea_quantity"], color="#E76F51")
    ax.set_title("Fresh Fruit Tea Quantity by Promo Day")
    ax.set_xlabel("")
    ax.set_ylabel("Fruit Tea Quantity")
    fig.subplots_adjust(bottom=0.26)
    fig.text(
        0.5,
        0.02,
        "Happy Hour window: Mon/Wed 2–5pm (PDT), 2025-10-20 to 2025-12-31",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    ax.tick_params(axis="x", rotation=15, labelsize=9)
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    _annotate_bars(ax, df["fruit_tea_quantity"].tolist(), "{:,.0f}")
    output = plots_dir / "fruit_tea_quantity_by_day.png"
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def plot_fruit_tea_net_sales(summary_path: Path, plots_dir: Path) -> Path:
    df = pd.read_csv(summary_path)
    df["promo_label"] = df["promo_day_type"].replace(
        {
            "monday_flat_4": "Monday: Flat $4 (Fruit Tea)",
            "wednesday_bogo_50": "Wednesday: BOGO 50% (Fruit Tea)",
        }
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    values = df["fruit_tea_net_sales"].tolist() if "fruit_tea_net_sales" in df.columns else [0] * len(df)
    ax.bar(df["promo_label"], values, color="#6D597A")
    ax.set_title("Fresh Fruit Tea Net Sales by Promo Day")
    ax.set_xlabel("")
    ax.set_ylabel("Net Sales")
    fig.subplots_adjust(bottom=0.26)
    fig.text(
        0.5,
        0.02,
        "Happy Hour window: Mon/Wed 2–5pm (PDT), 2025-10-20 to 2025-12-31",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    ax.tick_params(axis="x", rotation=15, labelsize=9)
    ax.yaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}"))
    _annotate_bars(ax, values, "${:,.0f}")
    output = plots_dir / "fruit_tea_net_sales_by_day.png"
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def plot_fruit_tea_effective_price(summary_path: Path, plots_dir: Path) -> Path:
    df = pd.read_csv(summary_path)
    df["promo_label"] = df["promo_day_type"].replace(
        {
            "monday_flat_4": "Monday: Flat $4 (Fruit Tea)",
            "wednesday_bogo_50": "Wednesday: BOGO 50% (Fruit Tea)",
        }
    )
    values = df["avg_price_per_item"].tolist() if "avg_price_per_item" in df.columns else [0] * len(df)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(df["promo_label"], values, color="#4A4E69")
    ax.set_title("Fresh Fruit Tea Effective Price per Drink")
    ax.set_xlabel("")
    ax.set_ylabel("Effective Price per Drink")
    fig.subplots_adjust(bottom=0.26)
    fig.text(
        0.5,
        0.02,
        "Happy Hour window: Mon/Wed 2–5pm (PDT), 2025-10-20 to 2025-12-31",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    ax.tick_params(axis="x", rotation=15, labelsize=9)
    ax.yaxis.set_major_formatter(StrMethodFormatter("${x:,.2f}"))
    _annotate_bars(ax, values, "${:,.2f}")
    output = plots_dir / "fruit_tea_effective_price_by_day.png"
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def plot_top_items(product_lift_path: Path, plots_dir: Path) -> list[Path]:
    df = pd.read_csv(product_lift_path)
    outputs: list[Path] = []
    for promo_day in df["promo_day_type"].dropna().unique().tolist():
        top = df[df["promo_day_type"] == promo_day].sort_values(
            "total_quantity", ascending=False
        ).head(5)
        fig, ax = plt.subplots(figsize=(9.5, 5.5))
        ax.barh(top["item_name"], top["total_quantity"], color="#264653")
        title_label = {
            "monday_flat_4": "Monday: Flat $4 (Fruit Tea)",
            "wednesday_bogo_50": "Wednesday: BOGO 50% (Fruit Tea)",
        }.get(promo_day, promo_day)
        ax.set_title(f"Top 5 Fresh Fruit Tea Items ({title_label})")
        ax.set_xlabel("Total Quantity")
        ax.set_ylabel("Item Name")
        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        for i, val in enumerate(top["total_quantity"].tolist()):
            ax.text(val, i, f" {val:,.0f}", va="center", fontsize=9)
        output = plots_dir / f"top_fruit_tea_items_{promo_day}.png"
        fig.subplots_adjust(top=0.9)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.savefig(output, dpi=200)
        plt.close(fig)
        outputs.append(output)
    return outputs


def main() -> None:
    project_root = get_project_root()
    exports_dir = project_root / "exports"
    plots_dir = ensure_plots_dir(project_root)

    set_plot_style()

    summary_by_day = exports_dir / "hh_summary_by_promo_day.csv"
    fruit_tea_summary = exports_dir / "fruit_tea_summary_by_promo_day.csv"
    product_lift = exports_dir / "fruit_tea_product_lift_by_promo_day.csv"

    saved: list[Path] = []
    saved.append(plot_summary_by_day(summary_by_day, plots_dir))
    saved.append(plot_total_items_by_day(summary_by_day, plots_dir))
    saved.append(plot_fruit_tea_summary(fruit_tea_summary, plots_dir))
    saved.append(plot_fruit_tea_net_sales(fruit_tea_summary, plots_dir))
    saved.append(plot_fruit_tea_effective_price(fruit_tea_summary, plots_dir))
    saved.extend(plot_top_items(product_lift, plots_dir))

    print("\nSaved plots:")
    for path in saved:
        print(f"- {path}")


if __name__ == "__main__":
    main()
