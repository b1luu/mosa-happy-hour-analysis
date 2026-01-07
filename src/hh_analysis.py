from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_processed_data(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    hh_path = project_root / "data" / "processed" / "items_happy_hour_window.csv"
    full_path = project_root / "data" / "processed" / "items_clean_full.csv"

    df_hh = pd.read_csv(hh_path)
    df_full = pd.read_csv(full_path)

    print(f"Loaded HH rows: {len(df_hh)}")
    print(f"Loaded full rows: {len(df_full)}")

    for col in ["is_happy_hour", "promo_day_type"]:
        if col not in df_hh.columns:
            raise KeyError(f"Required column missing in HH data: {col}")

    if "net_total" not in df_hh.columns and "net_sales" in df_hh.columns:
        df_hh["net_total"] = df_hh["net_sales"]
        print("Warning: net_total missing; using net_sales as net_total in HH data.")

    if "net_total" not in df_hh.columns or pd.to_numeric(df_hh["net_total"], errors="coerce").fillna(0).sum() == 0:
        if "price" in df_hh.columns and "quantity" in df_hh.columns:
            df_hh["net_total"] = pd.to_numeric(df_hh["price"], errors="coerce").fillna(0) * pd.to_numeric(
                df_hh["quantity"], errors="coerce"
            ).fillna(0)
            print("Warning: net_total missing/zero; computed from price * quantity.")
        elif "gross_sales" in df_hh.columns and "discounts" in df_hh.columns:
            df_hh["net_total"] = pd.to_numeric(df_hh["gross_sales"], errors="coerce").fillna(0) - pd.to_numeric(
                df_hh["discounts"], errors="coerce"
            ).fillna(0)
            print("Warning: net_total missing/zero; computed from gross_sales - discounts.")
        elif "gross_sales" in df_hh.columns:
            df_hh["net_total"] = pd.to_numeric(df_hh["gross_sales"], errors="coerce")
            print("Warning: net_total missing/zero; using gross_sales as net_total.")

    if not df_hh["is_happy_hour"].eq(True).all():
        print("Warning: Not all rows in df_hh have is_happy_hour == True")

    return df_hh, df_full


def is_fruit_tea_category(category: str) -> bool:
    if category is None or pd.isna(category):
        return False
    value = str(category).lower()
    keywords = ["fresh fruit tea", "fruit tea", "鲜果茶"]
    return any(key in value for key in keywords)


def add_fruit_tea_flag(df_hh: pd.DataFrame) -> pd.DataFrame:
    df_hh = df_hh.copy()
    df_hh["is_fruit_tea_series"] = df_hh["category"].apply(is_fruit_tea_category)
    print("Fruit tea flag counts:")
    print(df_hh["is_fruit_tea_series"].value_counts(dropna=False))
    return df_hh


def _fill_numeric_for_agg(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def summarize_by_promo_day(df_hh: pd.DataFrame) -> pd.DataFrame:
    df = _fill_numeric_for_agg(df_hh, ["quantity", "gross_sales", "net_total", "price"])
    if "price" in df.columns and "quantity" in df.columns:
        df["estimated_revenue"] = df["price"] * df["quantity"]

    agg_map: dict[str, tuple[str, str]] = {
        "total_items": ("quantity", "sum"),
        "total_gross_sales": ("gross_sales", "sum"),
        "total_net_sales": ("net_total", "sum"),
        "num_rows": ("promo_day_type", "size"),
    }
    if "estimated_revenue" in df.columns:
        agg_map["estimated_revenue"] = ("estimated_revenue", "sum")

    grouped = df.groupby("promo_day_type", dropna=False).agg(**agg_map)
    grouped = grouped.reset_index()
    grouped["avg_price_per_item"] = np.where(
        grouped["total_items"] > 0,
        grouped["total_net_sales"] / grouped["total_items"],
        np.nan,
    )
    grouped = grouped.sort_values("promo_day_type")

    print("\n=== Happy Hour Summary by promo_day_type ===")
    print(grouped.to_string(index=False))
    return grouped


def summarize_fruit_tea_by_promo_day(df_hh: pd.DataFrame) -> pd.DataFrame:
    df_ft = df_hh[df_hh["is_fruit_tea_series"] == True].copy()
    df_ft = _fill_numeric_for_agg(df_ft, ["quantity", "net_total"])

    grouped = df_ft.groupby("promo_day_type", dropna=False).agg(
        fruit_tea_quantity=("quantity", "sum"),
        fruit_tea_net_sales=("net_total", "sum"),
    )
    grouped = grouped.reset_index()
    grouped["avg_price_per_item"] = np.where(
        grouped["fruit_tea_quantity"] > 0,
        grouped["fruit_tea_net_sales"] / grouped["fruit_tea_quantity"],
        np.nan,
    )
    grouped = grouped.sort_values("promo_day_type")

    print("\n=== Fruit Tea Summary by promo_day_type ===")
    print(grouped.to_string(index=False))
    return grouped


def summarize_product_lift(df_hh: pd.DataFrame) -> pd.DataFrame:
    df_ft = df_hh[df_hh["is_fruit_tea_series"] == True].copy()
    df_ft = _fill_numeric_for_agg(df_ft, ["quantity", "net_total"])

    grouped = df_ft.groupby(["promo_day_type", "item_name"], dropna=False).agg(
        total_quantity=("quantity", "sum"),
        total_net_sales=("net_total", "sum"),
    )
    grouped = grouped.reset_index()
    grouped["rank_within_day"] = grouped.groupby("promo_day_type")["total_quantity"].rank(
        ascending=False, method="dense"
    )
    grouped = grouped.sort_values(["promo_day_type", "total_quantity"], ascending=[True, False])

    print("\n=== Top Fruit Tea Items by promo_day_type ===")
    for day in grouped["promo_day_type"].dropna().unique().tolist():
        top = grouped[grouped["promo_day_type"] == day].head(10)
        print(f"\n-- {day} (top 10 by quantity) --")
        print(top.to_string(index=False))

    return grouped


def summarize_basket_behavior(df_hh: pd.DataFrame) -> pd.DataFrame | None:
    candidate_ids = ["transaction_id", "order_id", "ticket_id"]
    order_id = next((col for col in candidate_ids if col in df_hh.columns), None)
    if order_id is None:
        print("No order-level ID column found; skipping basket-level analysis.")
        return None

    df = _fill_numeric_for_agg(df_hh, ["quantity", "net_total"])
    order_level = df.groupby(order_id, dropna=False).agg(
        total_items_per_order=("quantity", "sum"),
        total_sales_per_order=("net_total", "sum"),
        promo_day_type=("promo_day_type", "first"),
    )

    basket_summary = order_level.groupby("promo_day_type", dropna=False).agg(
        orders_count=("promo_day_type", "size"),
        avg_items_per_order=("total_items_per_order", "mean"),
        avg_sales_per_order=("total_sales_per_order", "mean"),
    )
    basket_summary = basket_summary.reset_index().sort_values("promo_day_type")

    print("\n=== Basket Behavior by promo_day_type ===")
    print(basket_summary.to_string(index=False))

    return basket_summary


def ensure_exports_dir(project_root: Path) -> Path:
    exports_dir = project_root / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    return exports_dir


def main() -> None:
    project_root = get_project_root()
    df_hh, df_full = load_processed_data(project_root)
    df_hh = add_fruit_tea_flag(df_hh)

    exports_dir = ensure_exports_dir(project_root)

    summary_by_day = summarize_by_promo_day(df_hh)
    ft_summary = summarize_fruit_tea_by_promo_day(df_hh)
    product_lift = summarize_product_lift(df_hh)
    basket_summary = summarize_basket_behavior(df_hh)

    summary_by_day.to_csv(exports_dir / "hh_summary_by_promo_day.csv", index=False)
    ft_summary.to_csv(exports_dir / "fruit_tea_summary_by_promo_day.csv", index=False)
    product_lift.to_csv(exports_dir / "fruit_tea_product_lift_by_promo_day.csv", index=False)
    if basket_summary is not None:
        basket_summary.to_csv(
            exports_dir / "hh_basket_behavior_by_promo_day.csv",
            index=False,
        )

    saved = [
        exports_dir / "hh_summary_by_promo_day.csv",
        exports_dir / "fruit_tea_summary_by_promo_day.csv",
        exports_dir / "fruit_tea_product_lift_by_promo_day.csv",
    ]
    if basket_summary is not None:
        saved.append(exports_dir / "hh_basket_behavior_by_promo_day.csv")

    print("\nSaved exports:")
    for path in saved:
        print(f"- {path}")


if __name__ == "__main__":
    main()
