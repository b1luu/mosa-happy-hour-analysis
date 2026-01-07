from __future__ import annotations

from pathlib import Path

import pandas as pd


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def validate_processed_data(df_full: pd.DataFrame, df_hh: pd.DataFrame) -> None:
    print("\n=== Validation: Schema Checks ===")
    required_cols = [
        "order_datetime",
        "order_date",
        "weekday_name",
        "order_hour",
        "category",
        "item_name",
        "quantity",
        "gross_sales",
        "discounts",
        "net_total",
        "promo_day_type",
        "is_happy_hour",
    ]
    missing = [c for c in required_cols if c not in df_full.columns]
    if missing:
        print(f"Missing required columns in full data: {missing}")
    else:
        print("All required columns present in full data.")

    print("\n=== Validation: Row Counts ===")
    print(f"Full rows: {len(df_full)}")
    print(f"HH rows: {len(df_hh)}")
    if len(df_hh) > len(df_full):
        print("Warning: HH rows exceed full rows.")

    print("\n=== Validation: Happy Hour Logic ===")
    if "is_happy_hour" in df_hh.columns:
        not_hh = df_hh[~df_hh["is_happy_hour"]]
        if not_hh.empty:
            print("All HH rows are flagged is_happy_hour == True.")
        else:
            print(f"Warning: {len(not_hh)} HH rows are not flagged correctly.")

    for col in ["order_hour", "weekday_name"]:
        if col in df_hh.columns:
            print(f"{col} unique values: {sorted(df_hh[col].dropna().unique().tolist())}")

    print("\n=== Validation: Numeric Sanity ===")
    numeric_cols = ["quantity", "gross_sales", "discounts", "net_total"]
    for col in numeric_cols:
        if col in df_full.columns:
            nan_count = int(pd.to_numeric(df_full[col], errors="coerce").isna().sum())
            print(f"{col} NaN count: {nan_count}")

    if all(col in df_full.columns for col in ["gross_sales", "discounts", "net_total"]):
        gross = pd.to_numeric(df_full["gross_sales"], errors="coerce").fillna(0)
        discounts = pd.to_numeric(df_full["discounts"], errors="coerce").fillna(0)
        net = pd.to_numeric(df_full["net_total"], errors="coerce").fillna(0)
        diff = (gross - discounts - net).abs()
        off = int((diff > 0.01).sum())
        print(f"Rows where gross - discounts != net_total (> $0.01): {off}")


def main() -> None:
    project_root = get_project_root()
    processed_dir = project_root / "data" / "processed"

    df_full = pd.read_csv(processed_dir / "items_clean_full.csv")
    df_hh = pd.read_csv(processed_dir / "items_happy_hour_window.csv")

    validate_processed_data(df_full, df_hh)


if __name__ == "__main__":
    main()
