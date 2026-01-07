from __future__ import annotations

from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_raw_items_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    print(f"Loaded raw CSV: {path}")
    print(f"Rows loaded: {len(df)}")
    print(f"Raw columns: {list(df.columns)}")
    return df


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "item": "item_name",
        "qty": "quantity",
        "point_name": "point_name",
        "modifiers_applied": "modifiers_applied",
        "gross_sales": "gross_sales",
        "net_total": "net_total",
        "net_sales": "net_total",
        "time_zone": "time_zone",
    }

    standardized = []
    for col in df.columns:
        clean = col.strip().lower().replace(" ", "_").replace("/", "_")
        clean = rename_map.get(clean, clean)
        standardized.append(clean)

    df = df.copy()
    df.columns = standardized
    print(f"Standardized columns: {list(df.columns)}")
    return df


def normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    text_cols = df.select_dtypes(include=["object"]).columns
    for col in text_cols:
        df[col] = (
            df[col]
            .astype(str)
            .map(lambda v: unicodedata.normalize("NFKC", v).strip())
        )
        df.loc[df[col] == "", col] = pd.NA
    return df


def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = ["quantity", "price", "gross_sales", "discounts", "net_total", "tax"]
    def _clean_numeric(series: pd.Series) -> pd.Series:
        cleaned = (
            series.astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.replace("(", "-", regex=False)
            .str.replace(")", "", regex=False)
        )
        cleaned = cleaned.where(~cleaned.str.strip().eq(""), pd.NA)
        return cleaned

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(_clean_numeric(df[col]), errors="coerce")
    nan_counts = {col: int(df[col].isna().sum()) for col in numeric_cols if col in df.columns}
    print(f"NaN counts for numeric columns: {nan_counts}")
    return df


def drop_empty_and_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    before = len(df)
    df = df.dropna(how="all")
    before_dupes = len(df)
    df = df.drop_duplicates()
    dupes_removed = before_dupes - len(df)
    print(f"Duplicate rows removed: {dupes_removed}")
    print(f"Rows after dropping empty/duplicates: {len(df)} (from {before})")
    return df


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "date_time" not in df.columns:
        if "date" in df.columns and "time" in df.columns:
            df["date_time"] = (
                df["date"].astype(str).str.strip() + " " + df["time"].astype(str).str.strip()
            )
        else:
            raise KeyError("date_time column is required for time parsing")

    parsed = pd.to_datetime(df["date_time"], errors="coerce")
    parsed = parsed.dt.tz_localize("America/Los_Angeles", ambiguous="NaT", nonexistent="NaT")

    df["order_datetime"] = parsed
    df = df.dropna(subset=["order_datetime"])
    df["order_date"] = df["order_datetime"].dt.date
    df["weekday_name"] = df["order_datetime"].dt.day_name()
    df["order_hour"] = df["order_datetime"].dt.hour

    print(
        f"Order datetime range: {df['order_datetime'].min()} -> {df['order_datetime'].max()}"
    )
    print(f"Weekdays present: {sorted(df['weekday_name'].dropna().unique().tolist())}")
    return df


def add_happy_hour_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    promo_start = pd.Timestamp("2025-10-20", tz="America/Los_Angeles")

    is_hh = (
        (df["order_datetime"] >= promo_start)
        & (df["weekday_name"].isin(["Monday", "Wednesday"]))
        & (df["order_hour"].between(14, 16, inclusive="both"))
    )
    df["is_happy_hour"] = is_hh

    df["promo_day_type"] = np.select(
        [df["is_happy_hour"] & (df["weekday_name"] == "Monday"),
         df["is_happy_hour"] & (df["weekday_name"] == "Wednesday")],
        ["monday_flat_4", "wednesday_bogo_50"],
        default="non_happy_hour",
    )

    print(f"Rows flagged as is_happy_hour: {int(df['is_happy_hour'].sum())}")
    print("promo_day_type counts:")
    print(df["promo_day_type"].value_counts(dropna=False))
    return df


def slim_to_allowed_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required_columns = [
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
    optional_columns = [
        "price",
        "modifiers_applied",
        "point_name",
    ]
    sensitive_columns = [
        "customer_name",
        "customer_id",
        "customer_phone_number",
        "courier_phone_number",
        "vehicle",
        "driver_name",
        "channel",
        "fulfillment_note",
        "token",
        "card_brand",
        "pan_suffix",
        "order_reference_id",
        "payment_id",
        "transaction_id",
        "owner_reference_id",
        "gtin",
        "commission",
        "employee",
    ]

    before_cols = set(df.columns)
    df = df.drop(columns=sensitive_columns, errors="ignore")
    keep_cols = [c for c in required_columns + optional_columns if c in df.columns]
    df = df[keep_cols]
    after_cols = set(df.columns)
    removed = sorted(before_cols - after_cols)

    print(f"Columns before slimming: {len(before_cols)}")
    print(f"Columns after slimming: {len(after_cols)}")
    print(f"Removed columns: {removed}")
    return df


def save_processed(df_full: pd.DataFrame, df_hh: pd.DataFrame, processed_dir: Path) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    full_path = processed_dir / "items_clean_full.csv"
    hh_path = processed_dir / "items_happy_hour_window.csv"
    df_full.to_csv(full_path, index=False)
    df_hh.to_csv(hh_path, index=False)
    print(f"Saved processed full dataset: {full_path}")
    print(f"Saved processed HH dataset: {hh_path}")


def main() -> None:
    project_root = get_project_root()
    raw_path = project_root / "data" / "raw" / "items-2025-10-20-2026-01-01.csv"

    df = load_raw_items_csv(raw_path)
    df = standardize_column_names(df)
    df = normalize_text_columns(df)
    df = convert_numeric_columns(df)
    df = drop_empty_and_duplicates(df)
    df = add_time_columns(df)
    df = add_happy_hour_flags(df)
    df = slim_to_allowed_columns(df)

    df_full_clean = df
    df_hh_window = df_full_clean[df_full_clean["is_happy_hour"]].copy()

    processed_dir = project_root / "data" / "processed"
    save_processed(df_full_clean, df_hh_window, processed_dir)

    print("\n=== Pipeline Summary ===")
    print(f"Total rows (clean full): {len(df_full_clean)}")
    print(f"Rows in Happy Hour window: {len(df_hh_window)}")
    print(f"Processed outputs at: {processed_dir}")


if __name__ == "__main__":
    main()
