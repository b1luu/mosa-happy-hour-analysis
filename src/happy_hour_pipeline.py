from __future__ import annotations

from pathlib import Path
import json
import re
import unicodedata

import numpy as np
import pandas as pd

BASELINE_START = pd.Timestamp("2025-08-29", tz="America/Los_Angeles")
BASELINE_END = pd.Timestamp("2025-10-19", tz="America/Los_Angeles")
PROMO_START = pd.Timestamp("2025-10-20", tz="America/Los_Angeles")
PROMO_END = pd.Timestamp("2025-12-31", tz="America/Los_Angeles")
BASELINE_START_DATE = BASELINE_START.date()
BASELINE_END_DATE = BASELINE_END.date()
PROMO_START_DATE = PROMO_START.date()
PROMO_END_DATE = PROMO_END.date()
HH_START_HOUR = 14
HH_END_HOUR = 17


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_raw_items_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    print(f"Loaded raw CSV: {path}")
    print(f"Rows loaded: {len(df)}")
    print(f"Raw columns: {list(df.columns)}")
    return df


def load_raw_items_csvs(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frames.append(load_raw_items_csv(path))
    combined = pd.concat(frames, ignore_index=True)
    print(f"Combined raw rows: {len(combined)}")
    return combined


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
    df["day_of_week"] = df["order_datetime"].dt.strftime("%a")
    df["order_hour"] = df["order_datetime"].dt.hour

    print(
        f"Order datetime range: {df['order_datetime'].min()} -> {df['order_datetime'].max()}"
    )
    print(f"Weekdays present: {sorted(df['weekday_name'].dropna().unique().tolist())}")
    return df


def add_period_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    order_date = pd.to_datetime(df["order_date"], errors="coerce").dt.date
    baseline_mask = (order_date >= BASELINE_START_DATE) & (order_date <= BASELINE_END_DATE)
    promo_mask = (order_date >= PROMO_START_DATE) & (order_date <= PROMO_END_DATE)

    df["period_label"] = np.select(
        [baseline_mask, promo_mask],
        ["baseline", "promo"],
        default="outside_scope",
    )
    print(df["period_label"].value_counts(dropna=False))
    return df


def add_happy_hour_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_hh_window"] = df["order_hour"].between(HH_START_HOUR, HH_END_HOUR - 1, inclusive="both")
    is_promo_hh = (
        df["is_hh_window"]
        & (df["period_label"] == "promo")
        & (df["weekday_name"].isin(["Monday", "Wednesday"]))
    )
    df["is_happy_hour"] = is_promo_hh

    df["promo_day_type"] = np.select(
        [
            df["is_happy_hour"] & (df["weekday_name"] == "Monday"),
            df["is_happy_hour"] & (df["weekday_name"] == "Wednesday"),
        ],
        ["monday_flat_4", "wednesday_bogo_50"],
        default="non_happy_hour",
    )

    print(f"Rows flagged as is_hh_window: {int(df['is_hh_window'].sum())}")
    print(f"Rows flagged as is_happy_hour: {int(df['is_happy_hour'].sum())}")
    print("promo_day_type counts:")
    print(df["promo_day_type"].value_counts(dropna=False))
    return df


def slim_to_allowed_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required_columns = [
        "order_id",
        "order_datetime",
        "order_date",
        "weekday_name",
        "day_of_week",
        "order_hour",
        "period_label",
        "category",
        "item_name",
        "sku",
        "quantity",
        "gross_sales",
        "discounts",
        "net_total",
        "promo_day_type",
        "is_hh_window",
        "is_happy_hour",
        "is_fresh_fruit_tea_item",
        "fresh_fruit_tea_qty",
        "fresh_fruit_tea_net_sales",
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


def load_fresh_fruit_tea_mapping(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Fresh fruit tea config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        mapping = json.load(f)
    return mapping


def flag_fresh_fruit_tea_items(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    df = df.copy()
    categories = [c.lower() for c in mapping.get("categories", []) if c]
    name_keywords = [k.lower() for k in mapping.get("name_keywords", []) if k]
    name_regex = [re.compile(rgx, flags=re.IGNORECASE) for rgx in mapping.get("name_regex", []) if rgx]
    sku_allowlist = [s.lower() for s in mapping.get("sku_allowlist", []) if s]

    def _match_row(row: pd.Series) -> bool:
        category = str(row.get("category") or "").lower()
        item_name = str(row.get("item_name") or "").lower()
        sku = str(row.get("sku") or "").lower()

        if categories and any(cat in category for cat in categories):
            return True
        if sku_allowlist and sku in sku_allowlist:
            return True
        if name_keywords and any(key in item_name for key in name_keywords):
            return True
        return any(rgx.search(item_name) for rgx in name_regex)

    df["is_fresh_fruit_tea_item"] = df.apply(_match_row, axis=1)
    df["fresh_fruit_tea_qty"] = np.where(df["is_fresh_fruit_tea_item"], df["quantity"], 0)
    df["fresh_fruit_tea_net_sales"] = np.where(df["is_fresh_fruit_tea_item"], df["net_total"], 0)
    print("Fresh fruit tea item counts:")
    print(df["is_fresh_fruit_tea_item"].value_counts(dropna=False))
    return df


def add_order_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "transaction_id" in df.columns:
        df["order_id"] = df["transaction_id"]
    elif "payment_id" in df.columns:
        df["order_id"] = df["payment_id"]
    else:
        df["order_id"] = pd.Series(range(len(df)), index=df.index).astype(str)
    missing = df["order_id"].isna().sum()
    if missing:
        df.loc[df["order_id"].isna(), "order_id"] = (
            df.loc[df["order_id"].isna(), "order_datetime"].astype(str)
            + "-"
            + df.loc[df["order_id"].isna()].index.astype(str)
        )
        print(f"Warning: filled {missing} missing order_id values with fallback IDs.")
    return df


def build_order_level_fact(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    agg_map = {
        "order_datetime": ("order_datetime", "min"),
        "order_date": ("order_date", "first"),
        "weekday_name": ("weekday_name", "first"),
        "day_of_week": ("day_of_week", "first"),
        "order_hour": ("order_hour", "first"),
        "period_label": ("period_label", "first"),
        "net_sales": ("net_total", "sum"),
        "gross_sales": ("gross_sales", "sum"),
        "discounts": ("discounts", "sum"),
        "item_count": ("quantity", "sum"),
        "contains_fresh_fruit_tea": ("is_fresh_fruit_tea_item", "max"),
        "fresh_fruit_tea_qty": ("fresh_fruit_tea_qty", "sum"),
        "fresh_fruit_tea_net_sales": ("fresh_fruit_tea_net_sales", "sum"),
    }
    order_level = df.groupby("order_id", dropna=False).agg(**agg_map).reset_index()
    order_level["is_hh_window"] = order_level["order_hour"].between(
        HH_START_HOUR, HH_END_HOUR - 1, inclusive="both"
    )
    order_level["is_happy_hour"] = (
        order_level["is_hh_window"]
        & (order_level["period_label"] == "promo")
        & (order_level["weekday_name"].isin(["Monday", "Wednesday"]))
    )
    order_level["promo_day_type"] = np.select(
        [
            order_level["is_happy_hour"] & (order_level["weekday_name"] == "Monday"),
            order_level["is_happy_hour"] & (order_level["weekday_name"] == "Wednesday"),
        ],
        ["monday_flat_4", "wednesday_bogo_50"],
        default="non_happy_hour",
    )
    return order_level


def save_processed(
    df_full: pd.DataFrame,
    df_hh: pd.DataFrame,
    orders_full: pd.DataFrame,
    orders_hh: pd.DataFrame,
    processed_dir: Path,
) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    full_path = processed_dir / "items_clean_full.csv"
    hh_path = processed_dir / "items_happy_hour_window.csv"
    orders_full_path = processed_dir / "orders_clean_full.csv"
    orders_hh_path = processed_dir / "orders_hh_window.csv"
    df_full.to_csv(full_path, index=False)
    df_hh.to_csv(hh_path, index=False)
    orders_full.to_csv(orders_full_path, index=False)
    orders_hh.to_csv(orders_hh_path, index=False)
    print(f"Saved processed full dataset: {full_path}")
    print(f"Saved processed HH dataset: {hh_path}")
    print(f"Saved processed orders dataset: {orders_full_path}")
    print(f"Saved processed orders HH dataset: {orders_hh_path}")


def main() -> None:
    project_root = get_project_root()
    baseline_path = project_root / "data" / "raw" / "items-2025-08-29-2025-10-20.csv"
    promo_path = project_root / "data" / "raw" / "items-2025-10-20-2026-01-01.csv"

    df = load_raw_items_csvs([baseline_path, promo_path])
    df = standardize_column_names(df)
    df = normalize_text_columns(df)
    df = convert_numeric_columns(df)
    df = drop_empty_and_duplicates(df)
    df = add_time_columns(df)
    df = add_period_label(df)
    df = add_order_id(df)
    config_path = project_root / "config" / "fresh_fruit_tea_mapping.json"
    mapping = load_fresh_fruit_tea_mapping(config_path)
    df = flag_fresh_fruit_tea_items(df, mapping)
    df = add_happy_hour_flags(df)
    df = slim_to_allowed_columns(df)

    df_full_clean = df
    df_hh_window = df_full_clean[df_full_clean["is_happy_hour"]].copy()

    orders_full = build_order_level_fact(df_full_clean)
    orders_hh = orders_full[orders_full["is_hh_window"]].copy()

    processed_dir = project_root / "data" / "processed"
    save_processed(df_full_clean, df_hh_window, orders_full, orders_hh, processed_dir)

    print("\n=== Pipeline Summary ===")
    print(f"Total rows (clean full): {len(df_full_clean)}")
    print(f"Rows in Happy Hour window: {len(df_hh_window)}")
    print(f"Order rows (clean full): {len(orders_full)}")
    print(f"Order rows in HH window: {len(orders_hh)}")
    print(f"Processed outputs at: {processed_dir}")


if __name__ == "__main__":
    main()
