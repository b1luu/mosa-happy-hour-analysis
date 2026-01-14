from __future__ import annotations

from pathlib import Path

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
HH_HOURS = 3
PRIMARY_DAYS = ["Monday", "Wednesday"]
BOOTSTRAP_SEED = 42
BOOTSTRAP_SAMPLES = 2000


METRIC_LABELS = {
    "total_transactions": "Total Transactions",
    "transactions_per_hour": "Transactions per Hour",
    "net_sales_hh": "Net Sales",
    "revenue_per_hour": "Revenue per Hour",
    "avg_order_value": "Avg Order Value",
    "items_per_transaction": "Items per Transaction",
    "fresh_fruit_tea_qty": "Fresh Fruit Tea Qty",
    "fresh_fruit_tea_net_sales": "Fresh Fruit Tea Net Sales",
    "fresh_fruit_tea_txn_share": "Fresh Fruit Tea Txn Share",
    "fresh_fruit_tea_qty_per_txn": "Fresh Fruit Tea Qty per Txn",
    "fresh_fruit_tea_sales_share": "Fresh Fruit Tea Sales Share",
    "discount_per_hour": "Discount per Hour",
    "discount_rate": "Discount Rate",
}

CI_METRICS = {"net_sales_hh", "transactions_per_hour"}


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_exports_dir(project_root: Path) -> Path:
    exports_dir = project_root / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    return exports_dir


def load_order_data(project_root: Path) -> pd.DataFrame:
    path = project_root / "data" / "processed" / "orders_clean_full.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing processed orders file: {path}. Run src/happy_hour_pipeline.py first."
        )
    df = pd.read_csv(path)
    parsed = pd.to_datetime(df["order_datetime"], errors="coerce", utc=True)
    df["order_datetime"] = parsed.dt.tz_convert("America/Los_Angeles")
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce").dt.date
    if "weekday_name" not in df.columns:
        df["weekday_name"] = pd.to_datetime(df["order_datetime"]).dt.day_name()
    if "day_of_week" not in df.columns:
        df["day_of_week"] = pd.to_datetime(df["order_datetime"]).dt.strftime("%a")
    if "order_hour" not in df.columns:
        df["order_hour"] = pd.to_datetime(df["order_datetime"]).dt.hour
    if "period_label" not in df.columns:
        df = add_period_label(df)

    numeric_cols = [
        "net_sales",
        "gross_sales",
        "discounts",
        "item_count",
        "fresh_fruit_tea_qty",
        "fresh_fruit_tea_net_sales",
        "total_transactions",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "contains_fresh_fruit_tea" in df.columns:
        raw = df["contains_fresh_fruit_tea"]
        numeric = pd.to_numeric(raw, errors="coerce")
        bool_map = raw.astype(str).str.lower().map({"true": 1, "false": 0})
        df["contains_fresh_fruit_tea"] = bool_map.fillna(numeric).fillna(0)
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
    return df


def ensure_hh_window_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "order_hour" not in df.columns:
        df["order_hour"] = pd.to_datetime(df["order_datetime"]).dt.hour
    if "is_hh_window" not in df.columns:
        df["is_hh_window"] = df["order_hour"].between(HH_START_HOUR, HH_END_HOUR - 1, inclusive="both")
    if "is_happy_hour" not in df.columns:
        df["is_happy_hour"] = (
            df["is_hh_window"]
            & (df["period_label"] == "promo")
            & df["weekday_name"].isin(PRIMARY_DAYS)
        )
    return df


def build_expected_dates() -> pd.DataFrame:
    date_index = pd.date_range(BASELINE_START_DATE, PROMO_END_DATE, freq="D")
    df = pd.DataFrame({"order_date": date_index.date})
    df["weekday_name"] = date_index.day_name()
    df["day_of_week"] = date_index.strftime("%a")
    df["period_label"] = np.select(
        [date_index <= pd.Timestamp(BASELINE_END_DATE), date_index >= pd.Timestamp(PROMO_START_DATE)],
        ["baseline", "promo"],
        default="outside_scope",
    )
    df = df[df["weekday_name"].isin(PRIMARY_DAYS)]
    df = df[df["period_label"].isin(["baseline", "promo"])]
    return df


def aggregate_daily(df_orders: pd.DataFrame) -> pd.DataFrame:
    df = ensure_hh_window_flags(df_orders)
    df = df[df["is_hh_window"] & df["weekday_name"].isin(PRIMARY_DAYS)].copy()

    agg_map = {
        "total_transactions": ("order_id", "nunique"),
        "net_sales_hh": ("net_sales", "sum"),
        "gross_sales_hh": ("gross_sales", "sum"),
        "discounts": ("discounts", "sum"),
        "total_item_qty": ("item_count", "sum"),
        "fresh_fruit_tea_qty": ("fresh_fruit_tea_qty", "sum"),
        "fresh_fruit_tea_net_sales": ("fresh_fruit_tea_net_sales", "sum"),
        "fresh_fruit_tea_txn_count": ("contains_fresh_fruit_tea", "sum"),
    }

    daily = df.groupby(["order_date", "weekday_name", "day_of_week", "period_label"], dropna=False).agg(**agg_map)
    daily = daily.reset_index()

    daily["transactions_per_hour"] = daily["total_transactions"] / HH_HOURS
    daily["revenue_per_hour"] = daily["net_sales_hh"] / HH_HOURS
    daily["avg_order_value"] = np.where(
        daily["total_transactions"] > 0, daily["net_sales_hh"] / daily["total_transactions"], np.nan
    )
    daily["items_per_transaction"] = np.where(
        daily["total_transactions"] > 0, daily["total_item_qty"] / daily["total_transactions"], np.nan
    )
    daily["fresh_fruit_tea_txn_share"] = np.where(
        daily["total_transactions"] > 0,
        daily["fresh_fruit_tea_txn_count"] / daily["total_transactions"],
        np.nan,
    )
    daily["fresh_fruit_tea_qty_per_txn"] = np.where(
        daily["total_transactions"] > 0,
        daily["fresh_fruit_tea_qty"] / daily["total_transactions"],
        np.nan,
    )
    daily["fresh_fruit_tea_sales_share"] = np.where(
        daily["net_sales_hh"] > 0,
        daily["fresh_fruit_tea_net_sales"] / daily["net_sales_hh"],
        np.nan,
    )
    daily["discount_per_hour"] = daily["discounts"] / HH_HOURS
    daily["discount_rate"] = np.where(
        daily["gross_sales_hh"] > 0,
        daily["discounts"] / daily["gross_sales_hh"],
        np.nan,
    )

    hour_counts = df.groupby(["order_date"], dropna=False)["order_hour"].nunique().rename("hh_hours_covered")
    hourly = (
        df.groupby(["order_date", "order_hour"], dropna=False).size().unstack(fill_value=0)
    )
    hourly = hourly.rename(
        columns={
            14: "transactions_2_3",
            15: "transactions_3_4",
            16: "transactions_4_5",
        }
    )
    daily = daily.merge(hour_counts, on="order_date", how="left")
    daily = daily.merge(hourly, on="order_date", how="left")

    expected = build_expected_dates()
    daily = expected.merge(daily, on=["order_date", "weekday_name", "day_of_week", "period_label"], how="left")

    daily["is_missing_day"] = daily["total_transactions"].isna()
    daily["hh_hours_covered"] = daily["hh_hours_covered"].fillna(0)
    daily["is_partial_hh"] = (daily["hh_hours_covered"] < HH_HOURS) & (~daily["is_missing_day"])

    return daily


def flag_outliers(df_daily: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    df = df_daily.copy()
    df["is_outlier"] = False
    df["outlier_metrics"] = ""

    for (period_label, weekday_name), group in df.groupby(["period_label", "weekday_name"], dropna=False):
        if period_label not in {"baseline", "promo"} or weekday_name not in PRIMARY_DAYS:
            continue
        for metric in metrics:
            if metric not in group.columns:
                continue
            series = group[metric].dropna()
            if len(series) < 4:
                continue
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            mask = (group[metric] < lower) | (group[metric] > upper)
            outlier_idx = group.loc[mask].index
            if len(outlier_idx) == 0:
                continue
            df.loc[outlier_idx, "is_outlier"] = True
            df.loc[outlier_idx, "outlier_metrics"] = (
                df.loc[outlier_idx, "outlier_metrics"].str.cat([metric] * len(outlier_idx), sep=", ")
            )

    df["outlier_metrics"] = df["outlier_metrics"].str.strip(", ")
    return df


def bootstrap_ci(
    baseline_mon: pd.Series,
    baseline_wed: pd.Series,
    promo_mon: pd.Series,
    promo_wed: pd.Series,
    seed: int = BOOTSTRAP_SEED,
    samples: int = BOOTSTRAP_SAMPLES,
) -> dict:
    baseline_mon = baseline_mon.dropna().to_numpy()
    baseline_wed = baseline_wed.dropna().to_numpy()
    promo_mon = promo_mon.dropna().to_numpy()
    promo_wed = promo_wed.dropna().to_numpy()

    if min(len(baseline_mon), len(baseline_wed), len(promo_mon), len(promo_wed)) < 2:
        return {
            "baseline_ratio_ci_low": np.nan,
            "baseline_ratio_ci_high": np.nan,
            "promo_ratio_ci_low": np.nan,
            "promo_ratio_ci_high": np.nan,
            "lift_mon_ci_low": np.nan,
            "lift_mon_ci_high": np.nan,
            "lift_wed_ci_low": np.nan,
            "lift_wed_ci_high": np.nan,
            "delta_lift_ci_low": np.nan,
            "delta_lift_ci_high": np.nan,
        }

    rng = np.random.default_rng(seed)
    baseline_ratios = []
    promo_ratios = []
    lift_mon_vals = []
    lift_wed_vals = []
    delta_lift_vals = []

    for _ in range(samples):
        base_mon = rng.choice(baseline_mon, size=len(baseline_mon), replace=True)
        base_wed = rng.choice(baseline_wed, size=len(baseline_wed), replace=True)
        promo_m = rng.choice(promo_mon, size=len(promo_mon), replace=True)
        promo_w = rng.choice(promo_wed, size=len(promo_wed), replace=True)

        base_mon_mean = np.mean(base_mon)
        base_wed_mean = np.mean(base_wed)
        promo_mon_mean = np.mean(promo_m)
        promo_wed_mean = np.mean(promo_w)

        baseline_ratio = base_mon_mean / base_wed_mean if base_wed_mean else np.nan
        promo_ratio = promo_mon_mean / promo_wed_mean if promo_wed_mean else np.nan
        lift_mon = (promo_mon_mean - base_mon_mean) / base_mon_mean if base_mon_mean else np.nan
        lift_wed = (promo_wed_mean - base_wed_mean) / base_wed_mean if base_wed_mean else np.nan
        delta_lift = lift_mon - lift_wed if np.isfinite(lift_mon) and np.isfinite(lift_wed) else np.nan

        baseline_ratios.append(baseline_ratio)
        promo_ratios.append(promo_ratio)
        lift_mon_vals.append(lift_mon)
        lift_wed_vals.append(lift_wed)
        delta_lift_vals.append(delta_lift)

    def _ci(values: list[float]) -> tuple[float, float]:
        arr = np.array(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            return np.nan, np.nan
        return np.percentile(arr, 2.5), np.percentile(arr, 97.5)

    base_ci = _ci(baseline_ratios)
    promo_ci = _ci(promo_ratios)
    lift_mon_ci = _ci(lift_mon_vals)
    lift_wed_ci = _ci(lift_wed_vals)
    delta_ci = _ci(delta_lift_vals)

    return {
        "baseline_ratio_ci_low": base_ci[0],
        "baseline_ratio_ci_high": base_ci[1],
        "promo_ratio_ci_low": promo_ci[0],
        "promo_ratio_ci_high": promo_ci[1],
        "lift_mon_ci_low": lift_mon_ci[0],
        "lift_mon_ci_high": lift_mon_ci[1],
        "lift_wed_ci_low": lift_wed_ci[0],
        "lift_wed_ci_high": lift_wed_ci[1],
        "delta_lift_ci_low": delta_ci[0],
        "delta_lift_ci_high": delta_ci[1],
    }


def summarize_metric(
    df: pd.DataFrame,
    metric: str,
    sample_label: str,
    mask: pd.Series,
) -> dict:
    def _subset(period: str, day: str) -> pd.Series:
        return df.loc[
            mask & (df["period_label"] == period) & (df["weekday_name"] == day), metric
        ].dropna()

    baseline_mon = _subset("baseline", "Monday")
    baseline_wed = _subset("baseline", "Wednesday")
    promo_mon = _subset("promo", "Monday")
    promo_wed = _subset("promo", "Wednesday")

    baseline_mon_mean = baseline_mon.mean()
    baseline_wed_mean = baseline_wed.mean()
    promo_mon_mean = promo_mon.mean()
    promo_wed_mean = promo_wed.mean()

    baseline_ratio = baseline_mon_mean / baseline_wed_mean if baseline_wed_mean else np.nan
    promo_ratio = promo_mon_mean / promo_wed_mean if promo_wed_mean else np.nan
    lift_mon = (promo_mon_mean - baseline_mon_mean) / baseline_mon_mean if baseline_mon_mean else np.nan
    lift_wed = (promo_wed_mean - baseline_wed_mean) / baseline_wed_mean if baseline_wed_mean else np.nan
    delta_lift = lift_mon - lift_wed if np.isfinite(lift_mon) and np.isfinite(lift_wed) else np.nan

    row = {
        "metric": metric,
        "metric_label": METRIC_LABELS.get(metric, metric),
        "sample": sample_label,
        "n_baseline_mon": int(baseline_mon.count()),
        "n_baseline_wed": int(baseline_wed.count()),
        "n_promo_mon": int(promo_mon.count()),
        "n_promo_wed": int(promo_wed.count()),
        "baseline_mon_mean": baseline_mon_mean,
        "baseline_wed_mean": baseline_wed_mean,
        "promo_mon_mean": promo_mon_mean,
        "promo_wed_mean": promo_wed_mean,
        "baseline_mon_median": baseline_mon.median(),
        "baseline_wed_median": baseline_wed.median(),
        "promo_mon_median": promo_mon.median(),
        "promo_wed_median": promo_wed.median(),
        "baseline_mon_std": baseline_mon.std(),
        "baseline_wed_std": baseline_wed.std(),
        "promo_mon_std": promo_mon.std(),
        "promo_wed_std": promo_wed.std(),
        "baseline_ratio": baseline_ratio,
        "promo_ratio": promo_ratio,
        "lift_mon": lift_mon,
        "lift_wed": lift_wed,
        "delta_lift": delta_lift,
    }

    if metric in CI_METRICS:
        row.update(bootstrap_ci(baseline_mon, baseline_wed, promo_mon, promo_wed))

    return row


def build_summary_table(df_daily: pd.DataFrame) -> pd.DataFrame:
    metrics = [metric for metric in METRIC_LABELS if metric in df_daily.columns]

    base_mask = (
        df_daily["period_label"].isin(["baseline", "promo"])
        & df_daily["weekday_name"].isin(PRIMARY_DAYS)
        & (~df_daily["is_missing_day"])
    )

    all_days_mask = base_mask
    no_outliers_mask = base_mask & (~df_daily["is_outlier"]) & (~df_daily["is_partial_hh"])

    rows = []
    for metric in metrics:
        rows.append(summarize_metric(df_daily, metric, "all_days", all_days_mask))
        rows.append(summarize_metric(df_daily, metric, "no_outliers", no_outliers_mask))

    summary = pd.DataFrame(rows)
    return summary


def classify_result(baseline_ratio: float, promo_ratio: float, delta_lift: float) -> str:
    if not np.isfinite(baseline_ratio) or not np.isfinite(promo_ratio) or not np.isfinite(delta_lift):
        return "Insufficient data for a confident call."

    ratio_change = (promo_ratio - baseline_ratio) / baseline_ratio if baseline_ratio else np.nan
    if abs(ratio_change) <= 0.05 and abs(delta_lift) <= 0.05:
        return "Mostly baseline traffic differences (promo lift is similar by day)."
    if ratio_change < -0.1 or delta_lift < -0.1:
        return "Promo appeal likely weaker on Monday than expected from baseline."
    if ratio_change > 0.1 or delta_lift > 0.1:
        return "Promo appears stronger on Monday than baseline would suggest."
    return "Mixed signal: baseline and promo effects both contribute."


def format_pct(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value * 100:.1f}%"


def format_ratio(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.2f}"


def build_answer_to_anna(summary: pd.DataFrame) -> str:
    def _get(metric: str, sample: str) -> pd.Series | None:
        rows = summary[(summary["metric"] == metric) & (summary["sample"] == sample)]
        if rows.empty:
            return None
        return rows.iloc[0]

    sample = "no_outliers" if (summary["sample"] == "no_outliers").any() else "all_days"
    net_row = _get("net_sales_hh", sample)
    tph_row = _get("transactions_per_hour", sample)

    if net_row is None or tph_row is None:
        return "Insufficient data to compute baseline-adjusted Monday vs Wednesday comparison."

    classification = classify_result(
        baseline_ratio=tph_row["baseline_ratio"],
        promo_ratio=tph_row["promo_ratio"],
        delta_lift=tph_row["delta_lift"],
    )

    lines = [
        "Answer to Anna: Is Monday underperformance mostly baseline traffic or promo appeal?",
        f"Sample used: {sample} (HH window 2-5pm, Mon/Wed only).",
        f"Baseline Mon/Wed ratio (Net Sales): {format_ratio(net_row['baseline_ratio'])}; TPH: {format_ratio(tph_row['baseline_ratio'])}.",
        f"Promo Mon/Wed ratio (Net Sales): {format_ratio(net_row['promo_ratio'])}; TPH: {format_ratio(tph_row['promo_ratio'])}.",
        f"Lift vs baseline (Net Sales): Mon {format_pct(net_row['lift_mon'])} vs Wed {format_pct(net_row['lift_wed'])} (delta {format_pct(net_row['delta_lift'])}).",
        f"Lift vs baseline (TPH): Mon {format_pct(tph_row['lift_mon'])} vs Wed {format_pct(tph_row['lift_wed'])} (delta {format_pct(tph_row['delta_lift'])}).",
        f"Interpretation: {classification}",
    ]

    if (summary["sample"] == "all_days").any() and sample != "all_days":
        net_all = _get("net_sales_hh", "all_days")
        tph_all = _get("transactions_per_hour", "all_days")
        if net_all is not None and tph_all is not None:
            classification_all = classify_result(
                baseline_ratio=tph_all["baseline_ratio"],
                promo_ratio=tph_all["promo_ratio"],
                delta_lift=tph_all["delta_lift"],
            )
            lines.append(
                "All-days sensitivity: "
                f"Net Sales ratio {format_ratio(net_all['promo_ratio'])} vs baseline {format_ratio(net_all['baseline_ratio'])}; "
                f"TPH ratio {format_ratio(tph_all['promo_ratio'])} vs baseline {format_ratio(tph_all['baseline_ratio'])}."
            )
            lines.append(f"Sensitivity interpretation: {classification_all}")

    return "\n".join(lines)


def _fill_numeric_for_agg(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def summarize_by_promo_day(df_items: pd.DataFrame) -> pd.DataFrame:
    df = _fill_numeric_for_agg(df_items, ["quantity", "gross_sales", "net_total"])
    agg_map = {
        "total_items": ("quantity", "sum"),
        "total_gross_sales": ("gross_sales", "sum"),
        "total_net_sales": ("net_total", "sum"),
        "num_rows": ("promo_day_type", "size"),
    }
    grouped = df.groupby("promo_day_type", dropna=False).agg(**agg_map).reset_index()
    grouped["avg_price_per_item"] = np.where(
        grouped["total_items"] > 0,
        grouped["total_net_sales"] / grouped["total_items"],
        np.nan,
    )
    return grouped.sort_values("promo_day_type")


def summarize_fruit_tea_by_promo_day(df_items: pd.DataFrame) -> pd.DataFrame:
    df_ft = df_items[df_items["is_fresh_fruit_tea_item"] == True].copy()
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
    return grouped.sort_values("promo_day_type")


def summarize_product_lift(df_items: pd.DataFrame) -> pd.DataFrame:
    df_ft = df_items[df_items["is_fresh_fruit_tea_item"] == True].copy()
    df_ft = _fill_numeric_for_agg(df_ft, ["quantity", "net_total"])
    grouped = df_ft.groupby(["promo_day_type", "item_name"], dropna=False).agg(
        total_quantity=("quantity", "sum"),
        total_net_sales=("net_total", "sum"),
    )
    grouped = grouped.reset_index()
    grouped["rank_within_day"] = grouped.groupby("promo_day_type")["total_quantity"].rank(
        ascending=False, method="dense"
    )
    return grouped.sort_values(["promo_day_type", "total_quantity"], ascending=[True, False])


def summarize_basket_behavior(df_items: pd.DataFrame) -> pd.DataFrame | None:
    if "order_id" not in df_items.columns:
        return None
    df = _fill_numeric_for_agg(df_items, ["quantity", "net_total"])
    order_level = df.groupby("order_id", dropna=False).agg(
        total_items_per_order=("quantity", "sum"),
        total_sales_per_order=("net_total", "sum"),
        promo_day_type=("promo_day_type", "first"),
    )
    basket_summary = order_level.groupby("promo_day_type", dropna=False).agg(
        orders_count=("promo_day_type", "size"),
        avg_items_per_order=("total_items_per_order", "mean"),
        avg_sales_per_order=("total_sales_per_order", "mean"),
    )
    return basket_summary.reset_index().sort_values("promo_day_type")


def main() -> None:
    project_root = get_project_root()
    exports_dir = ensure_exports_dir(project_root)

    df_orders = load_order_data(project_root)
    daily = aggregate_daily(df_orders)
    daily = flag_outliers(daily, metrics=["transactions_per_hour", "net_sales_hh"])

    summary = build_summary_table(daily)
    answer_text = build_answer_to_anna(summary)

    daily_path = exports_dir / "daily_aggregates.csv"
    summary_path = exports_dir / "summary_table.csv"
    answer_path = exports_dir / "answer_to_anna.txt"

    daily.to_csv(daily_path, index=False)
    summary.to_csv(summary_path, index=False)
    answer_path.write_text(answer_text, encoding="utf-8")

    saved = [daily_path, summary_path, answer_path]

    items_hh_path = project_root / "data" / "processed" / "items_happy_hour_window.csv"
    if items_hh_path.exists():
        df_items = pd.read_csv(items_hh_path)
        if "is_fresh_fruit_tea_item" not in df_items.columns and "is_fruit_tea_series" in df_items.columns:
            df_items = df_items.rename(columns={"is_fruit_tea_series": "is_fresh_fruit_tea_item"})
        summary_by_day = summarize_by_promo_day(df_items)
        ft_summary = summarize_fruit_tea_by_promo_day(df_items)
        product_lift = summarize_product_lift(df_items)
        basket_summary = summarize_basket_behavior(df_items)

        summary_by_day_path = exports_dir / "hh_summary_by_promo_day.csv"
        ft_summary_path = exports_dir / "fruit_tea_summary_by_promo_day.csv"
        product_lift_path = exports_dir / "fruit_tea_product_lift_by_promo_day.csv"
        summary_by_day.to_csv(summary_by_day_path, index=False)
        ft_summary.to_csv(ft_summary_path, index=False)
        product_lift.to_csv(product_lift_path, index=False)
        saved.extend([summary_by_day_path, ft_summary_path, product_lift_path])
        if basket_summary is not None:
            basket_path = exports_dir / "hh_basket_behavior_by_promo_day.csv"
            basket_summary.to_csv(basket_path, index=False)
            saved.append(basket_path)

    print("Saved outputs:")
    for path in saved:
        print(f"- {path}")


if __name__ == "__main__":
    main()
