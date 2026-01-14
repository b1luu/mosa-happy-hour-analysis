# mosa-happy-hour-analysis

## Overview
This project analyzes Happy Hour performance at Mosa Tea and separates baseline day-of-week traffic from promo effects.

Promos under review:
- `monday_flat_4`: Fruit Tea series drinks are flat $4
- `wednesday_bogo_50`: Buy One Get One 50% off (same product)

Time window:
- Monday & Wednesday only
- 2:00 pm – 5:00 pm (PDT)

Periods:
- Baseline (pre-promo): 2025-08-29 to 2025-10-19
- Promo: 2025-10-20 to 2025-12-31

Core business question:
Is Monday Happy Hour underperformance mostly baseline foot traffic, or does the promo underperform vs Wednesday?

## Definitions
- Promo Day Type: The specific Happy Hour rule applied to a row. Values are `monday_flat_4`, `wednesday_bogo_50`, or `non_happy_hour`.
- Promo Day: A shorthand used in charts/tables for Promo Day Type (Monday vs Wednesday Happy Hour).
- Happy Hour Window: Monday & Wednesday only, 2:00–5:00 pm (PDT).
- Period Label: `baseline` (pre-promo) or `promo` (Happy Hour running).
- Baseline Ratio: `Mon / Wed` for the same metric during the baseline period.
- Promo Ratio: `Mon / Wed` for the same metric during the promo period.
- Fruit Tea Series: Identified via `config/fresh_fruit_tea_mapping.json` (category and name rules).
- Quantity: Number of cups/items sold for that line item.
- Gross Sales: Pre-discount sales amount.
- Discounts: Total discounts applied.
- Net Total / Net Sales: Gross sales minus discounts.
- Effective Price per Drink: `net_total / quantity` (after discounts).
- Estimated Revenue: `price * quantity` (used if net sales are missing or zero).

## How to run
1) `python3 src/happy_hour_pipeline.py` (raw -> processed)
2) `python3 src/hh_analysis.py` (daily aggregates + baseline-adjusted summary)
3) `python3 src/hh_visualizations.py` (PNG charts)
4) `python3 src/hh_pptx.py` (PPTX deck)

## Outputs
These files are generated locally and are excluded from Git for privacy:
- `data/processed/items_clean_full.csv`: Cleaned, slimmed dataset (all rows).
- `data/processed/items_happy_hour_window.csv`: Cleaned Happy Hour-only rows.
- `data/processed/orders_clean_full.csv`: Order-level fact table (all rows).
- `data/processed/orders_hh_window.csv`: Order-level HH window rows.
- `exports/daily_aggregates.csv`: Day-level HH metrics for baseline and promo periods.
- `exports/summary_table.csv`: Baseline vs promo ratios, lifts, and CIs.
- `exports/answer_to_anna.txt`: Auto-generated summary text for the core question.
- `exports/hh_summary_by_promo_day.csv`: Summary metrics by promo day.
- `exports/fruit_tea_summary_by_promo_day.csv`: Fruit Tea-only summary metrics by promo day.
- `exports/fruit_tea_product_lift_by_promo_day.csv`: Fruit Tea item-level performance by promo day.
- `exports/plots/`: Generated charts.
- `exports/happy_hour_presentation.pptx`: Slide deck.

## Privacy & Data Handling
- Raw and processed data exports are excluded from Git in `.gitignore` (entire `data/` and `exports/`).
- Notebooks are excluded from Git to avoid saving outputs and data snapshots.
- Keep raw POS exports private; only share aggregated summaries or charts.

## Validation Notes
- Schema and Happy Hour logic checks pass (HH rows are only Mon/Wed 2–5pm).
- `gross_sales - discounts` does not always equal `net_total` for all rows; use `net_total` as the source of truth.
- Baseline vs promo comparisons use the same HH window and the same metrics for Mon/Wed.

## Conclusions (Draft)
- This analysis compares baseline Mon/Wed traffic to promo-period Mon/Wed performance during Happy Hour.
- Use the ratio and lift tables to distinguish baseline foot traffic from promo appeal.
- Fruit Tea charts and product lift tables show which items drive the promo results.
