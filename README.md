# mosa-happy-hour-analysis

## Overview
This project analyzes Happy Hour performance at Mosa Tea to compare two promo designs:

- `monday_flat_4`: Fruit Tea series drinks are flat $4
- `wednesday_bogo_50`: Buy One Get One 50% off (same product)

Happy Hour window:
- Monday & Wednesday only
- 2:00 pm – 5:00 pm (PDT)
- Start date: 2025-10-20

Core business question:
Which Happy Hour promo is more effective for Fruit Tea — Flat $4 Monday, or BOGO 50% Wednesday — and in what way?

## Definitions
- Promo Day Type: The specific Happy Hour rule applied to a row. Values are `monday_flat_4`, `wednesday_bogo_50`, or `non_happy_hour`.
- Promo Day: A shorthand used in charts/tables for Promo Day Type (Monday vs Wednesday Happy Hour).
- Happy Hour Window: Monday & Wednesday only, 2:00–5:00 pm (PDT), starting 2025-10-20.
- Fruit Tea Series: Rows whose `category` contains keywords like "fresh fruit tea", "fruit tea", or "鲜果茶".
- Quantity: Number of cups/items sold for that line item.
- Gross Sales: Pre-discount sales amount.
- Discounts: Total discounts applied.
- Net Total / Net Sales: Gross sales minus discounts.
- Effective Price per Drink: `net_total / quantity` (after discounts).
- Estimated Revenue: `price * quantity` (used if net sales are missing or zero).

## Outputs
These files are generated locally and are excluded from Git for privacy:
- `data/processed/items_clean_full.csv`: Cleaned, slimmed dataset (all rows).
- `data/processed/items_happy_hour_window.csv`: Cleaned Happy Hour-only rows.
- `exports/hh_summary_by_promo_day.csv`: Summary metrics by promo day.
- `exports/fruit_tea_summary_by_promo_day.csv`: Fruit Tea-only summary metrics by promo day.
- `exports/fruit_tea_product_lift_by_promo_day.csv`: Fruit Tea item-level performance by promo day.
- `exports/plots/`: Generated charts.
- `exports/happy_hour_presentation.pdf`: Slide-style PDF presentation.

## Privacy & Data Handling
- Raw and processed data exports are excluded from Git in `.gitignore`.
- Keep raw POS exports private; only share aggregated summaries or charts.

## Validation Notes
- Schema and Happy Hour logic checks pass (HH rows are only Mon/Wed 2–5pm).
- `gross_sales - discounts` does not always equal `net_total` for all rows; use `net_total` as the source of truth.

## Conclusions (Draft)
- This analysis compares Fruit Tea performance between Monday flat $4 and Wednesday BOGO 50% during Happy Hour.
- Use the Fruit Tea-only charts and product lift tables to see which promo drives higher volume and which items lead.
- Pairing behavior (BOGO effect) is inferred from higher quantity per item on Wednesdays, not from explicit order IDs.
