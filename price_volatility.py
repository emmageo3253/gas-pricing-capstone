# =========================
# Weekly Price Volatility + Magnitude Script (FIXED)
# =========================

import os
import pandas as pd

print("Working directory:", os.getcwd())

FILE_PATH = "panel_daily_gasolina95.csv"

# -------------------------
# 1) Load price panel
# -------------------------
panel_daily_g95 = pd.read_csv(FILE_PATH)
print("Loaded panel shape:", panel_daily_g95.shape)
print(panel_daily_g95.head())

# -------------------------
# 2) Clean types
# -------------------------
panel_daily_g95["date"] = pd.to_datetime(panel_daily_g95["date"], errors="coerce")
panel_daily_g95["precio_last"] = pd.to_numeric(panel_daily_g95["precio_last"], errors="coerce")

# Drop rows missing essentials (prevents weird groupby behavior)
panel_daily_g95 = panel_daily_g95.dropna(subset=["idEstacion", "date", "precio_last"]).copy()

# -------------------------
# 3) Sort + compute daily changes
# -------------------------
df = panel_daily_g95.sort_values(["idEstacion", "date"]).copy()

# Daily price change value (magnitude, signed)
df["price_change_value"] = df.groupby("idEstacion")["precio_last"].diff()

# Indicator: did price change vs previous day
df["price_change"] = df["price_change_value"].ne(0).fillna(False).astype(int)

# -------------------------
# 4) Week identifiers
# -------------------------
iso = df["date"].dt.isocalendar()
df["year"] = iso.year.astype(int)
df["week"] = iso.week.astype(int)

# -------------------------
# 5) Weekly frequency (# of changes)
# -------------------------
weekly_volatility = (
    df.groupby(["idEstacion", "year", "week"], as_index=False)
      .agg(
          weekly_price_changes=("price_change", "sum"),
          days_observed=("price_change", "count"),
      )
)
weekly_volatility["change_rate"] = weekly_volatility["weekly_price_changes"] / weekly_volatility["days_observed"]

# -------------------------
# 6) Weekly magnitude A: variance of price levels within week
# -------------------------
weekly_price_variance = (
    df.groupby(["idEstacion", "year", "week"], as_index=False)
      .agg(
          price_variance=("precio_last", "var"),
          price_std=("precio_last", "std"),
      )
)

# -------------------------
# 7) Weekly magnitude B: variance of daily price changes within week
# -------------------------
# Note: weeks with 0 or 1 non-null change will have NaN variance -> we fill later
weekly_change_variance = (
    df.groupby(["idEstacion", "year", "week"], as_index=False)
      .agg(
          change_variance=("price_change_value", "var"),
          change_std=("price_change_value", "std"),
      )
)
df["abs_change"] = df["price_change_value"].abs()

weekly_magnitude = (
    df.groupby(["idEstacion", "year", "week"], as_index=False)
      .agg(avg_abs_change=("abs_change", "mean"))
)

# -------------------------
# 8) Merge all into one weekly panel
# -------------------------
weekly_full = weekly_volatility.merge(
    weekly_price_variance,
    on=["idEstacion", "year", "week"],
    how="left",
)

weekly_full = weekly_full.merge(
    weekly_change_variance,
    on=["idEstacion", "year", "week"],
    how="left",
)

# Fill change_variance NaN with 0 (interpretation: no within-week variation in changes)
weekly_full["change_variance"] = weekly_full["change_variance"].fillna(0)

print("\nWeekly panel preview:")
print(weekly_full.head())

print("\nSummary:")
print(weekly_full[["weekly_price_changes", "price_variance", "change_variance"]].describe())

print(weekly_full.groupby("idEstacion")["weekly_price_changes"].mean().describe())
print(weekly_full.groupby("idEstacion")["price_std"].mean().describe())


# -------------------------
# 9) Save result
# -------------------------
#weekly_full.to_csv("weekly_price_volatility_and_magnitude.csv", index=False)
#print("\nSaved: weekly_price_volatility_and_magnitude.csv")
