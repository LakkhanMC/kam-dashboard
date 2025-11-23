import pandas as pd
import numpy as np


def normalize(series):
    if series.isna().all():
        return series.fillna(0.5)
    min_v, max_v = series.min(), series.max()
    if min_v == max_v:
        return series.apply(lambda _: 0.5)
    return (series - min_v) / (max_v - min_v)


def compute_health_score(sales, claims, crm, inv):
    # ===================
    # SALES TREND
    # ===================
    sales["date"] = pd.to_datetime(sales["date"])
    sales["month"] = sales["date"].dt.to_period("M")
    monthly = sales.groupby(["dealer_id", "month"])["units_sold"].sum().reset_index()
    monthly = monthly.sort_values(["dealer_id", "month"])

    monthly["rolling3"] = monthly.groupby("dealer_id")["units_sold"].rolling(3, min_periods=1).sum().reset_index(level=0, drop=True)
    monthly["prev3"] = monthly.groupby("dealer_id")["rolling3"].shift(3)
    monthly["trend"] = (monthly["rolling3"] - monthly["prev3"]) / (monthly["prev3"].replace(0, np.nan))
    trend_df = monthly.groupby("dealer_id").tail(1)[["dealer_id", "trend"]]

    # ===================
    # CLAIMS
    # ===================
    if not claims.empty:
        claims_agg = claims.groupby("dealer_id").agg(
            severity=("severity", "mean"),
            count=("claim_id", "count")
        ).reset_index()
    else:
        claims_agg = pd.DataFrame(columns=["dealer_id", "severity", "count"])

    # ===================
    # ENGAGEMENT (CR M)
    # ===================
    crm["date"] = pd.to_datetime(crm["date"])
    cutoff = crm["date"].max() - pd.Timedelta(days=90)
    recent = crm[crm["date"] >= cutoff]
    crm_agg = recent.groupby("dealer_id").agg(
        contacts=("interaction_type", "count"),
        avg_duration=("duration_mins", "mean")
    ).reset_index()

    # ===================
    # INVENTORY
    # ===================
    inv_agg = inv.groupby("dealer_id").agg(
        age=("ageing_days", "mean"),
        stock=("stock_units", "mean")
    ).reset_index()

    df = trend_df.merge(claims_agg, on="dealer_id", how="left") \
                 .merge(crm_agg, on="dealer_id", how="left") \
                 .merge(inv_agg, on="dealer_id", how="left")

    df.fillna(0, inplace=True)

    # normalize components
    df["s"] = normalize(df["trend"])
    df["c"] = 1 - normalize(df["severity"])     # lower severity = good
    df["e"] = normalize(df["contacts"])
    df["a"] = 1 - normalize(df["age"])          # lower age = good

    df["health_score"] = (
        0.35 * df["s"] +
        0.25 * df["e"] +
        0.20 * df["c"] +
        0.20 * df["a"]
    ) * 100

    df["health_bucket"] = pd.cut(
        df["health_score"],
        bins=[-1, 40, 70, 200],
        labels=["High Risk", "Watchlist", "Healthy"]
    )

    return df[["dealer_id", "health_score", "health_bucket"]]
