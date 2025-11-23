import pandas as pd
import numpy as np


def compute_sales_trend(sales_df: pd.DataFrame) -> pd.DataFrame:
    """Compute simple sales growth (last 3 months vs previous 3) per dealer."""
    sales_df["date"] = pd.to_datetime(sales_df["date"])
    sales_df["month"] = sales_df["date"].dt.to_period("M")

    monthly = (
        sales_df.groupby(["dealer_id", "month"])["units_sold"]
        .sum()
        .reset_index()
    )

    # sort by month
    monthly = monthly.sort_values(["dealer_id", "month"])

    # rolling windows
    monthly["rolling_3"] = (
        monthly.groupby("dealer_id")["units_sold"]
        .rolling(3, min_periods=3)
        .sum()
        .reset_index(level=0, drop=True)
    )

    # previous 3 (shifted)
    monthly["rolling_3_prev"] = (
        monthly.groupby("dealer_id")["rolling_3"]
        .shift(3)
    )

    monthly["sales_trend"] = (
        (monthly["rolling_3"] - monthly["rolling_3_prev"])
        / (monthly["rolling_3_prev"].replace(0, np.nan))
    )

    # keep last month’s trend per dealer
    trend = (
        monthly.sort_values(["dealer_id", "month"])
        .groupby("dealer_id")
        .tail(1)[["dealer_id", "sales_trend"]]
    )

    return trend


def compute_claim_severity(claims_df: pd.DataFrame) -> pd.DataFrame:
    if claims_df.empty:
        return pd.DataFrame(columns=["dealer_id", "avg_claim_severity", "claim_frequency"])

    claims_df["filed_date"] = pd.to_datetime(claims_df["filed_date"])
    agg = (
        claims_df.groupby("dealer_id")
        .agg(
            avg_claim_severity=("severity", "mean"),
            claim_frequency=("claim_id", "count")
        )
        .reset_index()
    )
    return agg


def compute_engagement_intensity(crm_df: pd.DataFrame) -> pd.DataFrame:
    crm_df["date"] = pd.to_datetime(crm_df["date"])
    latest_date = crm_df["date"].max()
    cutoff_90 = latest_date - pd.Timedelta(days=90)

    recent = crm_df[crm_df["date"] >= cutoff_90]
    agg = (
        recent.groupby("dealer_id")
        .agg(
            engagements_90d=("interaction_type", "count"),
            avg_duration_mins=("duration_mins", "mean")
        )
        .reset_index()
    )
    return agg


def compute_inventory_risk(inv_df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        inv_df.groupby("dealer_id")
        .agg(
            avg_stock_units=("stock_units", "mean"),
            avg_ageing_days=("ageing_days", "mean")
        )
        .reset_index()
    )
    return agg


def normalize(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series(0.5, index=series.index)
    return (series - min_val) / (max_val - min_val)


def compute_health_score(
    sales_df: pd.DataFrame,
    claims_df: pd.DataFrame,
    crm_df: pd.DataFrame,
    inv_df: pd.DataFrame
) -> pd.DataFrame:
    trend = compute_sales_trend(sales_df)
    claims = compute_claim_severity(claims_df)
    engage = compute_engagement_intensity(crm_df)
    inv = compute_inventory_risk(inv_df)

    df = trend.merge(claims, on="dealer_id", how="left") \
              .merge(engage, on="dealer_id", how="left") \
              .merge(inv, on="dealer_id", how="left")

    # fill empties
    df["avg_claim_severity"] = df["avg_claim_severity"].fillna(0)
    df["claim_frequency"] = df["claim_frequency"].fillna(0)
    df["engagements_90d"] = df["engagements_90d"].fillna(0)
    df["avg_duration_mins"] = df["avg_duration_mins"].fillna(0)
    df["avg_stock_units"] = df["avg_stock_units"].fillna(0)
    df["avg_ageing_days"] = df["avg_ageing_days"].fillna(0)
    df["sales_trend"] = df["sales_trend"].fillna(0)

    # normalize features
    df["sales_trend_norm"] = normalize(df["sales_trend"])
    df["claim_severity_norm"] = normalize(df["avg_claim_severity"])
    df["engagement_norm"] = normalize(df["engagements_90d"])
    df["age_norm"] = normalize(df["avg_ageing_days"])

    # higher is better => transform where needed
    claim_component = 1 - df["claim_severity_norm"]
    age_component = 1 - df["age_norm"]

    df["health_score"] = (
        0.35 * df["sales_trend_norm"] +
        0.20 * claim_component +
        0.25 * df["engagement_norm"] +
        0.20 * age_component
    ) * 100

    df["health_bucket"] = pd.cut(
        df["health_score"],
        bins=[-1, 40, 70, 101],
        labels=["High Risk", "Watchlist", "Healthy"]
    )

    return df[["dealer_id", "health_score", "health_bucket"]]
