import pandas as pd


def compute_simple_forecast(sales_df: pd.DataFrame, months_ahead: int = 3) -> pd.DataFrame:
    """
    Simple forecast: last 3 months average units per dealer + model,
    extended as flat forecast for next N months.
    """
    df = sales_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M")

    monthly = (
        df.groupby(["dealer_id", "model", "month"])["units_sold"]
        .sum()
        .reset_index()
    )

    # take last 3 months per dealer+model
    monthly = monthly.sort_values(["dealer_id", "model", "month"])
    last_3 = (
        monthly.groupby(["dealer_id", "model"])
        .tail(3)
        .groupby(["dealer_id", "model"])["units_sold"]
        .mean()
        .reset_index()
        .rename(columns={"units_sold": "avg_last_3_months"})
    )

    # get last actual month in dataset
    last_month = monthly["month"].max()

    # create forecast rows
    forecasts = []
    for _, row in last_3.iterrows():
        dealer_id = row["dealer_id"]
        model = row["model"]
        base = row["avg_last_3_months"]
        for i in range(1, months_ahead + 1):
            future_month = (last_month + i).to_timestamp()
            forecasts.append({
                "dealer_id": dealer_id,
                "model": model,
                "month": future_month.strftime("%Y-%m"),
                "forecast_units": round(base, 1)
            })

    forecast_df = pd.DataFrame(forecasts)
    return forecast_df
