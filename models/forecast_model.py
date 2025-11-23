import pandas as pd


def forecast_sales(sales, months=3):
    sales["date"] = pd.to_datetime(sales["date"])
    sales["month"] = sales["date"].dt.to_period("M")

    m = sales.groupby(["dealer_id", "model", "month"])["units_sold"].sum().reset_index()
    last3 = m.groupby(["dealer_id", "model"]).tail(3)
    avg = last3.groupby(["dealer_id", "model"])["units_sold"].mean().reset_index()

    last_month = m["month"].max()

    out = []
    for _, r in avg.iterrows():
        for i in range(1, months + 1):
            out.append({
                "dealer_id": r["dealer_id"],
                "model": r["model"],
                "month": (last_month + i).strftime("%Y-%m"),
                "forecast_units": round(r["units_sold"], 1)
            })

    return pd.DataFrame(out)
