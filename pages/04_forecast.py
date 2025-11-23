def generate_forecast(sales, dealer_id, months):
    df = sales[sales["dealer_id"] == dealer_id].copy()
    df["date"] = pd.to_datetime(df["date"])
    
    result = []

    for model_name in df["model"].unique():
        ts = (
            df[df["model"] == model_name]
            .set_index("date")["units_sold"]
            .resample("M")
            .sum()
        )

        fc = forecast_model(ts, periods=months)

        # build rows
        for i, val in enumerate(fc):
            result.append({
                "dealer_id": dealer_id,
                "model": model_name,
                "month": (ts.index[-1] + pd.DateOffset(months=i+1)).strftime("%Y-%m"),
                "forecast_units": round(val, 2)
            })

    return pd.DataFrame(result)
