import streamlit as st
import pandas as pd
from utils.paths import data_path
from statsmodels.tsa.statespace.sarimax import SARIMAX


@st.cache_data
def load_data():
    dealer = pd.read_csv(data_path("dealer_master.csv"))
    sales = pd.read_csv(data_path("sales_transactions.csv"))
    return dealer, sales


def forecast_ts(series, periods=3):
    """
    Train a SARIMA model on past dealer/model sales
    """
    model = SARIMAX(
        series,
        order=(1,1,1),
        seasonal_order=(1,1,1,12),  # yearly seasonality
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    results = model.fit(disp=False)
    return results.forecast(periods)


def build_forecast(sales_df, dealer_id, months):
    dealer_df = sales_df[sales_df["dealer_id"] == dealer_id].copy()
    dealer_df["date"] = pd.to_datetime(dealer_df["date"])

    results = []
    for model_name in dealer_df["model"].unique():
        ts = (
            dealer_df[dealer_df["model"] == model_name]
            .set_index("date")["units_sold"]
            .resample("M")
            .sum()
        )

        if len(ts) < 6:
            continue  # not enough data

        fc = forecast_ts(ts, periods=months)

        for i, val in enumerate(fc):
            results.append({
                "dealer_id": dealer_id,
                "model": model_name,
                "month": (ts.index[-1] + pd.DateOffset(months=i+1)).strftime("%Y-%m"),
                "forecast_units": float(val)
            })

    return pd.DataFrame(results)


def main():
    st.title("📈 Forecast & Growth Opportunities")

    dealer_df, sales_df = load_data()

    months = st.slider("Months Ahead", 1, 12, 3)
    dealer_id = st.selectbox("Dealer", dealer_df["dealer_id"])

    forecast = build_forecast(sales_df, dealer_id, months)

    if forecast.empty:
        st.warning("Not enough historical data to forecast.")
        return

    st.subheader("Forecast Units")
    st.dataframe(forecast, hide_index=True)

    chart_df = forecast.pivot(index="month", columns="model", values="forecast_units")
    st.line_chart(chart_df)


if __name__ == "__main__":
    main()
