import streamlit as st
import pandas as pd

from models.forecast_model import compute_simple_forecast


@st.cache_data
def load_data():
    dealer = pd.read_csv("data/dealer_master.csv")
    sales = pd.read_csv("data/sales_transactions.csv")
    return dealer, sales


def main():
    st.title("Forecast & Growth Opportunities")

    dealer, sales = load_data()

    months_ahead = st.slider("Forecast horizon (months)", min_value=1, max_value=6, value=3)
    forecast_df = compute_simple_forecast(sales, months_ahead=months_ahead)

    dealer_id = st.selectbox(
        "Select Dealer",
        options=dealer["dealer_id"],
        format_func=lambda d: f"{d} – {dealer.loc[dealer['dealer_id'] == d, 'dealer_name'].values[0]}"
    )

    d_forecast = forecast_df[forecast_df["dealer_id"] == dealer_id]
    if d_forecast.empty:
        st.write("No forecast data available for this dealer.")
        return

    model_choice = st.selectbox("Filter by model", options=["All"] + sorted(d_forecast["model"].unique().tolist()))
    if model_choice != "All":
        d_forecast = d_forecast[d_forecast["model"] == model_choice]

    st.subheader("Forecasted Units")
    chart_df = d_forecast.pivot_table(
        index="month",
        columns="model",
        values="forecast_units"
    ).reset_index()

    st.line_chart(chart_df, x="month")

    st.subheader("Opportunity Table (Simple Heuristic)")
    # Compare forecast vs portfolio average to highlight high-growth dealers later, kept local here
    opp = d_forecast.groupby("model")["forecast_units"].sum().reset_index()
    opp["comment"] = opp["forecast_units"].rank(ascending=False).apply(
        lambda r: "High potential – prioritize allocations" if r <= 2 else "Moderate"
    )
    st.dataframe(opp)


if __name__ == "__main__":
    main()
