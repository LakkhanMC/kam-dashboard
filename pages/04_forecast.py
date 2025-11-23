from utils.paths import data_path
import pandas as pd

@st.cache_data
def load_data():
    dealer = pd.read_csv(data_path("dealer_master.csv"))
    sales = pd.read_csv(data_path("sales_transactions.csv"))
    inv = pd.read_csv(data_path("inventory_stock.csv"))
    claims = pd.read_csv(data_path("warranty_claims.csv"))
    crm = pd.read_csv(data_path("crm_engagement.csv"))
    feedback = pd.read_csv(data_path("feedback_forms.csv"))
    return dealer, sales, inv, claims, crm, feedback
import streamlit as st
import pandas as pd
from utils.paths import data_path
from models.forecast_model import forecast_sales

@st.cache_data
def load_data():
    dealer = pd.read_csv(data_path("dealer_master.csv"))
    sales = pd.read_csv(data_path("sales_transactions.csv"))
    return dealer, sales


def main():
    st.title("Forecast & Growth Opportunities")

    dealer, sales = load_data()
    horizon = st.slider("Months Ahead", 1, 6, 3)

    fc = forecast_sales(sales, horizon)

    sel = st.selectbox("Dealer", dealer["dealer_id"])
    d = fc[fc["dealer_id"] == sel]

    if d.empty:
        st.warning("No forecast")
        return

    st.subheader("Forecast Units")
    st.dataframe(d)

    wide = d.pivot_table(index="month", columns="model", values="forecast_units")
    st.line_chart(wide)


if __name__ == "__main__":
    main()
