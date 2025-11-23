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
from models.health_score import compute_health_score
from models.churn_model import compute_churn
from models.sentiment_model import enrich_sentiment, dealer_sentiment

@st.cache_data
def load_data():
    dealer = pd.read_csv(data_path("dealer_master.csv"))
    sales = pd.read_csv(data_path("sales_transactions.csv"))
    inv = pd.read_csv(data_path("inventory_stock.csv"))
    claims = pd.read_csv(data_path("warranty_claims.csv"))
    crm = pd.read_csv(data_path("crm_engagement.csv"))
    feedback = pd.read_csv(data_path("feedback_forms.csv"))
    return dealer, sales, inv, claims, crm, feedback


def main():
    st.title("Account Explorer")

    dealer, sales, inv, claims, crm, feedback = load_data()
    health = compute_health_score(sales, claims, crm, inv)
    churn = compute_churn(sales, claims, crm, inv)

    f = enrich_sentiment(feedback)
    s_agg = dealer_sentiment(f)

    df = dealer.merge(health, on="dealer_id").merge(churn, on="dealer_id").merge(s_agg, on="dealer_id", how="left")

    sel = st.selectbox("Select Dealer", df["dealer_id"])

    d = df[df["dealer_id"] == sel].iloc[0]
    st.metric("Health", f"{d.health_score:.1f}")
    st.metric("Churn", f"{d.churn_prob:.2f}")
    st.metric("Sentiment", f"{d.sentiment_avg:.2f}")

    st.subheader("Sales Trend")
    sd = sales[sales["dealer_id"] == sel]
    sd = sd.groupby("date")["units_sold"].sum().reset_index()
    st.line_chart(sd, x="date", y="units_sold")

    st.subheader("Recent CRM Touchpoints")
    st.dataframe(crm[crm["dealer_id"] == sel].sort_values("date", ascending=False).head(20))

    st.subheader("Warranty")
    st.dataframe(claims[claims["dealer_id"] == sel].sort_values("filed_date", ascending=False))


if __name__ == "__main__":
    main()
