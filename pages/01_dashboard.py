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
    st.title("KAM AI Dashboard – Executive Overview")

    dealer, sales, inv, claims, crm, feedback = load_data()
    health = compute_health_score(sales, claims, crm, inv)
    churn = compute_churn(sales, claims, crm, inv)

    df = dealer.merge(health, on="dealer_id").merge(churn, on="dealer_id")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accounts", len(dealer))
    col2.metric("Avg Health", f"{df.health_score.mean():.1f}")
    col3.metric("Risk (High)", (df["risk_bucket"] == "High").sum())
    col4.metric("Avg Churn", f"{df.churn_prob.mean():.2f}")

    st.subheader("Health vs Churn")
    st.scatter_chart(df, x="health_score", y="churn_prob")

    st.subheader("Risk Accounts")
    st.dataframe(df.sort_values("churn_prob", ascending=False).head(10))


if __name__ == "__main__":
    main()
