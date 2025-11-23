import streamlit as st
import pandas as pd

from models.health_score import compute_health_score
from models.churn_model import compute_churn_probability


@st.cache_data
def load_data():
    dealer = pd.read_csv("data/dealer_master.csv")
    sales = pd.read_csv("data/sales_transactions.csv")
    inv = pd.read_csv("data/inventory_stock.csv")
    claims = pd.read_csv("data/warranty_claims.csv")
    crm = pd.read_csv("data/crm_engagement.csv")
    feedback = pd.read_csv("data/feedback_forms.csv")
    return dealer, sales, inv, claims, crm, feedback


def main():
    st.title("KAM AI Dashboard – Executive Overview")

    dealer, sales, inv, claims, crm, feedback = load_data()

    health = compute_health_score(sales, claims, crm, inv)
    churn = compute_churn_probability(sales, claims, crm, inv)
    summary = dealer.merge(health, on="dealer_id", how="left") \
                    .merge(churn, on="dealer_id", how="left")

    st.subheader("Portfolio KPIs")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Key Accounts", len(dealer))
    col2.metric("Avg Health Score", f"{summary['health_score'].mean():.1f}")
    col3.metric("High-Risk Accounts", int((summary["risk_bucket"] == "High").sum()))
    col4.metric("Avg Churn Probability", f"{summary['churn_probability'].mean():.2f}")

    st.subheader("Health vs Churn Scatter")
    st.caption("Each point = dealership. Bottom-right = high churn, low health.")

    scatter_df = summary[["dealer_id", "health_score", "churn_probability", "region", "tier"]]
    st.scatter_chart(
        scatter_df,
        x="health_score",
        y="churn_probability",
    )

    st.subheader("Top Risk Accounts")
    top_risk = summary.sort_values("churn_probability", ascending=False).head(10)
    st.dataframe(top_risk[["dealer_id", "dealer_name", "region", "health_score", "churn_probability", "risk_bucket"]])

    st.subheader("Health Distribution")
    st.bar_chart(summary["health_bucket"].value_counts())

    st.info(
        "🧠 Use the left sidebar to navigate to Account Explorer, Segmentation, and Forecast pages "
        "for deeper KAM insights."
    )


if __name__ == "__main__":
    main()
