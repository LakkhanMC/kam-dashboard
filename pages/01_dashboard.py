import streamlit as st
import pandas as pd

from utils.paths import data_path
from models.health_score import compute_health_score
from models.churn_model import compute_churn
from models.sentiment_model import enrich_sentiment, dealer_sentiment

st.set_page_config(page_title="🚗 OEM KAM Dashboard", layout="wide")


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
    st.title("🏢 OEM → Dealership Key Account Management AI Dashboard")

    # Load data
    dealer, sales, inv, claims, crm, feedback = load_data()

    # Enrich & compute
    health = compute_health_score(sales, claims, crm, inv)
    churn = compute_churn(sales, claims, crm, inv)
    feedback = enrich_sentiment(feedback)
    sent = dealer_sentiment(feedback)

    df = (
        dealer
        .merge(health, on="dealer_id", how="left")
        .merge(churn, on="dealer_id", how="left")
        .merge(sent, on="dealer_id", how="left")
    )

    # ====================
    # TOP KPIs
    # ====================
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Active Dealers", len(df))
    col2.metric("Avg Health Score", f"{df.health_score.mean():.1f}")
    col3.metric("⚠️ High Risk Dealers", (df.risk_bucket == "High").sum())
    col4.metric("Avg Sentiment", f"{df.sentiment_avg.mean():.2f}")

    st.divider()

    # ====================
    # BUSINESS TRENDS
    # ====================
    st.subheader("📊 Vehicle Sales Trend (Total OEM)")

    sales["date"] = pd.to_datetime(sales["date"])
    monthly = sales.groupby(sales["date"].dt.to_period("M"))["units_sold"].sum().reset_index()
    monthly.columns = ["month", "units"]
    st.line_chart(monthly, x="month", y="units")

    st.caption("Shows macro OEM demand across all dealers.")

    st.divider()

    # ====================
    # HEALTH DISTRIBUTION
    # ====================
    st.subheader("🏥 Dealer Health Distribution")
    st.bar_chart(df.groupby("health_bucket")["dealer_id"].count())

    st.caption("Detect how many dealers are at risk vs stable.")

    st.divider()

    # ====================
    # Satisfaction & Complaints
    # ====================
    st.subheader("🛠 Warranty Complaints Severity")

    sev = claims.groupby("severity")["claim_id"].count().reset_index()
    st.bar_chart(sev, x="severity", y="claim_id")

    st.caption("Spikes in severity directly correlate with churn.")

    st.divider()

    # ====================
    # TOP / BOTTOM ACCOUNTS
    # ====================
    st.subheader("⭐️ Top 10 Dealers by Sales Contribution")

    vol = sales.groupby("dealer_id")["units_sold"].sum().reset_index()
    top10 = vol.sort_values("units_sold", ascending=False).head(10)
    st.dataframe(
        top10.merge(dealer[["dealer_id", "city", "region"]], on="dealer_id"),
        hide_index=True
    )

    st.divider()

    st.subheader("🔥 Dealers Requiring Attention (Health < 45 & High Churn)")

    risk_df = df[(df.health_score < 45) & (df.churn_prob > 0.55)]
    st.dataframe(risk_df[["dealer_id", "region", "health_score", "risk_bucket", "churn_prob"]], hide_index=True)

    st.caption("These accounts may require visits, marketing push, or service intervention.")

    st.divider()

    st.subheader("🚀 Dealers with Latent Demand (High Sales + Poor Inventory)")

    latent = (
        vol.merge(inv.groupby("dealer_id")["ageing_days"].mean().reset_index(), on="dealer_id")
    )
    latent = latent[latent["ageing_days"] > 40].sort_values("units_sold", ascending=False)

    st.dataframe(latent.head(10), hide_index=True)

    st.caption("These dealers could increase sales if inventory is refreshed / allocated.")


if __name__ == "__main__":
    main()
