import streamlit as st
import pandas as pd

from models.health_score import compute_health_score
from models.churn_model import compute_churn_probability
from models.sentiment_model import enrich_feedback_sentiment, aggregate_dealer_sentiment


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
    st.title("Account Explorer – Dealer Drill-Down")

    dealer, sales, inv, claims, crm, feedback = load_data()

    health = compute_health_score(sales, claims, crm, inv)
    churn = compute_churn_probability(sales, claims, crm, inv)
    feedback_enriched = enrich_feedback_sentiment(feedback)
    sentiment_agg = aggregate_dealer_sentiment(feedback_enriched)

    summary = dealer.merge(health, on="dealer_id", how="left") \
                    .merge(churn, on="dealer_id", how="left") \
                    .merge(sentiment_agg, on="dealer_id", how="left")

    dealer_id = st.selectbox(
        "Select Dealer",
        options=summary["dealer_id"],
        format_func=lambda d: f"{d} – {summary.loc[summary['dealer_id'] == d, 'dealer_name'].values[0]}"
    )

    d_row = summary[summary["dealer_id"] == dealer_id].iloc[0]
    st.subheader(f"{d_row['dealer_name']} ({dealer_id})")

    col1, col2, col3 = st.columns(3)
    col1.metric("Health Score", f"{d_row['health_score']:.1f}", help="Higher is better")
    col2.metric("Churn Probability", f"{d_row['churn_probability']:.2f}", help="0–1 range")
    col3.metric("Avg Sentiment", f"{d_row.get('avg_sentiment_score', 0):.2f}")

    st.write("**Profile**")
    st.write({
        "Region": d_row["region"],
        "State": d_row["state"],
        "City": d_row["city"],
        "Tier": d_row["tier"],
        "Years Partnered": d_row["years_partnered"],
        "Ownership": d_row["ownership_type"]
    })

    st.markdown("---")
    st.subheader("Sales Trend (Units)")

    sales_d = sales[sales["dealer_id"] == dealer_id].copy()
    sales_d["date"] = pd.to_datetime(sales_d["date"])
    sales_d = sales_d.sort_values("date")

    model_choice = st.selectbox("Filter by model", options=["All"] + sorted(sales_d["model"].unique().tolist()))
    if model_choice != "All":
        sales_d = sales_d[sales_d["model"] == model_choice]

    chart_df = sales_d.groupby("date")["units_sold"].sum().reset_index()
    st.line_chart(chart_df, x="date", y="units_sold")

    st.subheader("Warranty Claims & Issues")
    claims_d = claims[claims["dealer_id"] == dealer_id]
    if claims_d.empty:
        st.write("No claims recorded for this dealer.")
    else:
        st.dataframe(claims_d.sort_values("filed_date", ascending=False).head(20))

    st.subheader("Engagement Timeline (Recent)")
    crm_d = crm[crm["dealer_id"] == dealer_id].copy()
    crm_d["date"] = pd.to_datetime(crm_d["date"])
    crm_d = crm_d.sort_values("date", ascending=False)

    st.dataframe(
        crm_d[["date", "interaction_type", "notes", "duration_mins"]]
        .head(20)
    )

    st.subheader("Feedback & Sentiment")
    fb_d = feedback_enriched[feedback_enriched["dealer_id"] == dealer_id]
    if fb_d.empty:
        st.write("No feedback available.")
    else:
        st.dataframe(
            fb_d[["feedback_date", "feedback_source", "sentiment", "sentiment_score", "comments"]]
            .sort_values("feedback_date", ascending=False)
        )

    st.markdown("---")
    st.subheader("AI Recommendation (Heuristic)")

    recs = []
    if d_row["health_score"] < 50:
        recs.append("Prioritize an in-person review meeting within 2 weeks.")
    if d_row["churn_probability"] > 0.6:
        recs.append("Offer targeted incentives or flexible credit terms to stabilize volumes.")
    if d_row.get("avg_sentiment_score", 0) < 0:
        recs.append("Address top negative themes from feedback, especially around support/service.")
    if claims_d["severity"].mean() > 2 if not claims_d.empty else False:
        recs.append("Investigate root causes for high-severity warranty claims with service teams.")

    if not recs:
        recs.append("Maintain current engagement plan. Monitor quarterly performance.")

    for i, r in enumerate(recs, start=1):
        st.write(f"**{i}. {r}**")


if __name__ == "__main__":
    main()
