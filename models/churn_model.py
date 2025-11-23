import pandas as pd
from .health_score import compute_health_score


def compute_churn_probability(
    sales_df: pd.DataFrame,
    claims_df: pd.DataFrame,
    crm_df: pd.DataFrame,
    inv_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Simple rule-based "AI" churn model derived from health score and claims.
    Returns churn_probability between 0 and 1.
    """
    health = compute_health_score(sales_df, claims_df, crm_df, inv_df)

    # base churn ≈ inverse of health
    health["churn_probability"] = 1 - (health["health_score"] / 100.0)

    # small bump if high complaints
    claims_agg = (
        claims_df.groupby("dealer_id")["severity"]
        .mean()
        .reset_index()
        .rename(columns={"severity": "avg_severity"})
    ) if not claims_df.empty else pd.DataFrame(columns=["dealer_id", "avg_severity"])

    df = health.merge(claims_agg, on="dealer_id", how="left")
    df["avg_severity"] = df["avg_severity"].fillna(0)

    df["churn_probability"] = df["churn_probability"] + 0.1 * (df["avg_severity"] / 3.0)
    df["churn_probability"] = df["churn_probability"].clip(0, 1)

    df["risk_bucket"] = pd.cut(
        df["churn_probability"],
        bins=[-0.01, 0.33, 0.66, 1.01],
        labels=["Low", "Medium", "High"]
    )

    return df[["dealer_id", "churn_probability", "risk_bucket"]]
