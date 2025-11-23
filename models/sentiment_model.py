import pandas as pd


POSITIVE_WORDS = {
    "good", "satisfied", "appreciated", "timely",
    "support", "strong", "happy", "excellent"
}

NEGATIVE_WORDS = {
    "delay", "delays", "insufficient", "bad", "poor",
    "complaint", "complaints", "issue", "problem", "pressure",
    "trust", "negative"
}


def score_text_sentiment(text: str) -> float:
    """
    Very simple lexicon-based sentiment:
    returns score between -1 (very negative) and +1 (very positive)
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0

    tokens = text.lower().replace(".", " ").replace(",", " ").split()
    pos = sum(t in POSITIVE_WORDS for t in tokens)
    neg = sum(t in NEGATIVE_WORDS for t in tokens)

    if pos == 0 and neg == 0:
        return 0.0

    score = (pos - neg) / (pos + neg)
    return max(-1.0, min(1.0, score))


def enrich_feedback_sentiment(feedback_df: pd.DataFrame) -> pd.DataFrame:
    feedback_df = feedback_df.copy()
    feedback_df["sentiment_score"] = feedback_df["comments"].apply(score_text_sentiment)
    # Map to simple labels
    feedback_df["sentiment"] = feedback_df["sentiment_score"].apply(
        lambda x: "Positive" if x > 0.2 else ("Negative" if x < -0.2 else "Neutral")
    )
    return feedback_df


def aggregate_dealer_sentiment(feedback_df: pd.DataFrame) -> pd.DataFrame:
    if feedback_df.empty:
        return pd.DataFrame(columns=["dealer_id", "avg_sentiment_score"])

    agg = (
        feedback_df.groupby("dealer_id")["sentiment_score"]
        .mean()
        .reset_index()
        .rename(columns={"sentiment_score": "avg_sentiment_score"})
    )
    return agg
