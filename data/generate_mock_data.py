import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)


def ensure_data_dir():
    os.makedirs("data", exist_ok=True)


def generate_dealer_master(n_dealers=50):
    regions = ["North", "South", "East", "West", "Central"]
    states = ["Delhi", "Maharashtra", "Karnataka", "Gujarat", "Tamil Nadu", "Uttar Pradesh"]
    cities = ["Delhi", "Mumbai", "Bengaluru", "Ahmedabad", "Chennai", "Lucknow", "Pune", "Jaipur"]
    tiers = ["T1", "T2", "T3"]
    ownership_types = ["Franchise", "Company Owned", "Partner"]

    rows = []
    for i in range(1, n_dealers + 1):
        dealer_id = f"D{i:03d}"
        rows.append({
            "dealer_id": dealer_id,
            "dealer_name": f"Dealer_{dealer_id}",
            "region": random.choice(regions),
            "state": random.choice(states),
            "city": random.choice(cities),
            "tier": random.choice(tiers),
            "years_partnered": np.random.randint(1, 15),
            "ownership_type": random.choice(ownership_types)
        })

    df = pd.DataFrame(rows)
    df.to_csv("data/dealer_master.csv", index=False)
    return df


def generate_sales_transactions(dealers, start_month="2024-01-01", n_months=12):
    models = ["Hatch-A", "Sedan-Z", "SUV-X", "MPV-Y", "EV-E1"]
    base_volume = {
        "Hatch-A": 30,
        "Sedan-Z": 20,
        "SUV-X": 15,
        "MPV-Y": 10,
        "EV-E1": 5
    }

    start_date = datetime.strptime(start_month, "%Y-%m-%d")
    rows = []
    for m in range(n_months):
        month_date = start_date + pd.DateOffset(months=m)
        for dealer_id in dealers["dealer_id"]:
            for model in models:
                # volume with randomness, higher for T1/T2
                tier = dealers.loc[dealers["dealer_id"] == dealer_id, "tier"].values[0]
                tier_factor = {"T1": 1.3, "T2": 1.0, "T3": 0.7}[tier]
                units = max(0, int(np.random.normal(base_volume[model] * tier_factor, 5)))
                if units == 0:
                    continue
                wholesale_value = units * np.random.randint(500000, 1200000) / 10  # per unit approx
                rows.append({
                    "date": month_date.strftime("%Y-%m-%d"),
                    "dealer_id": dealer_id,
                    "model": model,
                    "units_sold": units,
                    "wholesale_value": round(wholesale_value, 2)
                })

    df = pd.DataFrame(rows)
    df.to_csv("data/sales_transactions.csv", index=False)
    return df


def generate_inventory_stock(dealers, models=None):
    if models is None:
        models = ["Hatch-A", "Sedan-Z", "SUV-X", "MPV-Y", "EV-E1"]

    rows = []
    for dealer_id in dealers["dealer_id"]:
        for model in models:
            stock_units = max(0, int(np.random.normal(20, 8)))
            ageing_days = max(0, int(np.random.normal(30, 15)))
            rows.append({
                "dealer_id": dealer_id,
                "model": model,
                "stock_units": stock_units,
                "ageing_days": ageing_days
            })

    df = pd.DataFrame(rows)
    df.to_csv("data/inventory_stock.csv", index=False)
    return df


def generate_warranty_claims(dealers, sales_df, avg_claim_rate=0.03):
    rows = []
    claim_id_counter = 1
    issue_types = ["Engine", "AC", "Brakes", "Electrical", "Body", "Other"]
    severities = [1, 2, 3]  # 3 = high

    # Approximate number of claims based on sales volume
    total_sales = sales_df["units_sold"].sum()
    n_claims = int(total_sales * avg_claim_rate)

    for _ in range(n_claims):
        record = sales_df.sample(1).iloc[0]
        dealer_id = record["dealer_id"]
        model = record["model"]
        sale_date = datetime.strptime(record["date"], "%Y-%m-%d")
        filed_date = sale_date + timedelta(days=np.random.randint(5, 120))
        resolution_days = max(1, int(np.random.normal(7, 3)))

        rows.append({
            "dealer_id": dealer_id,
            "claim_id": f"C{claim_id_counter:05d}",
            "model": model,
            "issue_type": random.choice(issue_types),
            "severity": random.choice(severities),
            "filed_date": filed_date.strftime("%Y-%m-%d"),
            "resolution_days": resolution_days
        })
        claim_id_counter += 1

    df = pd.DataFrame(rows)
    df.to_csv("data/warranty_claims.csv", index=False)
    return df


def generate_crm_engagement(dealers, start_date="2024-01-01", end_date="2024-12-31"):
    interaction_types = ["Call", "Dealer Visit", "Video Call", "Email", "Review Meeting"]
    notes_templates = [
        "Discussed sales performance and incentives.",
        "Reviewed service quality and customer feedback.",
        "Talked about EV potential in the region.",
        "Aligned on quarterly targets and marketing support.",
        "Addressed complaints related to warranty delays."
    ]

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    delta_days = (end - start).days

    rows = []
    for dealer_id in dealers["dealer_id"]:
        n_interactions = np.random.randint(5, 25)
        for _ in range(n_interactions):
            date = start + timedelta(days=np.random.randint(0, delta_days + 1))
            rows.append({
                "dealer_id": dealer_id,
                "date": date.strftime("%Y-%m-%d"),
                "interaction_type": random.choice(interaction_types),
                "notes": random.choice(notes_templates),
                "duration_mins": np.random.randint(10, 90)
            })

    df = pd.DataFrame(rows)
    df.to_csv("data/crm_engagement.csv", index=False)
    return df


def generate_feedback_forms(dealers):
    feedback_sources = ["Annual Review", "Survey", "Ad-hoc Call", "Review Meeting"]
    positive_comments = [
        "Inventory allocation is good.",
        "Satisfied with marketing support.",
        "Product quality is appreciated.",
        "Timely delivery of vehicles."
    ]
    negative_comments = [
        "Warranty delays affecting trust.",
        "Marketing support is insufficient.",
        "Training for staff is needed.",
        "More EV models required.",
        "Price discounting pressure in region."
    ]

    rows = []
    for dealer_id in dealers["dealer_id"]:
        n_feedbacks = np.random.randint(1, 5)
        for _ in range(n_feedbacks):
            date = datetime(2024, np.random.randint(1, 13), np.random.randint(1, 28))
            if np.random.rand() < 0.6:
                comment = random.choice(positive_comments)
            else:
                comment = random.choice(negative_comments)

            rows.append({
                "dealer_id": dealer_id,
                "feedback_date": date.strftime("%Y-%m-%d"),
                "feedback_source": random.choice(feedback_sources),
                "sentiment": "",  # to be filled by sentiment model later
                "comments": comment
            })

    df = pd.DataFrame(rows)
    df.to_csv("data/feedback_forms.csv", index=False)
    return df


if __name__ == "__main__":
    ensure_data_dir()
    dealers = generate_dealer_master(n_dealers=50)
    sales = generate_sales_transactions(dealers)
    inventory = generate_inventory_stock(dealers)
    claims = generate_warranty_claims(dealers, sales)
    crm = generate_crm_engagement(dealers)
    feedback = generate_feedback_forms(dealers)
    print("Mock data generated in /data directory.")
