"""
Step 5: export a star schema for Power BI.

Reads  complaints_clean.parquet, complaint_components.parquet,
       complaint_clusters.parquet, cluster_labels.parquet,
       label_review.csv (the human review of the LLM labels)
Writes powerbi/fact_complaint.csv     one row per complaint
       powerbi/bridge_component.csv   one row per complaint x NHTSA component code
       powerbi/dim_problem.csv        one row per reviewed problem
       powerbi/dim_cluster.csv        one row per cluster (detail behind a problem)
       powerbi/dim_severity.csv       severity levels and their weights
       powerbi/dim_date.csv           calendar, one row per day

Usage:
    python step5_export_powerbi.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
OUT = HERE / "powerbi"

# Severity is the impact proxy: public data has no repair cost or labour hours.
# The weights are an assumption and live in their own table so they can be changed.
SEVERITY = pd.DataFrame([
    (4, "Injury or death", 10),
    (3, "Crash or fire", 5),
    (2, "Vehicle towed", 2),
    (1, "No crash, fire, injury or tow", 1),
], columns=["severity_level", "severity", "weight"])

UNCLUSTERED_ID, TOO_SHORT_ID = 900, 901


def one_line(s: pd.Series) -> pd.Series:
    """CSV safety: no line breaks or tabs inside text fields."""
    return s.fillna("").str.replace(r"[\r\n\t]+", " ", regex=True).str.strip()


def main() -> None:
    OUT.mkdir(exist_ok=True)
    c = pd.read_parquet(HERE / "complaints_clean.parquet")
    comp = pd.read_parquet(HERE / "complaint_components.parquet")
    cc = pd.read_parquet(HERE / "complaint_clusters.parquet")
    labels = pd.read_parquet(HERE / "cluster_labels.parquet")
    review = pd.read_csv(HERE / "label_review.csv", encoding="utf-8-sig")

    # ---- dim_problem: reviewed problems, numbered by volume (1 = largest)
    rv = review[review["cluster_id"] > 0]
    prob = (rv.groupby(["problem", "system", "issue_type"], as_index=False)
              .agg(n_complaints=("n_complaints", "sum"), n_clusters=("cluster_id", "count"))
              .sort_values("n_complaints", ascending=False, ignore_index=True))
    prob["problem_id"] = np.arange(1, len(prob) + 1)
    # description: from the largest cluster of each problem
    desc = (rv.merge(labels[["cluster_id", "description"]], on="cluster_id")
              .sort_values("n_complaints", ascending=False)
              .drop_duplicates("problem")[["problem", "description"]])
    prob = prob.merge(desc, on="problem", how="left")
    prob["is_recurring_problem"] = True
    rest = pd.DataFrame([
        {"problem_id": UNCLUSTERED_ID, "problem": "Unclustered (one-off complaints)",
         "system": "Not classified", "issue_type": "Not classified", "is_recurring_problem": False,
         "description": "Complaints that do not match any recurring problem."},
        {"problem_id": TOO_SHORT_ID, "problem": "Text too short to classify",
         "system": "Not classified", "issue_type": "Not classified", "is_recurring_problem": False,
         "description": "Complaints with fewer than five words of text."},
    ])
    dim_problem = pd.concat([prob, rest], ignore_index=True)
    for col in ["n_complaints", "n_clusters"]:
        dim_problem[col] = dim_problem[col].astype("Int64")
    dim_problem = dim_problem[["problem_id", "problem", "system", "issue_type", "description",
                               "n_complaints", "n_clusters", "is_recurring_problem"]]

    # ---- dim_cluster: the clusters behind each problem
    cl_problem = rv[["cluster_id", "problem"]].merge(prob[["problem", "problem_id"]], on="problem")
    dim_cluster = (labels[labels["cluster_id"] > 0][["cluster_id", "label", "keywords", "label_source"]]
                   .rename(columns={"label": "llm_label"})
                   .merge(cl_problem[["cluster_id", "problem_id"]], on="cluster_id")
                   .merge(rv[["cluster_id", "review", "note"]], on="cluster_id"))

    # ---- fact_complaint
    f = c.merge(cc[["complaint_id", "cluster_id", "assignment", "centroid_similarity"]],
                on="complaint_id", how="left")
    f = f.merge(cl_problem[["cluster_id", "problem_id"]], on="cluster_id", how="left")
    f.loc[f["cluster_id"] == -1, "problem_id"] = UNCLUSTERED_ID
    f.loc[f["cluster_id"] == -2, "problem_id"] = TOO_SHORT_ID
    f["problem_id"] = f["problem_id"].astype(int)

    f["severity_level"] = np.select(
        [(f["injured"] > 0) | (f["deaths"] > 0), f["crash"] | f["fire"], f["towed"]],
        [4, 3, 2], default=1)

    fact = pd.DataFrame({
        "complaint_id": f["complaint_id"],
        "received_date": f["received_date"].dt.date,
        "failure_date": f["failure_date"].dt.date,
        "model": f["model"].str.title(),
        "model_year": f["model_year"],
        "state": f["state"],
        "channel": f["channel"],
        "problem_id": f["problem_id"],
        "cluster_id": f["cluster_id"],
        "cluster_assignment": f["assignment"],
        "severity_level": f["severity_level"],
        "crash": f["crash"].astype(int),
        "fire": f["fire"].astype(int),
        "injured": f["injured"],
        "deaths": f["deaths"],
        "towed": f["towed"].astype(int),
        "mentions_recall": f["mentions_recall"].astype(int),
        "recall_parts_delay": f["recall_parts_delay"].astype(int),
        "code_unknown_only": f["code_unknown_only"].astype(int),
        "complaint_text": one_line(f["text_raw"]),
    })

    # ---- bridge_component
    bridge = comp.rename(columns={"component": "nhtsa_component", "component_top": "nhtsa_system"})
    bridge["component_unknown"] = bridge["component_unknown"].astype(int)

    # ---- dim_date
    days = pd.date_range(fact["received_date"].min(), fact["received_date"].max(), freq="D")
    dim_date = pd.DataFrame({
        "date": days.date,
        "year": days.year,
        "quarter": "Q" + days.quarter.astype(str),
        "month_start": days.to_period("M").to_timestamp().date,
        "month_label": days.strftime("%b %Y"),
        "month_sort": days.year * 100 + days.month,
    })

    # ---- write
    tables = {
        "fact_complaint": fact, "bridge_component": bridge, "dim_problem": dim_problem,
        "dim_cluster": dim_cluster, "dim_severity": SEVERITY, "dim_date": dim_date,
    }
    for name, t in tables.items():
        t.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")
        print(f"{name:<18} {len(t):>8,} rows")

    # ---- checks
    assert fact["complaint_id"].is_unique
    assert fact["problem_id"].isin(dim_problem["problem_id"]).all()
    assert bridge["complaint_id"].isin(fact["complaint_id"]).all()
    assert set(fact["received_date"]) <= set(dim_date["date"])
    assert dim_problem.loc[dim_problem.is_recurring_problem, "n_complaints"].sum() == \
        fact["problem_id"].lt(UNCLUSTERED_ID).sum()
    print("\nChecks passed: keys unique, every key has a match, totals agree.")


if __name__ == "__main__":
    main()
