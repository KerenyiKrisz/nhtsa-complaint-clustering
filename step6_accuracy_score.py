"""
Step 6b: score the manual accuracy check.

Reads  accuracy_review.xlsx   (filled in by hand, Verdict column)
       accuracy_key.csv       (which stratum each row came from)
       complaint_clusters.parquet (population sizes, for weighting)
Writes accuracy_results.txt

Two accuracy figures per stratum:
  strict   = Correct / reviewed
  lenient  = (Correct + Partly) / reviewed
with Wilson 95% intervals. The overall figure weights core and nearest by
their share of all assigned complaints, since the sample takes 100 of each.

Usage:
    python step6_accuracy_score.py
"""
from math import sqrt
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
Z = 1.96


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + Z**2 / n
    c = (p + Z**2 / (2 * n)) / d
    h = Z * sqrt(p * (1 - p) / n + Z**2 / (4 * n**2)) / d
    return c - h, c + h


def main() -> None:
    rev = pd.read_excel(HERE / "accuracy_review.xlsx", sheet_name="Review")
    rev = rev.rename(columns={"#": "review_id", "Verdict": "verdict", "Better fit": "better_fit",
                              "Assigned problem": "assigned"})
    key = pd.read_csv(HERE / "accuracy_key.csv")
    df = key.merge(rev[["review_id", "verdict", "better_fit"]], on="review_id")
    df["verdict"] = df["verdict"].astype(str).str.strip().str.capitalize()
    done = df[df["verdict"].isin(["Correct", "Partly", "Wrong"])]

    pop = pd.read_parquet(HERE / "complaint_clusters.parquet")["assignment"].value_counts()
    total = len(pd.read_parquet(HERE / "complaint_clusters.parquet"))
    assigned = pop["core"] + pop["nearest"]
    weights = {"core": pop["core"] / assigned, "nearest": pop["nearest"] / assigned}

    lines = [f"Reviewed {len(done)} of {len(df)} sampled complaints.", ""]
    est = {}
    for stratum in ["core", "nearest"]:
        g = done[done["assignment"] == stratum]
        n = len(g)
        c = (g["verdict"] == "Correct").sum()
        cp = g["verdict"].isin(["Correct", "Partly"]).sum()
        lo, hi = wilson(c, n)
        lo2, hi2 = wilson(cp, n)
        est[stratum] = (c / n if n else float("nan"), cp / n if n else float("nan"), n)
        lines += [
            f"{stratum.upper()} ({pop[stratum]:,} complaints, {weights[stratum]:.0%} of assigned)",
            f"  reviewed {n}: correct {c}, partly {cp - c}, wrong {n - cp}",
            f"  strict  accuracy {c / n:.0%}  (95% CI {lo:.0%} to {hi:.0%})" if n else "  no rows reviewed",
            f"  lenient accuracy {cp / n:.0%}  (95% CI {lo2:.0%} to {hi2:.0%})" if n else "",
            "",
        ]

    if all(est[s][2] > 0 for s in est):
        for i, name in [(0, "strict"), (1, "lenient")]:
            p = sum(weights[s] * est[s][i] for s in est)
            se = sqrt(sum(weights[s] ** 2 * est[s][i] * (1 - est[s][i]) / est[s][2] for s in est))
            lines.append(f"OVERALL {name:<7} {p:.0%}  (95% CI {max(0, p - Z * se):.0%} to {min(1, p + Z * se):.0%}),"
                         f" weighted by stratum size")
        lines += ["", f"Coverage: {assigned:,} of {total:,} complaints ({assigned / total:.1%}) are assigned "
                      f"to a problem; the rest are one-offs or too short."]

    wrong = done[done["verdict"] != "Correct"]
    if len(wrong):
        lines += ["", "Most common misassignments (assigned -> better fit):"]
        pairs = (wrong.assign(better_fit=wrong["better_fit"].fillna("not given"))
                 .groupby(["problem", "better_fit"]).size().sort_values(ascending=False).head(10))
        lines += [f"  {n}x  {a}  ->  {b}" for (a, b), n in pairs.items()]

    text = "\n".join(lines)
    print(text)
    (HERE / "accuracy_results.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
