"""
Step 6a: draw a random sample of complaints for a manual accuracy check.

Question: when the pipeline puts a complaint under a problem, how often is
that problem actually what the complaint is about?

Design
- Two strata, checked separately, because they were assigned differently:
    core     complaints HDBSCAN placed inside a cluster
    nearest  noise points that joined their nearest cluster (similarity >= 0.60)
- 100 random complaints from each, shuffled together. The review sheet does not
  show which stratum a row is from, so the reviewer is not biased by it.
- The stratum key is written to a separate CSV that step 6b joins back.

Reads  complaints_clean.parquet, complaint_clusters.parquet,
       powerbi/dim_cluster.csv, powerbi/dim_problem.csv
Writes accuracy_review.xlsx   the sheet to fill in (one verdict per row)
       accuracy_key.csv       review_id -> complaint_id, stratum, similarity

Usage:
    python step6_accuracy_sample.py
"""
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.datavalidation import DataValidation

HERE = Path(__file__).parent
PER_STRATUM = 100
SEED = 42

FONT = "Arial"
HEADER_FILL = PatternFill("solid", fgColor="1F3A4D")
INPUT_FILL = PatternFill("solid", fgColor="FFF4C2")
THIN = Side(style="thin", color="D0D5DD")


def load() -> pd.DataFrame:
    cl = pd.read_parquet(HERE / "complaints_clean.parquet",
                         columns=["complaint_id", "model", "model_year", "text_clean"])
    cc = pd.read_parquet(HERE / "complaint_clusters.parquet")
    dc = pd.read_csv(HERE / "powerbi" / "dim_cluster.csv")[["cluster_id", "problem_id"]]
    dp = pd.read_csv(HERE / "powerbi" / "dim_problem.csv")[["problem_id", "problem", "description"]]
    df = (cc[cc["assignment"].isin(["core", "nearest"])]
          .merge(dc, on="cluster_id", how="left")
          .merge(dp, on="problem_id", how="left")
          .merge(cl, on="complaint_id", how="left"))
    assert df["problem"].notna().all(), "every assigned complaint needs a problem"
    return df, dp


def sample(df: pd.DataFrame) -> pd.DataFrame:
    parts = [df[df["assignment"] == s].sample(PER_STRATUM, random_state=SEED)
             for s in ["core", "nearest"]]
    s = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
    s.insert(0, "review_id", range(1, len(s) + 1))
    return s


def write_sheet(s: pd.DataFrame, problems: list[str]) -> None:
    wb = Workbook()

    # ------------------------------------------------ instructions
    ins = wb.active
    ins.title = "Instructions"
    rows = [
        ("How to fill in this sheet", True),
        ("", False),
        ("Go to the Review tab. For each complaint, read the text and the problem the pipeline assigned to it.", False),
        ("Fill in only the yellow columns: Verdict (required), Better fit and Note (optional).", False),
        ("", False),
        ("Verdict", True),
        ("Correct: the assigned problem describes the main failure in the complaint.", False),
        ("Partly: right system but a different symptom, or the complaint has several issues and this is one of them.", False),
        ("Wrong: the complaint is about something else.", False),
        ("", False),
        ("Better fit: if not Correct, pick the problem that fits better, or 'None of the list'.", False),
        ("Judge the text only. Don't look anything up; about 15 seconds per row is enough.", False),
        ("When done, save the file and run: python step6_accuracy_score.py", False),
        ("", False),
        ("Example (not part of the sample)", True),
    ]
    for i, (text, bold) in enumerate(rows, start=1):
        c = ins.cell(row=i, column=1, value=text)
        c.font = Font(name=FONT, size=11 if not bold else 12, bold=bold)
    ex_row = len(rows) + 1
    ex = [("Complaint", "while driving at highway speed the steering wheel became very hard to turn and a "
                        "message said steering assist fault, service required."),
          ("Assigned problem", "Power steering assist fails while driving"),
          ("Verdict", "Correct")]
    for j, (k, v) in enumerate(ex):
        ins.cell(row=ex_row + j, column=1, value=f"{k}: {v}").font = Font(name=FONT, size=10, italic=True)
    ins.column_dimensions["A"].width = 120

    # ------------------------------------------------ review
    ws = wb.create_sheet("Review")
    headers = ["#", "Model", "Year", "Complaint text", "Assigned problem",
               "Problem description", "Verdict", "Better fit", "Note"]
    widths = [5, 12, 7, 80, 30, 36, 11, 30, 24]
    for j, (h, w) in enumerate(zip(headers, widths), start=1):
        c = ws.cell(row=1, column=j, value=h)
        c.font = Font(name=FONT, bold=True, color="FFFFFF")
        c.fill = HEADER_FILL
        c.alignment = Alignment(vertical="center", wrap_text=True)
        ws.column_dimensions[c.column_letter].width = w
    ws.freeze_panes = "E2"

    for i, r in s.iterrows():
        row = i + 2
        vals = [r["review_id"], r["model"], None if pd.isna(r["model_year"]) else int(r["model_year"]),
                r["text_clean"], r["problem"], r["description"], None, None, None]
        for j, v in enumerate(vals, start=1):
            c = ws.cell(row=row, column=j, value=v)
            c.font = Font(name=FONT, size=10, bold=(j == 5))
            c.alignment = Alignment(vertical="top", wrap_text=j in (4, 5, 6, 8, 9))
            c.border = Border(bottom=THIN)
            if j >= 7:
                c.fill = INPUT_FILL
    last = len(s) + 1

    # dropdowns
    lists = wb.create_sheet("Lists")
    for k, p in enumerate(problems + ["None of the list"], start=1):
        lists.cell(row=k, column=1, value=p).font = Font(name=FONT, size=10)
    lists.sheet_state = "hidden"
    dv_verdict = DataValidation(type="list", formula1='"Correct,Partly,Wrong"', allow_blank=True)
    dv_better = DataValidation(type="list", formula1=f"=Lists!$A$1:$A${len(problems) + 1}", allow_blank=True)
    ws.add_data_validation(dv_verdict)
    ws.add_data_validation(dv_better)
    dv_verdict.add(f"G2:G{last}")
    dv_better.add(f"H2:H{last}")

    # progress counter on the instructions tab
    p = ex_row + len(ex) + 1
    ins.cell(row=p, column=1, value="Progress").font = Font(name=FONT, size=12, bold=True)
    labels = ["Reviewed", "Correct", "Partly", "Wrong"]
    formulas = [f'=COUNTA(Review!G2:G{last})&" of {len(s)}"',
                f'=COUNTIF(Review!G2:G{last},"Correct")',
                f'=COUNTIF(Review!G2:G{last},"Partly")',
                f'=COUNTIF(Review!G2:G{last},"Wrong")']
    for k, (lab, f) in enumerate(zip(labels, formulas), start=1):
        ins.cell(row=p + k, column=1, value=lab).font = Font(name=FONT, size=10)
        c = ins.cell(row=p + k, column=2, value=f)
        c.font = Font(name=FONT, size=10, bold=True)
        c.alignment = Alignment(horizontal="left")
    ins.column_dimensions["B"].width = 14

    wb.active = 1
    wb.save(HERE / "accuracy_review.xlsx")


def main() -> None:
    df, dp = load()
    s = sample(df)
    problems = (dp[dp["problem_id"] < 900]
                .sort_values("problem")["problem"].tolist())
    write_sheet(s, problems)
    s[["review_id", "complaint_id", "assignment", "centroid_similarity",
       "cluster_id", "problem_id", "problem"]].to_csv(HERE / "accuracy_key.csv", index=False)

    pop = df["assignment"].value_counts()
    print(f"Sampled {PER_STRATUM} core and {PER_STRATUM} nearest complaints (seed {SEED}).")
    print(f"Population: core {pop['core']:,}, nearest {pop['nearest']:,}")
    print("Wrote accuracy_review.xlsx and accuracy_key.csv")


if __name__ == "__main__":
    main()
