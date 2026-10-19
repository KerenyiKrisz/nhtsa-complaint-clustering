"""
Step 1: load the NHTSA complaint flat files and take a first look.

Usage:
    python step1_explore.py COMPLAINTS_RECEIVED_2020-2024.txt COMPLAINTS_RECEIVED_2025-2026.txt

Writes complaints_raw.parquet next to this script for step 2.
"""
import csv
import sys
from pathlib import Path

import pandas as pd

# Column order from CMPL.txt (fields 1..51). The files have no header row.
COLUMNS = [
    "CMPLID", "ODINO", "MFR_NAME", "MAKETXT", "MODELTXT", "YEARTXT", "CRASH",
    "FAILDATE", "FIRE", "INJURED", "DEATHS", "COMPDESC", "CITY", "STATE", "VIN",
    "DATEA", "LDATE", "MILES", "OCCURENCES", "CDESCR", "CMPL_TYPE",
    "POLICE_RPT_YN", "PURCH_DT", "ORIG_OWNER_YN", "ANTI_BRAKES_YN",
    "CRUISE_CONT_YN", "NUM_CYLS", "DRIVE_TRAIN", "FUEL_SYS", "FUEL_TYPE",
    "TRANS_TYPE", "VEH_SPEED", "DOT", "TIRE_SIZE", "LOC_OF_TIRE",
    "TIRE_FAIL_TYPE", "ORIG_EQUIP_YN", "MANUF_DT", "SEAT_TYPE",
    "RESTRAINT_TYPE", "DEALER_NAME", "DEALER_TEL", "DEALER_CITY",
    "DEALER_STATE", "DEALER_ZIP", "PROD_TYPE", "REPAIRED_YN", "MEDICAL_ATTN",
    "VEHICLES_TOWED_YN", "STATE_OF_INCIDENT", "VEHICLE_OPERATOR",
]

# Personal data we never need for this analysis. Drop on load.
PII = ["CITY", "VIN", "DEALER_NAME", "DEALER_TEL", "DEALER_CITY",
       "DEALER_ZIP", "VEHICLE_OPERATOR"]

# Tire and child seat fields: irrelevant once we keep vehicles only.
NOT_NEEDED = ["DOT", "TIRE_SIZE", "LOC_OF_TIRE", "TIRE_FAIL_TYPE",
              "SEAT_TYPE", "RESTRAINT_TYPE", "REPAIRED_YN"]


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=COLUMNS,
        dtype=str,
        quoting=csv.QUOTE_NONE,   # the text contains stray quote marks
        encoding="utf-8",
        keep_default_na=False,    # keep empty strings as "", not NaN
    )
    df = df.drop(columns=PII + NOT_NEEDED)
    df["source_file"] = Path(path).name
    return df


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main(paths: list[str]) -> None:
    df = pd.concat([load(p) for p in paths], ignore_index=True)

    for c in ["LDATE", "FAILDATE"]:
        df[c] = pd.to_datetime(df[c], format="%Y%m%d", errors="coerce")
    for c in ["INJURED", "DEATHS", "MILES"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    section("Size")
    print(f"rows (complaint x component): {len(df):,}")
    print(f"unique complaints (ODINO):    {df['ODINO'].nunique():,}")
    print(f"rows per complaint: "
          f"{len(df) / df['ODINO'].nunique():.2f} on average")
    print(df.groupby("ODINO").size().value_counts().sort_index().head(6)
          .rename("complaints with n component rows").to_string())

    section("Date range (LDATE = date NHTSA received it)")
    print(df["LDATE"].min().date(), "to", df["LDATE"].max().date())
    monthly = df.drop_duplicates("ODINO").set_index("LDATE").resample("MS").size()
    print("complaints per month, last 6:")
    print(monthly.tail(6).to_string())

    section("Product type (V vehicle, T tires, E equipment, C child seat)")
    print(df.drop_duplicates("ODINO")["PROD_TYPE"].value_counts().to_string())

    section("Free text (CDESCR), one row per complaint")
    txt = df.drop_duplicates("ODINO")["CDESCR"].str.strip()
    words = txt.str.split().str.len().fillna(0)
    chars = txt.str.len()
    print(f"empty:              {(chars == 0).mean():.1%}")
    print(f"under 10 words:     {(words < 10).mean():.1%}")
    print("words, quantiles:")
    print(words.quantile([.1, .25, .5, .75, .9, .99]).round(0).to_string())
    print(f"hit the 2048 char cap (truncated): {(chars >= 2040).mean():.1%}")
    print(f"total words:        {int(words.sum()):,}")

    section("Same text on every component row of a complaint?")
    per = df.groupby("ODINO")["CDESCR"].nunique()
    print(f"complaints with >1 distinct text: {(per > 1).sum():,}")

    section("Component codes (COMPDESC), row level")
    df["COMP_TOP"] = df["COMPDESC"].str.split(":").str[0].str.strip()
    print(f"distinct full codes: {df['COMPDESC'].nunique():,}")
    print(f"distinct top level:  {df['COMP_TOP'].nunique():,}")
    print(df["COMP_TOP"].value_counts().head(15).to_string())

    section("Manufacturers, complaint level, top 12")
    c = df.drop_duplicates("ODINO")
    print(c["MFR_NAME"].value_counts().head(12).to_string())

    section("Make + model, complaint level, top 12")
    print((c["MAKETXT"] + " " + c["MODELTXT"]).value_counts().head(12).to_string())

    section("Severity flags, complaint level")
    print(f"crash:   {(c['CRASH'] == 'Y').mean():.1%}")
    print(f"fire:    {(c['FIRE'] == 'Y').mean():.1%}")
    print(f"injured >0: {(c['INJURED'] > 0).mean():.1%}")
    print(f"deaths  >0: {(c['DEATHS'] > 0).mean():.2%}")
    print(f"towed:   {(c['VEHICLES_TOWED_YN'] == 'Y').mean():.1%}")

    section("Fill rate of the fields we will use")
    use = ["MFR_NAME", "MAKETXT", "MODELTXT", "YEARTXT", "FAILDATE", "MILES",
           "COMPDESC", "STATE", "CMPL_TYPE", "VEHICLES_TOWED_YN"]
    fill = {}
    for col in use:
        s = c[col]
        if pd.api.types.is_numeric_dtype(s):
            empty = s.isna() | (s == 0)
        elif pd.api.types.is_datetime64_any_dtype(s):
            empty = s.isna()
        else:  # text columns: works on pandas 2.x and 3.x
            empty = s.fillna("").str.strip().isin(["", "9999"])
        fill[col] = f"{1 - empty.mean():.0%}"
    print(pd.Series(fill).to_string())

    section("Complaint source (CMPL_TYPE)")
    print(c["CMPL_TYPE"].value_counts().head(6).to_string())

    out = Path(__file__).with_name("complaints_raw.parquet")
    df.to_parquet(out, index=False)
    print(f"\nSaved {out.name}: {len(df):,} rows, {df.shape[1]} columns")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
