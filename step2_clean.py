"""
Step 2: scope, clean and reshape the complaints for embedding.

Reads  complaints_raw.parquet (from step 1)
Writes complaints_clean.parquet      one row per complaint, cleaned text + flags
       complaint_components.parquet  one row per complaint x component code

Usage:
    python step2_clean.py
"""
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent

# ---------------------------------------------------------------- scope
MANUFACTURER = "Ford Motor Company"
DATE_FROM = "2020-01-01"
DATE_TO = "2026-09-30"          # October 2026 is a partial month
MIN_WORDS = 5                   # shorter texts are kept but not clustered

# ---------------------------------------------------------------- text rules
# Hotline complaints (CMPL_TYPE = EVOQ) are written by NHTSA staff from a
# template. These sentences describe the reporting process, not the defect.
# Left in, they make hotline texts cluster together by writing style.
TEMPLATE_SENTENCES = [
    r"The contact (?:owns|owned|leases|leased) an? [^.]*?\.",
    r"The (?:approximate )?failure mileage was[^.]*\.",
    r"The vehicle was (?:not )?(?:yet )?(?:diagnosed or )?repaired[^.]*\.",
    r"The vehicle was not taken to [^.]*\.",
    r"(?:Neither|Both) the (?:local )?dealer (?:nor|and) the manufacturer (?:was|were) (?:not )?(?:notified|made aware|informed|contacted)[^.]*\.",
    r"The manufacturer was (?:not )?(?:notified|made aware|informed|contacted)[^.]*\.",
    r"(?:The|A) (?:local )?dealer(?:ship)? was (?:not )?(?:notified|made aware|informed|contacted)[^.]*\.",
    r"(?:An|The) independent mechanic was (?:not )?contacted\.",
    r"(?:No|There (?:was|were) no) warning lights? (?:was |were )?illuminated[^.]*\.",
    r"The VIN was (?:not available|unavailable|unknown)[^.]*\.",
    r"[^.]*NHTSA Hotline[^.]*\.",
    r"The contact had not experienced a failure\.",
    r"VIN tool confirms parts not available\.",
    r"See attached document[^.]*\.",
]
TEMPLATE_RE = re.compile("|".join(TEMPLATE_SENTENCES), flags=re.I)

REDACTION_RE = re.compile(
    r"INFORMATION REDACTED PURSUANT TO THE FREEDOM OF INFORMATION ACT \(FOIA\),"
    r"\s*5 U\.S\.C\.\s*552\(B\)\(6\)|\[?X{3,}\]?",
    flags=re.I,
)

# NHTSA campaign numbers (24V031000) and Ford recall codes (21S11, 22C05).
# The number itself means nothing to an embedding model; the word recall does.
CAMPAIGN_RE = re.compile(
    r"NHTSA Campaign Number:?\s*\d{2}V\d{3}(?:000)?|\b\d{2}V\d{3}(?:000)?\b|\b\d{2}[SCNE]\d{2}\b",
    flags=re.I,
)

# Flags we keep before the text is cleaned.
RECALL_RE = re.compile(r"\brecall|NHTSA Campaign Number", flags=re.I)
PARTS_DELAY_RE = re.compile(
    r"parts?\b[^.;]{0,40}?\b(?:was|were|are|is)\s+(?:not\s+(?:yet\s+)?available|unavailable)"
    r"|parts not available|back[\s-]?order|no parts|waiting (?:on|for) (?:the )?parts?"
    r"|exceeded a reasonable amount of time for the recall",
    flags=re.I,
)
NO_FAILURE_RE = re.compile(r"had not experienced a failure", flags=re.I)

CHANNEL = {"IVOQ": "Web", "MIVQ": "Web", "MAVQ": "Web", "EVOQ": "Hotline"}

# Hotline staff write in third person and NHTSA jargon; owners write in first
# person and plain words. A first clustering run split the same failure into a
# "hotline" cluster and a "web" cluster. Mapping the jargon to plain wording
# makes the clusters form around the failure instead of the reporting channel.
HOTLINE_TO_PLAIN = [
    (r"^\s*TL\s*[*-]\s*", ""),                       # old hotline record prefix
    (r"\bthe contact's\b", "my"),
    (r"\bthe contact\b", "I"),
    (r"\blost motive power\b", "lost power"),
    (r"\bmotive power\b", "power"),
    (r"\b(?:was|were) illuminated\b", "came on"),
    (r"\billuminated\b", "came on"),
    (r"\bwas informed\b", "was told"),
    (r"\bThe vehicle was taken to\b", "I took the vehicle to"),
    (r"\bwhere it was diagnosed (?:that|with)\b", "where they found"),
    (r"\bwho diagnosed (?:that|with)\b", "who found"),
    (r"\bwhile driving (?:at )?(?:an undisclosed speed|various speeds|approximately)\b,?", "while driving"),
]


def model_pattern(model: str) -> re.Pattern:
    """Own model name with optional year and make in front, e.g.
    '2018 Ford F-150', 'Ford F150', 'F 150', 'my Bronco' for BRONCO SPORT."""
    words = model.split()
    variants = [r"\s+".join(re.escape(w) for w in words)]
    if len(words) > 1 and len(words[0]) >= 4:
        variants.append(re.escape(words[0]))
    body = "|".join(variants).replace(r"\-", r"[-\s]?")
    return re.compile(
        rf"(?:\b(?:19|20)\d\d\s+)?(?:\b(?:ford|lincoln|mercury)\s+)?(?<![\w-])(?:{body})(?:'?s)?(?!\w)",
        flags=re.I,
    )


def clean_text(s: pd.Series) -> pd.Series:
    s = s.str.replace(r"^\s*TL\s*[*-]\s*", "", regex=True)
    s = s.str.replace(REDACTION_RE, " ", regex=True)
    s = s.str.replace(TEMPLATE_RE, " ", regex=True)
    s = s.str.replace(CAMPAIGN_RE, "recall", regex=True)
    s = s.str.replace(r"\bThe contact stated(?: that)?,?\s*", "", regex=True, case=False)
    for pattern, plain in HOTLINE_TO_PLAIN:
        s = s.str.replace(pattern, plain, regex=True, case=False)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s


def mask_models(df: pd.DataFrame) -> pd.Series:
    """Replace each complaint's own model name with 'the vehicle', so clusters
    form around the failure and model stays a filter in Power BI."""
    out = df["text_clean"].copy()
    for model, idx in df.groupby("model").groups.items():
        if not model:
            continue
        out.loc[idx] = out.loc[idx].str.replace(model_pattern(model), "the vehicle", regex=True)
    return out.str.replace(r"\s+", " ", regex=True).str.strip()


def main() -> None:
    raw = pd.read_parquet(HERE / "complaints_raw.parquet")

    scope = raw[
        (raw["MFR_NAME"] == MANUFACTURER)
        & (raw["PROD_TYPE"] == "V")
        & (raw["LDATE"] >= DATE_FROM)
        & (raw["LDATE"] <= DATE_TO)
    ].copy()
    print(f"In scope: {scope['ODINO'].nunique():,} complaints, {len(scope):,} rows")

    # ---- component table (complaint x component)
    comp = scope[["ODINO", "COMPDESC"]].drop_duplicates().copy()
    comp["component_top"] = comp["COMPDESC"].str.split(":").str[0].str.strip()
    comp = comp.rename(columns={"ODINO": "complaint_id", "COMPDESC": "component"})
    comp["component_unknown"] = comp["component_top"].eq("UNKNOWN OR OTHER")

    # ---- complaint table (one row per complaint)
    c = scope.drop_duplicates("ODINO").copy()
    out = pd.DataFrame({
        "complaint_id": c["ODINO"],
        "received_date": c["LDATE"].dt.normalize(),
        "failure_date": c["FAILDATE"],
        "make": c["MAKETXT"].str.strip(),
        "model": c["MODELTXT"].str.strip(),
        "model_year": pd.to_numeric(c["YEARTXT"], errors="coerce").replace(9999, pd.NA).astype("Int64"),
        "state": c["STATE"].str.strip(),
        "channel": c["CMPL_TYPE"].map(CHANNEL).fillna("Other"),
        "crash": c["CRASH"].eq("Y"),
        "fire": c["FIRE"].eq("Y"),
        "injured": c["INJURED"].fillna(0).astype(int),
        "deaths": c["DEATHS"].fillna(0).astype(int),
        "towed": c["VEHICLES_TOWED_YN"].eq("Y"),
        "medical_attention": c["MEDICAL_ATTN"].eq("Y"),
        "text_raw": c["CDESCR"].str.strip(),
    })

    # implausible failure dates: before 1990 or after the complaint was received
    bad = (out["failure_date"] < "1990-01-01") | (out["failure_date"] > out["received_date"])
    out.loc[bad, "failure_date"] = pd.NaT

    # flags from the raw text
    out["mentions_recall"] = out["text_raw"].str.contains(RECALL_RE)
    out["recall_parts_delay"] = out["text_raw"].str.contains(PARTS_DELAY_RE)
    out["no_failure_reported"] = out["text_raw"].str.contains(NO_FAILURE_RE)

    # only unknown/other component code?
    known = comp.loc[~comp["component_unknown"], "complaint_id"].unique()
    out["code_unknown_only"] = ~out["complaint_id"].isin(known)

    # clean text
    out["text_clean"] = clean_text(out["text_raw"])
    out["text_clean"] = mask_models(out)
    out["words_raw"] = out["text_raw"].str.split().str.len().fillna(0).astype(int)
    out["words_clean"] = out["text_clean"].str.split().str.len().fillna(0).astype(int)
    out["usable_for_clustering"] = out["words_clean"] >= MIN_WORDS

    # exact duplicate texts: keep them as complaints, embed the text once
    key = out["text_clean"].str.lower()
    out["duplicate_text"] = key.duplicated(keep="first") & out["usable_for_clustering"]

    out = out.sort_values("received_date").reset_index(drop=True)

    # ---- report
    n = len(out)
    hot = out["channel"].eq("Hotline")
    print(f"\nComplaints:              {n:,}")
    print(f"Months:                  {out['received_date'].dt.to_period('M').nunique()}")
    print(f"Hotline share:           {hot.mean():.1%}")
    print(f"Mentions a recall:       {out['mentions_recall'].mean():.1%}")
    print(f"Recall parts delay:      {out['recall_parts_delay'].mean():.1%}")
    print(f"No failure reported:     {out['no_failure_reported'].mean():.1%}")
    print(f"Code UNKNOWN only:       {out['code_unknown_only'].mean():.1%}")
    print(f"Median words, raw/clean: web {out.loc[~hot,'words_raw'].median():.0f}/"
          f"{out.loc[~hot,'words_clean'].median():.0f}, hotline "
          f"{out.loc[hot,'words_raw'].median():.0f}/{out.loc[hot,'words_clean'].median():.0f}")
    print(f"Too short to cluster:    {(~out['usable_for_clustering']).sum():,}")
    print(f"Duplicate texts:         {out['duplicate_text'].sum():,}")
    to_embed = out["usable_for_clustering"] & ~out["duplicate_text"]
    print(f"Texts to embed (step 3): {to_embed.sum():,}")

    print("\nTop models:")
    print(out["model"].value_counts().head(8).to_string())

    print("\nExample, hotline text before and after:")
    ex = out[hot & out["words_raw"].between(60, 110)].sample(1, random_state=7).iloc[0]
    print("BEFORE:", ex["text_raw"])
    print("AFTER: ", ex["text_clean"])

    out.to_parquet(HERE / "complaints_clean.parquet", index=False)
    comp.to_parquet(HERE / "complaint_components.parquet", index=False)
    print(f"\nSaved complaints_clean.parquet ({n:,} rows) and "
          f"complaint_components.parquet ({len(comp):,} rows)")


if __name__ == "__main__":
    main()
