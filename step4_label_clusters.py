"""
Step 4: name each cluster with a local LLM (Ollama), fully offline.

Reads  clusters.parquet, complaint_clusters.parquet, complaints_clean.parquet
Writes cluster_labels.parquet / cluster_labels.csv   one row per cluster
       cache/labels_cache.json                       progress, so a rerun resumes

Needs Ollama running on this machine with the model pulled:
    ollama pull qwen3.5:4b

Usage:
    python step4_label_clusters.py
"""
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
CACHE_FILE = HERE / "cache" / "labels_cache.json"

# ---------------------------------------------------------------- settings
MODEL = "qwen3.5:4b"
HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
if not HOST.startswith("http"):
    HOST = "http://" + HOST
N_EXAMPLES = 12            # complaints shown to the model per cluster
MAX_CHARS = 300            # each complaint is cut to this length
MAX_TRIES = 3

SYSTEMS = [
    "Engine & cooling",
    "Transmission & driveline",
    "Steering",
    "Brakes",
    "Suspension, wheels & tires",
    "Fuel system & emissions",
    "Electrical & battery",
    "Cameras & driver assistance",
    "Airbags & seat belts",
    "Body, doors & glass",
    "Lights, climate & infotainment",
    "Fire & crash reports",
    "General / multiple systems",
]
ISSUE_TYPES = ["Product defect", "Recall or service process"]

SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string"},
        "description": {"type": "string"},
        "system": {"type": "string", "enum": SYSTEMS},
        "issue_type": {"type": "string", "enum": ISSUE_TYPES},
    },
    "required": ["label", "description", "system", "issue_type"],
}

SYSTEM_PROMPT = (
    "You name recurring problems found in vehicle owner complaints for a "
    "service quality team. Be specific and factual. Use only what the "
    "complaints say."
)

PROMPT = """Below are complaints from one cluster of similar complaints about Ford vehicles, with the cluster's most distinctive keywords.

Cluster facts: {n:,} complaints. {recall:.0%} mention a recall; {parts:.0%} say recall parts or repairs are not available.
Keywords: {keywords}

Complaints:
{examples}

Return JSON with:
- label: the specific recurring problem in 3 to 8 words, for example "Rear axle bolt shears off" or "Backup camera image goes blank". Name the part and what goes wrong. No vehicle model names, no brand names.
- description: one sentence, at most 25 words, on what owners experience.
- system: the vehicle system, one of: {systems}.
- issue_type: "Product defect" if owners describe a failure of the vehicle; "Recall or service process" if most complaints are about recall notices, recall parts not being available, repair delays, or a VIN not covered by a recall.
"""


# ---------------------------------------------------------------- ollama
def ollama(path: str, payload: dict | None = None, timeout: int = 600) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(HOST + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def check_ollama() -> None:
    try:
        tags = ollama("/api/tags", timeout=10)
    except (urllib.error.URLError, OSError):
        raise SystemExit(f"Ollama is not reachable at {HOST}. Start the Ollama app and try again.")
    names = [m["name"] for m in tags.get("models", [])]
    if MODEL not in names:
        raise SystemExit(f"Model {MODEL} not found. Run:  ollama pull {MODEL}")


def ask(prompt: str, temperature: float) -> dict:
    resp = ollama("/api/chat", {
        "model": MODEL,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": prompt}],
        "think": False,
        "stream": False,
        "format": SCHEMA,
        "options": {"temperature": temperature, "seed": 42, "num_ctx": 8192},
    })
    return json.loads(resp["message"]["content"])


def valid(out: dict) -> bool:
    words = len(str(out.get("label", "")).split())
    return (2 <= words <= 10
            and len(str(out.get("description", ""))) > 10
            and out.get("system") in SYSTEMS
            and out.get("issue_type") in ISSUE_TYPES)


# ---------------------------------------------------------------- main
def main() -> None:
    check_ollama()
    clusters = pd.read_parquet(HERE / "clusters.parquet")
    cc = pd.read_parquet(HERE / "complaint_clusters.parquet")
    flags = pd.read_parquet(HERE / "complaints_clean.parquet",
                            columns=["complaint_id", "mentions_recall", "recall_parts_delay"])
    stats = cc.merge(flags, on="complaint_id").groupby("cluster_id")[
        ["mentions_recall", "recall_parts_delay"]].mean()

    cache = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
    rows, t0 = [], time.time()
    total = len(clusters)

    for i, r in enumerate(clusters.itertuples(index=False), start=1):
        examples = [t[:MAX_CHARS] for t in list(r.example_texts)[:N_EXAMPLES]]
        key = f"{r.cluster_id}:" + hashlib.sha1("\n".join(examples).encode()).hexdigest()[:10]

        if key in cache:
            out = cache[key]
        else:
            prompt = PROMPT.format(
                n=r.n_complaints,
                recall=stats.loc[r.cluster_id, "mentions_recall"],
                parts=stats.loc[r.cluster_id, "recall_parts_delay"],
                keywords=r.keywords,
                examples="\n".join(f"{k}. {t}" for k, t in enumerate(examples, 1)),
                systems=", ".join(f'"{s}"' for s in SYSTEMS),
            )
            t = time.time()
            out = None
            for attempt in range(MAX_TRIES):
                try:
                    cand = ask(prompt, temperature=0 if attempt == 0 else 0.4)
                    if valid(cand):
                        out = {**cand, "label_source": "llm"}
                        break
                except (json.JSONDecodeError, KeyError, urllib.error.URLError, TimeoutError):
                    pass
            if out is None:   # fall back to keywords rather than stop the run
                out = {"label": ", ".join(r.keywords.split(", ")[:3]),
                       "description": "", "system": "General / multiple systems",
                       "issue_type": "Product defect", "label_source": "keywords"}
            cache[key] = out
            CACHE_FILE.parent.mkdir(exist_ok=True)
            CACHE_FILE.write_text(json.dumps(cache, indent=1))
            print(f"[{i}/{total}] #{r.cluster_id:<3} {out['label']}  "
                  f"({out['system']}; {out['issue_type']})  {time.time() - t:.0f}s", flush=True)

        rows.append({"cluster_id": int(r.cluster_id), "n_complaints": int(r.n_complaints),
                     **{k: out[k] for k in ["label", "description", "system", "issue_type", "label_source"]},
                     "keywords": r.keywords})

    labels = pd.DataFrame(rows)
    # complete the dimension table for Power BI
    extra = pd.DataFrame([
        {"cluster_id": -1, "label": "Unclustered (one-off complaints)", "system": "General / multiple systems",
         "issue_type": "Product defect", "description": "Complaints that do not match any recurring problem.",
         "label_source": "rule"},
        {"cluster_id": -2, "label": "Text too short to classify", "system": "General / multiple systems",
         "issue_type": "Product defect", "description": "Complaints with fewer than five words of text.",
         "label_source": "rule"},
    ])
    counts = cc["cluster_id"].value_counts()
    extra["n_complaints"] = extra["cluster_id"].map(counts).fillna(0).astype(int)
    extra["keywords"] = ""
    labels = pd.concat([labels, extra], ignore_index=True)
    labels["model"] = MODEL

    labels.to_parquet(HERE / "cluster_labels.parquet", index=False)
    labels.to_csv(HERE / "cluster_labels.csv", index=False, encoding="utf-8-sig")

    # ---- report
    print(f"\nLabeled {total} clusters in {(time.time() - t0) / 60:.1f} min "
          f"({(labels['label_source'] == 'keywords').sum()} fell back to keywords)")
    print("\nComplaints by system:")
    by_sys = labels[labels.cluster_id > 0].groupby("system")["n_complaints"].sum().sort_values(ascending=False)
    print(by_sys.to_string())
    print("\nComplaints by issue type:")
    print(labels[labels.cluster_id > 0].groupby("issue_type")["n_complaints"].sum().to_string())
    dup = labels[labels.duplicated("label", keep=False) & (labels.cluster_id > 0)]
    if len(dup):
        print(f"\nSame label on more than one cluster ({len(dup)} clusters), merged in Power BI by label:")
        print(dup.sort_values("label")[["cluster_id", "label", "n_complaints"]].to_string(index=False))
    print("\nSaved cluster_labels.parquet and cluster_labels.csv")


if __name__ == "__main__":
    main()
