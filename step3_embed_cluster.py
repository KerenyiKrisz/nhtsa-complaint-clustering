"""
Step 3: embed the cleaned complaint texts and cluster them.

Reads  complaints_clean.parquet (from step 2)
Writes cache/embeddings.npy        sentence embeddings, reused on rerun
       cache/umap_*.npy            reduced embeddings, reused on rerun
       complaint_clusters.parquet  one row per complaint: cluster_id, assignment
       clusters.parquet            one row per cluster: size, keywords, examples
       cluster_preview.txt         readable check of every cluster

Usage:
    python step3_embed_cluster.py

The first run downloads the embedding model once (about 90 MB) and embeds
all texts on the CPU. Later runs reuse the cache, so tuning the clustering
settings below only takes a few minutes.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

HERE = Path(__file__).parent
CACHE = HERE / "cache"

# ---------------------------------------------------------------- settings
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"   # small, fast on CPU
BATCH_SIZE = 64

UMAP_PARAMS = dict(n_neighbors=30, n_components=10, min_dist=0.0,
                   metric="cosine", random_state=42)

MIN_CLUSTER_SIZE = 150      # smallest group we call a recurring problem
MIN_SAMPLES = 5             # higher = stricter, more complaints left as noise
MAX_CLUSTER_SIZE = 3000     # caps broad groups: without it "transmission" became one
                            # cluster of 14k complaints, i.e. the NHTSA code again
REASSIGN_MIN_SIM = 0.60     # noise joins its nearest cluster above this cosine similarity
                            # (checked by hand: below ~0.6 the matches are mostly wrong)
N_EXAMPLES = 20             # texts per cluster handed to the LLM in step 4

DOMAIN_STOP_WORDS = {
    "vehicle", "vehicles", "car", "truck", "owner", "ford", "lincoln",
    "issue", "issues", "problem", "problems", "said", "told", "just", "like",
    "got", "get", "did", "does", "would", "could", "time", "times", "know",
    "going", "went", "also", "able", "back",
}


# ---------------------------------------------------------------- stages
def embed(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    return model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)


def cached_embeddings(texts: list[str]) -> np.ndarray:
    """Embed each distinct text once. The cache is keyed by the text itself,
    so after a change in step 2 only the changed texts are encoded again."""
    CACHE.mkdir(exist_ok=True)
    vec_path = CACHE / "embeddings.npy"
    txt_path = CACHE / "embedding_texts.parquet"
    if vec_path.exists() and txt_path.exists():
        vecs = np.load(vec_path)
        known = pd.read_parquet(txt_path)["text"].tolist()
    else:
        vecs, known = np.zeros((0, 384), dtype=np.float32), []
    index = {t: i for i, t in enumerate(known)}
    missing = [t for t in dict.fromkeys(texts) if t not in index]

    if missing:
        print(f"Embeddings: {len(texts) - len(missing):,} from cache, "
              f"encoding {len(missing):,} new texts with {MODEL_NAME} on CPU ...")
        t = time.time()
        vecs = np.vstack([vecs, embed(missing)])
        known = known + missing
        np.save(vec_path, vecs)
        pd.DataFrame({"text": known}).to_parquet(txt_path, index=False)
        index = {t: i for i, t in enumerate(known)}
        print(f"Embeddings: done in {(time.time() - t) / 60:.1f} min")
    else:
        print(f"Embeddings: all {len(texts):,} texts loaded from cache")
    return vecs[[index[t] for t in texts]]


def cached_umap(emb: np.ndarray) -> np.ndarray:
    key = hashlib.sha1(
        (json.dumps(UMAP_PARAMS, sort_keys=True) + str(emb.shape) + str(emb[:5].sum())).encode()
    ).hexdigest()[:12]
    path = CACHE / f"umap_{key}.npy"
    if path.exists():
        print(f"UMAP: loaded from cache ({path.name})")
        return np.load(path)
    import umap
    print("UMAP: reducing to 10 dimensions (a few minutes) ...")
    t = time.time()
    red = umap.UMAP(**UMAP_PARAMS).fit_transform(emb)
    np.save(path, red)
    print(f"UMAP: done in {(time.time() - t) / 60:.1f} min")
    return red


def cluster(red: np.ndarray) -> np.ndarray:
    print("HDBSCAN: clustering ...")
    t = time.time()
    labels = HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        max_cluster_size=MAX_CLUSTER_SIZE,
        cluster_selection_method="eom",
        n_jobs=-1,
    ).fit_predict(red)
    print(f"HDBSCAN: done in {(time.time() - t) / 60:.1f} min")
    return labels


def centroids_of(emb: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ids = np.array(sorted(set(labels) - {-1}))
    cents = np.vstack([emb[labels == i].mean(axis=0) for i in ids])
    cents /= np.linalg.norm(cents, axis=1, keepdims=True)
    return ids, cents


def keywords_by_cluster(texts: pd.Series, labels: np.ndarray, top_n: int = 10) -> dict:
    """Class-based TF-IDF: words frequent in one cluster and rare in the others."""
    vec = CountVectorizer(
        stop_words=list(ENGLISH_STOP_WORDS | DOMAIN_STOP_WORDS),
        ngram_range=(1, 2), min_df=5, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )
    X = vec.fit_transform(texts)
    terms = np.array(vec.get_feature_names_out())
    ids = sorted(set(labels) - {-1})
    tf = np.vstack([np.asarray(X[labels == i].sum(axis=0)).ravel() for i in ids])
    tf_norm = tf / tf.sum(axis=1, keepdims=True)
    avg_words = tf.sum() / len(ids)
    idf = np.log(1 + avg_words / tf.sum(axis=0).clip(min=1))
    score = tf_norm * idf
    return {i: ", ".join(terms[np.argsort(-score[k])[:top_n]]) for k, i in enumerate(ids)}


# ---------------------------------------------------------------- main
class Tee:
    """Print to the screen and to step3_output.txt at the same time."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()
    def flush(self):
        for st in self.streams:
            st.flush()


def main() -> None:
    log = open(HERE / "step3_output.txt", "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log)
    t0 = time.time()
    df = pd.read_parquet(HERE / "complaints_clean.parquet")
    df["text_key"] = df["text_clean"].str.lower()

    # embed each distinct usable text once
    usable = df[df["usable_for_clustering"]]
    uniq = usable.drop_duplicates("text_key").reset_index(drop=True)
    texts = uniq["text_clean"].tolist()
    emb = cached_embeddings(texts)
    red = cached_umap(emb)
    labels = cluster(red)

    n_clusters = len(set(labels) - {-1})
    noise = labels == -1
    print(f"\nClusters found: {n_clusters}; noise before reassignment: {noise.mean():.1%}")

    # renumber clusters by size, 1 = largest
    sizes = pd.Series(labels[~noise]).value_counts()
    remap = {old: new for new, old in enumerate(sizes.index, start=1)}
    labels = np.array([remap.get(l, -1) for l in labels])

    # noise: join the nearest cluster if it is close enough, else stay unclustered
    ids, cents = centroids_of(emb, labels)
    sims = emb @ cents.T
    nearest = ids[sims.argmax(axis=1)]
    best = sims.max(axis=1)
    assignment = np.where(labels != -1, "core", "unclustered")
    reassign = (labels == -1) & (best >= REASSIGN_MIN_SIM)
    labels = np.where(reassign, nearest, labels)
    assignment[reassign] = "nearest"
    # clusters are numbered 1..n, so cluster c sits in column c - 1
    own_sim = np.where(labels > 0, sims[np.arange(len(labels)), (labels - 1).clip(min=0)], np.nan)

    uniq["cluster_id"] = labels
    uniq["assignment"] = assignment
    uniq["centroid_similarity"] = np.round(own_sim, 3)

    # map back to every complaint: duplicates share their text's cluster
    out = df[["complaint_id", "text_key", "usable_for_clustering"]].merge(
        uniq[["text_key", "cluster_id", "assignment", "centroid_similarity"]],
        on="text_key", how="left",
    )
    short = ~out["usable_for_clustering"]
    out.loc[short, "cluster_id"] = -2
    out.loc[short, "assignment"] = "too_short"
    out["cluster_id"] = out["cluster_id"].astype(int)
    out = out.drop(columns=["text_key", "usable_for_clustering"])

    # cluster table: size, keywords, the texts closest to each centroid
    core = uniq["assignment"].eq("core").to_numpy()
    kw = keywords_by_cluster(uniq.loc[core, "text_clean"], labels[core])
    counts = out["cluster_id"].value_counts()
    rows = []
    for k, cid in enumerate(ids):
        members = np.where(core & (labels == cid))[0]
        order = members[np.argsort(-sims[members, k])][:N_EXAMPLES]
        rows.append({
            "cluster_id": int(cid),
            "n_complaints": int(counts.get(cid, 0)),
            "n_core_texts": int(len(members)),
            "keywords": kw[cid],
            "example_ids": uniq.loc[order, "complaint_id"].tolist(),
            "example_texts": uniq.loc[order, "text_clean"].tolist(),
        })
    clusters = pd.DataFrame(rows).sort_values("n_complaints", ascending=False)

    out.to_parquet(HERE / "complaint_clusters.parquet", index=False)
    clusters.to_parquet(HERE / "clusters.parquet", index=False)

    # ---- report
    n = len(out)
    shares = out["assignment"].value_counts(normalize=True)
    print(f"Complaints: {n:,}")
    for a in ["core", "nearest", "unclustered", "too_short"]:
        print(f"  {a:<12} {shares.get(a, 0):.1%}")
    in_cluster = out["cluster_id"] > 0
    top10 = clusters["n_complaints"].head(10).sum() / in_cluster.sum()
    print(f"Largest cluster: {clusters['n_complaints'].iloc[0]:,} complaints "
          f"({clusters['n_complaints'].iloc[0] / n:.1%} of all)")
    print(f"Smallest cluster: {clusters['n_complaints'].iloc[-1]:,}")
    print(f"Top 10 clusters hold {top10:.0%} of clustered complaints")
    # quality check: do clusters form around the failure or the reporting channel?
    ch = out[in_cluster].merge(df[["complaint_id", "channel"]], on="complaint_id")
    hot = ch.groupby("cluster_id")["channel"].apply(lambda c: (c == "Hotline").mean())
    print(f"Channel check: hotline share overall {(ch['channel'] == 'Hotline').mean():.0%}; "
          f"clusters over 85% hotline: {(hot > 0.85).sum()}, under 5% hotline: {(hot < 0.05).sum()}")
    print("\nTop 15 clusters:")
    for _, r in clusters.head(15).iterrows():
        print(f"  #{r.cluster_id:<3} {r.n_complaints:>6,}  {r.keywords}")

    with open(HERE / "cluster_preview.txt", "w", encoding="utf-8") as f:
        for _, r in clusters.iterrows():
            f.write(f"\n#{r.cluster_id}  n={r.n_complaints:,}  [{r.keywords}]\n")
            for t in r.example_texts[:5]:
                f.write(f"   - {t[:220]}\n")
    print(f"\nSaved complaint_clusters.parquet, clusters.parquet ({len(clusters)} clusters), "
          f"cluster_preview.txt. Total {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
