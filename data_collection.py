"""
Step 1 — Data Collection
========================
Faithfulness Reasoning Evaluator project

Loads 5 sentiment datasets, samples a balanced subset from each,
normalises labels to {0: negative, 1: positive}, and saves everything
to a single CSV ready for Stage 2 (prompt engineering).

Datasets
--------
- stanfordnlp/sst2             → short movie review sentences (binary)
- stanfordnlp/imdb             → long movie reviews (binary)
- fancyzhx/amazon_polarity     → product reviews (binary)
- GooglePlay_2021-2023_reviews → app reviews (binary, from local xlsx)
- lmassaron/FinancialPhraseBank→ financial news (3-class, neutral kept as label=2)

Output
------
data/sentiment_samples.csv  with columns:
  sample_id | dataset | text | label | label_str | split
"""

import os
import random
import pandas as pd
from datasets import load_dataset
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLES_PER_CLASS  = 100
RANDOM_SEED        = 42
OUTPUT_DIR         = "data"
OUTPUT_FILE        = os.path.join(OUTPUT_DIR, "sentiment_samples.csv")
GOOGLE_PLAY_PATH   = "data/GooglePlay 2021-2023 reviews.xlsx"
# ─────────────────────────────────────────────────────────────────────────────

random.seed(RANDOM_SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

all_rows = []


def balanced_sample(rows: list[dict], label_key: str, n: int, seed: int) -> list[dict]:
    """Return up to n rows per unique label value, randomly sampled."""
    buckets: dict = defaultdict(list)
    for r in rows:
        buckets[r[label_key]].append(r)
    rng = random.Random(seed)
    out = []
    for lbl, items in buckets.items():
        rng.shuffle(items)
        out.extend(items[:n])
    return out


# ── 1. SST-2 ──────────────────────────────────────────────────────────────────
print("Loading SST-2 …")
sst2 = load_dataset("stanfordnlp/sst2", split="train")

sst2_rows = [
    {"text": r["sentence"].strip(), "label": r["label"]}
    for r in sst2
    if r["label"] in (0, 1) and len(r["sentence"].strip()) > 10
]

for r in balanced_sample(sst2_rows, "label", SAMPLES_PER_CLASS, RANDOM_SEED):
    all_rows.append({
        "dataset":   "sst2",
        "text":      r["text"],
        "label":     r["label"],
        "label_str": "positive" if r["label"] == 1 else "negative",
        "split":     "train",
    })

print(f"  → {sum(1 for r in all_rows if r['dataset']=='sst2')} rows")


# ── 2. IMDb ───────────────────────────────────────────────────────────────────
print("Loading IMDb …")
imdb = load_dataset("stanfordnlp/imdb", split="train")

imdb_rows = [
    {"text": r["text"].strip()[:1500], "label": r["label"]}
    for r in imdb
    if r["label"] in (0, 1) and len(r["text"].strip()) > 40
]

for r in balanced_sample(imdb_rows, "label", SAMPLES_PER_CLASS, RANDOM_SEED):
    all_rows.append({
        "dataset":   "imdb",
        "text":      r["text"],
        "label":     r["label"],
        "label_str": "positive" if r["label"] == 1 else "negative",
        "split":     "train",
    })

print(f"  → {sum(1 for r in all_rows if r['dataset']=='imdb')} rows")


# ── 3. Amazon Polarity ────────────────────────────────────────────────────────
print("Loading Amazon Polarity …")
amazon = load_dataset("fancyzhx/amazon_polarity", split="train")

amazon_rows = [
    {
        "text":  (r["title"] + " " + r["content"]).strip()[:1500],
        "label": r["label"],
    }
    for r in amazon
    if r["label"] in (0, 1) and len(r["content"].strip()) > 20
]

for r in balanced_sample(amazon_rows, "label", SAMPLES_PER_CLASS, RANDOM_SEED):
    all_rows.append({
        "dataset":   "amazon",
        "text":      r["text"],
        "label":     r["label"],
        "label_str": "positive" if r["label"] == 1 else "negative",
        "split":     "train",
    })

print(f"  → {sum(1 for r in all_rows if r['dataset']=='amazon')} rows")


# ── 4. Google Play ────────────────────────────────────────────────────────────
print("Loading Google Play reviews …")

if not os.path.exists(GOOGLE_PLAY_PATH):
    raise FileNotFoundError(
        f"Google Play file not found at '{GOOGLE_PLAY_PATH}'.\n"
        "Please copy GooglePlay_2021-2023_reviews.xlsx into your data/ folder."
    )

gp_df = pd.read_excel(GOOGLE_PLAY_PATH)

# Drop 3-star reviews (ambiguous), map 1-2 → negative, 4-5 → positive
gp_df = gp_df[gp_df["score"].isin([1, 2, 4, 5])].copy()
gp_df["label"] = gp_df["score"].apply(lambda x: 0 if x in [1, 2] else 1)
gp_df = gp_df[gp_df["text"].notna() & (gp_df["text"].str.len() > 10)]

gp_rows = [
    {"text": str(row["text"]).strip()[:1500], "label": int(row["label"])}
    for _, row in gp_df.iterrows()
]

for r in balanced_sample(gp_rows, "label", SAMPLES_PER_CLASS, RANDOM_SEED):
    all_rows.append({
        "dataset":   "googleplay",
        "text":      r["text"],
        "label":     r["label"],
        "label_str": "positive" if r["label"] == 1 else "negative",
        "split":     "train",
    })

print(f"  → {sum(1 for r in all_rows if r['dataset']=='googleplay')} rows")


# ── 5. FinancialPhraseBank ────────────────────────────────────────────────────
print("Loading FinancialPhraseBank …")
fpb = load_dataset("lmassaron/FinancialPhraseBank", split="train")

# Labels are integers: 0=negative, 1=positive, 2=neutral
FPB_LABEL_MAP = {0: "negative", 1: "positive", 2: "neutral"}

fpb_rows = [
    {"text": r["sentence"].strip(), "label": r["label"]}
    for r in fpb
    if len(r["sentence"].strip()) > 10 and r["label"] in FPB_LABEL_MAP
]

for r in balanced_sample(fpb_rows, "label", SAMPLES_PER_CLASS, RANDOM_SEED):
    lbl = r["label"]
    all_rows.append({
        "dataset":   "financial",
        "text":      r["text"],
        "label":     lbl,
        "label_str": FPB_LABEL_MAP[lbl],
        "split":     "train",
    })

print(f"  → {sum(1 for r in all_rows if r['dataset']=='financial')} rows")


# ── Assemble & save ───────────────────────────────────────────────────────────
df = pd.DataFrame(all_rows)
df.insert(0, "sample_id", [f"{r['dataset']}_{i:04d}" for i, r in df.iterrows()])

df.to_csv(OUTPUT_FILE, index=False)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n── Dataset summary ──────────────────────────────────────────────")
print(df.groupby(["dataset", "label_str"]).size().rename("count").to_string())
print(f"\nTotal rows : {len(df)}")
print(f"Saved to   : {OUTPUT_FILE}")
print("\nSample rows:")
print(df.sample(5, random_state=RANDOM_SEED)[["sample_id", "dataset", "label_str", "text"]].to_string(max_colwidth=80))