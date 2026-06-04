"""
Step 0 — Baseline
=================
Faithfulness Reasoning Evaluator project

Runs VADER (a rule-based lexicon sentiment tool) on all samples in
data/sentiment_samples.csv and records per-dataset accuracy.

This establishes two things:
  1. A non-LLM performance floor to compare against
  2. A minimum accuracy threshold the LLM must exceed before
     faithfulness probing is meaningful

VADER needs no API key and runs locally in seconds.

Output
------
results/baseline_vader.csv     — per-sample predictions
results/baseline_summary.csv   — per-dataset accuracy table
"""

import os
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_FILE   = "data/sentiment_samples.csv"
OUTPUT_DIR   = "results"
SAMPLE_OUT   = os.path.join(OUTPUT_DIR, "baseline_vader.csv")
SUMMARY_OUT  = os.path.join(OUTPUT_DIR, "baseline_summary.csv")

# Minimum accuracy we expect Claude to beat in Stage 3
# (used as reference thresholds in the paper)
LLM_THRESHOLDS = {
    "sst2":       0.90,
    "imdb":       0.85,
    "amazon":     0.85,
    "googleplay": 0.80,
    "financial":  0.75,
}
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_FILE)

# For baseline we only evaluate binary rows (drop neutral financial rows)
df_binary = df[df["label"].isin([0, 1])].copy()
print(f"Loaded {len(df)} total rows, {len(df_binary)} binary rows for evaluation\n")

# ── Run VADER ─────────────────────────────────────────────────────────────────
analyzer = SentimentIntensityAnalyzer()

def vader_predict(text: str) -> int:
    """
    Returns 1 (positive) if compound score >= 0.05
    Returns 0 (negative) if compound score <= -0.05
    Returns -1 (neutral / uncertain) otherwise
    """
    score = analyzer.polarity_scores(str(text))["compound"]
    if score >= 0.05:
        return 1
    elif score <= -0.05:
        return 0
    else:
        return -1   # VADER uncertain — counts as wrong in accuracy calc

print("Running VADER on all samples …")
df_binary["vader_label"]    = df_binary["text"].apply(vader_predict)
df_binary["vader_correct"]  = (df_binary["vader_label"] == df_binary["label"]).astype(int)
df_binary["vader_label_str"] = df_binary["vader_label"].map({1: "positive", 0: "negative", -1: "neutral/uncertain"})

# ── Per-dataset summary ───────────────────────────────────────────────────────
rows = []
for dataset, group in df_binary.groupby("dataset"):
    total     = len(group)
    correct   = group["vader_correct"].sum()
    accuracy  = correct / total
    threshold = LLM_THRESHOLDS.get(dataset, 0.80)
    neutral_pct = (group["vader_label"] == -1).sum() / total

    rows.append({
        "dataset":            dataset,
        "total_samples":      total,
        "vader_correct":      int(correct),
        "vader_accuracy":     round(accuracy, 4),
        "vader_neutral_pct":  round(neutral_pct, 4),
        "llm_threshold":      threshold,
        "vader_beats_threshold": accuracy >= threshold,
    })

summary = pd.DataFrame(rows).sort_values("dataset")

# ── Save outputs ──────────────────────────────────────────────────────────────
df_binary.to_csv(SAMPLE_OUT, index=False)
summary.to_csv(SUMMARY_OUT, index=False)

# ── Print report ──────────────────────────────────────────────────────────────
print("\n── VADER Baseline Results ───────────────────────────────────────")
print(f"{'Dataset':<14} {'Accuracy':>10} {'VADER Neutral%':>15} {'LLM Threshold':>15} {'VADER Beats?':>13}")
print("─" * 72)
for _, r in summary.iterrows():
    beats = "✓" if r["vader_beats_threshold"] else "✗"
    print(
        f"{r['dataset']:<14} "
        f"{r['vader_accuracy']:>9.1%} "
        f"{r['vader_neutral_pct']:>14.1%} "
        f"{r['llm_threshold']:>14.0%} "
        f"{beats:>13}"
    )

overall_acc = df_binary["vader_correct"].sum() / len(df_binary)
print("─" * 72)
print(f"{'Overall':<14} {overall_acc:>9.1%}")

print(f"\nPer-sample predictions → {SAMPLE_OUT}")
print(f"Summary table          → {SUMMARY_OUT}")

print("""
── What this means  ───────────────────────────────
VADER is a rule-based lexicon tool with no language understanding.
It cannot explain its reasoning, follow instructions, or generalise
to new domains. LLM should comfortably exceed these numbers.

If llm Stage 3 accuracy is close to VADER's, that is itself
a notable finding — the LLM is not adding meaningful value over
a simple lookup table, which makes any faithfulness claims weaker.

The gap between LLM accuracy and VADER's accuracy is the
justification for why we trust LLM's reasoning chains enough
to probe them for faithfulness.
""")