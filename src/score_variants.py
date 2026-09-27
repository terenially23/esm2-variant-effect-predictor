"""Scores each variant in the fetched ClinVar table with ESM2's zero-shot
masked-marginal method and evaluates AUROC against the pathogenic/benign labels."""

import argparse

import pandas as pd
from sklearn.metrics import roc_auc_score

from scoring import score_variant


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/variants.csv")
    parser.add_argument("--output", default="data/variant_scores.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    scores = []
    for _, row in df.iterrows():
        score = score_variant(row["sequence"], row["pos"], row["wt"], row["mt"])
        scores.append(score)
        print(f"{row['gene']} {row['wt']}{row['pos']}{row['mt']} ({row['label']}): {score:+.3f}")

    df["esm2_score"] = scores
    y_true = (df["label"] == "Pathogenic").astype(int)
    auc = roc_auc_score(y_true, -df["esm2_score"])  # more negative score -> more pathogenic
    print(f"\nESM2 zero-shot AUROC (pathogenic vs benign): {auc:.3f}")

    df.drop(columns=["sequence"]).to_csv(args.output, index=False)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
