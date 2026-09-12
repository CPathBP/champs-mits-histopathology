"""Score every slide with the stain classifier.

Mean-pools each slide's Virchow2 feature file and writes the class
probabilities; ``p_non_HE`` is the value the cohort construction thresholds.
Mean-pooled embeddings are cached in ``--cache``.
"""

import argparse
import os

import numpy as np
import pandas as pd

from champs_pipeline.data_prep.feature_index import load_feature_index
from champs_pipeline.data_prep.stain_classifier import (
    StainClassifier, load_embeddings, virchow2_files,
)


def score(classifier, embedding):
    """Probabilities of one slide, as a row of the output table."""
    x = embedding.reshape(1, -1).astype(np.float64)
    proba = classifier.clf.predict_proba(classifier.scaler.transform(x))[0]
    classes = list(classifier.clf.classes_)
    row = {
        "pred_stain": classes[int(np.argmax(proba))],
        "confidence": float(proba.max()),
        "p_non_HE": 1.0 - float(proba[classes.index("he")]),
    }
    for stain, p in zip(classes, proba):
        row[f"p_{stain}"] = float(p)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    files = virchow2_files(load_feature_index(args.index))
    paths = dict(zip(files["slide_id"], files["path"]))
    embeddings = load_embeddings(paths, args.cache, args.workers)
    classifier = StainClassifier.load(args.model)

    rows = []
    for slide_id in sorted(paths):
        if slide_id in embeddings:
            rows.append({"slide_id": slide_id, **score(classifier, embeddings[slide_id])})
    scores = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    scores.to_csv(args.out, index=False)
    print(f"{len(scores)} slides scored; p_non_HE >= 0.5: {int((scores['p_non_HE'] >= 0.5).sum())}")


if __name__ == "__main__":
    main()
