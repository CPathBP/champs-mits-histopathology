"""Train the stain classifier on mean-pooled Virchow2 slide embeddings.

Multinomial logistic regression on standardised mean-pooled tile features.
Labels come from the slide name: the stain token when present and ``he``
when the name carries none. Stains with fewer than ``--min-cases`` cases
are folded into ``other``; H&E is capped. Slides are grouped by case for
the out-of-fold report.
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from champs_pipeline.data_prep.feature_index import load_feature_index
from champs_pipeline.data_prep.stain_classifier import (
    ENCODER, STAIN_TYPES, load_embeddings, save, virchow2_files,
)
from champs_pipeline.data_prep.stains import detect_stain


def stain_label(slide_id):
    token = detect_stain(slide_id)
    if token is None:
        return "he"
    if token.lower() in STAIN_TYPES:
        return token.lower()
    return "other"


def new_model():
    return LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")


def select_training_slides(files, args):
    """Label the slides, fold rare stains into ``other``, and cap H&E."""
    df = files[files["champs_deid"].notna()].copy()
    df["label"] = df["slide_id"].map(stain_label)
    cases_per_label = df.groupby("label")["champs_deid"].nunique()
    supported = cases_per_label[cases_per_label >= args.min_cases].index
    df["label"] = df["label"].where(df["label"].isin(supported), "other")

    he = df[df["label"] == "he"].sample(frac=1.0, random_state=args.seed)
    he = he.groupby("champs_deid", group_keys=False).head(args.he_per_case)
    if len(he) > args.he_cap:
        he = he.sample(n=args.he_cap, random_state=args.seed)
    return pd.concat([he, df[df["label"] != "he"]], ignore_index=True)


def out_of_fold_predictions(X, y, groups, n_splits, seed):
    """Predicted class and its probability for every slide, from a grouped cross-validation."""
    predicted = np.empty(len(y), dtype=object)
    confidence = np.zeros(len(y))
    folds = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train, test in folds.split(X, y, groups):
        pipeline = make_pipeline(StandardScaler(), new_model()).fit(X[train], y[train])
        proba = pipeline.predict_proba(X[test])
        predicted[test] = pipeline.classes_[proba.argmax(axis=1)]
        confidence[test] = proba.max(axis=1)
    return predicted, confidence


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", required=True, help="Feature index with a champs_deid column.")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pred-csv", required=True)
    ap.add_argument("--min-cases", type=int, default=10)
    ap.add_argument("--he-cap", type=int, default=5000)
    ap.add_argument("--he-per-case", type=int, default=2)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    files = virchow2_files(load_feature_index(args.index))
    df = select_training_slides(files, args)
    embeddings = load_embeddings(dict(zip(df["slide_id"], df["path"])), args.cache, args.workers)
    df = df[df["slide_id"].isin(embeddings)]
    X = np.stack([embeddings[s] for s in df["slide_id"]]).astype(np.float64)
    y = df["label"].to_numpy()
    groups = df["champs_deid"].to_numpy()
    classes = sorted(set(y))

    fewest_cases = int(df.groupby("label")["champs_deid"].nunique().min())
    n_splits = max(2, min(args.n_splits, fewest_cases))
    predicted, confidence = out_of_fold_predictions(X, y, groups, n_splits, args.seed)
    print(classification_report(y, predicted, labels=classes, zero_division=0))
    balanced_accuracy = balanced_accuracy_score(y, predicted)
    cm = confusion_matrix(y, predicted, labels=classes)
    recall = {}
    for i, stain in enumerate(classes):
        recall[stain] = float(cm[i, i] / cm[i].sum()) if cm[i].sum() else None

    os.makedirs(os.path.dirname(os.path.abspath(args.pred_csv)), exist_ok=True)
    pd.DataFrame({
        "slide_id": df["slide_id"].values,
        "champs_deid": groups,
        "true_stain": y,
        "pred_stain": predicted,
        "confidence": confidence,
    }).to_csv(args.pred_csv, index=False)

    scaler = StandardScaler().fit(X)
    model = new_model().fit(scaler.transform(X), y)
    save({
        "scaler": scaler,
        "clf": model,
        "classes": list(model.classes_),
        "feature_dim": X.shape[1],
        "encoder": ENCODER,
        "n_train": len(y),
        "min_cases": args.min_cases,
        "he_cap": args.he_cap,
        "cv": {
            "n_splits": n_splits,
            "balanced_accuracy": float(balanced_accuracy),
            "per_class_recall": recall,
        },
    }, args.out)
    print(f"balanced accuracy (out of fold, {n_splits} folds): {balanced_accuracy:.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
