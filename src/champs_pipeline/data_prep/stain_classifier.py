"""Stain classifier over mean-pooled slide embeddings.

A slide's stain is predicted from the mean of its tile features (Virchow2,
2560 dimensions): StandardScaler, then multinomial LogisticRegression. The
trainer is ``scripts/slides/train_stain_classifier.py``; the artifact is a
pickled dict ``{scaler, clf, classes, feature_dim, encoder, n_train,
min_cases, he_cap, cv}``.
"""

import pickle
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np

STAIN_TYPES = ("he", "ihc", "gram", "grocott", "trichrome", "gms", "pas", "afb", "iron", "zn", "ws",
               "verhoeff", "mucicarmine", "hall", "fontana", "fm", "other")
ENCODER = "virchow2"
RESOLUTION = "20x_224px_0px_overlap"


def meanpool_h5(path):
    """Mean of the ``features`` dataset, or None for an empty bag or a non-finite mean."""
    with h5py.File(path, "r") as f:
        x = f["features"][:]
    if x.size == 0:
        return None
    m = x.astype(np.float64).mean(axis=0).astype(np.float32)
    return m if np.all(np.isfinite(m)) else None


class StainClassifier:
    def __init__(self, payload):
        self.scaler, self.clf = payload["scaler"], payload["clf"]
        self.classes = list(payload["classes"])
        self.feature_dim = int(payload["feature_dim"])
        self.payload = payload

    def predict_from_features(self, x):
        """``(stain, confidence)`` from one mean-pooled embedding."""
        x = np.asarray(x, dtype=np.float64).reshape(1, -1)
        if x.shape[1] != self.feature_dim:
            raise ValueError(f"feature dim {x.shape[1]} != {self.feature_dim}")
        proba = self.clf.predict_proba(self.scaler.transform(x))[0]
        i = int(np.argmax(proba))
        return str(self.clf.classes_[i]), float(proba[i])

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return cls(pickle.load(f))


def save(payload, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def virchow2_files(index):
    from champs_pipeline.data_prep.feature_index import usable_files

    return usable_files(index, ENCODER, RESOLUTION)


def _meanpool(item):
    slide_id, path = item
    try:
        return slide_id, meanpool_h5(path)
    except Exception:  # noqa: BLE001
        return slide_id, None


def load_embeddings(slide_paths, cache_file, workers):
    """Mean-pooled embedding per slide, cached in ``cache_file`` (.npz) across runs."""
    cache_file = Path(cache_file)
    emb = {}
    if cache_file.exists():
        z = np.load(cache_file, allow_pickle=True)
        emb = {str(i): v for i, v in zip(z["slide_ids"], z["embs"])}
    todo = [(s, p) for s, p in slide_paths.items() if s not in emb]
    if todo:
        with Pool(workers) as pool:
            for s, v in pool.imap_unordered(_meanpool, todo, chunksize=8):
                if v is not None:
                    emb[s] = np.asarray(v, dtype=np.float32)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        ids = np.array(list(emb))
        np.savez(cache_file, slide_ids=ids, embs=np.stack([emb[i] for i in ids]))
    return emb
