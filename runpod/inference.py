"""Classify audio using exact training-time 38-feature extraction."""
import json
from pathlib import Path
import librosa
import numpy as np
import pandas as pd
from feature_extraction import extract_tabular
from model_loader import load_model

BASE = Path(__file__).resolve().parent.parent
CONFIG = json.loads((BASE / "model/config.json").read_text())
FEATURES = json.loads((BASE / "model/feature_columns.json").read_text())
CLASSES = CONFIG["classes"]
VISIBLE = CONFIG["deployed_classes"]
SR = CONFIG["sample_rate"]
CLIP_LEN = int(SR * CONFIG["clip_duration"])
MODEL = None

def classify_file(path):
    global MODEL
    if MODEL is None:
        MODEL = load_model(BASE / "model/xgboost_model.pkl")
        if list(MODEL.classes_) != CLASSES:
            raise ValueError("Model class order mismatch")
    y, sr = librosa.load(str(path), sr=SR, mono=True)
    if len(y) == 0 or not np.isfinite(y).all():
        raise ValueError("Invalid or empty audio")
    n = (len(y) + CLIP_LEN - 1) // CLIP_LEN
    rows = []
    for i in range(n):
        clip = y[i * CLIP_LEN:(i + 1) * CLIP_LEN]
        if len(clip) < CLIP_LEN:
            clip = np.pad(clip, (0, CLIP_LEN - len(clip)))
        rows.append(extract_tabular(clip, sr))
    X = pd.DataFrame(rows)[FEATURES].to_numpy()
    if X.shape[1] != 38 or not np.isfinite(X).all():
        raise ValueError("Feature extraction failed")
    probabilities = np.asarray(MODEL.predict_proba(X))
    if probabilities.shape != (n, len(CLASSES)):
        raise ValueError("Unexpected model output shape")
    flags = probabilities > CONFIG["threshold"]
    percentages = {c: round(float(100 * flags[:, CLASSES.index(c)].mean()), 2) for c in VISIBLE}
    timeline = [{"clip": i + 1, "start_seconds": i * CONFIG["clip_duration"],
                 "end_seconds": round(min((i + 1) * CONFIG["clip_duration"], len(y)/sr), 3),
                 "predictions": {c: bool(flags[i, CLASSES.index(c)]) for c in VISIBLE}}
                for i in range(n)]
    return {"duration_seconds": round(len(y)/sr, 3), "clips": n,
            "percentages": percentages, "timeline": timeline,
            "beta_warning": "quacking_queen_bee: Beta - may have false positives",
            "excluded_from_display": CONFIG["excluded_classes"]}
