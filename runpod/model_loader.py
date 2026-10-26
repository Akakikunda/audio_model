"""Trusted joblib loader for the notebook-defined custom wrapper."""
import __main__
import joblib
import numpy as np
from pathlib import Path

class MultiBinaryXGB:
    def __init__(self, estimators, classes):
        self.estimators_ = estimators
        self.classes_ = classes
    def predict_proba(self, X):
        return np.array([e.predict_proba(X)[:, 1] for e in self.estimators_]).T
    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)


def load_model(path):
    # Notebook saved this class under __main__; register before unpickling.
    __main__.MultiBinaryXGB = MultiBinaryXGB
    model = joblib.load(Path(path))  # Load trusted artifacts ONLY.
    if not isinstance(model, MultiBinaryXGB) or len(model.estimators_) != 6:
        raise ValueError("Unexpected model structure")
    return model
