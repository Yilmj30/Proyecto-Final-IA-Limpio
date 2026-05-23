"""
src/models/predict.py
Carga el modelo entrenado y expone funciones de predicción.
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

CKPT = Path(__file__).parents[2] / "models" / "checkpoints" / "lightgbm_dropout_model.pkl"


def load_model():
    """Carga el modelo LightGBM serializado."""
    return joblib.load(CKPT)


def predict(X: pd.DataFrame, model=None):
    """Retorna etiquetas binarias (0/1)."""
    if model is None:
        model = load_model()
    return model.predict(X)


def predict_proba(X: pd.DataFrame, model=None) -> np.ndarray:
    """Retorna probabilidad de deserción (clase 1) para cada muestra."""
    if model is None:
        model = load_model()
    return model.predict_proba(X)[:, 1]
