"""
src/data/loader.py
Utilidades de carga y acceso a los datos procesados.
"""
from pathlib import Path
import pandas as pd

PROCESSED = Path(__file__).parents[2] / "data" / "processed"
RAW       = Path(__file__).parents[2] / "data" / "raw"


def load_splits():
    """Carga los 6 splits procesados (X/y × train/val/test)."""
    X_train = pd.read_csv(PROCESSED / "X_train.csv")
    X_val   = pd.read_csv(PROCESSED / "X_val.csv")
    X_test  = pd.read_csv(PROCESSED / "X_test.csv")
    y_train = pd.read_csv(PROCESSED / "y_train.csv").squeeze()
    y_val   = pd.read_csv(PROCESSED / "y_val.csv").squeeze()
    y_test  = pd.read_csv(PROCESSED / "y_test.csv").squeeze()
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_feature_names() -> list[str]:
    """Retorna la lista de nombres de features en el mismo orden que las matrices."""
    return pd.read_csv(PROCESSED / "feature_names.csv").iloc[:, 0].tolist()


def load_raw() -> pd.DataFrame:
    """Carga el dataset crudo original."""
    return pd.read_csv(RAW / "dataset.csv")
