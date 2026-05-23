"""
src/evaluation/metrics.py
Funciones de evaluación del modelo predictivo y del componente LLM.
"""
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
)


def compute_clf_metrics(y_true, y_pred, y_proba) -> dict:
    """Calcula el conjunto completo de métricas de clasificación."""
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred),         4),
        "precision": round(precision_score(y_true, y_pred),        4),
        "recall":    round(recall_score(y_true, y_pred),           4),
        "f1":        round(f1_score(y_true, y_pred),               4),
        "roc_auc":   round(roc_auc_score(y_true, y_proba),         4),
    }


def compare_models(metrics_csv_path: str) -> pd.DataFrame:
    """
    Carga el CSV de métricas de validación generado en 03_modeling
    y retorna un DataFrame ordenado por F1 descendente.
    """
    df = pd.read_csv(metrics_csv_path)
    return df.sort_values("f1", ascending=False).reset_index(drop=True)


def print_report(y_true, y_pred):
    print(classification_report(y_true, y_pred,
                                 target_names=["No Deserción", "Deserción"]))
