"""
src/agents/explainer.py
Agente de explicabilidad: SHAP + LLM (Groq) -> lenguaje natural.
"""
import os
import json
import numpy as np
import shap
from groq import Groq

LLM_MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = """Eres un orientador académico de una universidad colombiana.
Tu función es explicar, en lenguaje claro y empático, por qué un estudiante
podría estar en riesgo de deserción, basándote en los factores más influyentes
identificados por un modelo predictivo.

Reglas:
- Dirige la explicación a un consejero universitario (no a un experto en ML).
- Menciona los 3 factores más importantes por nombre, explica su efecto y da
  una recomendación concreta para cada uno.
- Tono: profesional, empático, nunca alarmista.
- Longitud: máximo 180 palabras.
- Responde siempre en español."""


class DropoutExplainer:
    """
    Combina SHAP (explicaciones locales del modelo LightGBM) con un LLM
    para generar explicaciones en lenguaje natural sobre el riesgo de deserción.
    """

    def __init__(self, model, feature_names: list[str], api_key: str | None = None):
        self.model         = model
        self.feature_names = feature_names
        self.explainer     = shap.TreeExplainer(model)
        self.client        = Groq(api_key=api_key or os.environ["GROQ_API_KEY"])

    def _shap_profile(self, X_row, top_n: int = 3) -> list[dict]:
        """Calcula SHAP para una fila y retorna los top_n features."""
        import pandas as pd
        sv = self.explainer.shap_values(X_row)
        sv = sv[1] if isinstance(sv, list) else sv
        sv = sv.flatten()
        top_idx = np.argsort(np.abs(sv))[::-1][:top_n]
        return [
            {
                "feature":   self.feature_names[i],
                "shap":      round(float(sv[i]), 4),
                "direction": "↑ riesgo" if sv[i] > 0 else "↓ riesgo",
            }
            for i in top_idx
        ]

    def _build_prompt(self, proba: float, features: list[dict]) -> str:
        nivel = "alto" if proba >= 0.6 else ("moderado" if proba >= 0.4 else "bajo")
        lines = [
            f"Perfil del estudiante:",
            f"- Probabilidad de deserción: {proba:.2f} (riesgo {nivel})",
            "- Factores principales:",
        ]
        for i, f in enumerate(features, 1):
            lines.append(f"  {i}. {f['feature']}: {f['direction']}")
        return "\n".join(lines)

    def explain(self, X_row, proba: float) -> str:
        """
        Genera una explicación en lenguaje natural para un estudiante.

        Parameters
        ----------
        X_row : array-like (1, n_features)
        proba : float — probabilidad de deserción predicha por el modelo

        Returns
        -------
        str — explicación en español
        """
        features = self._shap_profile(X_row)
        prompt   = self._build_prompt(proba, features)

        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.4,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
