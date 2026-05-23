# Guía de Usuario — Sistema de Predicción de Deserción Estudiantil

## ¿Qué hace este sistema?

Predice el riesgo de deserción de un estudiante universitario a partir de
variables académicas, demográficas y financieras, y genera una explicación
en lenguaje natural dirigida a consejeros universitarios.

---

## Requisitos

- Python 3.10 o superior
- API key gratuita de Groq: [https://console.groq.com](https://console.groq.com)

---

## Instalación

```bash
git clone https://github.com/usuario/proyecto-ia-eafit.git
cd proyecto-ia-eafit
python -m venv venv

# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

---

## Configurar la API key de Groq

```powershell
# Windows PowerShell
$env:GROQ_API_KEY = "gsk_TU_KEY_AQUI"

# Linux / macOS
export GROQ_API_KEY="gsk_TU_KEY_AQUI"
```

---

## Orden de ejecución de notebooks

```
jupyter notebook
```

| # | Notebook | Qué hace | Produce |
|---|----------|----------|---------|
| 1 | `01_eda.ipynb` | Exploración y análisis de datos | Figuras fig1–fig7 |
| 2 | `02_preprocessing.ipynb` | Limpieza, encoding, splits | `data/processed/` |
| 3 | `03_modeling.ipynb` | Baseline + LightGBM + tuning | `models/checkpoints/lightgbm_dropout_model.pkl` |
| 4 | `04_llm_rag_agents.ipynb` | SHAP + LLM (Groq) → explicaciones | `models/llm_eval_results.json` |

> **Nota:** el archivo `lightgbm_dropout_model.pkl` está en `.gitignore` por su tamaño.
> Se regenera ejecutando `03_modeling.ipynb`.

---

## Usar el módulo `src` directamente

```python
from src.data.loader import load_splits, load_feature_names
from src.models.predict import load_model, predict_proba
from src.agents.explainer import DropoutExplainer

# Cargar datos y modelo
X_train, X_val, X_test, y_train, y_val, y_test = load_splits()
feature_names = load_feature_names()
model = load_model()

# Predecir riesgo
probas = predict_proba(X_test, model)

# Explicar un estudiante
explainer = DropoutExplainer(model, feature_names)
explanation = explainer.explain(X_test.iloc[[0]], proba=probas[0])
print(explanation)
```

---

## Estructura del repositorio

```
proyecto-ia-eafit/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/           ← dataset.csv original + FUENTE.md
│   └── processed/     ← splits, figuras
├── docs/
│   ├── informe_final.pdf
│   └── guia_usuario.md   ← este archivo
├── models/
│   ├── checkpoints/   ← lightgbm_dropout_model.pkl (generado localmente)
│   └── *.json / *.csv ← artefactos de validación y tuning
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_llm_rag_agents.ipynb
└── src/
    ├── data/          ← loader.py
    ├── models/        ← predict.py
    ├── agents/        ← explainer.py (SHAP + Groq)
    └── evaluation/    ← metrics.py
```
