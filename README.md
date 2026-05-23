# Proyecto-Final---IA-

# Predicción de Deserción Estudiantil

> **¿Puede un clasificador LightGBM predecir el riesgo de deserción estudiantil superando un baseline lineal en F1, y pueden sus predicciones complementarse con explicaciones en lenguaje natural generadas por un LLM?**

---

## Integrantes

| Nombre| Correo                  | Contribución                                |
|-----------------------------------|-------------------------|---------------------------------------------|
| Helen  Valentina Sanabria Guevara | hvsanabrig@eafit.edu.co | EDA, preprocesamiento, visualizaciones      |
| Juan Esteban Orrego Gomez         | jeorregog1@eafit.edu.co | Baseline, LightGBM, tuning, secciones 3 y 4 |
| Jeronimo Rodriguez Restrepo       | jrodrigur3@eafit.edu.co | LLM/SHAP/Groq, evaluación, repo, informe    |   

---

## Resultados principales

| Modelo | Accuracy | F1 | AUC-ROC | Conjunto |
|------------------------------|----------|----|---------|----------|
| Dummy (prior) | 0.678 | 0.000 | 0.500 | val |
| LightGBM base | 0.861 | 0.779 | 0.922 | val |
| LogReg balanceada (baseline) | 0.871 | 0.805 | 0.926 | val |
| **LightGBM tuned** | **0.877** | **0.806** | **0.924** | val |
| **LightGBM tuned** | **0.874** | **0.804** | **0.929** | **test** |

---

## Dataset

- **Fuente:** [Predict Students' Dropout and Academic Success — Kaggle](https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention)
- **Registros:** 4 424 estudiantes · **Features:** 35 variables
- **Variable objetivo:** riesgo de deserción (binaria)
- **Licencia:** CC BY 4.0

---

## Instalación

```bash
git clone https://github.com/Valesague52/Proyecto-Final---IA-.git
cd Proyecto-Final---IA-
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### API key de Groq (requerida para `04_llm_rag_agents.ipynb`)

```powershell
$env:GROQ_API_KEY = "gsk_TU_KEY_AQUI"
```

Obtener key gratuita en [console.groq.com](https://console.groq.com) — sin tarjeta.

---

## Orden de ejecución

```bash
jupyter notebook
```

| # | Notebook | Produce |
|---|----------|---------|
| 1 | `01_eda.ipynb` | Figuras EDA (fig1–fig7) |
| 2 | `02_preprocessing.ipynb` | Splits en `data/processed/` |
| 3 | `03_modeling.ipynb` | `models/checkpoints/lightgbm_dropout_model.pkl` |
| 4 | `04_llm_rag_agents.ipynb` | Explicaciones LLM + `models/llm_eval_results.json` |

> `lightgbm_dropout_model.pkl` está en `.gitignore`. Se regenera ejecutando `03_modeling.ipynb`.

---

## Estructura del repositorio

    Proyecto-Final---IA-/
    ├── README.md
    ├── requirements.txt
    ├── data/
    │   ├── raw/                  ← FUENTE.md
    │   └── processed/            ← splits + figuras
    ├── docs/
    |   ├── Informe-Proyecto Final IA
    │   └── guia_usuario.md
    ├── models/
    │   ├── checkpoints/          ← lightgbm_dropout_model.pkl
    │   ├── lightgbm_best_params.json
    │   ├── metrics_validation_candidates.csv
    │   ├── modeling_handoff_summary.json
    │   └── test_metrics_final.json
    ├── notebooks/
    │   ├── 01_eda.ipynb
    │   ├── 02_preprocessing.ipynb
    │   ├── 03_modeling.ipynb
    │   └── 04_llm_rag_agents.ipynb
    └── src/
        ├── data/loader.py
        ├── models/predict.py
        ├── agents/explainer.py
        └── evaluation/metrics.py


---

## Video demo

https://www.youtube.com/watch?v=aOkuu5GL9f8

---

## Informe

📄 `Informe-Proyecto Final IA` · 
