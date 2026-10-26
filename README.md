# Enterprise AI Data Analyst Suite PRO
An automated BI dashboard, statistical inference lab and ML pipeline with an **LLM narrator**,
an **image quality & vision auditor**, and a **pipeline audit trail**.

## Run it (Windows, Python 3.11 recommended)
```powershell
py -3.11 -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
venv\Scripts\python.exe -m streamlit run app.py
```
Demo data: `sample_data.csv` (tabular) and `sample_images/` (images).

## The three workspace modes
| Mode | What it does |
|---|---|
| **Analytical Engine Core** (CSV) | Overview, ETL studio, 15 BI visuals, statistics lab, ML lab, and the **AI Dashboard PDF Publisher** (LLM narrated, 8 pages) |
| **Multimodal CV Matrix Auditor** (images) | Multi-image upload; exposure, blur, contrast, clipping, entropy, edge density, colourfulness; dominant colours; RGB histogram; edge map; 0-100 quality score; **AI vision description**; **PDF audit report** |
| **Enterprise Pipeline Manifest Logs** | **System health** (uptime, memory, environment, integrations), **data lineage** (raw → ETL → filtered, per-column change report), **activity timeline** (audit trail with charts and CSV/JSON export), **session inspector**, **run manifest JSON** |

## Architecture
```
app.py ── Streamlit UI and routing
 ├─ etl_pipeline.py      cleaning, imputation, scaling, outlier engines
 ├─ bi_engine.py         15 native BI visuals
 ├─ insights.py          Shapiro / ANOVA / chi-square statistics lab
 ├─ automl.py            regression + classification lab
 ├─ dashboard_report.py  facts engine, charts, 8-page dashboard PDF
 ├─ llm_narrator.py      Claude / OpenAI / Groq / Gemini client + offline fallback
 ├─ cv_auditor.py        image analysis, vision LLM, image-audit PDF
 ├─ manifest_logs.py     audit trail, lineage, health, session inspector
 └─ reporting.py         original tabular PDF (kept as "Classic PDF")
```

## LLM / API keys
Paste a key in the Publisher tab (or the AI Vision tab), set an environment variable
(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GROQ_API_KEY`, `GEMINI_API_KEY`), or copy
`.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`. No key? Use **Offline**.
Data report: only aggregated statistics are sent. Image description: the resized image is sent
to the chosen provider (only when you click the button).

## Fixes made to the original code
- `automl.py`: invalid `stratify=None` argument removed; classification targets limited to 2-30 classes;
  report/confusion matrix use the full label set; deprecated `multi_class` removed.
- `app.py`: uploading a *different* CSV now reloads the data (previously the first file stayed forever);
  ETL runs, dataset loads and reports are written to the audit trail.
- `requirements.txt`: added `requests`, `statsmodels` (needed by Plotly `trendline="ols"`).
