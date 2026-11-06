# CausalLens AI — Research Decision Intelligence & Causal AI Platform
**A Full-Stack Research-Level Data Science + Causal Inference + Generative AI Application**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit%20Cloud-FF4B4B?style=for-the-badge&logo=streamlit)](https://casuallens-ai.streamlit.app/)
[![Tests](https://img.shields.io/badge/pytest-66%20passed-brightgreen.svg)](tests/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](requirements.txt)
[![Framework](https://img.shields.io/badge/Streamlit-1.40%2B-red.svg)](app.py)
[![VectorDB](https://img.shields.io/badge/ChromaDB-Persistent-purple.svg)](src/rag.py)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> 🚀 **Live Production Deployment:** [https://casuallens-ai.streamlit.app/](https://casuallens-ai.streamlit.app/)
> 
> Explore the live interactive application directly in your browser with pre-loaded benchmarks (Lalonde job training, synthetic ground truth) and ChromaDB document intelligence.

CausalLens AI bridges the gap between observational causal inference, structural decision-making, and generative AI. It combines automated data quality profiling, Judea Pearl's Structural Causal Models (DAGs), multi-model treatment effect estimation (OLS, Matching, IPW, Doubly Robust AIPW, Double Machine Learning), sensitivity analysis (E-values, refutation tests), Monte Carlo benchmarking, and an evidence-grounded LLM assistant audited for numerical faithfulness.

---

## 🏛️ Technical Architecture & Modular Separation

CausalLens AI enforces a strict separation of concerns across scientific computing, document intelligence, and interactive presentation:

```
CausalLens_AI/
├── app.py                      # Presentation Layer: Streamlit UI, theme, 8 dedicated workspaces, routing
├── src/
│   ├── gemini_client.py        # Centralized GenAI Client: Official google-genai SDK, key masking, error sanitization, model cascade
│   ├── profiler.py             # Dataset Processing: missingness, descriptive stats, Tukey IQR, correlation
│   ├── dag.py                  # Causal Identification: Pearl SCM, d-separation, backdoor criterion, cycle checks
│   ├── causal.py               # Causal Estimators: OLS (HC3), 1:1 PSM (ATT), IPW, AIPW, Double ML
│   ├── sensitivity.py          # Sensitivity & Refutations: VanderWeele E-Values, placebo & subsample tests
│   ├── benchmark.py            # Simulation Lab: Synthetic DGPs, finite-sample Monte Carlo benchmarking
│   ├── rag.py                  # Document Intelligence & RAG: Multi-format extractors, chunking, ChromaDB, BM25Okapi RRF
│   ├── assistant.py            # Conversational AI Assistant: Dual-source engine (Dataset, Docs, Hybrid)
│   ├── eval.py                 # Factual Evaluation: Regex numerical claim auditor & causal assumption checker
│   └── reporting.py            # Reproducibility Suite: Markdown, JSON bundles, and standalone HTML reports
├── tests/                      # Automated Test Suite: 66 comprehensive unit, e2e, and live Gemini integration tests
└── data/                       # Persistent Data Storage (ChromaDB vectors & local SQLite indices)
```

### Module Responsibilities

| Subsystem | Source Module | Architectural Function |
| :--- | :--- | :--- |
| **Frontend & UI** | [`app.py`](file:///c:/Users/shiva/OneDrive/Desktop/CausalLens_AI/app.py) | 8 dedicated workspaces, reactive routing (`switch_page`), responsive CSS glassmorphism. |
| **Document Ingestion** | [`src/rag.py`](file:///c:/Users/shiva/OneDrive/Desktop/CausalLens_AI/src/rag.py) | Multi-format ingestor for PDF, Word (.docx, .doc), Text (.txt), Markdown (.md), CSV, Excel (.xlsx, .xls), and PowerPoint (.pptx). |
| **Text Extraction** | [`src/rag.py`](file:///c:/Users/shiva/OneDrive/Desktop/CausalLens_AI/src/rag.py) | Dedicated format parsers extracting per-page / per-sheet content with clean whitespace normalization. |
| **Intelligent Chunking** | [`src/rag.py`](file:///c:/Users/shiva/OneDrive/Desktop/CausalLens_AI/src/rag.py) | Heading-aware chunker with academic section classification (Abstract, Intro, SCM, Methods, Results, Sensitivity). |
| **Embedding Generation** | [`src/rag.py`](file:///c:/Users/shiva/OneDrive/Desktop/CausalLens_AI/src/rag.py) | Google Gemini embeddings with high-entropy offline hash vectors for air-gapped execution. |
| **Vector Storage** | [`src/rag.py`](file:///c:/Users/shiva/OneDrive/Desktop/CausalLens_AI/src/rag.py) | Persistent ChromaDB storage (`CHROMADB_PERSIST_DIR`), SQLite-backed metadata, SHA-256 deduplication. |
| **Hybrid Retrieval** | [`src/rag.py`](file:///c:/Users/shiva/OneDrive/Desktop/CausalLens_AI/src/rag.py) | Dense vector cosine similarity fused with in-memory BM25Okapi lexical matching via Reciprocal Rank Fusion ($k=60$). |
| **Conversational Assistant** | [`src/assistant.py`](file:///c:/Users/shiva/OneDrive/Desktop/CausalLens_AI/src/assistant.py) | Dual-source conversational engine querying structured datasets, uploaded documents, or hybrid contexts. |
| **Factual Claim Auditor** | [`src/eval.py`](file:///c:/Users/shiva/OneDrive/Desktop/CausalLens_AI/src/eval.py) | Automated regex parsing cross-verifying numbers against computed empirical tensors. |
| **Dataset Profiling** | [`src/profiler.py`](file:///c:/Users/shiva/OneDrive/Desktop/CausalLens_AI/src/profiler.py) | Tabular diagnostics, Tukey IQR anomaly detection, pairwise Pearson correlations, baseline balance. |
| **Causal Inference** | [`src/causal.py`](file:///c:/Users/shiva/OneDrive/Desktop/CausalLens_AI/src/causal.py), [`src/dag.py`](file:///c:/Users/shiva/OneDrive/Desktop/CausalLens_AI/src/dag.py) | Pearl SCM graph d-separation, backdoor sets, OLS (HC3), 1:1 PSM, IPW, Doubly Robust AIPW, Double ML. |
| **Reporting & Exports** | [`src/reporting.py`](file:///c:/Users/shiva/OneDrive/Desktop/CausalLens_AI/src/reporting.py) | Reproducibility exports in Markdown, machine-readable JSON bundles, and standalone HTML documents. |

---

## 🌟 Research Contribution & Hypotheses

Observational causal inference relies on untestable structural assumptions (unconfoundedness, positivity, consistency, SUTVA). Standard LLMs frequently hallucinate numerical results or overclaim causation from mere observational correlation.

CausalLens AI implements and evaluates:
- **H1 (Numerical Faithfulness):** Constraining LLM generation to cite computed evidence objects reduces unsupported numerical claims to near zero.
- **H2 (Assumption Disclosure):** Contextualizing prompts with causal identification requirements forces explicit disclosure of exchangeability, common support, and model specification assumptions.
- **H3 (Ablation Comparison):** Enables direct empirical comparison across three prompting regimes:
  1. *Evidence-Grounded Assistant:* Strict citation constraints (`[Evidence: metric=value]`).
  2. *Methodologist Auditor:* Skeptical peer-review persona evaluating covariate balance (Rubin's criteria) and E-values.
  3. *Unconstrained LLM:* Baseline generating intuition without computational bounds.

---

## ⚖️ Causal Epistemology: Assumptions vs. Empirical Evidence

CausalLens AI establishes a strict scientific distinction between what must be **theoretically assumed** by the researcher and what is **empirically verified** by data:

- **Untestable Structural Assumptions:**
  - *Causal Directionality ($T \to Y$):* Cannot be proven from observational distribution $P(T, Y)$ alone without temporal or experimental intervention.
  - *Unconfoundedness ($Y(t) \perp T \mid X$):* Unmeasured confounders ($U$) cannot be tested in observed data.
  - *SUTVA (No Interference):* Treatment assigned to one unit does not affect potential outcomes of other units.
- **Empirically Verified Statistical Findings:**
  - *Covariate Balance Tables & Love Plot:* Evaluates empirical $|SMD| < 0.10$ against Rubin's criteria.
  - *Common Support Density:* Propensity score histograms reveal empirical overlap $[min, max]$.
  - *Point Estimates & Robust SEs:* OLS (HC3), 1:1 PSM (ATT), IPW, AIPW, and 5-Fold Double ML.
  - *VanderWeele E-Values:* Quantifies required unmeasured confounding strength to nullify the effect.
  - *Falsification Refutations:* Placebo treatment permutations, random noise injections, and subsample draws.

---

## 🚀 Key Workspaces & Capabilities

### 1. Overview / Commercial Homepage
- Futuristic Hero with headline, product description, and 3D Isometric Causal Nexus SVG.
- Prominent AI Research Hub with 4 instant prompt chips and verified synthesis.
- Active research project summary card and recently indexed documents card.
- 6 interactive workspace jump cards providing quick access across the platform.

### 2. Data Analysis & Ingestion Studio
- Comprehensive feature profiling table, missingness analysis, and cardinality metrics.
- Tukey's IQR ($1.5 \times IQR$) and $Z$-score ($|Z| > 3.0$) outlier detection.
- Pearson correlation matrix heatmap and top-ranked pairwise correlations.
- In-page CSV and Excel dataset uploader supporting `.csv`, `.xlsx`, and `.xls`.

### 3. Causal Analysis & DAG Workbench
- **4-Step Guided Workflow:** SCM Identification $\to$ Balance & Positivity $\to$ Multi-Estimator Suite $\to$ Sensitivity & Refutations.
- Canonical SCM presets (Confounding Triangle, Mediation, M-Bias Collider, Instrumental Variable).
- Interactive node and edge studio with automated cycle prevention.
- Love plot, common support overlap histogram, and comparative Forest Plot (ATE vs ATT).

### 4. Dual-Source AI Research Assistant
- Inquire structured dataset tensors, uploaded multi-format documents, or both in hybrid mode.
- Automated claim auditor verifying all extracted numbers against computed data context.
- Causal assumptions auditor surfacing potential violations of unconfoundedness, positivity, or SUTVA.

### 5. Universal Document Intelligence & RAG
- Universal ingestion across PDF, DOCX, XLSX, CSV, PPTX, TXT, and Markdown files.
- Persistent ChromaDB vector database with configurable storage location.
- Multi-document comparison matrix, research paper summarizer, and key information extractor.

### 6. Simulation & Monte Carlo Benchmark Studio
- Evaluates finite-sample performance across known ground-truth treatment effects ($\tau$).
- Measures empirical bias, sampling variance, RMSE, and 95% confidence interval coverage rate.
- Interactive boxplot visualization and one-click benchmark CSV download.

### 7. Scientific Reporting & Reproducibility Suite
- Automated Markdown research report with LaTeX formulas, Mermaid DAG, and balance tables.
- Machine-readable JSON reproducibility bundle with parameters, random seeds, and arrays.
- Publication-ready standalone HTML report with zero external CDN dependencies.

### 8. System Settings & Diagnostics
- Secure API key configuration masked in the UI.
- Model selection (`gemini-2.5-flash`, `gemini-2.5-pro`), temperature, and persona selection.
- In-memory cache purge and environment telemetry.

---

## 🛠️ Installation & Setup

### 1. Clone & Set Up Virtual Environment
```bash
git clone https://github.com/ShivamkumarMantri/CasualLens-AI.git
cd CasualLens-AI

# Create virtual environment
python -m venv .venv

# Activate environment
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On macOS / Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables
```bash
cp .env.example .env
```
Edit `.env` to configure your credentials:
```env
# Google Gemini API Key (Official Google GenAI SDK)
GEMINI_API_KEY="your-gemini-api-key"

# Optional backwards compatibility
GOOGLE_API_KEY="your-gemini-api-key"

# Active Gemini Model (gemini-3.5-flash-lite, gemini-3.1-flash-lite, gemini-3.5-flash, gemini-3.1-pro-preview)
GEMINI_MODEL="gemini-3.5-flash-lite"

# Persistent Vector Store Directory
CHROMADB_PERSIST_DIR="data/chroma_db"
```
*Note: All data profiling, DAG identification, 5 causal estimators, sensitivity E-values, Monte Carlo simulations, multi-format file ingestion, and report generation operate 100% locally without an API key. When an API key is provided, CausalLens AI activates live Google GenAI generation with automatic model cascades, dense embeddings (gemini-embedding-001), and factual claim verification.*

### 3. Launch Streamlit Application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 🐳 Production & Cloud Deployment

### 🌐 Streamlit Community Cloud (Live)
CausalLens AI is hosted and live at:
👉 **[https://casuallens-ai.streamlit.app/](https://casuallens-ai.streamlit.app/)**

To deploy your own instance to Streamlit Community Cloud:
1. Fork or push this repository to GitHub.
2. Sign in to [share.streamlit.io](https://share.streamlit.io).
3. Connect repository `CasualLens-AI`, select branch `main`, and main file path `app.py`.
4. Under **Advanced settings → Secrets**, paste your API credentials:
   ```toml
   GEMINI_API_KEY = "your_actual_key_here"
   GEMINI_MODEL = "gemini-3.5-flash-lite"
   CHROMADB_PERSIST_DIR = "data/chroma_db"
   ```
5. Click **Deploy**.

### Docker Containerization
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Persistent storage mount point for vector database
ENV CHROMADB_PERSIST_DIR=/app/data/chroma_db
VOLUME ["/app/data"]

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.headless=true", "--server.address=0.0.0.0"]
```

### Build & Run Container
```bash
# Build image
docker build -t causallens-ai:latest .

# Run container with persistent volume
docker run -d -p 8501:8501 -v causallens_data:/app/data --name causallens causallens-ai:latest
```

---

## 🧪 Automated Test Suite

Run all 66 unit, regression, UI navigation, and integration tests:
```bash
pytest -v
```

All 66 tests pass with 100% success across:
- `tests/test_assistant_and_eval.py` (Numerical claim verification & assumptions audit)
- `tests/test_benchmark.py` (Synthetic data generation & Monte Carlo coverage)
- `tests/test_causal.py` (OLS, PSM, IPW, AIPW, DML, and multi-format dataset ingestion)
- `tests/test_core.py` (Profiling, correlations, outliers, and SMDs)
- `tests/test_dag.py` (Identification, d-separation, backdoor paths, cycle prevention)
- `tests/test_e2e_pipeline.py` (Full research pipeline AST and page routing)
- `tests/test_gemini_integration.py` (Live Google GenAI client, masking, and RAG synthesis)
- `tests/test_rag.py` (Universal document ingestion, chunking, ChromaDB, and genuine evidence QA)
- `tests/test_reporting.py` (Markdown, JSON reproducibility bundles, and HTML exports)
- `tests/test_sensitivity.py` (VanderWeele E-Values and refutation battery)
- `tests/test_ui_navigation.py` (App lifecycle, 8 workspaces, and execution flows)

---

## 📚 Foundational Scientific Bibliography

1. **Pearl, J. (2009).** *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press.
2. **Rubin, D. B. (1974).** Estimating causal effects of treatments in randomized and nonrandomized studies. *Journal of Educational Psychology*, 66(5), 688–701.
3. **Rosenbaum, P. R., & Rubin, D. B. (1983).** The central role of the propensity score in observational studies for causal effects. *Biometrika*, 70(1), 41–55.
4. **Robins, J. M., Rotnitzky, A., & Zhao, L. P. (1994).** Estimation of regression coefficients when some regressors are not always observed. *Journal of the American Statistical Association*, 89(427), 846–866.
5. **Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018).** Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1–C68.
6. **VanderWeele, T. J., & Ding, P. (2017).** Sensitivity analysis in observational research: Introducing the E-value. *Annals of Internal Medicine*, 167(4), 268–274.

---

## ⚖️ Academic Integrity & Disclaimer
CausalLens AI is an evidence-grounded research decision-intelligence tool designed for empirical exploration and methodological research. It strictly separates untestable structural assumptions from empirical statistical calculations. All findings should be interpreted in conjunction with domain expertise and sensitivity analyses.

