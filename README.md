# CP-SSX: Causal Prescriptive Student Success eXplainer Engine

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Optimization](https://img.shields.io/badge/PuLP-MILP-green.svg)](https://coin-or.github.io/pulp/)
[![UI](https://img.shields.io/badge/Streamlit-1.25%2B-ff4b4b.svg)](https://streamlit.io/)

**CP-SSX** is a world-class, novel educational technology platform designed for *Managerial Machine Learning in Python*. Unlike standard early-warning software that merely outputs passive dropout risk scores, **CP-SSX** shifts the paradigm to **Causal Prescriptive Analytics with Mechanistic Neural Network Explainability and Constrained Resource Optimization**.

---

## 🏛️ System Architecture

```
[ Synthetic Data Generator ] (Module 1: OULAD + UCI Schema)
            │
            ▼
┌────────────────────────────────────────────────────────┐
│  Dynamic Daily Clickstream (VLE, Forum, Quiz)         │
│  + Static Socio-Demographics & Baseline Indicators     │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ Module 2: PyTorch Multi-Task Bi-LSTM                   │
│  ├── Multi-Task Predictions: Risk P(Y) & Expected GPA  │
│  └── Bottleneck Feature Extractor: Latent h_{i,t} ∈ R³²│
└───────────────────────────┬────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
┌───────────────────────────┐   ┌───────────────────────────┐
│ Module 3: MC-PRE Explainer│   │ Module 4: Uplift Engine   │
│ Concept Prober (C1...C4)  │   │ Multi-Treatment CATE      │
│  - Comprehension          │   │  - Advising Uplift τ_adv  │
│  - Procrastination        │   │  - Tutoring Uplift τ_tut  │
│  - Peer Isolation         │   │  - Micro-Grant τ_grant    │
│  - Financial Hardship     │   │ Counterfactual Recourse   │
└─────────────┬─────────────┘   └─────────────┬─────────────┘
              │                               │
              └───────────────┬───────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────┐
│ Module 5: MILP Resource Allocator (PuLP)              │
│  Maximize total risk reduction subject to advising,    │
│  tutoring, & grant capacity constraints                │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ Module 6: Narrative Compiler & Streamlit Dashboard    │
│  ├── Local SLM Generator (Ollama Llama3.2/Phi4 + fallback)│
│  └── 3 Role Views: Advisor, Admin/Dean, Student       │
└───────────────────────────┴────────────────────────────┘
```

---

## 🧩 The 6 Core Modules

1. **Module 1: Data Generator & Feature Engineering (`data/generate_data.py`)**
   - Synthesizes dynamic 12-week OULAD clickstream sequences and static UCI student demographics.
   - Computes engineered signals: Weekly Engagement Velocity ($\Delta E_t$), Submission Procrastination Lag ($\mu_{pro}$), and Academic Preparedness Index ($API$).
2. **Module 2: PyTorch Bi-LSTM Base Model & Latent Extractor (`src/models/base_lstm.py`)**
   - Multi-task sequence neural network predicting dropout risk $P(Y_i)$ and final GPA.
   - Intercepts 32-dimensional bottleneck latent representation $h_{i,t} \in \mathbb{R}^{32}$.
3. **Module 3: Mechanistic Concept-Bottleneck Explainer (`src/models/concept_probing.py`)**
   - Concept Probing Layer attaching to frozen $h_{i,t}$ vectors to decode 4 managerial concepts ($C_1 \dots C_4$).
4. **Module 4: Causal Uplift (CATE) & Counterfactual Recourse Engine (`src/causal/uplift_engine.py`)**
   - T-Learners estimating multi-treatment heterogeneous effects ($\tau_{i,a}$) for Advising, Tutoring, and Micro-Grants.
   - Computes minimal behavioral recourse targets to cross safe risk thresholds ($P(Y) < 0.20$).
5. **Module 5: Prescriptive Integer Programming Allocator (`src/optimization/milp_allocator.py`)**
   - PuLP Mixed-Integer Linear Program (MILP) maximizing aggregate risk reduction subject to advising hours, tutoring capacity, and financial aid budgets.
6. **Module 6: Privacy-Preserving SLM Narrative & Streamlit Dashboard (`src/llm/local_narrative.py` & `app.py`)**
   - Zero-hallucination local SLM engine (Ollama with fallback).
   - Multi-stakeholder dashboard with Advisor, Admin/Dean, and Student views.

---

## 🚀 Getting Started & Execution

### 1. Installation
```bash
cd cp_ssx_engine
pip install -r requirements.txt
```

### 2. Pre-Compute & Build Cache (Recommended)
To ensure the Streamlit web app launches instantly without waiting for neural network training and Causal T-Learner fitting, run the pre-computation build cache script:
```bash
python build_cache.py
```
* **What `build_cache.py` does:**
  1. Loads student demographic and 12-week clickstream data into memory.
  2. Trains the PyTorch Multi-Task Bi-LSTM sequence model for 15 epochs.
  3. Fits the **Unsupervised PCA Factor Engine** on the 32-dimensional bottleneck vectors $h_{i,t}$.
  4. Fits the **Causal T-Learner Meta-Algorithms** for Advising, Tutoring, and Micro-Grant interventions.
  5. Serializes the fitted model pipeline into `data_cache/pipeline_cache.pkl` for instantaneous web dashboard loading.

### 3. Generate Report & Executive Charts
To generate all 12 high-resolution charts used in `paper.pdf` and `presentation.pdf`:
```bash
python generate_charts.py
```

### 4. Test Modules Individually
```bash
python src/models/base_lstm.py
python src/models/pca_factor_engine.py
python src/causal/uplift_engine.py
python src/optimization/milp_allocator.py
python src/llm/local_narrative.py
```

### 5. Launch Interactive Streamlit Web Application
```bash
streamlit run app.py
```

---

## 💻 Tech Stack
- **Deep Learning & Math:** `PyTorch`, `NumPy`, `Pandas`, `SciPy`
- **Machine Learning & Causal Inference:** `Scikit-Learn`, `XGBoost`
- **Optimization:** `PuLP` (CBC Solver)
- **Local SLM:** `Ollama` (`llama3.2:3b` / `phi4:mini`) with fallback engine
- **Web App & Visualization:** `Streamlit`, `Matplotlib`, `Seaborn`

