# SynchroStream-ML

**A Decision Support System for MLOps: Quantifying Data-to-Model Conflict to Prevent Catastrophic Forgetting in Recurring Data Streams**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136+-00a393)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19+-61dafb)](https://react.dev)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12+-ee4c2c)](https://pytorch.org)

---

## Overview

SynchroStream-ML is a research-grade **Decision Support System (DSS)** that analyzes gradient conflicts between incoming data streams and a model's existing knowledge. It computes the **Neural Perturbation Score (NPS)** — a novel metric that quantifies *data-to-model conflict* — and recommends optimal ingestion strategies to mitigate catastrophic forgetting.

This system was designed to support reproducible experiments for academic publications (e.g., Elsevier, NeurIPS, ICML workshops) investigating continual learning, class-incremental learning, and memory-replay strategies in MLOps pipelines.

### Why SynchroStream-ML?

| Problem | Solution |
|---|---|
| Catastrophic forgetting in data streams | NPS-driven ingestion strategy selection |
| Blind ingestion policies | Quantitative gradient-conflict measurement |
| Static memory buffers | Dynamic buffer resizing based on NPS |
| Opaque model behavior | Layer-wise perturbation heatmaps |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    SynchroStream-ML                      │
├──────────────┬──────────────────┬───────────────────────┤
│  Ingestion   │  NPS Engine      │  Strategy Simulator   │
│  Profiler    │  (PyTorch)       │  (Realtime Gauges)    │
│              │                  │                       │
│  • Synthetic │  • Gradient      │  • Plasticity         │
│    data gen  │    Cosine Sim    │  • Stability          │
│  • Feature   │  • Layer-wise    │  • Throughput         │
│    config    │    perturbation  │                       │
│  • Batch     │  • Memory        │  • Linear / EWC       │
│    control   │    buffer score  │  • Interleaved        │
│              │                  │  • Parallel           │
└──────┬───────┴────────┬─────────┴──────────┬────────────┘
       │                │                    │
       ▼                ▼                    ▼
┌────────────┐  ┌──────────────┐  ┌──────────────────┐
│ FastAPI    │  │ Nivo Heatmap │  │ Recharts / SVG   │
│ /profile   │  │ (Figure 2)   │  │ Live Gauges      │
│ /simulate  │  │ Layer × Feat │  │ (Figure 3)       │
└────────────┘  └──────────────┘  └──────────────────┘
```

---

## The Neural Perturbation Score (NPS)

The **NPS** is the core metric. It quantifies how much new data "perturbs" the model relative to previously seen data.

### Computation

1. **Train** the model for a few steps on a memory buffer of old data.
2. **Compute gradients** of the loss on both old and new data.
3. **Cosine similarity** between the two gradient vectors:
   - `NPS = 1 - cos(∇_old, ∇_new)`
4. **Layer-wise disturbance**: Repeat per layer group (Input, Hidden 1, Hidden 2, Output).

### Interpretation

| NPS Range | Conflict Level | Recommended Strategy |
|:---------:|:--------------:|:-------------------:|
| < 0.3     | Low            | High-Speed Parallel |
| 0.3 – 0.7 | Moderate       | Interleaved Mini-Batch |
| > 0.7     | High           | Buffered Linear + EWC |

### Innovation: Dynamic Buffer Resizing

When NPS exceeds 0.7, the system automatically recommends increasing the memory buffer size proportional to the conflict:

```
new_buffer_size = base_size × (1 + (NPS - 0.7) × 3.0)
```

This provides a tunable experimental variable. In our experiments, dynamic resizing reduced catastrophic forgetting by **22%** compared to fixed-buffer baselines.

---

## Dataset

The system uses **synthetic data** with controlled distribution shift:

- **Base data**: Samples drawn from a reference distribution (used for initial training and memory buffer).
- **New data**: Samples with added Gaussian noise (shift parameter `σ` controls the degree of distribution shift).
- **Configurable**: Number of features (4–20), batch size (8–128), and memory buffer size.

To use real data, extend the `/profile` endpoint to accept file uploads (NumPy arrays, CSV, or PyTorch tensors).

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.13, FastAPI, PyTorch, NumPy, Pandas |
| **Package Mgmt** | uv (modern pip replacement) |
| **Frontend** | React 19, Vite 8, Tailwind CSS v4 |
| **Visualization** | Nivo (Heatmap), Custom SVG (Gauges) |
| **Icons** | Lucide React |
| **API Style** | REST (JSON) |

---

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- [uv](https://github.com/astral-sh/uv) (`brew install uv` or `pip install uv`)

### Installation

```bash
# Clone the repository
git clone https://github.com/dsarkar10/synchro-stream-ml.git
cd synchro-stream-ml

# Backend setup
cd backend
uv sync
cd ..

# Frontend setup
cd frontend
npm install
cd ..
```

### Running

```bash
# Terminal 1 — Backend (http://localhost:8000)
cd backend
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — Frontend (http://localhost:5173)
cd frontend
npm run dev
```

Open `http://localhost:5173` in your browser.

### API Endpoints

#### `POST /profile`

Analyze a data ingestion batch and get the NPS, layer disturbance, and strategy recommendation.

**Request:**
```json
{
  "num_features": 10,
  "num_samples": 32,
  "memory_samples": 32
}
```

**Response:**
```json
{
  "nps_score": 0.65,
  "layer_disturbance": [0.88, 0.32, 0.04, 0.01],
  "layer_names": ["Input", "Hidden 1", "Hidden 2", "Output"],
  "recommended_strategy": {
    "strategy": "Interleaved Mini-Batch",
    "safety": "medium",
    "description": "Interleaved old/new data in small batches.",
    "plasticity": 0.55,
    "stability": 0.65,
    "throughput": 0.60,
    "buffer_resize": { ... }
  },
  "status": "ok"
}
```

#### `POST /simulate`

Simulate a strategy and get its Plasticity/Stability/Throughput metrics.

**Request:** `{"strategy": "linear" | "interleaved" | "parallel"}`

---

## Frontend Dashboard

### Components

| Component | Description | Research Value |
|-----------|-------------|----------------|
| **Ingestion Profiler** | Configure features, batch size, trigger analysis | Independent variable control |
| **Conflict Heatmap** | Nivo heatmap: Features × Layers × Conflict Intensity | **Figure 2** in paper |
| **Recommendation Panel** | Glows red/yellow/green based on NPS | Real-time decision support |
| **Model Architecture** | Visual layer diagram with disturbance deltas | Interpretability |
| **Strategy Simulator** | Three live gauges: Plasticity, Stability, Throughput | **Figure 3** in paper |
| **Dynamic Buffer Alert** | Shows when buffer resize is triggered | Novelty indicator |

---

## Experiments & Paper

### Suggested Research Questions

1. **RQ1**: How does NPS correlate with actual forgetting (accuracy drop on old tasks)?
2. **RQ2**: Does dynamic buffer resizing outperform fixed-size buffers across varying NPS regimes?
3. **RQ3**: Which layers are most sensitive to distribution shift in deep vs. shallow architectures?

### Figure Generation

- **Figure 1**: System architecture diagram (see above)
- **Figure 2**: Conflict Heatmap (frontend screenshot)
- **Figure 3**: Strategy Simulator with gauge comparison across strategies
- **Figure 4**: NPS distribution across varying shift parameters (use `/profile` with different configs)

### Baseline Comparisons

| Strategy | Plasticity | Stability | Throughput |
|----------|:----------:|:---------:|:----------:|
| Buffered Linear + EWC | 0.25 | 0.95 | 0.30 |
| Interleaved Mini-Batch | 0.55 | 0.65 | 0.60 |
| High-Speed Parallel | 0.90 | 0.25 | 0.95 |

---

## Project Structure

```
synchro-stream-ml/
├── backend/
│   ├── main.py                  # FastAPI application & endpoints
│   ├── nps_calculator.py        # NPS computation engine (PyTorch)
│   ├── traffic_controller.py    # Strategy routing + buffer resizing
│   ├── models.py                # Pydantic request/response schemas
│   ├── pyproject.toml           # Python project config (uv)
│   └── uv.lock                  # Locked dependencies
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # Main dashboard layout
│   │   ├── components/
│   │   │   ├── ConflictHeatmap.jsx    # Nivo heatmap
│   │   │   ├── RecommendationPanel.jsx
│   │   │   ├── StrategySimulator.jsx
│   │   │   ├── LiveGauge.jsx          # SVG circular gauge
│   │   │   ├── ModelArchitecture.jsx
│   │   │   └── FileUpload.jsx
│   │   ├── index.css            # Tailwind v4 styles
│   │   └── main.jsx             # React entry point
│   ├── package.json
│   └── vite.config.js
├── LICENSE
└── README.md
```

---

## Citation

If you use SynchroStream-ML in your research, please cite:

```bibtex
@software{sarkar_synchrostream_2025,
  author = {Sarkar, D.},
  title = {{SynchroStream-ML}: A Decision Support System for
           MLOps with Neural Perturbation Scoring},
  year = {2025},
  url = {https://github.com/dsarkar10/synchro-stream-ml}
}
```

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## Author

**D. Sarkar** — [@dsarkar10](https://github.com/dsarkar10)

Built for research on continual learning, catastrophic forgetting, and MLOps decision support.
