# Deep Learning for Time Series Modeling — Summer School

**TU Berlin | M. Stender & M. Brzosko**

A hands-on summer school covering deep learning methods for time series modeling, including sequence-to-sequence architectures, forecasting, and data-driven modeling of dynamical systems.

---

## Repository Structure

```
DL_TimeSeries_SummerSchool/
├── README.md                  ← this file
├── resources.md               ← curated links: Python tutorials, papers, tools
├── teaching_material/
│   ├── slides/                ← lecture slide decks (PDF / PPTX)
│   └── notebooks/             ← Jupyter notebooks accompanying the lectures
└── projects/
    ├── README.md              ← overview of all projects
    └── project_<name>/        ← one folder per project
        ├── instructions.md    ← project-specific task description
        └── ...                ← starter code, data references, etc.
```

### `teaching_material/`

All lecture slides and accompanying Jupyter notebooks. Notebooks are self-contained and can be run on Google Colab or locally (see setup instructions below).

### `projects/`

Individual modeling projects assigned during the summer school. Each project folder contains a dedicated `instructions.md` with the task description, data sources, and deliverables. Students fork or copy the relevant project folder into their own workspace.

### `resources.md`

A curated list of external links: Python/NumPy/PyTorch tutorials, reference papers, useful tools and datasets.

---

## What Students Need for the First Class

### Hardware

- A laptop capable of running Jupyter notebooks (any OS).  
- GPU access is *not* required for the first session — Google Colab (free tier) is sufficient for all exercises.

### Software — please install *before* the first class

| Tool | Version | Install |
|------|---------|---------|
| Python | ≥ 3.10 | [python.org](https://www.python.org/downloads/) or via `conda` |
| PyTorch | ≥ 2.0 | `pip install torch` — see [pytorch.org/get-started](https://pytorch.org/get-started/locally/) for your OS/CUDA combo |
| Jupyter | latest | `pip install jupyterlab` |
| NumPy | latest | `pip install numpy` |
| Matplotlib | latest | `pip install matplotlib` |
| pandas | latest | `pip install pandas` |

**Quick-start (recommended):**

```bash
# 1. Create a dedicated environment (conda example)
conda create -n dl_timeseries python=3.11
conda activate dl_timeseries

# 2. Install PyTorch (CPU-only — works everywhere; swap for a CUDA build if you have a GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 3. Install remaining dependencies
pip install jupyterlab numpy matplotlib pandas scikit-learn
```

### Verify your setup

Run the following snippet in a Python shell or notebook cell — if it prints without errors, you are ready:

```python
import torch, numpy, matplotlib, pandas, sklearn
print("PyTorch", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
```

### Google Colab (alternative)

If local installation is not feasible, all notebooks can be opened directly in [Google Colab](https://colab.research.google.com/). A Google account is all you need — no additional setup required.

### Prior knowledge

Students are expected to have:
- Basic Python programming skills (loops, functions, classes).
- Familiarity with NumPy arrays and basic data manipulation.
- Some exposure to machine learning concepts (helpful but not strictly required).

See [`resources.md`](resources.md) for self-study material on any of the above.

---

## License

Content in this repository is intended for educational use within the summer school.
Contact the instructors before redistributing any material.
