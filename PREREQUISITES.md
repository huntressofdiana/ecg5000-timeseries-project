# Prerequisites: Before You Arrive

Complete this checklist before the first session, class time is for learning, not troubleshooting installs.

## Hardware

- A laptop, any operating system, able to run Jupyter notebooks and plain Python scripts.
- A GPU is not required. Without one, use a free cloud provider instead, see Accounts below.

## Accounts

- [GitHub](https://github.com/): required, for course material and project submissions.
- [Google Colab](https://colab.research.google.com/): recommended free GPU alternative, needs a Google account.
- [Kaggle Notebooks](https://www.kaggle.com/code): backup free GPU alternative, needs a Kaggle account.

## Software

| Tool | Version | Install |
|------|---------|---------|
| Python | ≥ 3.10 | [python.org](https://www.python.org/downloads/) or via `conda` |
| PyTorch | ≥ 2.0 | [pytorch.org/get-started](https://pytorch.org/get-started/locally/) for your OS/CUDA combo |
| Jupyter | latest | `pip install jupyterlab` |
| Git | latest | [git-scm.com](https://git-scm.com/downloads), configured with your GitHub account |
| NumPy, pandas, Matplotlib | latest | `pip install numpy pandas matplotlib` |

Work inside a dedicated, clean virtual environment (`venv` or `conda`), kept separate from other projects.

**Quick start (conda example):**

```bash
conda create -n dl_timeseries python=3.11
conda activate dl_timeseries
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install jupyterlab numpy matplotlib pandas scikit-learn
```

**Verify:**

```python
import torch, numpy, matplotlib, pandas, sklearn
print("PyTorch", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
```

If it prints without errors, you are ready. Test this before arriving, some corporate or university laptops restrict installs or block package downloads.

## Skills

You do not need to be an expert, but you should arrive comfortable with:

- **Python:** variables, loops, functions, classes, and basic NumPy/pandas usage.
- **Git:** clone, add, commit, push, pull, and working with a remote repository such as GitHub.
- **Package management:** creating a virtual environment and installing packages with `pip` or `conda`.
- **Deep learning fundamentals:** what a neural network is (layers, weights, activation functions), and what training means (forward pass, loss function, backpropagation, gradient descent).
- **PyTorch basics:** tensors, and how to define and train a simple model, including the training loop.

New to any of this? See [`RESOURCES.md`](RESOURCES.md) for tutorials, ideally before the course starts.

## Checklist

- [ ] Laptop ready, or a Colab/Kaggle account set up
- [ ] GitHub account created
- [ ] Python, PyTorch, Jupyter, and Git installed and verified in a clean virtual environment
- [ ] Comfortable with basic Python, git, package installation, and "what is a neural network"
