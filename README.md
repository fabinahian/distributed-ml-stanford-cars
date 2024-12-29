```markdown
# Distributed ML on Stanford Cars

This repository demonstrates **data-parallel distributed neural network training** on the **Stanford Cars** dataset, focusing on **fine-grained car classification**. Instead of manually downloading the dataset, we utilize **`torchvision.datasets.StanfordCars`** to automatically fetch images and bounding boxes.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [How to Clone and Set Up](#how-to-clone-and-set-up)  
3. [Repository Structure](#repository-structure)  
4. [Using the Virtual Environment](#using-the-virtual-environment)  
5. [Installing Dependencies](#installing-dependencies)  
6. [Working with the Stanford Cars Dataset via TorchVision](#working-with-the-stanford-cars-dataset-via-torchvision)  
   - [Quick EDA Outline](#quick-eda-outline)  
7. [Running This Project](#running-this-project)  
   - [Single-GPU Training](#single-gpu-training)  
   - [Distributed Training](#distributed-training)  
8. [References](#references)

---

## Project Overview

- **Dataset**: [Stanford Cars in TorchVision](https://pytorch.org/vision/main/generated/torchvision.datasets.StanfordCars.html), featuring 16,185 images & 196 car classes.  
- **Focus**: Distinguishing subtle differences in car makes, models, and years (fine-grained classification).  
- **Tech Stack**:  
  - **PyTorch** and `torchvision` for deep learning and dataset handling.  
  - **Virtual Environment** (venv) for reproducible Python dependencies.  
  - **DistributedDataParallel (DDP)** for multi-GPU speedup.

---

## How to Clone and Set Up

1. **Clone the Repository**
   ```bash
   git clone https://github.com/YourUsername/distributed-ml-stanford-cars.git
   cd distributed-ml-stanford-cars
   ```
   *(Replace the URL with your own if needed.)*

2. **Create a Virtual Environment**  
   Make sure you have a suitable Python version (e.g., **Python 3.11** on Ubuntu 24). Then:
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```
   > This local `.venv` ensures all installations stay isolated from system Python.

3. **Install Dependencies**  
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```
   Once installed, you can confirm with `pip list`.

---

## Repository Structure

```bash
distributed-ml-stanford-cars/
├── .gitignore
├── README.md
├── data/
│   ├── raw/          # (Optional) if you store data manually
│   └── processed/    # (Optional) for processed or augmented images
├── docs/
│   ├── literature_review.md
│   └── project_proposal.md
├── notebooks/
│   └── 01-eda.ipynb  # Exploratory Data Analysis examples
└── src/
    ├── __init__.py
    ├── data_preprocessing.py
    ├── distributed_train.py   # Multi-GPU or multi-node training
    ├── train.py               # Single-GPU training script
    └── utils.py               # Helpers (e.g., metrics, logging)
```

---

## Using the Virtual Environment

- **Activate**:
  ```bash
  source .venv/bin/activate
  ```
- **Deactivate**:
  ```bash
  deactivate
  ```
- **Update dependencies**:
  ```bash
  pip install -r requirements.txt
  ```

---

## Installing Dependencies

In your `requirements.txt`, you might see entries like:
```
torch==2.0.1
torchvision==0.15.2
matplotlib==3.7.2
pandas==2.0.3
seaborn==0.12.2
```
These cover **PyTorch**, **torchvision**, and common data science libraries. After activating the `.venv`, just run:
```bash
pip install -r requirements.txt
```

---

## Working with the Stanford Cars Dataset via TorchVision

Since **`torchvision.datasets.StanfordCars`** automatically downloads and manages the dataset for you, you can load the train/test splits and bounding box information without manual steps. For instance, in a Python script or Jupyter Notebook:

```python
import torchvision
from torchvision.datasets import StanfordCars
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_data = StanfordCars(
    root="./data",
    split="train",
    download=True,
    transform=transform
)
test_data = StanfordCars(
    root="./data",
    split="test",
    download=True,
    transform=transform
)
```

### Quick EDA Outline

- **Dataset size**:
  ```python
  print("Train images:", len(train_data))
  print("Test images:", len(test_data))
  ```
- **Classes**:
  ```python
  print("Number of classes:", len(train_data.classes))
  print(train_data.classes[:5])  # sample class names
  ```
- **Annotations** (bounding boxes, etc.):
  ```python
  print(train_data.annotations[0])  # dict with keys like 'bbox', 'class', 'fname'
  ```
- **Visualization**: See `notebooks/01-eda.ipynb` for examples of drawing bounding boxes and class distributions.

---

## Running This Project

### Single-GPU Training

1. **Adjust hyperparameters** in `src/train.py` (e.g., epochs, batch size, learning rate).  
2. **Run** the training:
   ```bash
   python src/train.py
   ```
   This will train on a single GPU if available (or default to CPU).

### Distributed Training

1. In `src/distributed_train.py`, you’ll find a **PyTorch DistributedDataParallel** setup.  
2. Launch with multiple processes:
   ```bash
   # Example: 4 GPUs on a single machine
   torchrun --nproc_per_node=4 src/distributed_train.py
   ```
3. Monitor logs and metrics to ensure gradients are syncing properly.

---

## References

- **Stanford Cars Dataset**:  
  - [PyTorch Docs for StanfordCars](https://pytorch.org/vision/main/generated/torchvision.datasets.StanfordCars.html)
- **PyTorch & TorchVision**:  
  - [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)
- **DistributedDataParallel** (DDP):  
  - [Distributed Overview](https://pytorch.org/docs/stable/distributed.html)
- **Docs**:  
  - See `docs/project_proposal.md` and `docs/literature_review.md` for methodology, related works, and conceptual background.

---

**Happy coding!** If you have any issues or suggestions, please open a GitHub issue or submit a pull request. Contributions are always welcome.
```