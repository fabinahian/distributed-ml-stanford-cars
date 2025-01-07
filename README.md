# Designing ML System using Data-Parallel Distributed Neural Network Training

This project demonstrates distributed training simulation using a single GPU by partitioning GPU memory. It shows how multi-node training can be more efficient than single-node training, even when simulated on a single device.

## 🎯 Project Overview

This implementation simulates distributed training on a single GPU (RTX 3080) by:

- Partitioning GPU memory among simulated nodes
- Using PyTorch's distributed training capabilities
- Managing system resources efficiently
- Comparing performance metrics between single and multi-node setups

## 🚀 Quick Start

### Prerequisites

```bash
# Clone the repository
git clone <repository-url>
cd <repository-name>

# Create and activate conda environment
conda create -n distributed-sim python=3.8
conda activate distributed-sim

# Install requirements
pip install -r requirements.txt
```

### Running Training

Single GPU Training:

```bash
python src/scripts/launcher.py --mode single
```

Distributed Training Simulation:

```bash
python src/scripts/launcher.py --mode distributed --num-epochs 100 --num-nodes 4 --batch-size 8
```

## 🏗️ Project Structure

```
project_root/
├── data/
│   └── car_data/          # Dataset directory
├── src/
│   ├── utils/             # Utility functions
│   ├── data/              # Data processing
│   ├── model/             # Model architecture
│   ├── training/          # Training implementations
│   └── scripts/           # Training scripts
├── configs/               # Configuration files
└── requirements.txt
```

## 🛠️ Components

### 1. Memory Management (`src/utils/memory_utils.py`)

- GPU memory partitioning
- Resource monitoring
- Memory optimization strategies

### 2. Data Processing (`src/data/data_preprocessing.py`)

- Custom dataset implementation
- Data loading and augmentation
- Distributed sampling

### 3. Model Architecture (`src/model/model_setup.py`)

- ResNet18-based classifier
- Distributed model wrapper
- Model checkpoint handling

### 4. Training Implementation

- Single GPU (`src/training/trainer.py`)
- Distributed (`src/training/distributed_trainer.py`)
- Training metrics and monitoring

## 📊 Performance Monitoring

The training progress is monitored through:

- TensorBoard logging
- Memory usage tracking
- Training metrics visualization
- Resource utilization stats

Access TensorBoard:

```bash
tensorboard --logdir runs/
```

## ⚙️ Configuration

### Single Node Configuration

```yaml
# configs/single_node_config.yaml
batch_size: 8
num_epochs: 100
learning_rate: 0.0001
```

### Distributed Configuration

```yaml
# configs/distributed_config.yaml
num_nodes: 4
batch_size: 8 # Per node
memory_fraction: 0.23 # Per node
```

## 📈 Results Analysis

Compare performance metrics between single and multi-node setups:

- Training time per epoch
- Memory utilization
- Model convergence rate
- Final model accuracy
- Resource efficiency

## 🔧 Customization

### Adjusting Memory Allocation

```python
# Modify memory fraction per node
python src/scripts/train_single.py --num-epochs=100 --batch-size 8 --memory-fraction 0.20
```

### Batch Size Tuning

```python
# Adjust batch size per node
python src/scripts/train_distributed.py --num-nodes 4 --num-epochs=100 --batch-size 8 --memory-fraction 0.20
```

## 📝 License

This project is a open-source project developed as a coursework for CSE707 Distributed Computing Systems, MSCCSE, Brac University.

## 🤝 Contributing

1. Fork the Project
2. Create your Feature Branch
3. Commit your Changes
4. Push to the Branch
5. Open a Pull Request

## 🚧 Known Limitations

- Limited by single GPU memory (10GB RTX 3080)
- System RAM constraints (16GB)
- Simulation overhead vs actual multi-GPU setup
- Fault tolarance testing
