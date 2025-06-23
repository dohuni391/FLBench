# FLSimulation

This project quantifies the local computation costs and convergence speed for a ResNet-18 model trained with Federated Averaging (FedAvg) across various client scales and datasets. The primary focus is on the compute time, ignoring network latency, to establish a baseline for training performance.

---

## 1. Objective

The goal is to measure two key metrics:
1.  **Local computation cost:** The time taken for a single client to complete one round of FedAvg (5 local epochs).
2.  **Convergence time:** The number of global rounds required to reach **90% test accuracy**.

These metrics are evaluated under six different client-scale settings: 10, 50, 100, 200, 500, and 1000 clients.

---

## 2. Methodology

### Model Architecture

A single, standardized model is used for all experiments to ensure a fair comparison.

*   **Model:** `torchvision.models.resnet18`
*   **Modification:** For 32x32 input images (CIFAR-10, upscaled FEMNIST), the initial 7x7 convolution is replaced with a 3x3 kernel (`nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)`). The initial max-pooling layer is replaced with an identity function.
*   **Parameter Count:** Approx. 11.7 million (~45 MB per model update).

### Datasets & Client Scaling

The choice of dataset is determined by the number of clients in the simulation to ensure meaningful data partitions.

| Client Count        | Dataset(s)                       | Rationale                                                              |
| ------------------- | -------------------------------- | ---------------------------------------------------------------------- |
| **≤ 100**           | **MNIST** & **CIFAR-10**           | Classic CV baselines where clients have sufficient data (≥500 images). |
| **> 100**           | **FEMNIST**                      | A large-scale, writer-separated dataset suitable for up to 1000+ clients. |

### Training Hyperparameters

*   **Local Epochs (`E`):** 5
*   **Batch Size:** 32
*   **Optimizer:** SGD (learning rate: 0.1, momentum: 0.9)
*   **Client Fraction (`fraction_fit`):** 1.0 (all clients participate in each round)

---

## 3. How to Run

### 3.1. Environment Setup

Create and activate the Conda environment, then install the required Python packages.

```bash
# Create and activate the environment
conda create -n fl_bench python=3.10 -y
conda activate fl_bench

# Install dependencies
pip install torch torchvision flwr[simulation] flwr-datasets[vision]
```

### 3.2. Running Experiments

Use the provided scripts to run the local timing benchmarks and the full federated simulations.

```bash
# --- Local Timing Benchmark ---
# Purpose: Measure the time for one client to train for 5 local epochs.
# Example for CIFAR-10 with settings that would match a 100-client run:
python local_timing.py --dataset CIFAR10 --epochs 5 --clients 100


# --- Federated Simulation ---
# Purpose: Run a full FedAvg simulation to find rounds to 90% accuracy.
# The script can auto-select the dataset based on the --clients flag.
# Example for FEMNIST with 500 clients:
python federated_simulation.py --dataset FEMNIST --clients 500 --epochs 5 --target_acc 0.9
```

---

## 4. Results

The experiments will populate the following tables.

### Table 1: Local Compute Time (5 Epochs)

This table records the wall-clock time for a single client to perform its local training update.

| Dataset               | Avg. Samples / Client | Time (seconds) |
| --------------------- | --------------------- | -------------- |
| MNIST (≤ 100 clients) | *fill this in*        | *fill this in* |
| CIFAR-10 (≤ 100)      | *fill this in*        | *fill this in* |
| FEMNIST (> 100)       | *fill this in*        | *fill this in* |

### Table 2: Global Rounds to 90% Test Accuracy

This table records the number of federated rounds required to hit the target accuracy.

| # Clients | MNIST | CIFAR-10 | FEMNIST |
| --------- | ----- | -------- | ------- |
| 10        | *TBD* | *TBD*    | n/a     |
| 50        | *TBD* | *TBD*    | n/a     |
| 100       | *TBD* | *TBD*    | n/a     |
| 200       | n/a   | n/a      | *TBD*   |
| 500       | n/a   | n/a      | *TBD*   |
| 1000      | n/a   | n/a      | *TBD*   |

### Estimating Total Compute-Only Runtime

The total estimated time (ignoring communication) can be calculated as:

```
Total Runtime = (Time from Table 1) × (Rounds from Table 2)
```

---

## 5. Assumptions

*   **Data Partitioning:** MNIST/CIFAR-10 are split IID. FEMNIST uses its natural, non-IID writer split.
*   **Hardware:** All clients are assumed to have identical computational resources. The round time is dictated by the slowest client (though they are identical here).
*   **Communication:** Network latency, bandwidth, and model aggregation time at the server are explicitly **ignored**.

---

## 6. References

1.  McMahan, H. B., et al. (2017). "Communication-Efficient Learning of Deep Networks from Decentralized Data." AISTATS.
2.  LEAF Benchmark Suite. "FEMNIST dataset." GitHub.
3.  Flower Documentation. `flwr_datasets` usage for FEMNIST.
4.  Torchvision Model Zoo. ResNet-18.