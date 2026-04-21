# GPU-Accelerated Monte Carlo Simulation for Financial Modeling

An enterprise-grade, High-Performance Computing (HPC) pipeline that leverages Apple Silicon's Unified Memory Architecture to simulate millions of stock price paths using Geometric Brownian Motion (GBM). The engine prices European Call Options and calculates 99% Value at Risk (VaR) in seconds.

## 🚀 Project Highlights
* **Hardware Acceleration:** Offloads parallel mathematical computations to the Apple M4 GPU using PyTorch's Metal Performance Shaders (MPS).
* **OOM-Safe Architecture:** Implements chunk-based batching and aggressive Python/MPS garbage collection to successfully process 10,000,000+ paths without triggering Out-Of-Memory swap crashes.
* **Three Hardware Layers:** Features benchmarking across a single-threaded CPU (Serial), Symmetric Multiprocessing (Parallel CPU via `joblib`), and Massively Parallel SIMD execution (GPU).
* **Quantitative Finance:** Evaluates final price distributions to calculate European Option premiums and tail-risk VaR directly on the GPU.

## 📂 Repository Structure

```text
gpu_monte_carlo_hpc/
├── requirements.txt
├── README.md
├── results/                  # Generated benchmark graphs (Execution Time, Speedup)
└── src/
    ├── gbm_engine.py         # The core mathematical matrix generators
    ├── risk_metrics.py       # Option pricing and VaR calculations
    ├── main.py               # Production CLI pipeline with memory management
    └── gbm_benchmark.py      # Hardware benchmarking and visualization script
```

## ⚙️ Installation
Clone the repository and navigate to the project root.

Create a virtual environment (Python 3.12 recommended for Apple Silicon stability):

```Bash
python3.12 -m venv .venv
source .venv/bin/activate
```
Install the required dependencies:
```
Bash
pip install --upgrade pip
pip install -r requirements.txt
```
💻 Usage
1. Run the Production Pipeline (main.py)
Use the Command Line Interface (CLI) to run the simulation engine. The pipeline automatically chunks the workload into memory-safe batches.

Bash
python src/main.py --sims 10000000 --batch 500000 --strike 110 --vol 0.20
CLI Arguments:

--sims: Total number of paths to simulate (default: 10000000)

--batch: Maximum batch size per GPU dispatch to prevent OOM errors (default: 500000)

--s0: Initial stock price (default: 100.0)

--strike: Strike price for the European Call Option (default: 110.0)

--time: Time horizon in years (default: 1.0)

--rate: Risk-free interest rate (default: 0.05)

--vol: Volatility / Sigma (default: 0.20)

--steps: Number of time steps / trading days (default: 252)

2. Run the Hardware Benchmarks (gbm_benchmark.py)
To test the "Cold Start" anomaly, Amdahl's Law, and the crossover point between Compute-Bound and Memory-Bound execution, run the benchmark script:

```
Bash
python src/gbm_benchmark.py
```
This will race the Serial CPU, Parallel CPU, and MPS GPU against each other at varying workloads (10k to 500k) and automatically save logarithmic graphs to the results/ folder.

## 📊 Benchmark Results & Findings
On an Apple M4 Chip (10 CPU Cores), this project successfully demonstrated:

The "Sweet Spot": At a workload of 100,000 simulations, the MPS GPU hit peak efficiency, executing the mathematics ~15.6x faster than the Serial baseline.

The IPC Bottleneck: At smaller workloads, the Inter-Process Communication (IPC) overhead of spinning up 10 CPU cores made the Parallel CPU slower than a single-threaded execution.

Memory Limits: At workloads exceeding 500,000 continuous unbatched paths, hardware memory latency dominates compute speed, proving the absolute necessity of the batching pipeline built into main.py.
