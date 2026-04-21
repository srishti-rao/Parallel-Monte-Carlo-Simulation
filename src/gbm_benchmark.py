import time
import os
import gc
import torch
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

from gbm_engine import simulate_gbm_serial, simulate_gbm_parallel_cpu, simulate_gbm_mps

def run_benchmarks():
    print("Starting HPC Monte Carlo Benchmarks on Mac M4...")
    print(f"CPU Cores detected: {multiprocessing.cpu_count()}")
    print("-" * 50)
    
    S0, T, r, sigma, steps = 100.0, 1.0, 0.05, 0.2, 252
    os.makedirs('results', exist_ok=True)
    
    # Adjusted scales for MacBook Air Memory Limits (No SSD Swapping)
    sim_counts = [10_000, 50_000, 100_000, 250_000, 500_000]
    
    times_serial = []
    times_par_cpu = []
    times_gpu = []
    
    print("Warming up processors and libraries (Cold Start elimination)...")
    _ = simulate_gbm_serial(2000, S0, T, r, sigma, steps)
    _ = simulate_gbm_parallel_cpu(2000, S0, T, r, sigma, steps)
    _ = simulate_gbm_mps(2000, S0, T, r, sigma, steps)
    torch.mps.synchronize()
    print("Warmup complete. Beginning official benchmarks...\n")
    
    for n in sim_counts:
        print(f"Running {n:,} simulations...")
        
        # 1. Serial CPU
        t0 = time.perf_counter()
        _ = simulate_gbm_serial(n, S0, T, r, sigma, steps)
        t_serial = time.perf_counter() - t0
        times_serial.append(t_serial)
        print(f"  Serial CPU:      {t_serial:.4f} seconds")
        
        # 2. Parallel CPU
        t0 = time.perf_counter()
        _ = simulate_gbm_parallel_cpu(n, S0, T, r, sigma, steps)
        t_par_cpu = time.perf_counter() - t0
        times_par_cpu.append(t_par_cpu)
        print(f"  Parallel CPU:    {t_par_cpu:.4f} seconds")
        
        # 3. GPU (MPS)
        t0 = time.perf_counter()
        _ = simulate_gbm_mps(n, S0, T, r, sigma, steps)
        torch.mps.synchronize() 
        t_gpu = time.perf_counter() - t0
        times_gpu.append(t_gpu)
        print(f"  Parallel GPU:    {t_gpu:.4f} seconds")
        
        # Aggressive memory cleanup to prevent swap
        torch.mps.empty_cache()
        gc.collect()
        
        speedup_cpu = t_serial / t_par_cpu
        speedup_gpu = t_serial / t_gpu
        print(f"  -> Achieved {speedup_cpu:.2f}x speedup using Parallel CPU.")
        print(f"  -> Achieved {speedup_gpu:.2f}x speedup using MPS GPU.\n")

    # Generate Graphs
    plt.figure(figsize=(10, 6))
    plt.plot(sim_counts, times_serial, marker='o', label='Serial CPU', linewidth=2)
    plt.plot(sim_counts, times_par_cpu, marker='s', label='Parallel CPU', linewidth=2)
    plt.plot(sim_counts, times_gpu, marker='^', label='Parallel GPU (MPS)', linewidth=2)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of Simulations (Paths)')
    plt.ylabel('Execution Time (Seconds)')
    plt.title('Execution Time vs. Workload Size (Lower is Better)')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/execution_time.png')
    
    speedup_cpu_arr = np.array(times_serial) / np.array(times_par_cpu)
    speedup_gpu_arr = np.array(times_serial) / np.array(times_gpu)
    
    plt.figure(figsize=(10, 6))
    plt.plot(sim_counts, speedup_cpu_arr, marker='s', label='Parallel CPU Speedup', linewidth=2)
    plt.plot(sim_counts, speedup_gpu_arr, marker='^', label='GPU (MPS) Speedup', linewidth=2)
    plt.axhline(y=1, color='r', linestyle='--', label='Baseline (Serial)')
    plt.xscale('log')
    plt.xlabel('Number of Simulations (Paths)')
    plt.ylabel('Speedup Factor (Higher is Better)')
    plt.title('Hardware Acceleration Speedup vs. Serial Baseline')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/speedup_factor.png')
    print("Benchmarks complete! Plots saved to the 'results' directory.")

if __name__ == "__main__":
    run_benchmarks()