import torch
import numpy as np
import multiprocessing
from joblib import Parallel, delayed

# ---------------------------------------------------------
# 1. Serial (Standard CPU) Implementation
# ---------------------------------------------------------
def simulate_gbm_serial(num_sims, S0, T, r, sigma, num_steps):
    dt = T / num_steps
    nu = (r - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)
    
    Z = np.random.standard_normal((num_sims, num_steps))
    increments = nu + vol * Z
    log_paths = np.cumsum(increments, axis=1)
    
    paths = S0 * np.exp(log_paths)
    paths = np.hstack((np.full((num_sims, 1), S0), paths))
    return paths

# ---------------------------------------------------------
# 2. Parallel CPU Implementation
# ---------------------------------------------------------
def _simulate_chunk(chunk_size, S0, T, r, sigma, num_steps):
    return simulate_gbm_serial(chunk_size, S0, T, r, sigma, num_steps)

def simulate_gbm_parallel_cpu(num_sims, S0, T, r, sigma, num_steps):
    num_cores = multiprocessing.cpu_count()
    chunk_size = num_sims // num_cores
    chunks = [chunk_size] * num_cores
    chunks[-1] += num_sims % num_cores
    
    results = Parallel(n_jobs=num_cores)(
        delayed(_simulate_chunk)(c, S0, T, r, sigma, num_steps) for c in chunks
    )
    return np.vstack(results)

# ---------------------------------------------------------
# 3. Parallel GPU (Mac M4 MPS) Implementation
# ---------------------------------------------------------
def simulate_gbm_mps(num_sims, S0, T, r, sigma, num_steps):
    device = torch.device("mps")
    dt = T / num_steps
    
    nu = torch.tensor((r - 0.5 * sigma**2) * dt, device=device, dtype=torch.float32)
    vol = torch.tensor(sigma * np.sqrt(dt), device=device, dtype=torch.float32)
    S0_tensor = torch.tensor(S0, device=device, dtype=torch.float32)
    
    Z = torch.randn((num_sims, num_steps), device=device, dtype=torch.float32)
    
    increments = nu + vol * Z
    log_paths = torch.cumsum(increments, dim=1)
    
    paths = S0_tensor * torch.exp(log_paths)
    
    start_prices = torch.full((num_sims, 1), S0_tensor, device=device, dtype=torch.float32)
    paths = torch.cat((start_prices, paths), dim=1)
    
    torch.mps.synchronize()
    return paths