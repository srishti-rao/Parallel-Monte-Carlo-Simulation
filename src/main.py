import argparse
import logging
import time
import torch
import gc  # NEW: Import Python's garbage collector
from gbm_engine import simulate_gbm_mps
from risk_metrics import calculate_european_call_mps, calculate_value_at_risk_mps

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def run_hpc_pipeline(total_sims, batch_size, S0, K, T, r, sigma, steps):
    logger.info(f"Starting GPU Pipeline: {total_sims:,} total simulations.")
    logger.info(f"Hardware: Apple Silicon MPS")
    
    device = torch.device("mps")
    
    # Warmup
    _ = simulate_gbm_mps(100, S0, T, r, sigma, steps)
    torch.mps.synchronize()
    
    batches = total_sims // batch_size
    remainder = total_sims % batch_size
    
    total_call_value = 0.0
    all_terminal_prices = []
    
    start_time = time.perf_counter()
    
    for i in range(batches):
        paths = simulate_gbm_mps(batch_size, S0, T, r, sigma, steps)
        
        batch_call = calculate_european_call_mps(paths, K, r, T)
        total_call_value += batch_call * batch_size
        
        all_terminal_prices.append(paths[:, -1].clone())
        
        # ---------------------------------------------------------
        # NEW: Aggressive Memory Management
        # ---------------------------------------------------------
        del paths                   # Delete the massive 2GB tensor from Python
        torch.mps.empty_cache()     # Force PyTorch to release GPU memory back to OS
        gc.collect()                # Force Python to clean up unreferenced variables
        # ---------------------------------------------------------
        
        if (i + 1) % max(1, (batches // 10)) == 0:
            logger.info(f"Processed batch {i+1}/{batches}...")

    if remainder > 0:
        paths = simulate_gbm_mps(remainder, S0, T, r, sigma, steps)
        batch_call = calculate_european_call_mps(paths, K, r, T)
        total_call_value += batch_call * remainder
        all_terminal_prices.append(paths[:, -1].clone())
        del paths
        torch.mps.empty_cache()
        gc.collect()

    torch.mps.synchronize()
    execution_time = time.perf_counter() - start_time
    
    final_call_price = total_call_value / total_sims
    flat_terminals = torch.cat(all_terminal_prices)
    
    dummy_paths = torch.zeros((total_sims, 2), device=device)
    dummy_paths[:, 0] = S0
    dummy_paths[:, -1] = flat_terminals
    
    var_99 = calculate_value_at_risk_mps(dummy_paths, 0.99)
    
    logger.info(f"Pipeline completed in {execution_time:.4f} seconds.")
    logger.info("-" * 40)
    logger.info(f"European Call Price (K={K}): ${final_call_price:.4f}")
    logger.info(f"99% Value at Risk (VaR):     {var_99 * 100:.2f}% loss")
    logger.info("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sims", type=int, default=10_000_000)
    parser.add_argument("--batch", type=int, default=500_000) # Lowered default batch size
    parser.add_argument("--s0", type=float, default=100.0)
    parser.add_argument("--strike", type=float, default=110.0)
    parser.add_argument("--time", type=float, default=1.0)
    parser.add_argument("--rate", type=float, default=0.05)
    parser.add_argument("--vol", type=float, default=0.20)
    parser.add_argument("--steps", type=int, default=252)
    
    args = parser.parse_args()
    run_hpc_pipeline(args.sims, args.batch, args.s0, args.strike, args.time, args.rate, args.vol, args.steps)