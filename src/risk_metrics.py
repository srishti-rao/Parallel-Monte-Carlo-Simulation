import torch

def calculate_european_call_mps(paths, strike_price, risk_free_rate, time_horizon):
    """
    Calculates the European Call Option price using the simulated paths on the GPU.
    Formula: C = e^{-rT} * E[max(S_T - K, 0)]
    """
    # Extract the final terminal prices (last column of the paths matrix)
    terminal_prices = paths[:, -1]
    
    # Calculate payoff: max(S_T - K, 0)
    # Using torch.clamp is a highly optimized, vectorized operation on the GPU
    payoffs = torch.clamp(terminal_prices - strike_price, min=0.0)
    
    # Expected value (mean of payoffs)
    expected_payoff = torch.mean(payoffs)
    
    # Discount back to present value
    discount_factor = torch.exp(torch.tensor(-risk_free_rate * time_horizon, device=paths.device))
    call_price = discount_factor * expected_payoff
    
    return call_price.item()

def calculate_value_at_risk_mps(paths, confidence_level=0.99):
    """
    Calculates the Value at Risk (VaR) at a given confidence level.
    Determines the maximum expected loss over the time horizon.
    """
    initial_price = paths[0, 0]
    terminal_prices = paths[:, -1]
    
    # Calculate percentage returns
    returns = (terminal_prices - initial_price) / initial_price
    
    # Sort the returns to find the percentile
    sorted_returns, _ = torch.sort(returns)
    
    # Find the index for the tail risk (e.g., the worst 1% of outcomes)
    index = int((1.0 - confidence_level) * len(sorted_returns))
    
    var_percentage = sorted_returns[index].item()
    
    # VaR is typically expressed as a positive number representing the loss amount
    return -var_percentage