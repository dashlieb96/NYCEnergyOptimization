"""
Battery Dispatch Optimization with Demand Charges

This module compares:
1. Heuristic (rule-based) battery dispatch
2. Optimized dispatch using convex optimization (CVXPY)

Both approaches include Con Edison demand charges in addition to energy costs.

The key insight: Standard energy arbitrage optimization ignores demand charges,
which can dominate costs for commercial buildings. A single 15-minute spike
can set your demand charge for the entire month.
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt


@dataclass
class BatteryParams:
    """Battery system parameters."""
    capacity_kwh: float = 1200      # Total capacity (kWh)
    min_soc_kwh: float = 120        # Minimum SOC (kWh) - 10%
    max_soc_kwh: float = 1200       # Maximum SOC (kWh)
    max_charge_kw: float = 300      # Max charging power (kW)
    max_discharge_kw: float = 300   # Max discharging power (kW)
    charge_efficiency: float = 0.9  # Charging efficiency
    discharge_efficiency: float = 0.9  # Discharging efficiency
    initial_soc_kwh: float = 600    # Initial SOC (kWh)


@dataclass
class DemandChargeParams:
    """Con Edison demand charge parameters (Standby Rate)."""
    # Contract demand charge ($/kW/month)
    contract_demand_rate: float = 8.50
    contract_demand_kw: float = 400  # Contracted standby capacity

    # Daily as-used demand charge ($/kW/day) - summer
    daily_demand_rate_summer: float = 0.85
    daily_demand_rate_winter: float = 0.45

    # Demand window: 8 AM - 10 PM (hours 8-21 inclusive)
    demand_window_start: int = 8
    demand_window_end: int = 22  # exclusive

    # Rider Q 4-hour window (optional)
    rider_q_start: int = 14
    rider_q_end: int = 18


def load_data(filepath: str = 'test2.xlsx') -> Dict:
    """Load data from the Excel file."""
    xlsx = pd.ExcelFile(filepath)

    # Energy prices ($/kWh)
    ep_buy = pd.read_excel(xlsx, sheet_name='ep_b', header=None).iloc[0].astype(float).values
    ep_sell = pd.read_excel(xlsx, sheet_name='ep_s', header=None).iloc[0].astype(float).values

    # Renewable generation (kW) - this is PV
    pv_generation = pd.read_excel(xlsx, sheet_name='rg', header=None).iloc[0].astype(float).values

    # Energy demand - from RESULTS sheet if available
    results = pd.read_excel(xlsx, sheet_name='RESULTS')
    edem_row = results[results.iloc[:, 0] == 'EDEM']
    if len(edem_row) > 0:
        demand = edem_row.iloc[0, 1:].astype(float).values
    else:
        # Fall back to 24-hour profile repeated
        ed_24 = pd.read_excel(xlsx, sheet_name='ed', header=None).iloc[0].astype(float).values
        demand = np.tile(ed_24, 365)[:len(pv_generation)]

    # Battery parameters
    battery = BatteryParams(
        capacity_kwh=float(pd.read_excel(xlsx, sheet_name='emax', header=None).iloc[0, 0]),
        min_soc_kwh=float(pd.read_excel(xlsx, sheet_name='emin', header=None).iloc[0, 0]),
        max_soc_kwh=float(pd.read_excel(xlsx, sheet_name='emax', header=None).iloc[0, 0]),
        max_charge_kw=float(pd.read_excel(xlsx, sheet_name='pch', header=None).iloc[0, 0]),
        max_discharge_kw=float(pd.read_excel(xlsx, sheet_name='pdis', header=None).iloc[0, 0]),
        charge_efficiency=float(pd.read_excel(xlsx, sheet_name='nch', header=None).iloc[0, 0]),
        discharge_efficiency=float(pd.read_excel(xlsx, sheet_name='ndis', header=None).iloc[0, 0]),
        initial_soc_kwh=float(pd.read_excel(xlsx, sheet_name='einit', header=None).iloc[0, 0]),
    )

    # Create timestamps (hourly, starting Jan 1)
    start = datetime(2024, 1, 1)
    timestamps = [start + timedelta(hours=i) for i in range(len(pv_generation))]

    return {
        'timestamps': timestamps,
        'demand': demand,  # Building load (kW)
        'pv_generation': pv_generation,  # PV output (kW)
        'price_buy': ep_buy,  # $/kWh to buy from grid
        'price_sell': ep_sell,  # $/kWh to sell to grid
        'battery': battery,
        'n_hours': len(pv_generation),
    }


def get_demand_window_mask(timestamps, params: DemandChargeParams) -> np.ndarray:
    """Return boolean mask for hours within demand charge window."""
    hours = np.array([t.hour for t in timestamps])
    return (hours >= params.demand_window_start) & (hours < params.demand_window_end)


def heuristic_dispatch(
    data: Dict,
    demand_params: DemandChargeParams = None,
    peak_target_kw: float = 300,
) -> Dict:
    """
    Heuristic (rule-based) battery dispatch strategy.

    Strategy:
    1. During demand window (8 AM - 10 PM): discharge to cap grid demand at peak_target
    2. Outside demand window: charge from cheap grid or excess PV
    3. Always charge from excess PV when available

    Args:
        data: Dictionary with load, PV, prices, battery params
        demand_params: Demand charge parameters
        peak_target_kw: Target maximum grid demand (kW)

    Returns:
        Dictionary with dispatch results
    """
    if demand_params is None:
        demand_params = DemandChargeParams()

    battery = data['battery']
    n = data['n_hours']

    demand = data['demand']
    pv = data['pv_generation']
    price_buy = data['price_buy']

    # Initialize arrays
    soc = np.zeros(n)
    charge = np.zeros(n)  # Positive = charging
    discharge = np.zeros(n)  # Positive = discharging
    grid_buy = np.zeros(n)
    grid_sell = np.zeros(n)

    # Get demand window mask
    window_mask = get_demand_window_mask(data['timestamps'], demand_params)

    # Initial SOC
    current_soc = battery.initial_soc_kwh

    for t in range(n):
        hour = data['timestamps'][t].hour

        # Net load before battery
        net_load = demand[t] - pv[t]

        # Decision logic
        if net_load < 0:
            # Excess PV - charge battery
            excess = -net_load
            max_charge = min(
                battery.max_charge_kw,
                (battery.max_soc_kwh - current_soc) / battery.charge_efficiency
            )
            charge[t] = min(excess, max_charge)
            current_soc += charge[t] * battery.charge_efficiency

            # Sell remaining excess
            remaining_excess = excess - charge[t]
            grid_sell[t] = remaining_excess

        elif window_mask[t]:
            # During demand window - peak shaving
            if net_load > peak_target_kw:
                # Need to discharge to reduce grid demand
                needed = net_load - peak_target_kw
                max_discharge = min(
                    battery.max_discharge_kw,
                    (current_soc - battery.min_soc_kwh) * battery.discharge_efficiency
                )
                discharge[t] = min(needed, max_discharge)
                current_soc -= discharge[t] / battery.discharge_efficiency

                # Buy remaining from grid
                grid_buy[t] = net_load - discharge[t]
            else:
                # Demand already below target
                grid_buy[t] = net_load
        else:
            # Outside demand window
            if price_buy[t] < 0.10 and current_soc < 0.8 * battery.max_soc_kwh:
                # Cheap power - charge
                max_charge = min(
                    battery.max_charge_kw,
                    (battery.max_soc_kwh - current_soc) / battery.charge_efficiency
                )
                charge[t] = min(100, max_charge)  # Moderate charging
                current_soc += charge[t] * battery.charge_efficiency
                grid_buy[t] = net_load + charge[t]
            else:
                grid_buy[t] = net_load

        # Ensure non-negative and record SOC
        grid_buy[t] = max(0, grid_buy[t])
        soc[t] = current_soc

    return {
        'soc': soc,
        'charge': charge,
        'discharge': discharge,
        'grid_buy': grid_buy,
        'grid_sell': grid_sell,
        'net_grid_demand': grid_buy,  # For demand charge calculation
    }


def optimize_dispatch(
    data: Dict,
    demand_params: DemandChargeParams = None,
    time_slice: Tuple[int, int] = None,
    demand_charge_weight: float = 1.0,
    verbose: bool = True,
) -> Dict:
    """
    Optimal battery dispatch using convex optimization (CVXPY).

    The optimization minimizes:
        Energy Cost + Demand Charge Cost

    Where:
        Energy Cost = sum(price_buy * grid_buy - price_sell * grid_sell)
        Demand Charge = daily_rate * max(grid_buy during demand window)

    The max() is handled by introducing a variable P_peak and constraining
    all grid_buy values to be <= P_peak.

    Args:
        data: Dictionary with load, PV, prices, battery params
        demand_params: Demand charge parameters
        time_slice: Optional (start, end) tuple to optimize subset of hours
        demand_charge_weight: Weight for demand charges vs energy (default 1.0)
        verbose: Print solver status

    Returns:
        Dictionary with optimal dispatch results
    """
    if demand_params is None:
        demand_params = DemandChargeParams()

    battery = data['battery']

    # Time slice
    if time_slice is None:
        t_start, t_end = 0, data['n_hours']
    else:
        t_start, t_end = time_slice

    n = t_end - t_start

    # Extract data for this slice
    demand = data['demand'][t_start:t_end]
    pv = data['pv_generation'][t_start:t_end]
    price_buy = data['price_buy'][t_start:t_end]
    price_sell = data['price_sell'][t_start:t_end]
    timestamps = data['timestamps'][t_start:t_end]

    # Get demand window mask
    window_mask = get_demand_window_mask(timestamps, demand_params)

    # =====================
    # CVXPY OPTIMIZATION
    # =====================

    # Decision variables
    charge = cp.Variable(n, nonneg=True)      # Battery charging (kW)
    discharge = cp.Variable(n, nonneg=True)   # Battery discharging (kW)
    grid_buy = cp.Variable(n, nonneg=True)    # Grid purchase (kW)
    grid_sell = cp.Variable(n, nonneg=True)   # Grid export (kW)
    soc = cp.Variable(n)                       # State of charge (kWh)
    peak_demand = cp.Variable(nonneg=True)     # Peak demand variable for minimax

    # Constraints
    constraints = []

    # Energy balance: demand + charge + sell = pv + discharge + buy
    constraints.append(
        demand + charge + grid_sell == pv + discharge + grid_buy
    )

    # Battery SOC dynamics
    # SOC[t] = SOC[t-1] + charge*eff - discharge/eff
    for t in range(n):
        if t == 0:
            prev_soc = battery.initial_soc_kwh
        else:
            prev_soc = soc[t-1]

        constraints.append(
            soc[t] == prev_soc + charge[t] * battery.charge_efficiency
                              - discharge[t] / battery.discharge_efficiency
        )

    # SOC limits
    constraints.append(soc >= battery.min_soc_kwh)
    constraints.append(soc <= battery.max_soc_kwh)

    # Power limits
    constraints.append(charge <= battery.max_charge_kw)
    constraints.append(discharge <= battery.max_discharge_kw)

    # Peak demand constraint (minimax formulation)
    # For all t in demand window: grid_buy[t] <= peak_demand
    for t in range(n):
        if window_mask[t]:
            constraints.append(grid_buy[t] <= peak_demand)

    # =====================
    # OBJECTIVE FUNCTION
    # =====================

    # Energy cost ($/hour, summed)
    energy_cost = cp.sum(cp.multiply(price_buy, grid_buy)) - cp.sum(cp.multiply(price_sell, grid_sell))

    # Demand charge cost
    # For a single day: daily_rate * peak
    # For multiple days: we use the overall peak (conservative)
    n_days = max(1, n // 24)

    # Determine season (summer if any day is in June-Sept)
    months = [t.month for t in timestamps]
    is_summer = any(m in [6, 7, 8, 9] for m in months)
    daily_rate = demand_params.daily_demand_rate_summer if is_summer else demand_params.daily_demand_rate_winter

    demand_charge_cost = daily_rate * peak_demand * n_days

    # Total objective
    objective = cp.Minimize(energy_cost + demand_charge_weight * demand_charge_cost)

    # Solve
    problem = cp.Problem(objective, constraints)

    try:
        problem.solve(solver=cp.ECOS, verbose=False)

        if problem.status not in ['optimal', 'optimal_inaccurate']:
            if verbose:
                print(f"Warning: Solver status = {problem.status}")
            # Try another solver
            problem.solve(solver=cp.SCS, verbose=False)
    except Exception as e:
        if verbose:
            print(f"Solver error: {e}")
        problem.solve(solver=cp.SCS, verbose=False)

    if verbose:
        print(f"Optimization status: {problem.status}")
        print(f"Optimal cost: ${problem.value:.2f}")

    # Extract results
    return {
        'soc': soc.value if soc.value is not None else np.zeros(n),
        'charge': charge.value if charge.value is not None else np.zeros(n),
        'discharge': discharge.value if discharge.value is not None else np.zeros(n),
        'grid_buy': grid_buy.value if grid_buy.value is not None else np.zeros(n),
        'grid_sell': grid_sell.value if grid_sell.value is not None else np.zeros(n),
        'peak_demand': peak_demand.value if peak_demand.value is not None else 0,
        'energy_cost': energy_cost.value if energy_cost.value is not None else 0,
        'demand_charge_cost': demand_charge_cost.value if demand_charge_cost.value is not None else 0,
        'total_cost': problem.value if problem.value is not None else 0,
        'status': problem.status,
        'net_grid_demand': grid_buy.value if grid_buy.value is not None else np.zeros(n),
    }


def calculate_costs(
    grid_buy: np.ndarray,
    grid_sell: np.ndarray,
    price_buy: np.ndarray,
    price_sell: np.ndarray,
    timestamps: list,
    demand_params: DemandChargeParams = None,
) -> Dict:
    """
    Calculate energy and demand costs from dispatch results.

    Returns breakdown of all cost components.
    """
    if demand_params is None:
        demand_params = DemandChargeParams()

    n = len(grid_buy)

    # Energy costs
    energy_buy_cost = np.sum(price_buy * grid_buy)
    energy_sell_revenue = np.sum(price_sell * grid_sell)
    net_energy_cost = energy_buy_cost - energy_sell_revenue

    # Demand charges - need to find peak in each day's demand window
    window_mask = get_demand_window_mask(timestamps, demand_params)

    # Group by day
    dates = [t.date() for t in timestamps]
    unique_dates = sorted(set(dates))

    daily_peaks = []
    for date in unique_dates:
        day_mask = np.array([d == date for d in dates])
        combined_mask = day_mask & window_mask
        if np.any(combined_mask):
            daily_peak = np.max(grid_buy[combined_mask])
            daily_peaks.append(daily_peak)

    # Determine season
    months = [t.month for t in timestamps]
    is_summer = any(m in [6, 7, 8, 9] for m in months)
    daily_rate = demand_params.daily_demand_rate_summer if is_summer else demand_params.daily_demand_rate_winter

    # Total demand charges
    demand_charge = sum(daily_rate * peak for peak in daily_peaks)

    # Contract demand (prorated)
    n_days = len(unique_dates)
    contract_charge = demand_params.contract_demand_rate * demand_params.contract_demand_kw * n_days / 30

    return {
        'energy_buy_cost': energy_buy_cost,
        'energy_sell_revenue': energy_sell_revenue,
        'net_energy_cost': net_energy_cost,
        'demand_charge': demand_charge,
        'contract_charge': contract_charge,
        'total_cost': net_energy_cost + demand_charge + contract_charge,
        'daily_peaks': daily_peaks,
        'max_peak': max(daily_peaks) if daily_peaks else 0,
        'n_days': n_days,
    }


def compare_strategies(
    data: Dict,
    time_slice: Tuple[int, int] = None,
    heuristic_target: float = 300,
) -> Dict:
    """
    Compare heuristic vs optimized dispatch strategies.

    Args:
        data: Input data dictionary
        time_slice: Optional (start, end) to analyze subset
        heuristic_target: Peak target for heuristic strategy

    Returns:
        Comparison results dictionary
    """
    demand_params = DemandChargeParams()

    if time_slice is None:
        t_start, t_end = 0, min(data['n_hours'], 168)  # Default to 1 week
    else:
        t_start, t_end = time_slice

    n = t_end - t_start

    # Slice data
    sliced_data = {
        'timestamps': data['timestamps'][t_start:t_end],
        'demand': data['demand'][t_start:t_end],
        'pv_generation': data['pv_generation'][t_start:t_end],
        'price_buy': data['price_buy'][t_start:t_end],
        'price_sell': data['price_sell'][t_start:t_end],
        'battery': data['battery'],
        'n_hours': n,
    }

    print("=" * 70)
    print("COMPARING DISPATCH STRATEGIES")
    print("=" * 70)
    print(f"Time period: {sliced_data['timestamps'][0]} to {sliced_data['timestamps'][-1]}")
    print(f"Hours: {n} ({n/24:.1f} days)")
    print()

    # 1. No battery baseline
    print("1. Calculating NO BATTERY baseline...")
    no_battery_buy = np.maximum(sliced_data['demand'] - sliced_data['pv_generation'], 0)
    no_battery_sell = np.maximum(sliced_data['pv_generation'] - sliced_data['demand'], 0)

    no_battery_costs = calculate_costs(
        no_battery_buy, no_battery_sell,
        sliced_data['price_buy'], sliced_data['price_sell'],
        sliced_data['timestamps'], demand_params
    )

    # 2. Heuristic dispatch
    print(f"2. Running HEURISTIC dispatch (target={heuristic_target} kW)...")
    heuristic_result = heuristic_dispatch(sliced_data, demand_params, heuristic_target)

    heuristic_costs = calculate_costs(
        heuristic_result['grid_buy'], heuristic_result['grid_sell'],
        sliced_data['price_buy'], sliced_data['price_sell'],
        sliced_data['timestamps'], demand_params
    )

    # 3. Optimized dispatch
    print("3. Running OPTIMIZED dispatch (this may take a moment)...")
    optimized_result = optimize_dispatch(sliced_data, demand_params, verbose=True)

    optimized_costs = calculate_costs(
        optimized_result['grid_buy'], optimized_result['grid_sell'],
        sliced_data['price_buy'], sliced_data['price_sell'],
        sliced_data['timestamps'], demand_params
    )

    # Print comparison
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)

    print("\n{:<25} {:>15} {:>15} {:>15}".format(
        "Metric", "No Battery", "Heuristic", "Optimized"))
    print("-" * 70)

    print("{:<25} {:>15.2f} {:>15.2f} {:>15.2f}".format(
        "Energy Buy Cost ($)",
        no_battery_costs['energy_buy_cost'],
        heuristic_costs['energy_buy_cost'],
        optimized_costs['energy_buy_cost']))

    print("{:<25} {:>15.2f} {:>15.2f} {:>15.2f}".format(
        "Energy Sell Revenue ($)",
        no_battery_costs['energy_sell_revenue'],
        heuristic_costs['energy_sell_revenue'],
        optimized_costs['energy_sell_revenue']))

    print("{:<25} {:>15.2f} {:>15.2f} {:>15.2f}".format(
        "Net Energy Cost ($)",
        no_battery_costs['net_energy_cost'],
        heuristic_costs['net_energy_cost'],
        optimized_costs['net_energy_cost']))

    print("-" * 70)

    print("{:<25} {:>15.2f} {:>15.2f} {:>15.2f}".format(
        "Peak Demand (kW)",
        no_battery_costs['max_peak'],
        heuristic_costs['max_peak'],
        optimized_costs['max_peak']))

    print("{:<25} {:>15.2f} {:>15.2f} {:>15.2f}".format(
        "Demand Charges ($)",
        no_battery_costs['demand_charge'],
        heuristic_costs['demand_charge'],
        optimized_costs['demand_charge']))

    print("{:<25} {:>15.2f} {:>15.2f} {:>15.2f}".format(
        "Contract Charges ($)",
        no_battery_costs['contract_charge'],
        heuristic_costs['contract_charge'],
        optimized_costs['contract_charge']))

    print("-" * 70)

    print("{:<25} {:>15.2f} {:>15.2f} {:>15.2f}".format(
        "TOTAL COST ($)",
        no_battery_costs['total_cost'],
        heuristic_costs['total_cost'],
        optimized_costs['total_cost']))

    print("\n" + "-" * 70)
    print("SAVINGS vs NO BATTERY")
    print("-" * 70)

    heuristic_savings = no_battery_costs['total_cost'] - heuristic_costs['total_cost']
    optimized_savings = no_battery_costs['total_cost'] - optimized_costs['total_cost']

    print(f"Heuristic savings:  ${heuristic_savings:,.2f} ({100*heuristic_savings/no_battery_costs['total_cost']:.1f}%)")
    print(f"Optimized savings:  ${optimized_savings:,.2f} ({100*optimized_savings/no_battery_costs['total_cost']:.1f}%)")
    print(f"Optimizer vs Heuristic: ${optimized_savings - heuristic_savings:,.2f} additional savings")

    return {
        'sliced_data': sliced_data,
        'no_battery': {'costs': no_battery_costs, 'grid_buy': no_battery_buy, 'grid_sell': no_battery_sell},
        'heuristic': {'result': heuristic_result, 'costs': heuristic_costs},
        'optimized': {'result': optimized_result, 'costs': optimized_costs},
    }


def plot_comparison(comparison: Dict, save_path: str = 'dispatch_comparison.png'):
    """Create visualization comparing dispatch strategies."""

    data = comparison['sliced_data']
    n = data['n_hours']
    hours = np.arange(n)

    fig, axes = plt.subplots(5, 1, figsize=(16, 18), sharex=True)

    # Plot 1: Load and PV
    ax1 = axes[0]
    ax1.fill_between(hours, 0, data['demand'], alpha=0.5, color='gray', label='Building Demand')
    ax1.fill_between(hours, 0, data['pv_generation'], alpha=0.5, color='gold', label='PV Generation')
    ax1.set_ylabel('Power (kW)')
    ax1.set_title('Building Load and PV Generation', fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Grid demand comparison
    ax2 = axes[1]

    # Shade demand windows
    for day in range(n // 24 + 1):
        start = day * 24 + 8
        end = day * 24 + 22
        if start < n:
            ax2.axvspan(start, min(end, n), alpha=0.1, color='red')

    ax2.plot(hours, comparison['no_battery']['grid_buy'],
             color='gray', linewidth=1.5, alpha=0.7, label='No Battery')
    ax2.plot(hours, comparison['heuristic']['result']['grid_buy'],
             color='blue', linewidth=1.5, label='Heuristic')
    ax2.plot(hours, comparison['optimized']['result']['grid_buy'],
             color='green', linewidth=2, label='Optimized')

    # Mark peaks
    no_batt_peak = comparison['no_battery']['costs']['max_peak']
    heur_peak = comparison['heuristic']['costs']['max_peak']
    opt_peak = comparison['optimized']['costs']['max_peak']

    ax2.axhline(y=no_batt_peak, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(y=heur_peak, color='blue', linestyle='--', alpha=0.5)
    ax2.axhline(y=opt_peak, color='green', linestyle='--', alpha=0.5)

    ax2.set_ylabel('Grid Demand (kW)')
    ax2.set_title(f'Grid Demand Comparison (Peaks: No Batt={no_batt_peak:.0f}, Heur={heur_peak:.0f}, Opt={opt_peak:.0f} kW)',
                  fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Battery SOC comparison
    ax3 = axes[2]
    ax3.plot(hours, comparison['heuristic']['result']['soc'],
             color='blue', linewidth=1.5, label='Heuristic SOC')
    ax3.plot(hours, comparison['optimized']['result']['soc'],
             color='green', linewidth=2, label='Optimized SOC')
    ax3.axhline(y=data['battery'].max_soc_kwh, color='gray', linestyle='--', alpha=0.5, label='Max/Min SOC')
    ax3.axhline(y=data['battery'].min_soc_kwh, color='gray', linestyle='--', alpha=0.5)
    ax3.set_ylabel('Battery SOC (kWh)')
    ax3.set_title('Battery State of Charge', fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Battery power (charge/discharge)
    ax4 = axes[3]

    heur_power = comparison['heuristic']['result']['discharge'] - comparison['heuristic']['result']['charge']
    opt_power = comparison['optimized']['result']['discharge'] - comparison['optimized']['result']['charge']

    ax4.plot(hours, heur_power, color='blue', linewidth=1.5, alpha=0.7, label='Heuristic')
    ax4.plot(hours, opt_power, color='green', linewidth=2, label='Optimized')
    ax4.axhline(y=0, color='black', linewidth=0.5)
    ax4.set_ylabel('Battery Power (kW)\n(+discharge, -charge)')
    ax4.set_title('Battery Dispatch', fontweight='bold')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)

    # Plot 5: Energy prices
    ax5 = axes[4]
    ax5.plot(hours, data['price_buy'], color='red', linewidth=1, label='Buy Price')
    ax5.plot(hours, data['price_sell'], color='green', linewidth=1, label='Sell Price')
    ax5.axhline(y=0, color='black', linewidth=0.5)
    ax5.set_ylabel('Price ($/kWh)')
    ax5.set_xlabel('Hour')
    ax5.set_title('Energy Prices', fontweight='bold')
    ax5.legend(loc='upper right')
    ax5.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nPlot saved to: {save_path}")


def main():
    """Main function to run the comparison."""

    print("Loading data from test2.xlsx...")
    data = load_data('test2.xlsx')

    print(f"\nData loaded:")
    print(f"  Total hours: {data['n_hours']} ({data['n_hours']/24:.0f} days)")
    print(f"  Peak building demand: {data['demand'].max():.0f} kW")
    print(f"  Peak PV generation: {data['pv_generation'].max():.0f} kW")
    print(f"  Battery: {data['battery'].capacity_kwh} kWh, {data['battery'].max_discharge_kw} kW")

    # Find an interesting week with high demand/price spikes
    # Let's try a summer week (July)
    july_start = 24 * 182  # ~July 1
    july_end = july_start + 24 * 7

    print(f"\nAnalyzing week starting at hour {july_start} (July)...")

    # Run comparison
    comparison = compare_strategies(
        data,
        time_slice=(july_start, july_end),
        heuristic_target=300,
    )

    # Create visualization
    plot_comparison(comparison, 'dispatch_comparison.png')

    return comparison


if __name__ == "__main__":
    comparison = main()
