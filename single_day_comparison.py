"""
Single 24-hour comparison: Heuristic vs Optimized Battery Dispatch
"""

import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


def load_single_day(filepath: str = 'test2.xlsx', day_index: int = 182):
    """
    Load a single day of data.

    day_index: 0 = Jan 1, 182 = July 1 (approx)
    """
    xlsx = pd.ExcelFile(filepath)

    start_hour = day_index * 24
    end_hour = start_hour + 24

    # Energy prices
    ep_buy = pd.read_excel(xlsx, sheet_name='ep_b', header=None).iloc[0].astype(float).values[start_hour:end_hour]
    ep_sell = pd.read_excel(xlsx, sheet_name='ep_s', header=None).iloc[0].astype(float).values[start_hour:end_hour]

    # PV generation
    pv = pd.read_excel(xlsx, sheet_name='rg', header=None).iloc[0].astype(float).values[start_hour:end_hour]

    # Building demand - from 'ed' sheet (24-hour template)
    # NOTE: This is synthetic test data, not real metered demand
    ed_template = pd.read_excel(xlsx, sheet_name='ed', header=None).iloc[0].astype(float).values
    demand = ed_template[:24]  # Use the 24-hour template

    # Replace any NaN with interpolation
    demand = np.nan_to_num(demand, nan=np.nanmean(demand))

    # Battery parameters
    battery = {
        'capacity': 1200,       # kWh
        'min_soc': 120,         # kWh
        'max_soc': 1200,        # kWh
        'max_charge': 300,      # kW
        'max_discharge': 300,   # kW
        'eff_charge': 0.9,
        'eff_discharge': 0.9,
        'initial_soc': 600,     # kWh
    }

    return {
        'hours': np.arange(24),
        'demand': demand,
        'pv': pv,
        'price_buy': ep_buy,
        'price_sell': ep_sell,
        'battery': battery,
        'date': datetime(2024, 1, 1) + timedelta(days=day_index),
    }


def heuristic_dispatch(data, peak_target=None, allow_export=False):
    """
    IMPROVED Rule-based heuristic dispatch.

    Strategy:
    1. First pass: scan day to find the "natural" peak (without battery)
    2. Set target = natural_peak - available_battery_energy / peak_hours
    3. Charge aggressively overnight and from excess PV
    4. Discharge during demand window to cap grid demand at target

    Args:
        allow_export: If False (default), no selling back to grid - excess PV is curtailed
    """
    n = 24
    battery = data['battery']
    demand = data['demand']
    pv = data['pv']
    price_buy = data['price_buy']

    # =====================
    # FIRST PASS: Analyze the day
    # =====================
    net_load = demand - pv  # Positive = need grid

    # Find peak in demand window (8-22)
    window_net_load = net_load.copy()
    window_net_load[:8] = 0
    window_net_load[22:] = 0
    natural_peak = np.max(np.maximum(window_net_load, 0))

    # Calculate achievable target based on battery capacity
    available_energy = (battery['initial_soc'] - battery['min_soc']) * battery['eff_discharge']

    # Estimate hours where we'll need to discharge
    threshold = natural_peak * 0.5
    peak_hours = np.sum((window_net_load > threshold) & (np.arange(24) >= 8) & (np.arange(24) < 22))
    peak_hours = max(peak_hours, 1)

    # Auto-calculate target if not provided
    if peak_target is None:
        max_shave_per_hour = battery['max_discharge']
        potential_reduction = min(available_energy / peak_hours, max_shave_per_hour)
        peak_target = max(natural_peak - potential_reduction, 100)

    print(f"  Heuristic: natural_peak={natural_peak:.0f} kW, target={peak_target:.0f} kW")

    # =====================
    # SECOND PASS: Dispatch
    # =====================
    soc = np.zeros(n)
    charge = np.zeros(n)
    discharge = np.zeros(n)
    grid_buy = np.zeros(n)
    grid_sell = np.zeros(n)
    curtailed = np.zeros(n)  # Track curtailed PV

    current_soc = battery['initial_soc']

    for t in range(n):
        hour = t
        current_net_load = net_load[t]

        # --- EXCESS PV (net_load < 0) ---
        if current_net_load < 0:
            excess = -current_net_load
            max_charge = min(
                battery['max_charge'],
                (battery['max_soc'] - current_soc) / battery['eff_charge']
            )
            charge[t] = min(excess, max_charge)
            current_soc += charge[t] * battery['eff_charge']

            remaining_excess = excess - charge[t]
            if allow_export:
                grid_sell[t] = remaining_excess
            else:
                curtailed[t] = remaining_excess  # Can't export - PV is curtailed

        # --- OVERNIGHT (before demand window) ---
        elif hour < 8:
            if price_buy[t] < 0.10 and current_soc < battery['max_soc'] * 0.95:
                max_charge = min(
                    battery['max_charge'],
                    (battery['max_soc'] - current_soc) / battery['eff_charge']
                )
                charge[t] = max_charge
                current_soc += charge[t] * battery['eff_charge']
                grid_buy[t] = current_net_load + charge[t]
            else:
                grid_buy[t] = max(0, current_net_load)

        # --- DEMAND WINDOW (8-22) ---
        elif 8 <= hour < 22:
            if current_net_load > peak_target:
                needed = current_net_load - peak_target
                max_discharge = min(
                    battery['max_discharge'],
                    (current_soc - battery['min_soc']) * battery['eff_discharge']
                )
                discharge[t] = min(needed, max_discharge)
                current_soc -= discharge[t] / battery['eff_discharge']
                grid_buy[t] = current_net_load - discharge[t]
            else:
                grid_buy[t] = max(0, current_net_load)

        # --- EVENING (after demand window) ---
        else:
            if price_buy[t] < 0.15 and current_soc < battery['max_soc'] * 0.9:
                max_charge = min(
                    battery['max_charge'],
                    (battery['max_soc'] - current_soc) / battery['eff_charge']
                )
                charge[t] = min(200, max_charge)
                current_soc += charge[t] * battery['eff_charge']
                grid_buy[t] = max(0, current_net_load) + charge[t]
            else:
                grid_buy[t] = max(0, current_net_load)

        grid_buy[t] = max(0, grid_buy[t])
        soc[t] = current_soc

    return {
        'soc': soc,
        'charge': charge,
        'discharge': discharge,
        'grid_buy': grid_buy,
        'grid_sell': grid_sell,
        'curtailed': curtailed,
        'peak_target': peak_target,
    }


def optimized_dispatch(data, demand_charge_rate=0.85, allow_export=False):
    """
    Optimal dispatch using CVXPY.

    Minimizes: energy_cost + demand_charge * peak_demand

    Args:
        allow_export: If False (default for NYC), no selling back to grid
    """
    n = 24
    battery = data['battery']
    demand = data['demand']
    pv = data['pv']
    price_buy = data['price_buy']
    price_sell = data['price_sell']

    # Decision variables
    charge = cp.Variable(n, nonneg=True)
    discharge = cp.Variable(n, nonneg=True)
    grid_buy = cp.Variable(n, nonneg=True)
    soc = cp.Variable(n)
    peak_demand = cp.Variable(nonneg=True)

    # Curtailment variable (excess PV that can't be stored or exported)
    curtailed = cp.Variable(n, nonneg=True)

    if allow_export:
        grid_sell = cp.Variable(n, nonneg=True)
    else:
        grid_sell = np.zeros(n)  # No export allowed

    constraints = []

    # Energy balance: demand + charge + sell + curtailed == pv + discharge + buy
    # Rearranged: what we need = what we have
    if allow_export:
        constraints.append(demand + charge + grid_sell + curtailed == pv + discharge + grid_buy)
    else:
        # No export: excess PV is either stored or curtailed
        constraints.append(demand + charge + curtailed == pv + discharge + grid_buy)

    # SOC dynamics
    for t in range(n):
        if t == 0:
            prev_soc = battery['initial_soc']
        else:
            prev_soc = soc[t-1]
        constraints.append(
            soc[t] == prev_soc + charge[t] * battery['eff_charge']
                             - discharge[t] / battery['eff_discharge']
        )

    # SOC limits
    constraints.append(soc >= battery['min_soc'])
    constraints.append(soc <= battery['max_soc'])

    # Power limits
    constraints.append(charge <= battery['max_charge'])
    constraints.append(discharge <= battery['max_discharge'])

    # Peak demand constraint (8 AM - 10 PM demand window)
    for t in range(n):
        if 8 <= t < 22:
            constraints.append(grid_buy[t] <= peak_demand)

    # Objective: minimize energy cost + demand charges
    energy_cost = cp.sum(cp.multiply(price_buy, grid_buy))
    if allow_export:
        energy_cost -= cp.sum(cp.multiply(price_sell, grid_sell))

    demand_cost = demand_charge_rate * peak_demand

    # Small penalty on curtailment to prefer storing over wasting
    curtail_penalty = 0.001 * cp.sum(curtailed)

    objective = cp.Minimize(energy_cost + demand_cost + curtail_penalty)

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS, verbose=False)

    return {
        'soc': soc.value,
        'charge': charge.value,
        'discharge': discharge.value,
        'grid_buy': grid_buy.value,
        'grid_sell': grid_sell if allow_export else np.zeros(n),
        'curtailed': curtailed.value,
        'peak_demand': peak_demand.value,
        'status': problem.status,
    }


def plot_single_day(data, heuristic, optimized):
    """Create detailed single-day comparison plot."""

    hours = data['hours']

    fig, axes = plt.subplots(6, 1, figsize=(14, 20))

    # Common x-axis formatting
    hour_labels = [f'{h:02d}:00' for h in range(24)]

    # =========================================
    # SUBPLOT 1: Building Demand (Raw Input)
    # =========================================
    ax1 = axes[0]
    ax1.bar(hours, data['demand'], color='steelblue', alpha=0.8, edgecolor='navy', linewidth=1)
    ax1.set_ylabel('Power (kW)', fontsize=11)
    ax1.set_title('① BUILDING DEMAND (from "ed" sheet - synthetic test data)',
                  fontsize=12, fontweight='bold', color='steelblue')
    ax1.set_xticks(hours)
    ax1.set_xticklabels(hour_labels, rotation=45, ha='right', fontsize=8)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_xlim(-0.5, 23.5)

    # Annotate
    ax1.annotate(f'Peak: {data["demand"].max():.0f} kW\nTotal: {data["demand"].sum():.0f} kWh',
                 xy=(0.02, 0.95), xycoords='axes fraction', fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # =========================================
    # SUBPLOT 2: PV Generation (Raw Input)
    # =========================================
    ax2 = axes[1]
    ax2.fill_between(hours, 0, data['pv'], color='gold', alpha=0.7, label='PV Generation')
    ax2.plot(hours, data['pv'], color='orange', linewidth=2)
    ax2.set_ylabel('Power (kW)', fontsize=11)
    ax2.set_title('② PV GENERATION (from "rg" sheet - solar output)',
                  fontsize=12, fontweight='bold', color='orange')
    ax2.set_xticks(hours)
    ax2.set_xticklabels(hour_labels, rotation=45, ha='right', fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xlim(-0.5, 23.5)

    ax2.annotate(f'Peak: {data["pv"].max():.0f} kW\nTotal: {data["pv"].sum():.0f} kWh',
                 xy=(0.02, 0.95), xycoords='axes fraction', fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    # =========================================
    # SUBPLOT 3: Net Load (Demand - PV)
    # =========================================
    ax3 = axes[2]
    net_load = data['demand'] - data['pv']

    # Color positive (need grid) and negative (excess PV) differently
    colors = ['red' if x > 0 else 'green' for x in net_load]
    ax3.bar(hours, net_load, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax3.axhline(y=0, color='black', linewidth=1)
    ax3.set_ylabel('Net Load (kW)', fontsize=11)
    ax3.set_title('③ NET LOAD = Demand - PV (Red=need grid, Green=excess PV)',
                  fontsize=12, fontweight='bold')
    ax3.set_xticks(hours)
    ax3.set_xticklabels(hour_labels, rotation=45, ha='right', fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_xlim(-0.5, 23.5)

    # =========================================
    # SUBPLOT 4: Grid Demand Comparison
    # =========================================
    ax4 = axes[3]

    # Shade demand window
    ax4.axvspan(8, 22, alpha=0.15, color='red', label='Demand Window (8AM-10PM)')

    # No battery baseline
    no_battery = np.maximum(net_load, 0)

    ax4.step(hours, no_battery, where='mid', color='gray', linewidth=2,
             linestyle='--', label=f'No Battery (peak={no_battery.max():.0f} kW)')
    ax4.step(hours, heuristic['grid_buy'], where='mid', color='blue', linewidth=2,
             label=f'Heuristic (peak={heuristic["grid_buy"].max():.0f} kW)')
    ax4.step(hours, optimized['grid_buy'], where='mid', color='green', linewidth=2.5,
             label=f'Optimized (peak={optimized["grid_buy"].max():.0f} kW)')

    ax4.set_ylabel('Grid Purchase (kW)', fontsize=11)
    ax4.set_title('④ GRID DEMAND COMPARISON (what you pay demand charges on)',
                  fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=9)
    ax4.set_xticks(hours)
    ax4.set_xticklabels(hour_labels, rotation=45, ha='right', fontsize=8)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_xlim(-0.5, 23.5)
    ax4.set_ylim(bottom=0)

    # =========================================
    # SUBPLOT 5: Battery State of Charge
    # =========================================
    ax5 = axes[4]

    ax5.plot(hours, heuristic['soc'], color='blue', linewidth=2, marker='o',
             markersize=4, label='Heuristic SOC')
    ax5.plot(hours, optimized['soc'], color='green', linewidth=2, marker='s',
             markersize=4, label='Optimized SOC')
    ax5.axhline(y=data['battery']['max_soc'], color='gray', linestyle='--',
                alpha=0.5, label='Max SOC (1200 kWh)')
    ax5.axhline(y=data['battery']['min_soc'], color='gray', linestyle='--',
                alpha=0.5, label='Min SOC (120 kWh)')
    ax5.fill_between(hours, data['battery']['min_soc'], optimized['soc'],
                     alpha=0.2, color='green')

    ax5.set_ylabel('SOC (kWh)', fontsize=11)
    ax5.set_title('⑤ BATTERY STATE OF CHARGE', fontsize=12, fontweight='bold')
    ax5.legend(loc='upper right', fontsize=9)
    ax5.set_xticks(hours)
    ax5.set_xticklabels(hour_labels, rotation=45, ha='right', fontsize=8)
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_xlim(-0.5, 23.5)
    ax5.set_ylim(0, 1400)

    # =========================================
    # SUBPLOT 6: Battery Dispatch (Charge/Discharge)
    # =========================================
    ax6 = axes[5]

    # Optimized dispatch
    opt_power = optimized['discharge'] - optimized['charge']
    heur_power = heuristic['discharge'] - heuristic['charge']

    width = 0.35
    ax6.bar(hours - width/2, heur_power, width, color='blue', alpha=0.7,
            label='Heuristic', edgecolor='navy')
    ax6.bar(hours + width/2, opt_power, width, color='green', alpha=0.7,
            label='Optimized', edgecolor='darkgreen')
    ax6.axhline(y=0, color='black', linewidth=1)

    ax6.set_ylabel('Battery Power (kW)\n(+)Discharge  (-)Charge', fontsize=11)
    ax6.set_xlabel('Hour of Day', fontsize=11)
    ax6.set_title('⑥ BATTERY DISPATCH ACTIONS', fontsize=12, fontweight='bold')
    ax6.legend(loc='upper right', fontsize=9)
    ax6.set_xticks(hours)
    ax6.set_xticklabels(hour_labels, rotation=45, ha='right', fontsize=8)
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.set_xlim(-0.5, 23.5)

    plt.tight_layout()
    plt.savefig('single_day_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("Plot saved to: single_day_comparison.png")


def calculate_costs(grid_buy, grid_sell, price_buy, price_sell, demand_rate=0.85,
                    curtailed=None, allow_export=False):
    """Calculate all cost components."""

    # Energy costs
    energy_buy = np.sum(price_buy * grid_buy)

    if allow_export and grid_sell is not None:
        energy_sell = np.sum(price_sell * grid_sell)
    else:
        energy_sell = 0.0

    # Demand charge (peak during 8-22)
    peak_in_window = np.max(grid_buy[8:22])
    demand_charge = demand_rate * peak_in_window

    # Track curtailed energy
    total_curtailed = np.sum(curtailed) if curtailed is not None else 0.0

    return {
        'energy_buy': energy_buy,
        'energy_sell': energy_sell,
        'net_energy': energy_buy - energy_sell,
        'peak_demand': peak_in_window,
        'demand_charge': demand_charge,
        'total': energy_buy - energy_sell + demand_charge,
        'curtailed_kwh': total_curtailed,
    }


def main():
    print("=" * 70)
    print("SINGLE DAY ANALYSIS: HEURISTIC vs OPTIMIZED DISPATCH")
    print("=" * 70)
    print("\n*** NYC MODE: No grid export allowed (excess PV is curtailed) ***")

    # Load a summer day (July 1 = day 182)
    data = load_single_day('test2.xlsx', day_index=182)

    print(f"\nDate: {data['date'].strftime('%Y-%m-%d')} (Day 182)")
    print(f"\nINPUT DATA:")
    print(f"  Building Demand: {data['demand'].min():.0f} - {data['demand'].max():.0f} kW")
    print(f"  PV Generation:   {data['pv'].min():.0f} - {data['pv'].max():.0f} kW")
    print(f"  Buy Price:       ${data['price_buy'].min():.3f} - ${data['price_buy'].max():.3f}/kWh")

    print(f"\nBATTERY:")
    print(f"  Capacity: {data['battery']['capacity']} kWh")
    print(f"  Max Power: {data['battery']['max_charge']} kW charge/discharge")

    # Run dispatches (NO EXPORT)
    allow_export = False

    print("\n" + "-" * 70)
    print("Running heuristic dispatch...")
    heuristic = heuristic_dispatch(data, peak_target=None, allow_export=allow_export)

    print("Running optimized dispatch...")
    optimized = optimized_dispatch(data, allow_export=allow_export)
    print(f"  Optimizer status: {optimized['status']}")

    # =====================
    # CALCULATE ALL COSTS
    # =====================

    # 1. BASELINE: Grid only (no PV, no battery) - pay for ALL demand from grid
    baseline_buy = data['demand'].copy()  # All demand from grid
    baseline_peak = np.max(baseline_buy[8:22])  # Peak in demand window

    costs_baseline = {
        'energy_buy': np.sum(data['price_buy'] * baseline_buy),
        'energy_sell': 0,
        'net_energy': np.sum(data['price_buy'] * baseline_buy),
        'peak_demand': baseline_peak,
        'demand_charge': 0.85 * baseline_peak,
        'curtailed_kwh': 0,
        'pv_used_kwh': 0,
    }
    costs_baseline['total'] = costs_baseline['net_energy'] + costs_baseline['demand_charge']

    # 2. PV ONLY (no battery) - use PV when available, curtail excess
    pv_only_buy = np.maximum(data['demand'] - data['pv'], 0)
    pv_only_curtailed = np.maximum(data['pv'] - data['demand'], 0)
    pv_used_no_batt = np.minimum(data['pv'], data['demand'])

    costs_pv_only = calculate_costs(
        pv_only_buy, None, data['price_buy'], data['price_sell'],
        curtailed=pv_only_curtailed, allow_export=False
    )
    costs_pv_only['pv_used_kwh'] = np.sum(pv_used_no_batt)

    # 3. PV + BATTERY (Heuristic)
    costs_heur = calculate_costs(
        heuristic['grid_buy'], None, data['price_buy'], data['price_sell'],
        curtailed=heuristic.get('curtailed'), allow_export=False
    )
    # PV used = total PV - curtailed
    costs_heur['pv_used_kwh'] = np.sum(data['pv']) - costs_heur['curtailed_kwh']

    # 4. PV + BATTERY (Optimized)
    costs_opt = calculate_costs(
        optimized['grid_buy'], None, data['price_buy'], data['price_sell'],
        curtailed=optimized.get('curtailed'), allow_export=False
    )
    costs_opt['pv_used_kwh'] = np.sum(data['pv']) - costs_opt['curtailed_kwh']

    # =====================
    # PRINT COMPARISON TABLE
    # =====================
    print("\n" + "=" * 80)
    print("COST COMPARISON (Single Day) - NYC MODE (NO GRID EXPORT)")
    print("=" * 80)

    print("\n{:<25} {:>12} {:>12} {:>12} {:>12}".format(
        "", "Grid Only", "PV Only", "PV+Batt", "PV+Batt"))
    print("{:<25} {:>12} {:>12} {:>12} {:>12}".format(
        "", "(Baseline)", "(No Batt)", "(Heuristic)", "(Optimized)"))
    print("-" * 80)

    print("{:<25} {:>12.2f} {:>12.2f} {:>12.2f} {:>12.2f}".format(
        "Energy Purchase ($)",
        costs_baseline['energy_buy'], costs_pv_only['energy_buy'],
        costs_heur['energy_buy'], costs_opt['energy_buy']))

    print("{:<25} {:>12.0f} {:>12.0f} {:>12.0f} {:>12.0f}".format(
        "PV Used (kWh)",
        costs_baseline.get('pv_used_kwh', 0), costs_pv_only['pv_used_kwh'],
        costs_heur['pv_used_kwh'], costs_opt['pv_used_kwh']))

    print("{:<25} {:>12.0f} {:>12.0f} {:>12.0f} {:>12.0f}".format(
        "PV Curtailed (kWh)",
        0, costs_pv_only['curtailed_kwh'],
        costs_heur['curtailed_kwh'], costs_opt['curtailed_kwh']))

    print("-" * 80)

    print("{:<25} {:>12.0f} {:>12.0f} {:>12.0f} {:>12.0f}".format(
        "Peak Demand (kW)",
        costs_baseline['peak_demand'], costs_pv_only['peak_demand'],
        costs_heur['peak_demand'], costs_opt['peak_demand']))

    print("{:<25} {:>12.2f} {:>12.2f} {:>12.2f} {:>12.2f}".format(
        "Demand Charge ($)",
        costs_baseline['demand_charge'], costs_pv_only['demand_charge'],
        costs_heur['demand_charge'], costs_opt['demand_charge']))

    print("-" * 80)

    print("{:<25} {:>12.2f} {:>12.2f} {:>12.2f} {:>12.2f}".format(
        "TOTAL DAILY COST ($)",
        costs_baseline['total'], costs_pv_only['total'],
        costs_heur['total'], costs_opt['total']))

    # =====================
    # SAVINGS BREAKDOWN
    # =====================
    print("\n" + "=" * 80)
    print("SAVINGS BREAKDOWN (vs Grid-Only Baseline)")
    print("=" * 80)

    for name, costs in [("PV Only", costs_pv_only),
                        ("PV + Battery (Heuristic)", costs_heur),
                        ("PV + Battery (Optimized)", costs_opt)]:
        savings = costs_baseline['total'] - costs['total']
        energy_savings = costs_baseline['energy_buy'] - costs['energy_buy']
        demand_savings = costs_baseline['demand_charge'] - costs['demand_charge']

        print(f"\n{name}:")
        print(f"  Energy savings:     ${energy_savings:>10.2f}  (from using PV)")
        print(f"  Demand savings:     ${demand_savings:>10.2f}  (from peak shaving)")
        print(f"  TOTAL SAVINGS:      ${savings:>10.2f}  ({100*savings/costs_baseline['total']:.1f}%)")

    # =====================
    # CREATE COST COMPARISON CHART
    # =====================
    plot_cost_comparison(data, costs_baseline, costs_pv_only, costs_heur, costs_opt)

    # Create dispatch plot
    print("\n" + "-" * 70)
    plot_single_day(data, heuristic, optimized)

    return data, heuristic, optimized, {
        'baseline': costs_baseline,
        'pv_only': costs_pv_only,
        'heuristic': costs_heur,
        'optimized': costs_opt,
    }


def plot_cost_comparison(data, costs_baseline, costs_pv_only, costs_heur, costs_opt):
    """Create a bar chart comparing costs across all scenarios."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Data for plots
    scenarios = ['Grid Only\n(Baseline)', 'PV Only\n(No Battery)', 'PV + Battery\n(Heuristic)', 'PV + Battery\n(Optimized)']
    energy_costs = [costs_baseline['energy_buy'], costs_pv_only['energy_buy'],
                    costs_heur['energy_buy'], costs_opt['energy_buy']]
    demand_costs = [costs_baseline['demand_charge'], costs_pv_only['demand_charge'],
                    costs_heur['demand_charge'], costs_opt['demand_charge']]
    total_costs = [costs_baseline['total'], costs_pv_only['total'],
                   costs_heur['total'], costs_opt['total']]

    colors = ['#d62728', '#ff7f0e', '#1f77b4', '#2ca02c']

    # =====================
    # LEFT PLOT: Stacked bar chart
    # =====================
    ax1 = axes[0]
    x = np.arange(len(scenarios))
    width = 0.6

    bars1 = ax1.bar(x, energy_costs, width, label='Energy Cost', color='steelblue', alpha=0.8)
    bars2 = ax1.bar(x, demand_costs, width, bottom=energy_costs, label='Demand Charge', color='coral', alpha=0.8)

    # Add total labels on top
    for i, (e, d) in enumerate(zip(energy_costs, demand_costs)):
        total = e + d
        ax1.annotate(f'${total:.0f}',
                     xy=(i, total + 50),
                     ha='center', va='bottom',
                     fontsize=11, fontweight='bold')

    ax1.set_ylabel('Daily Cost ($)', fontsize=12)
    ax1.set_title('Daily Cost Breakdown by Scenario', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, fontsize=10)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_ylim(0, max(total_costs) * 1.15)
    ax1.grid(True, alpha=0.3, axis='y')

    # =====================
    # RIGHT PLOT: Savings waterfall
    # =====================
    ax2 = axes[1]

    # Calculate incremental savings
    baseline = costs_baseline['total']
    pv_savings = baseline - costs_pv_only['total']
    batt_heur_savings = costs_pv_only['total'] - costs_heur['total']
    opt_vs_heur_savings = costs_heur['total'] - costs_opt['total']

    categories = ['Baseline\nCost', 'PV\nSavings', 'Battery\n(Heuristic)', 'Optimizer\nImprovement', 'Final\nCost']
    values = [baseline, -pv_savings, -batt_heur_savings, -opt_vs_heur_savings, costs_opt['total']]

    # Waterfall calculation
    cumulative = [baseline]
    for v in values[1:-1]:
        cumulative.append(cumulative[-1] + v)
    cumulative.append(costs_opt['total'])

    # Colors for waterfall
    waterfall_colors = ['#d62728', '#2ca02c', '#2ca02c', '#2ca02c', '#1f77b4']

    # Plot bars
    for i, (cat, val, cum) in enumerate(zip(categories, values, cumulative)):
        if i == 0:  # Baseline
            ax2.bar(i, val, width=0.6, color=waterfall_colors[i], alpha=0.8)
        elif i == len(categories) - 1:  # Final
            ax2.bar(i, val, width=0.6, color=waterfall_colors[i], alpha=0.8)
        else:  # Savings (negative)
            bottom = cum - val if val < 0 else cum
            ax2.bar(i, abs(val), width=0.6, bottom=bottom + val, color=waterfall_colors[i], alpha=0.8)

            # Add savings label
            ax2.annotate(f'-${abs(val):.0f}',
                         xy=(i, cum + 30),
                         ha='center', va='bottom',
                         fontsize=10, fontweight='bold', color='green')

    # Add connector lines
    for i in range(len(cumulative) - 1):
        ax2.plot([i + 0.3, i + 0.7], [cumulative[i], cumulative[i]],
                 color='gray', linewidth=1, linestyle='--')

    # Labels
    ax2.annotate(f'${baseline:.0f}', xy=(0, baseline + 30), ha='center', fontsize=11, fontweight='bold', color='red')
    ax2.annotate(f'${costs_opt["total"]:.0f}', xy=(4, costs_opt["total"] + 30), ha='center', fontsize=11, fontweight='bold', color='blue')

    ax2.set_ylabel('Daily Cost ($)', fontsize=12)
    ax2.set_title('Savings Waterfall: Baseline → Optimized', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(categories)))
    ax2.set_xticklabels(categories, fontsize=10)
    ax2.set_ylim(0, baseline * 1.15)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add total savings annotation
    total_savings = baseline - costs_opt['total']
    ax2.annotate(f'Total Savings: ${total_savings:.0f}/day\n({100*total_savings/baseline:.0f}%)',
                 xy=(0.5, 0.95), xycoords='axes fraction',
                 ha='center', va='top',
                 fontsize=12, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    plt.tight_layout()
    plt.savefig('cost_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\nCost comparison chart saved to: cost_comparison.png")


if __name__ == "__main__":
    results = main()
