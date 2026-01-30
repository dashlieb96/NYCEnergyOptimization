"""
Example: Standby Rate Demand Charges for Building with PV + Battery

This script demonstrates how Con Edison standby demand charges work
for a commercial building with solar PV and battery storage.

Key concepts:
1. Contract Demand: Fixed charge for your maximum potential grid draw
2. Daily As-Used Demand: Charged on the SINGLE highest 15-min peak
   during the demand window (not multiple periods)

Demand Windows:
- Conventional Standby: 8 AM - 10 PM (14 hours) - one peak matters
- Rider Q: 2 PM - 6 PM (4 hours, summer weekdays only) - easier to shave
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dataclasses import dataclass


@dataclass
class StandbyRates:
    """Con Edison Standby Rate parameters (approximate 2024 values)"""
    # Contract demand ($/kW/month) - for your contracted standby capacity
    contract_demand_monthly: float = 8.50

    # Daily as-used demand ($/kW/day) - summer vs winter
    daily_demand_summer: float = 0.85  # June-September
    daily_demand_winter: float = 0.45  # October-May

    # Rider Q daily demand ($/kW/day) - 4-hour window
    rider_q_peak_window: float = 1.50   # Within 2-6 PM window
    rider_q_off_peak: float = 0.25      # Outside window

    # Demand window hours (24-hour format)
    conventional_start: int = 8   # 8 AM
    conventional_end: int = 22    # 10 PM
    rider_q_start: int = 14       # 2 PM
    rider_q_end: int = 18         # 6 PM


def create_building_load_profile(num_intervals: int = 96) -> dict:
    """
    Create a realistic commercial building load profile with PV and battery.

    Returns 15-minute interval data for:
    - Building gross load (what the building needs)
    - PV generation (solar production)
    - Battery dispatch (positive = discharge, negative = charge)
    - Net grid demand (what's actually drawn from Con Edison)
    """
    np.random.seed(42)

    # Time array (15-minute intervals for 24 hours)
    hours = np.linspace(0, 24, num_intervals, endpoint=False)

    # === BUILDING GROSS LOAD ===
    # Base load + occupancy-driven HVAC + equipment
    base_load = 200  # kW (always-on: elevators, common areas, etc.)

    # Occupancy pattern (ramps up 7-9 AM, steady 9-5, ramps down 5-7 PM)
    occupancy = np.zeros(num_intervals)
    for i, h in enumerate(hours):
        if 7 <= h < 9:
            occupancy[i] = (h - 7) / 2  # Ramp up
        elif 9 <= h < 17:
            occupancy[i] = 1.0  # Full occupancy
        elif 17 <= h < 19:
            occupancy[i] = 1.0 - (h - 17) / 2  # Ramp down
        else:
            occupancy[i] = 0.1  # Minimal (security, etc.)

    # HVAC load (peaks in afternoon due to solar heat gain)
    hvac_base = 150  # kW at full occupancy
    afternoon_boost = 100 * np.exp(-((hours - 15)**2) / 8)  # Peak around 3 PM
    hvac_load = hvac_base * occupancy + afternoon_boost * occupancy

    # Equipment/plug loads
    equipment = 100 * occupancy

    # Total building load
    building_load = base_load + hvac_load + equipment
    building_load += np.random.normal(0, 15, num_intervals)  # Add noise
    building_load = np.maximum(building_load, 50)  # Minimum load

    # === SOLAR PV GENERATION ===
    # 300 kW PV system, summer day with some clouds
    pv_capacity = 300  # kW

    # Solar curve (sunrise ~6 AM, sunset ~8 PM in summer)
    solar_curve = np.zeros(num_intervals)
    for i, h in enumerate(hours):
        if 6 <= h <= 20:
            # Bell curve centered at solar noon (1 PM with DST)
            solar_curve[i] = np.exp(-((h - 13)**2) / 18)

    # Add cloud variability
    cloud_factor = 0.85 + 0.15 * np.random.random(num_intervals)
    pv_generation = pv_capacity * solar_curve * cloud_factor

    # === BATTERY DISPATCH STRATEGY ===
    # 500 kWh battery, 250 kW max power
    # Strategy: Charge overnight/from PV, discharge to shave peaks
    battery_capacity = 500  # kWh
    battery_power_max = 250  # kW

    battery_dispatch = np.zeros(num_intervals)  # Positive = discharge
    soc_profile = np.zeros(num_intervals)

    interval_hours = 0.25  # 15 minutes

    # Calculate net load without battery first
    net_load_no_battery = building_load - pv_generation

    # Determine target peak cap (aggressive peak shaving)
    # Target: keep net demand below 250 kW during peak hours
    peak_target = 250  # kW - target maximum grid demand

    # Start with battery at 80% SOC (charged overnight)
    battery_soc = 0.8 * battery_capacity

    for i, h in enumerate(hours):
        current_net = net_load_no_battery[i]

        # Charging logic
        if h < 8:  # Overnight - maintain/build charge
            if battery_soc < 0.95 * battery_capacity:
                charge = min(
                    100,  # Moderate overnight charging
                    battery_power_max,
                    (battery_capacity - battery_soc) / interval_hours
                )
                battery_dispatch[i] = -charge
                battery_soc += charge * interval_hours

        elif current_net < 0:  # Excess PV - charge
            charge = min(
                -current_net,
                battery_power_max,
                (battery_capacity - battery_soc) / interval_hours
            )
            battery_dispatch[i] = -charge
            battery_soc += charge * interval_hours

        # Discharge logic during peak hours
        elif 8 <= h < 22:  # Demand window
            if current_net > peak_target and battery_soc > 50:
                discharge = min(
                    current_net - peak_target,
                    battery_power_max,
                    (battery_soc - 50) / interval_hours  # Keep 50 kWh reserve
                )
                battery_dispatch[i] = discharge
                battery_soc -= discharge * interval_hours

        soc_profile[i] = max(0, min(battery_soc, battery_capacity))

    # === NET GRID DEMAND ===
    # What the building actually draws from Con Edison
    net_grid_demand = building_load - pv_generation - battery_dispatch
    net_grid_demand = np.maximum(net_grid_demand, 0)  # Can't export (simplification)

    return {
        'hours': hours,
        'building_load': building_load,
        'pv_generation': pv_generation,
        'battery_dispatch': battery_dispatch,
        'battery_soc': soc_profile,
        'net_grid_demand': net_grid_demand,
    }


def calculate_standby_demand_charges(
    hours: np.ndarray,
    net_demand: np.ndarray,
    contract_demand_kw: float,
    rates: StandbyRates = None,
    is_summer: bool = True
) -> dict:
    """
    Calculate standby rate demand charges for a single day.

    Key insight: The daily as-used demand charge is based on the SINGLE
    highest 15-minute peak within the demand window, not multiple peaks.
    """
    if rates is None:
        rates = StandbyRates()

    # Find indices for each demand window
    conventional_mask = (hours >= rates.conventional_start) & (hours < rates.conventional_end)
    rider_q_mask = (hours >= rates.rider_q_start) & (hours < rates.rider_q_end)

    # Peak demand in each window
    conventional_peak = np.max(net_demand[conventional_mask]) if np.any(conventional_mask) else 0
    rider_q_peak = np.max(net_demand[rider_q_mask]) if np.any(rider_q_mask) else 0
    off_peak_demand = np.max(net_demand[~rider_q_mask]) if np.any(~rider_q_mask) else 0
    overall_peak = np.max(net_demand)

    # Contract demand charge (prorated daily from monthly)
    contract_charge_daily = rates.contract_demand_monthly * contract_demand_kw / 30

    # CONVENTIONAL STANDBY: One peak for entire 8 AM - 10 PM window
    if is_summer:
        conventional_daily_charge = rates.daily_demand_summer * conventional_peak
    else:
        conventional_daily_charge = rates.daily_demand_winter * conventional_peak

    conventional_total = contract_charge_daily + conventional_daily_charge

    # RIDER Q: Peak within 4-hour window + off-peak
    rider_q_daily_charge = (
        rates.rider_q_peak_window * rider_q_peak +
        rates.rider_q_off_peak * off_peak_demand
    )
    rider_q_total = contract_charge_daily + rider_q_daily_charge

    return {
        # Peaks
        'overall_peak_kw': overall_peak,
        'conventional_window_peak_kw': conventional_peak,
        'rider_q_window_peak_kw': rider_q_peak,
        'off_peak_demand_kw': off_peak_demand,

        # Conventional standby charges
        'conventional': {
            'contract_charge': contract_charge_daily,
            'daily_as_used_charge': conventional_daily_charge,
            'total_daily': conventional_total,
            'window': f"{rates.conventional_start}:00 - {rates.conventional_end}:00",
            'billed_peak_kw': conventional_peak,
        },

        # Rider Q charges
        'rider_q': {
            'contract_charge': contract_charge_daily,
            'peak_window_charge': rates.rider_q_peak_window * rider_q_peak,
            'off_peak_charge': rates.rider_q_off_peak * off_peak_demand,
            'total_daily': rider_q_total,
            'window': f"{rates.rider_q_start}:00 - {rates.rider_q_end}:00",
            'billed_peak_kw': rider_q_peak,
        },
    }


def plot_load_profile(data: dict, demand_results: dict):
    """Create visualization of load profile and demand windows."""

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    hours = data['hours']

    # Colors
    colors = {
        'load': '#2ecc71',
        'pv': '#f1c40f',
        'battery': '#3498db',
        'net': '#e74c3c',
        'peak_window': '#ffcccc',
        'rider_q_window': '#ffe6cc',
    }

    # === Plot 1: Load components ===
    ax1 = axes[0]
    ax1.fill_between(hours, 0, data['building_load'], alpha=0.3, color=colors['load'], label='Building Load')
    ax1.plot(hours, data['building_load'], color=colors['load'], linewidth=2)
    ax1.fill_between(hours, 0, data['pv_generation'], alpha=0.3, color=colors['pv'], label='PV Generation')
    ax1.plot(hours, data['pv_generation'], color=colors['pv'], linewidth=2)

    ax1.set_ylabel('Power (kW)')
    ax1.set_title('Building Load vs. PV Generation', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 600)

    # === Plot 2: Net demand with demand windows ===
    ax2 = axes[1]

    # Shade demand windows
    ax2.axvspan(8, 22, alpha=0.15, color='red', label='Conventional Window (8AM-10PM)')
    ax2.axvspan(14, 18, alpha=0.25, color='orange', label='Rider Q Window (2PM-6PM)')

    # Net demand line
    ax2.plot(hours, data['net_grid_demand'], color=colors['net'], linewidth=2.5, label='Net Grid Demand')

    # Mark the peaks
    conv_peak = demand_results['conventional_window_peak_kw']
    rq_peak = demand_results['rider_q_window_peak_kw']

    # Find peak times
    conv_mask = (hours >= 8) & (hours < 22)
    rq_mask = (hours >= 14) & (hours < 18)

    conv_peak_idx = np.where(conv_mask)[0][np.argmax(data['net_grid_demand'][conv_mask])]
    rq_peak_idx = np.where(rq_mask)[0][np.argmax(data['net_grid_demand'][rq_mask])]

    ax2.scatter([hours[conv_peak_idx]], [conv_peak], color='red', s=150, zorder=5,
                marker='v', label=f'Conv. Peak: {conv_peak:.0f} kW')
    ax2.scatter([hours[rq_peak_idx]], [rq_peak], color='orange', s=150, zorder=5,
                marker='v', label=f'Rider Q Peak: {rq_peak:.0f} kW')

    # Horizontal lines showing peak levels
    ax2.axhline(y=conv_peak, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax2.axhline(y=rq_peak, color='orange', linestyle='--', alpha=0.5, linewidth=1)

    ax2.set_ylabel('Net Grid Demand (kW)')
    ax2.set_title('Net Demand with Demand Charge Windows', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 400)

    # === Plot 3: Battery operation ===
    ax3 = axes[2]

    # Battery dispatch (positive = discharge, negative = charge)
    discharge = np.maximum(data['battery_dispatch'], 0)
    charge = np.minimum(data['battery_dispatch'], 0)

    ax3.fill_between(hours, 0, discharge, alpha=0.5, color='green', label='Discharge')
    ax3.fill_between(hours, 0, charge, alpha=0.5, color='blue', label='Charge')
    ax3.plot(hours, data['battery_dispatch'], color='black', linewidth=1)

    # SOC on secondary axis
    ax3b = ax3.twinx()
    ax3b.plot(hours, data['battery_soc'], color='purple', linewidth=2, linestyle='--', label='Battery SOC')
    ax3b.set_ylabel('State of Charge (kWh)', color='purple')
    ax3b.tick_params(axis='y', labelcolor='purple')
    ax3b.set_ylim(0, 600)

    ax3.axvspan(14, 18, alpha=0.15, color='orange')
    ax3.set_xlabel('Hour of Day')
    ax3.set_ylabel('Battery Power (kW)')
    ax3.set_title('Battery Dispatch Strategy', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 24)
    ax3.set_xticks(range(0, 25, 2))

    plt.tight_layout()
    plt.savefig('standby_demand_example.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nPlot saved to: standby_demand_example.png")


def main():
    """Run the standby demand charge example."""

    print("=" * 70)
    print("STANDBY RATE DEMAND CHARGES - EXAMPLE WITH PV + BATTERY")
    print("=" * 70)

    # Create load profile
    data = create_building_load_profile()

    # Building parameters
    contract_demand = 400  # kW - max backup capacity contracted

    # Calculate demand charges
    results = calculate_standby_demand_charges(
        hours=data['hours'],
        net_demand=data['net_grid_demand'],
        contract_demand_kw=contract_demand,
        is_summer=True  # July
    )

    # Display results
    print("\n" + "-" * 70)
    print("BUILDING CONFIGURATION")
    print("-" * 70)
    print(f"  Building peak load:     {np.max(data['building_load']):.0f} kW")
    print(f"  PV system capacity:     300 kW")
    print(f"  Battery:                500 kWh / 250 kW")
    print(f"  Contract demand:        {contract_demand} kW")

    print("\n" + "-" * 70)
    print("LOAD PROFILE SUMMARY (Summer Weekday)")
    print("-" * 70)
    print(f"  Overall peak demand:    {results['overall_peak_kw']:.0f} kW")
    print(f"  Max PV generation:      {np.max(data['pv_generation']):.0f} kW")
    print(f"  Max battery discharge:  {np.max(data['battery_dispatch']):.0f} kW")

    print("\n" + "=" * 70)
    print("HOW STANDBY DEMAND CHARGES WORK")
    print("=" * 70)

    print("""
IMPORTANT: You are billed on ONE peak per day, not multiple peaks.

The demand window defines WHEN your peak matters, but you're only
charged for the SINGLE HIGHEST 15-minute average within that window.

Example: If your peaks are:
  - 9:00 AM:  200 kW
  - 2:30 PM:  350 kW  <-- This is your billed peak
  - 7:00 PM:  280 kW

You pay demand charges on 350 kW (the highest), NOT on all three.
""")

    print("\n" + "-" * 70)
    print("CONVENTIONAL STANDBY RATE")
    print(f"Demand Window: {results['conventional']['window']} (14 hours)")
    print("-" * 70)
    conv = results['conventional']
    print(f"""
  Peak demand in window:     {conv['billed_peak_kw']:.0f} kW

  Contract Demand Charge:    ${conv['contract_charge']:.2f}/day
    ({contract_demand} kW × $8.50/kW/month ÷ 30 days)

  Daily As-Used Charge:      ${conv['daily_as_used_charge']:.2f}/day
    ({conv['billed_peak_kw']:.0f} kW × $0.85/kW/day)

  ─────────────────────────────────────
  TOTAL DAILY DEMAND COST:   ${conv['total_daily']:.2f}
  Monthly (×30):             ${conv['total_daily'] * 30:.2f}
""")

    print("\n" + "-" * 70)
    print("RIDER Q STANDBY RATE")
    print(f"Peak Window: {results['rider_q']['window']} (4 hours, summer weekdays)")
    print("-" * 70)
    rq = results['rider_q']
    print(f"""
  Peak in 4-hour window:     {rq['billed_peak_kw']:.0f} kW
  Off-peak demand:           {results['off_peak_demand_kw']:.0f} kW

  Contract Demand Charge:    ${rq['contract_charge']:.2f}/day
    ({contract_demand} kW × $8.50/kW/month ÷ 30 days)

  Peak Window Charge:        ${rq['peak_window_charge']:.2f}/day
    ({rq['billed_peak_kw']:.0f} kW × $1.50/kW/day)

  Off-Peak Charge:           ${rq['off_peak_charge']:.2f}/day
    ({results['off_peak_demand_kw']:.0f} kW × $0.25/kW/day)

  ─────────────────────────────────────
  TOTAL DAILY DEMAND COST:   ${rq['total_daily']:.2f}
  Monthly (×30):             ${rq['total_daily'] * 30:.2f}
""")

    print("\n" + "-" * 70)
    print("COMPARISON: BATTERY IMPACT ON DEMAND CHARGES")
    print("-" * 70)

    # Calculate what demand would be WITHOUT battery
    net_no_battery = data['building_load'] - data['pv_generation']
    net_no_battery = np.maximum(net_no_battery, 0)

    results_no_battery = calculate_standby_demand_charges(
        hours=data['hours'],
        net_demand=net_no_battery,
        contract_demand_kw=contract_demand,
        is_summer=True
    )

    print(f"""
  WITHOUT Battery:
    Conventional window peak:  {results_no_battery['conventional_window_peak_kw']:.0f} kW
    Rider Q window peak:       {results_no_battery['rider_q_window_peak_kw']:.0f} kW

  WITH Battery (peak shaving):
    Conventional window peak:  {results['conventional_window_peak_kw']:.0f} kW  (↓{results_no_battery['conventional_window_peak_kw'] - results['conventional_window_peak_kw']:.0f} kW)
    Rider Q window peak:       {results['rider_q_window_peak_kw']:.0f} kW  (↓{results_no_battery['rider_q_window_peak_kw'] - results['rider_q_window_peak_kw']:.0f} kW)

  Daily Demand Charge Savings:
    Conventional rate:         ${results_no_battery['conventional']['total_daily'] - results['conventional']['total_daily']:.2f}/day
    Rider Q rate:              ${results_no_battery['rider_q']['total_daily'] - results['rider_q']['total_daily']:.2f}/day

  Monthly Savings:
    Conventional rate:         ${(results_no_battery['conventional']['total_daily'] - results['conventional']['total_daily']) * 30:.2f}/month
    Rider Q rate:              ${(results_no_battery['rider_q']['total_daily'] - results['rider_q']['total_daily']) * 30:.2f}/month
""")

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
1. ONE PEAK MATTERS: Within each demand window, only your single highest
   15-minute demand is billed. Multiple peaks don't multiply charges.

2. CONVENTIONAL vs RIDER Q:
   - Conventional: 14-hour window (harder to manage with battery)
   - Rider Q: 4-hour window (easier to shave with battery)

3. BATTERY STRATEGY: Focus discharge during the demand window,
   especially the Rider Q 2-6 PM period when rates are highest.

4. CONTRACT DEMAND: This is your "insurance" - the max you COULD pull
   from the grid. Set it based on worst-case (cloudy day, battery dead).

5. WHY RIDER Q? The concentrated window means higher $/kW rate BUT
   much easier to manage with storage, often resulting in lower total cost.
""")

    # Create visualization
    plot_load_profile(data, results)

    return data, results


if __name__ == "__main__":
    data, results = main()
