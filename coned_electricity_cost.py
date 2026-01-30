"""
Con Edison NYC Electricity Cost Calculator

This module calculates daily electricity costs based on Con Edison's rate structures,
including demand charges, time-of-use rates, and standby service rates.

References:
- Con Edison Electric Rate Schedules (PSC No. 10): https://www.coned.com/en/rates-tariffs/rates/electric-rates-schedule
- NYSERDA Energy Storage Customer Electric Rates Reference Guide:
  https://www.nyserda.ny.gov/-/media/Project/Nyserda/Files/Programs/Energy-Storage/energy-storage-customer-electric-rates-reference-guide.pdf
- OpenEI U.S. Utility Rate Database: https://apps.openei.org/USURDB/
- NY-BEST Standby Rate + Con Ed Rider Q Fact Sheet:
  https://www.nyserda.ny.gov/-/media/Project/Nyserda/Files/Programs/Energy-Storage/Rider-Q.pdf

Rate structures based on Con Edison Service Classification SC9 (General Large - Time of Day)
and Standby Service rates. Rates should be verified against current tariffs as they are
subject to change through regulatory filings with the NY Public Service Commission.

Note: Demand is measured as the average kW over 15-minute intervals. The peak demand
for billing purposes is typically the highest average demand recorded during the
applicable time period.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Union, Optional, List, Tuple, Dict
from dataclasses import dataclass, field
from enum import Enum


class ServiceClass(Enum):
    """Con Edison Service Classifications for electric service."""
    SC1 = "SC1"   # Residential and Religious
    SC2 = "SC2"   # General Small (under 10 kW demand)
    SC9 = "SC9"   # General Large (10 kW and over)
    SC12 = "SC12" # Multiple Dwelling Space Heating
    SC13 = "SC13" # Bulk Power (over 1,500 kW)


class RateType(Enum):
    """Rate types within service classifications."""
    RATE_I = "Rate I"           # Conventional (voluntary TOD for SC9 < 1500 kW)
    RATE_II = "Rate II"         # Mandatory TOD (SC9 > 1500 kW)
    STANDBY = "Standby"         # For customers with on-site generation (DERs)
    RIDER_Q = "Rider Q"         # Enhanced standby rate with 4-hour peak window


class Season(Enum):
    """Billing seasons for Con Edison rates."""
    SUMMER = "Summer"       # June - September
    WINTER = "Winter"       # October - May


@dataclass
class ConEdRateSchedule:
    """
    Con Edison rate schedule parameters.

    Rates are based on SC9 Rate II (Time-of-Day) as a reference.
    All rates in $/kWh or $/kW as indicated.

    Sources:
    - OpenEI USURDB (historical rates, adjusted for 2024)
    - Con Edison PSC No. 10 tariff schedules
    - NYSERDA Energy Storage Reference Guide

    Note: These rates are approximate and should be verified against
    current Con Edison tariff filings. Rates are subject to monthly
    adjustment clauses (MAC) and various surcharges.
    """

    # Monthly fixed charges ($)
    monthly_customer_charge: float = 143.09  # SC9 Rate II base

    # Delivery charges - Energy ($/kWh)
    delivery_energy_charge: float = 0.0079

    # Delivery charges - Demand ($/kW) by season
    # Summer: June - September, Winter: October - May
    delivery_demand_summer: float = 16.70    # Peak demand charge Jun-Aug
    delivery_demand_winter: float = 5.36     # Off-peak months

    # Time-of-Use Demand Charges ($/kW)
    # Period definitions vary; these are typical SC9 Rate II values
    tou_demand_on_peak: float = 23.89        # Summer on-peak
    tou_demand_off_peak: float = 11.48       # Off-peak periods

    # Supply charges (market-based, highly variable)
    # These are illustrative averages; actual supply varies hourly
    supply_energy_on_peak: float = 0.12      # $/kWh on-peak
    supply_energy_off_peak: float = 0.08     # $/kWh off-peak
    supply_demand: float = 3.50              # $/kW demand supply charge

    # Reactive power charge ($/kVar)
    reactive_power_charge: float = 0.69

    # System Benefits Charge ($/kWh)
    system_benefits_charge: float = 0.003985

    # Clean Energy Standard surcharge ($/kWh)
    clean_energy_surcharge: float = 0.002

    # Minimum demand for billing (kW)
    minimum_demand_kw: float = 5.0

    # Monthly Adjustment Clause (MAC) - varies monthly
    mac_adjustment: float = 0.01  # $/kWh approximate


@dataclass
class StandbyRateSchedule:
    """
    Standby service rate schedule for customers with on-site generation (DERs).

    Standby rates apply to customers with solar, CHP, battery storage, or other
    distributed energy resources who may need grid backup power.

    Key features:
    - Contract demand: Fixed charge based on agreed maximum backup capacity
    - Daily as-used demand: Charged based on actual daily peak demand
    - More granular than standard rates (daily vs monthly demand)

    References:
    - Con Edison SC9 Rate IV/V Standby provisions
    - NYSERDA Standby Rate + Rider Q Fact Sheet
    """

    # Contract demand charge ($/kW/month)
    # Based on contracted standby capacity
    contract_demand_charge: float = 8.50

    # Daily as-used demand charge ($/kW/day)
    # Conventional window: 10-14 hours
    daily_demand_charge_summer: float = 0.85  # $/kW/day Jun-Sep
    daily_demand_charge_winter: float = 0.45  # $/kW/day Oct-May

    # Delivery energy charges ($/kWh)
    delivery_energy_summer: float = 0.025
    delivery_energy_winter: float = 0.018

    # Supply charges ($/kWh) - market-based
    supply_energy: float = 0.10


@dataclass
class RiderQSchedule:
    """
    Rider Q rate schedule - enhanced standby rate for demand response.

    Rider Q concentrates peak demand charges into a 4-hour daily window,
    aligned with system peak times. This creates opportunities for load
    shifting and battery storage optimization.

    Window: Monday-Friday, June-September, specific 4-hour period
    varies by demand response network location.

    Reference:
    - NY-BEST/NYSERDA Rider Q Fact Sheet
    """

    # Contract demand charge ($/kW/month)
    contract_demand_charge: float = 8.50

    # Daily as-used demand within 4-hour window ($/kW/day)
    # Significantly higher than conventional due to concentrated window
    daily_demand_peak_window: float = 1.5019

    # Daily demand outside peak window ($/kW/day)
    daily_demand_off_peak: float = 0.25

    # Energy charges ($/kWh)
    energy_charge_peak: float = 0.035
    energy_charge_off_peak: float = 0.015


@dataclass
class TimeOfUsePeriods:
    """
    Time-of-use period definitions for Con Edison rates.

    Peak periods vary by rate schedule and season. These are typical
    definitions for SC9 commercial rates.
    """

    # Summer peak hours (June-September, weekdays)
    summer_peak_start: int = 8   # 8 AM
    summer_peak_end: int = 22    # 10 PM

    # Summer super-peak hours (additional premium)
    summer_super_peak_start: int = 14  # 2 PM
    summer_super_peak_end: int = 18    # 6 PM

    # Winter peak hours (October-May, weekdays)
    winter_peak_start: int = 8   # 8 AM
    winter_peak_end: int = 22    # 10 PM

    # Rider Q 4-hour window (varies by network, this is typical)
    rider_q_window_start: int = 14  # 2 PM
    rider_q_window_end: int = 18    # 6 PM


def is_summer_month(month: int) -> bool:
    """Check if month is in summer billing season (June-September)."""
    return month in [6, 7, 8, 9]


def is_peak_hour(hour: int, month: int, weekday: int,
                 periods: TimeOfUsePeriods = None) -> bool:
    """
    Determine if a given hour is within peak pricing period.

    Args:
        hour: Hour of day (0-23)
        month: Month (1-12)
        weekday: Day of week (0=Monday, 6=Sunday)
        periods: TimeOfUsePeriods configuration

    Returns:
        True if hour is during peak period
    """
    if periods is None:
        periods = TimeOfUsePeriods()

    # Weekends and holidays are off-peak
    if weekday >= 5:  # Saturday or Sunday
        return False

    if is_summer_month(month):
        return periods.summer_peak_start <= hour < periods.summer_peak_end
    else:
        return periods.winter_peak_start <= hour < periods.winter_peak_end


def is_super_peak_hour(hour: int, month: int, weekday: int,
                       periods: TimeOfUsePeriods = None) -> bool:
    """
    Determine if hour is within summer super-peak period.

    Super-peak applies only during summer weekdays, 2-6 PM.
    """
    if periods is None:
        periods = TimeOfUsePeriods()

    if weekday >= 5:  # Weekend
        return False

    if not is_summer_month(month):
        return False

    return periods.summer_super_peak_start <= hour < periods.summer_super_peak_end


def is_rider_q_peak(hour: int, month: int, weekday: int,
                    periods: TimeOfUsePeriods = None) -> bool:
    """
    Determine if hour is within Rider Q 4-hour peak window.

    Rider Q peak window: Mon-Fri, June-September, 2-6 PM (typical).
    """
    if periods is None:
        periods = TimeOfUsePeriods()

    # Only applies summer weekdays
    if weekday >= 5:
        return False

    if not is_summer_month(month):
        return False

    return periods.rider_q_window_start <= hour < periods.rider_q_window_end


def calculate_interval_energy(demands_kw: np.ndarray,
                              interval_minutes: Union[int, np.ndarray]) -> np.ndarray:
    """
    Calculate energy consumption (kWh) from demand readings.

    Args:
        demands_kw: Array of demand values in kW
        interval_minutes: Duration of each interval in minutes
                         (scalar or array matching demands_kw length)

    Returns:
        Array of energy values in kWh
    """
    interval_hours = np.asarray(interval_minutes) / 60.0
    return demands_kw * interval_hours


def calculate_daily_electricity_cost(
    timestamps: Union[List[datetime], np.ndarray],
    demands_kw: Union[List[float], np.ndarray],
    interval_minutes: Union[int, List[int], np.ndarray] = 15,
    service_class: ServiceClass = ServiceClass.SC9,
    rate_type: RateType = RateType.RATE_II,
    rate_schedule: Optional[ConEdRateSchedule] = None,
    standby_schedule: Optional[StandbyRateSchedule] = None,
    rider_q_schedule: Optional[RiderQSchedule] = None,
    contract_demand_kw: Optional[float] = None,
    power_factor: float = 0.90,
    include_supply: bool = True
) -> Dict:
    """
    Calculate daily electricity cost based on Con Edison rate structures.

    This function computes electricity costs including:
    - Energy charges (delivery + supply)
    - Demand charges (peak demand within billing period)
    - Time-of-use adjustments
    - Standby/Rider Q charges if applicable
    - Various surcharges and adjustments

    Args:
        timestamps: Array of datetime objects for each demand reading
        demands_kw: Array of demand values in kW (average for each interval)
        interval_minutes: Duration of each measurement interval in minutes.
                         Can be a scalar (all same) or array per interval.
                         Con Edison meters typically use 15-minute intervals.
        service_class: Con Edison service classification (default SC9)
        rate_type: Rate type within service class
        rate_schedule: Custom rate schedule (uses defaults if None)
        standby_schedule: Standby rate schedule (for STANDBY rate type)
        rider_q_schedule: Rider Q schedule (for RIDER_Q rate type)
        contract_demand_kw: Contract demand for standby service (kW)
        power_factor: Power factor for reactive power calculations (0-1)
        include_supply: Whether to include supply charges (False for delivery-only)

    Returns:
        Dictionary containing:
        - total_cost: Total daily electricity cost ($)
        - energy_cost: Energy charge component ($)
        - demand_cost: Demand charge component ($)
        - fixed_cost: Prorated daily fixed charges ($)
        - surcharges: Taxes and surcharges ($)
        - total_energy_kwh: Total energy consumed (kWh)
        - peak_demand_kw: Peak demand recorded (kW)
        - on_peak_energy_kwh: Energy during peak hours (kWh)
        - off_peak_energy_kwh: Energy during off-peak hours (kWh)
        - breakdown: Detailed cost breakdown by component

    Example:
        >>> from datetime import datetime, timedelta
        >>> import numpy as np
        >>>
        >>> # Generate sample data for one day (15-minute intervals)
        >>> base_time = datetime(2024, 7, 15, 0, 0)  # Summer weekday
        >>> timestamps = [base_time + timedelta(minutes=15*i) for i in range(96)]
        >>>
        >>> # Simulate typical commercial load profile
        >>> hours = np.array([t.hour + t.minute/60 for t in timestamps])
        >>> base_load = 500  # kW
        >>> demands = base_load + 300 * np.sin((hours - 6) * np.pi / 12)
        >>> demands = np.maximum(demands, 100)  # Minimum load
        >>>
        >>> result = calculate_daily_electricity_cost(
        ...     timestamps=timestamps,
        ...     demands_kw=demands,
        ...     interval_minutes=15
        ... )
        >>> print(f"Total daily cost: ${result['total_cost']:.2f}")

    References:
        - Con Edison PSC No. 10 Electric Rate Schedule
        - NYSERDA Energy Storage Customer Electric Rates Reference Guide
        - OpenEI U.S. Utility Rate Database
    """

    # Convert inputs to numpy arrays
    timestamps = np.asarray(timestamps)
    demands_kw = np.asarray(demands_kw, dtype=float)

    if np.isscalar(interval_minutes):
        interval_minutes = np.full(len(demands_kw), interval_minutes)
    else:
        interval_minutes = np.asarray(interval_minutes)

    # Validate inputs
    if len(timestamps) != len(demands_kw):
        raise ValueError("timestamps and demands_kw must have same length")

    if len(timestamps) == 0:
        raise ValueError("Empty input arrays")

    # Initialize rate schedules
    if rate_schedule is None:
        rate_schedule = ConEdRateSchedule()

    if standby_schedule is None:
        standby_schedule = StandbyRateSchedule()

    if rider_q_schedule is None:
        rider_q_schedule = RiderQSchedule()

    tou_periods = TimeOfUsePeriods()

    # Extract time components
    months = np.array([t.month for t in timestamps])
    hours = np.array([t.hour for t in timestamps])
    weekdays = np.array([t.weekday() for t in timestamps])

    # Determine season (use first timestamp as reference)
    is_summer = is_summer_month(months[0])

    # Calculate energy for each interval
    energy_kwh = calculate_interval_energy(demands_kw, interval_minutes)
    total_energy_kwh = np.sum(energy_kwh)

    # Classify intervals by time-of-use period
    is_peak = np.array([is_peak_hour(h, m, w, tou_periods)
                        for h, m, w in zip(hours, months, weekdays)])
    is_super = np.array([is_super_peak_hour(h, m, w, tou_periods)
                         for h, m, w in zip(hours, months, weekdays)])
    is_rider_q = np.array([is_rider_q_peak(h, m, w, tou_periods)
                           for h, m, w in zip(hours, months, weekdays)])

    # Calculate energy by period
    on_peak_energy = np.sum(energy_kwh[is_peak])
    off_peak_energy = np.sum(energy_kwh[~is_peak])
    super_peak_energy = np.sum(energy_kwh[is_super])

    # Calculate peak demands
    peak_demand_kw = np.max(demands_kw)
    on_peak_demand_kw = np.max(demands_kw[is_peak]) if np.any(is_peak) else 0
    off_peak_demand_kw = np.max(demands_kw[~is_peak]) if np.any(~is_peak) else 0

    # Apply minimum demand
    billed_demand_kw = max(peak_demand_kw, rate_schedule.minimum_demand_kw)

    # Initialize cost components
    energy_delivery_cost = 0.0
    energy_supply_cost = 0.0
    demand_delivery_cost = 0.0
    demand_supply_cost = 0.0
    surcharges = 0.0

    # Calculate costs based on rate type
    if rate_type == RateType.STANDBY:
        # Standby rate: daily demand charges + energy charges
        if contract_demand_kw is None:
            contract_demand_kw = peak_demand_kw

        # Daily contract demand (prorated from monthly)
        daily_contract = standby_schedule.contract_demand_charge * contract_demand_kw / 30

        # Daily as-used demand charge
        if is_summer:
            daily_demand = standby_schedule.daily_demand_charge_summer * peak_demand_kw
            energy_rate = standby_schedule.delivery_energy_summer
        else:
            daily_demand = standby_schedule.daily_demand_charge_winter * peak_demand_kw
            energy_rate = standby_schedule.delivery_energy_winter

        demand_delivery_cost = daily_contract + daily_demand
        energy_delivery_cost = total_energy_kwh * energy_rate

        if include_supply:
            energy_supply_cost = total_energy_kwh * standby_schedule.supply_energy

    elif rate_type == RateType.RIDER_Q:
        # Rider Q: concentrated 4-hour peak window
        if contract_demand_kw is None:
            contract_demand_kw = peak_demand_kw

        # Daily contract demand (prorated)
        daily_contract = rider_q_schedule.contract_demand_charge * contract_demand_kw / 30

        # Peak window demand (only in summer, M-F)
        rider_q_peak_demand = np.max(demands_kw[is_rider_q]) if np.any(is_rider_q) else 0
        rider_q_off_peak_demand = np.max(demands_kw[~is_rider_q]) if np.any(~is_rider_q) else 0

        daily_demand = (rider_q_schedule.daily_demand_peak_window * rider_q_peak_demand +
                       rider_q_schedule.daily_demand_off_peak * rider_q_off_peak_demand)

        demand_delivery_cost = daily_contract + daily_demand

        # Energy charges by period
        rider_q_energy = np.sum(energy_kwh[is_rider_q])
        non_rider_q_energy = np.sum(energy_kwh[~is_rider_q])

        energy_delivery_cost = (rider_q_schedule.energy_charge_peak * rider_q_energy +
                               rider_q_schedule.energy_charge_off_peak * non_rider_q_energy)

        if include_supply:
            energy_supply_cost = total_energy_kwh * standby_schedule.supply_energy

    else:
        # Standard SC9 Rate I or Rate II (Time-of-Day)
        # Energy delivery charges
        energy_delivery_cost = total_energy_kwh * rate_schedule.delivery_energy_charge

        # Demand delivery charges (seasonal)
        if is_summer:
            demand_delivery_cost = billed_demand_kw * rate_schedule.delivery_demand_summer
            # Add TOU demand premium for on-peak
            if rate_type == RateType.RATE_II and on_peak_demand_kw > 0:
                demand_delivery_cost += on_peak_demand_kw * (
                    rate_schedule.tou_demand_on_peak - rate_schedule.delivery_demand_summer)
        else:
            demand_delivery_cost = billed_demand_kw * rate_schedule.delivery_demand_winter
            # TOU demand for winter
            if rate_type == RateType.RATE_II and on_peak_demand_kw > 0:
                demand_delivery_cost += on_peak_demand_kw * (
                    rate_schedule.tou_demand_off_peak - rate_schedule.delivery_demand_winter)

        # Prorate daily (demand charges are typically monthly)
        demand_delivery_cost = demand_delivery_cost / 30

        # Supply charges (if applicable)
        if include_supply:
            energy_supply_cost = (
                on_peak_energy * rate_schedule.supply_energy_on_peak +
                off_peak_energy * rate_schedule.supply_energy_off_peak
            )
            demand_supply_cost = billed_demand_kw * rate_schedule.supply_demand / 30

    # Surcharges and adjustments (apply to all rate types)
    surcharges = total_energy_kwh * (
        rate_schedule.system_benefits_charge +
        rate_schedule.clean_energy_surcharge +
        rate_schedule.mac_adjustment
    )

    # Reactive power charges (if power factor < 0.9)
    reactive_power_cost = 0.0
    if power_factor < 0.9:
        # Calculate reactive demand
        apparent_power = peak_demand_kw / power_factor
        reactive_power = np.sqrt(apparent_power**2 - peak_demand_kw**2)
        # Charge for reactive power exceeding 30% of real power
        excess_reactive = max(0, reactive_power - 0.3 * peak_demand_kw)
        reactive_power_cost = excess_reactive * rate_schedule.reactive_power_charge / 30

    # Fixed charges (prorated daily)
    fixed_cost = rate_schedule.monthly_customer_charge / 30

    # Total costs
    total_energy_cost = energy_delivery_cost + energy_supply_cost
    total_demand_cost = demand_delivery_cost + demand_supply_cost
    total_cost = total_energy_cost + total_demand_cost + fixed_cost + surcharges + reactive_power_cost

    # Build detailed breakdown
    breakdown = {
        'energy_delivery': energy_delivery_cost,
        'energy_supply': energy_supply_cost,
        'demand_delivery': demand_delivery_cost,
        'demand_supply': demand_supply_cost,
        'fixed_charges': fixed_cost,
        'system_benefits_charge': total_energy_kwh * rate_schedule.system_benefits_charge,
        'clean_energy_surcharge': total_energy_kwh * rate_schedule.clean_energy_surcharge,
        'mac_adjustment': total_energy_kwh * rate_schedule.mac_adjustment,
        'reactive_power': reactive_power_cost,
    }

    return {
        'total_cost': total_cost,
        'energy_cost': total_energy_cost,
        'demand_cost': total_demand_cost,
        'fixed_cost': fixed_cost,
        'surcharges': surcharges,
        'total_energy_kwh': total_energy_kwh,
        'peak_demand_kw': peak_demand_kw,
        'billed_demand_kw': billed_demand_kw,
        'on_peak_energy_kwh': on_peak_energy,
        'off_peak_energy_kwh': off_peak_energy,
        'super_peak_energy_kwh': super_peak_energy,
        'on_peak_demand_kw': on_peak_demand_kw,
        'off_peak_demand_kw': off_peak_demand_kw,
        'is_summer': is_summer,
        'breakdown': breakdown
    }


def calculate_monthly_electricity_cost(
    timestamps: Union[List[datetime], np.ndarray],
    demands_kw: Union[List[float], np.ndarray],
    interval_minutes: Union[int, List[int], np.ndarray] = 15,
    **kwargs
) -> Dict:
    """
    Calculate monthly electricity cost by aggregating daily costs.

    This function processes an entire month's worth of interval data and
    calculates the total monthly bill including demand charges based on
    the monthly peak demand.

    Args:
        timestamps: Array of datetime objects for each demand reading
        demands_kw: Array of demand values in kW
        interval_minutes: Duration of each measurement interval
        **kwargs: Additional arguments passed to calculate_daily_electricity_cost

    Returns:
        Dictionary with monthly cost breakdown
    """
    timestamps = np.asarray(timestamps)
    demands_kw = np.asarray(demands_kw, dtype=float)

    if np.isscalar(interval_minutes):
        interval_minutes = np.full(len(demands_kw), interval_minutes)
    else:
        interval_minutes = np.asarray(interval_minutes)

    # Get rate schedule
    rate_schedule = kwargs.get('rate_schedule', ConEdRateSchedule())
    tou_periods = TimeOfUsePeriods()

    # Extract time components
    months = np.array([t.month for t in timestamps])
    hours = np.array([t.hour for t in timestamps])
    weekdays = np.array([t.weekday() for t in timestamps])
    dates = np.array([t.date() for t in timestamps])

    unique_dates = np.unique(dates)
    is_summer = is_summer_month(months[0])

    # Calculate total energy
    energy_kwh = calculate_interval_energy(demands_kw, interval_minutes)
    total_energy_kwh = np.sum(energy_kwh)

    # Peak demand for the month
    is_peak = np.array([is_peak_hour(h, m, w, tou_periods)
                        for h, m, w in zip(hours, months, weekdays)])

    monthly_peak_demand = np.max(demands_kw)
    on_peak_demand = np.max(demands_kw[is_peak]) if np.any(is_peak) else 0

    # Apply minimum demand
    billed_demand = max(monthly_peak_demand, rate_schedule.minimum_demand_kw)

    # Energy costs (using daily calculation logic but for whole month)
    on_peak_energy = np.sum(energy_kwh[is_peak])
    off_peak_energy = np.sum(energy_kwh[~is_peak])

    include_supply = kwargs.get('include_supply', True)

    # Energy charges
    energy_delivery = total_energy_kwh * rate_schedule.delivery_energy_charge

    if include_supply:
        energy_supply = (
            on_peak_energy * rate_schedule.supply_energy_on_peak +
            off_peak_energy * rate_schedule.supply_energy_off_peak
        )
    else:
        energy_supply = 0

    # Demand charges (monthly, not prorated)
    if is_summer:
        demand_delivery = billed_demand * rate_schedule.delivery_demand_summer
        rate_type = kwargs.get('rate_type', RateType.RATE_II)
        if rate_type == RateType.RATE_II and on_peak_demand > 0:
            demand_delivery += on_peak_demand * (
                rate_schedule.tou_demand_on_peak - rate_schedule.delivery_demand_summer
            )
    else:
        demand_delivery = billed_demand * rate_schedule.delivery_demand_winter

    if include_supply:
        demand_supply = billed_demand * rate_schedule.supply_demand
    else:
        demand_supply = 0

    # Fixed charges (full month)
    fixed_cost = rate_schedule.monthly_customer_charge

    # Surcharges
    surcharges = total_energy_kwh * (
        rate_schedule.system_benefits_charge +
        rate_schedule.clean_energy_surcharge +
        rate_schedule.mac_adjustment
    )

    total_cost = (energy_delivery + energy_supply +
                  demand_delivery + demand_supply +
                  fixed_cost + surcharges)

    return {
        'total_cost': total_cost,
        'energy_cost': energy_delivery + energy_supply,
        'demand_cost': demand_delivery + demand_supply,
        'fixed_cost': fixed_cost,
        'surcharges': surcharges,
        'total_energy_kwh': total_energy_kwh,
        'peak_demand_kw': monthly_peak_demand,
        'billed_demand_kw': billed_demand,
        'on_peak_energy_kwh': on_peak_energy,
        'off_peak_energy_kwh': off_peak_energy,
        'num_days': len(unique_dates),
        'is_summer': is_summer,
        'breakdown': {
            'energy_delivery': energy_delivery,
            'energy_supply': energy_supply,
            'demand_delivery': demand_delivery,
            'demand_supply': demand_supply,
            'fixed_charges': fixed_cost,
            'surcharges': surcharges,
        }
    }


def estimate_demand_charge_savings(
    timestamps: Union[List[datetime], np.ndarray],
    demands_kw: Union[List[float], np.ndarray],
    peak_shaving_kw: float,
    interval_minutes: int = 15,
    **kwargs
) -> Dict:
    """
    Estimate savings from peak demand reduction (e.g., via battery storage).

    Args:
        timestamps: Array of datetime objects
        demands_kw: Array of demand values in kW
        peak_shaving_kw: Amount of peak demand reduction in kW
        interval_minutes: Measurement interval duration
        **kwargs: Additional arguments for cost calculation

    Returns:
        Dictionary with baseline cost, reduced cost, and savings
    """
    demands_kw = np.asarray(demands_kw, dtype=float)

    # Calculate baseline cost
    baseline = calculate_daily_electricity_cost(
        timestamps, demands_kw, interval_minutes, **kwargs
    )

    # Apply peak shaving (reduce demand above threshold)
    threshold = np.max(demands_kw) - peak_shaving_kw
    reduced_demands = np.minimum(demands_kw, threshold)

    # Calculate reduced cost
    reduced = calculate_daily_electricity_cost(
        timestamps, reduced_demands, interval_minutes, **kwargs
    )

    return {
        'baseline_cost': baseline['total_cost'],
        'reduced_cost': reduced['total_cost'],
        'savings': baseline['total_cost'] - reduced['total_cost'],
        'baseline_peak_kw': baseline['peak_demand_kw'],
        'reduced_peak_kw': reduced['peak_demand_kw'],
        'demand_reduction_kw': baseline['peak_demand_kw'] - reduced['peak_demand_kw'],
    }


# Example usage and demonstration
if __name__ == "__main__":
    from datetime import datetime, timedelta

    # Generate sample data for a summer weekday
    base_time = datetime(2024, 7, 15, 0, 0)  # Monday in July
    num_intervals = 96  # 24 hours at 15-minute intervals

    timestamps = [base_time + timedelta(minutes=15*i) for i in range(num_intervals)]

    # Simulate typical commercial building load profile
    hours = np.array([t.hour + t.minute/60 for t in timestamps])

    # Base load with occupancy-driven pattern
    base_load = 400  # kW base load
    occupancy_factor = np.where(
        (hours >= 8) & (hours < 18),  # Business hours
        1.0 + 0.5 * np.sin((hours - 8) * np.pi / 10),  # Peak at noon-ish
        0.4  # Reduced load outside business hours
    )

    # Add some random variation
    np.random.seed(42)
    noise = np.random.normal(0, 20, len(hours))

    demands = base_load * occupancy_factor + noise
    demands = np.maximum(demands, 50)  # Minimum load

    print("=" * 70)
    print("Con Edison Electricity Cost Calculator - Example Output")
    print("=" * 70)
    print(f"\nDate: {base_time.strftime('%Y-%m-%d')} (Summer weekday)")
    print(f"Number of intervals: {num_intervals} (15-minute intervals)")
    print(f"Total hours: {num_intervals * 15 / 60}")

    # Calculate costs for different rate types
    print("\n" + "-" * 70)
    print("STANDARD SC9 RATE II (Time-of-Day)")
    print("-" * 70)

    result = calculate_daily_electricity_cost(
        timestamps=timestamps,
        demands_kw=demands,
        interval_minutes=15,
        rate_type=RateType.RATE_II
    )

    print(f"\nEnergy Consumption:")
    print(f"  Total Energy:      {result['total_energy_kwh']:.1f} kWh")
    print(f"  On-Peak Energy:    {result['on_peak_energy_kwh']:.1f} kWh")
    print(f"  Off-Peak Energy:   {result['off_peak_energy_kwh']:.1f} kWh")
    print(f"  Super-Peak Energy: {result['super_peak_energy_kwh']:.1f} kWh")

    print(f"\nDemand:")
    print(f"  Peak Demand:       {result['peak_demand_kw']:.1f} kW")
    print(f"  On-Peak Demand:    {result['on_peak_demand_kw']:.1f} kW")
    print(f"  Billed Demand:     {result['billed_demand_kw']:.1f} kW")

    print(f"\nCost Breakdown (Daily):")
    print(f"  Energy Charges:    ${result['energy_cost']:.2f}")
    print(f"  Demand Charges:    ${result['demand_cost']:.2f}")
    print(f"  Fixed Charges:     ${result['fixed_cost']:.2f}")
    print(f"  Surcharges:        ${result['surcharges']:.2f}")
    print(f"  -------------------------")
    print(f"  TOTAL DAILY COST:  ${result['total_cost']:.2f}")
    print(f"\n  Estimated Monthly: ${result['total_cost'] * 30:.2f}")

    # Compare with Standby rate
    print("\n" + "-" * 70)
    print("STANDBY RATE (for customers with on-site generation)")
    print("-" * 70)

    standby_result = calculate_daily_electricity_cost(
        timestamps=timestamps,
        demands_kw=demands,
        interval_minutes=15,
        rate_type=RateType.STANDBY,
        contract_demand_kw=500  # Contract for 500 kW standby
    )

    print(f"\nCost Breakdown (Daily):")
    print(f"  Energy Charges:    ${standby_result['energy_cost']:.2f}")
    print(f"  Demand Charges:    ${standby_result['demand_cost']:.2f}")
    print(f"  Fixed Charges:     ${standby_result['fixed_cost']:.2f}")
    print(f"  Surcharges:        ${standby_result['surcharges']:.2f}")
    print(f"  -------------------------")
    print(f"  TOTAL DAILY COST:  ${standby_result['total_cost']:.2f}")

    # Compare with Rider Q
    print("\n" + "-" * 70)
    print("RIDER Q RATE (enhanced standby with 4-hour peak window)")
    print("-" * 70)

    rider_q_result = calculate_daily_electricity_cost(
        timestamps=timestamps,
        demands_kw=demands,
        interval_minutes=15,
        rate_type=RateType.RIDER_Q,
        contract_demand_kw=500
    )

    print(f"\nCost Breakdown (Daily):")
    print(f"  Energy Charges:    ${rider_q_result['energy_cost']:.2f}")
    print(f"  Demand Charges:    ${rider_q_result['demand_cost']:.2f}")
    print(f"  Fixed Charges:     ${rider_q_result['fixed_cost']:.2f}")
    print(f"  Surcharges:        ${rider_q_result['surcharges']:.2f}")
    print(f"  -------------------------")
    print(f"  TOTAL DAILY COST:  ${rider_q_result['total_cost']:.2f}")

    # Demonstrate peak shaving analysis
    print("\n" + "-" * 70)
    print("PEAK SHAVING ANALYSIS (50 kW reduction)")
    print("-" * 70)

    savings = estimate_demand_charge_savings(
        timestamps=timestamps,
        demands_kw=demands,
        peak_shaving_kw=50,
        interval_minutes=15
    )

    print(f"\n  Baseline Peak:     {savings['baseline_peak_kw']:.1f} kW")
    print(f"  Reduced Peak:      {savings['reduced_peak_kw']:.1f} kW")
    print(f"  Demand Reduction:  {savings['demand_reduction_kw']:.1f} kW")
    print(f"\n  Baseline Cost:     ${savings['baseline_cost']:.2f}")
    print(f"  Reduced Cost:      ${savings['reduced_cost']:.2f}")
    print(f"  Daily Savings:     ${savings['savings']:.2f}")
    print(f"  Monthly Savings:   ${savings['savings'] * 30:.2f}")

    print("\n" + "=" * 70)
    print("Note: Rates are based on Con Edison SC9 schedules and should be")
    print("verified against current tariff filings with the NY PSC.")
    print("=" * 70)
