import csv
import datetime
import numpy as np
import matplotlib.pyplot as plt
import os

# Production capacities for 2022 (MW), variable costs (€/MWh), and combustion-only
# CO2 emission rates (gCO2eq/kWh), aligned with the provided course figures/tables.

TIME_SLOTS_PER_DAY = 48
SLOT_HOURS = 0.5
EV_MAX_POWER_KW = 7.0

FLEET_SIZE = 15_000_000
DEFAULT_SIM_FLEET_SIZE = 500

# Seed of the random off-peak start times (one generator per scenario).
OFF_PEAK_SEED = 42

# Capacity factors suggested in the TP statement to account for intermittency.
WIND_CAPACITY_FACTOR = 0.3
SOLAR_CAPACITY_FACTOR = 0.2
def get_model_context(nuclear_availability=0.5):

    # Define assets
    assets = [
        {"name": "Wind", "capacity": 19099 * WIND_CAPACITY_FACTOR, "variable_cost": 0, "co2_rate": 0},
        {"name": "Solar", "capacity": 13612 * SOLAR_CAPACITY_FACTOR, "variable_cost": 0, "co2_rate": 0},
        {"name": "Hydro pump.", "capacity": 25504, "variable_cost": 0, "co2_rate": 0},
        {"name": "Bioenergy", "capacity": 2234, "variable_cost": 0, "co2_rate": 494},
        {"name": "Nuclear", "capacity": 61370 * nuclear_availability, "variable_cost": 30, "co2_rate": 0},
        {"name": "Gas", "capacity": 12752, "variable_cost": 70, "co2_rate": 429},
        {"name": "Coal", "capacity": 1818, "variable_cost": 86, "co2_rate": 986},
        {"name": "Fuel", "capacity": 2890, "variable_cost": 162, "co2_rate": 777},
    ]
    
    # Sort assets
    assets.sort(key=lambda a: (a["variable_cost"], a["co2_rate"]))
    
    # Calculate cumulative capacities P_hat
    p_hat = [0]
    for asset in assets:
        p_hat.append(p_hat[-1] + asset["capacity"])
        
    # Calculate Cumulative CO2 Rates
    cumulative_rates = []
    current_rate = 0
    for asset in assets:
        current_rate += asset["co2_rate"]
        cumulative_rates.append(current_rate)

    return {
        "assets": assets,
        "p_hat": p_hat,
        "cumulative_rates": cumulative_rates,
        "f_co2": lambda load: _f_co2_impl(load, assets, p_hat, cumulative_rates),
        "cumulative_capacity_until": lambda name: _cumulative_capacity_until_impl(name, assets)
    }

def _cumulative_capacity_until_impl(asset_name, assets):
    total = 0.0
    for asset in assets:
        total += asset["capacity"]
        if asset["name"] == asset_name:
            return total
    return None

def _f_co2_impl(load, assets, p_hat, cumulative_rates):
    """Calculates total CO2 emissions (g/h) for a given total load (MW) using cumulative rates."""
    total_co2 = 0
    remaining_load = load
    
    # Iterate through assets in merit order
    for i, asset in enumerate(assets):
        capacity = asset["capacity"]
        rate = cumulative_rates[i]
        
        allocated_load = min(remaining_load, capacity)
        if allocated_load > 0:
            total_co2 += allocated_load * 1000 * rate
            remaining_load -= allocated_load
        
        if remaining_load <= 1e-9:
            break
            
    # If load exceeds total capacity, apply last asset's cumulative rate to excess
    if remaining_load > 0:
        total_co2 += remaining_load * 1000 * cumulative_rates[-1]
    return total_co2

def f_co2_total(load_curve, model_ctx):
    """Calculates total CO2 emissions (g) over a period given a load curve (MW)."""
    f = model_ctx["f_co2"]
    return sum(f(L) * SLOT_HOURS for L in load_curve)

def _parse_ec02mix_day(filename):
    with open(filename, "r") as f:
        reader = csv.DictReader(f, delimiter=";")
        first = next(reader, None)
        if not first:
            return None
        dt = datetime.datetime.strptime(first["date"], "%Y-%m-%d %H:%M:%S")
        return dt.strftime("%d/%m/%Y")

def _load_ec02mix_day_curve(filename, day=None):
    nf_load = []
    day_label = None
    try:
        with open(filename, "r") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                dt = datetime.datetime.strptime(row["date"], "%Y-%m-%d %H:%M:%S")
                row_day = dt.strftime("%d/%m/%Y")
                if day is None:
                    if day_label is None:
                        day_label = row_day
                    if row_day != day_label:
                        break
                else:
                    if row_day != day:
                        continue
                    if day_label is None:
                        day_label = row_day
                nf_load.append(float(row["forecast_day-1"]))
                if len(nf_load) >= TIME_SLOTS_PER_DAY:
                    break
    except FileNotFoundError:
        print(f"Warning: File {filename} not found.")
        return None, np.array([])

    if len(nf_load) != TIME_SLOTS_PER_DAY:
        print(f"Warning: Could not load {TIME_SLOTS_PER_DAY} slots for day={day!r} in {filename}.")
        return day_label, np.array([])

    return day_label, np.array(nf_load)

def load_data(year=2019, season="summer", day=None, ev_limit=DEFAULT_SIM_FLEET_SIZE):
    """Loads EV scenarios + non-flexible load curve for a single day (48 slots)."""
    filename = f"data/eCO2mix_RTE_Annuel-Definitif_{year}_{season}.csv"
    if day is None:
        day_label, nf_load = _load_ec02mix_day_curve(filename, day=None)
    else:
        try:
            day_label = datetime.datetime.strptime(day, "%d/%m/%Y").strftime("%d/%m/%Y")
        except ValueError:
            raise ValueError("day must be formatted as dd/mm/YYYY, e.g. 01/06/2019") from None
        _, nf_load = _load_ec02mix_day_curve(filename, day=day_label)

    if nf_load.size == 0:
        return [], np.array([])

    ev_day_label = day_label
    ev_rows = []
    with open("data/ev_scenarios.csv", "r") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            if ev_day_label is None or row.get("day") == ev_day_label:
                ev_rows.append(row)
            if len(ev_rows) >= ev_limit:
                break

    if len(ev_rows) < ev_limit:
        with open("data/ev_scenarios.csv", "r") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                if len(ev_rows) >= ev_limit:
                    break
                if ev_day_label is not None and row.get("day") == ev_day_label:
                    continue
                ev_rows.append(row)

    return ev_rows, nf_load

def run_scenario(year, season, day=None, ev_count=DEFAULT_SIM_FLEET_SIZE, fleet_size=FLEET_SIZE):
    ev_data, nf_load = load_data(year=year, season=season, day=day, ev_limit=ev_count)
    if not ev_data or nf_load.size == 0:
        return None
    
    test_evs = ev_data[:ev_count]
    sim_fleet_size = len(test_evs)
    fleet_scale = fleet_size / max(len(test_evs), 1)
    
    # Create CO2 model context for this scenario
    ctx = get_model_context() 
    
    # 1. Plug-and-Charge
    pc = [policy_plug_and_charge(r) for r in test_evs]
    pc_co2 = calculate_co2_scores(nf_load, pc, fleet_scale, ctx)
    
    # 2. Off-peak (seeded start times)
    op_rng = np.random.default_rng(OFF_PEAK_SEED)
    op = [policy_off_peak(r, op_rng) for r in test_evs]
    op_co2 = calculate_co2_scores(nf_load, op, fleet_scale, ctx)
    
    # 3. BRD
    print(f"\n--- Running BRD for {year} {season} ---")
    brd, history = brd_coordination(test_evs, nf_load, fleet_scale=fleet_scale, ctx=ctx, max_k=10)
    brd_co2 = calculate_co2_scores(nf_load, brd, fleet_scale, ctx)
    
    # Plot load curves for this scenario
    plot_load_curves(
        nf_load,
        pc,
        op,
        brd,
        year,
        season,
        fleet_scale,
        fleet_size,
        f"results/load_{year}_{season}.png",
        ctx=ctx
    )
    
    # Plot EV-only comparison
    plot_ev_load_comparison(pc, op, brd, year, season, len(test_evs), f"results/ev_only_{year}_{season}.png")

    # Water-filling view of the BRD equilibrium
    plot_water_filling(nf_load, brd, fleet_scale, year, season, f"results/water_filling_{year}_{season}.png")

    # Load-shape, emission and threshold indicators for every policy
    op22 = [policy_off_peak_22(r) for r in test_evs]
    policy_profiles = {"base": None, "plug_and_charge": pc, "off_peak": op, "off_peak_22": op22, "brd": brd}
    scenario_metrics = [
        policy_metrics(year, season, name, nf_load, profiles, fleet_scale, fleet_size, len(test_evs), ctx,
                       brd_history=history if name == "brd" else None)
        for name, profiles in policy_profiles.items()
    ]
    
    # Save first convergence history
    if not os.path.exists("results/brd_convergence.png"):
        plot_convergence(history)

    return {
        "year": year,
        "season": season,
        "sim_fleet_size": sim_fleet_size,
        "fleet_size": fleet_size,
        "pc_co2": pc_co2,
        "op_co2": op_co2,
        "brd_co2": brd_co2,
        "metrics": scenario_metrics,
    }

def total_load_mw(nf_load, profiles, fleet_scale):
    """Non-flexible load plus the scaled EV load (MW)."""
    if profiles is None or len(profiles) == 0:
        return np.asarray(nf_load, dtype=float)
    return nf_load + (np.sum(profiles, axis=0) * fleet_scale) / 1000

def policy_metrics(year, season, policy, nf_load, profiles, fleet_scale, fleet_size, sim_fleet_size, ctx,
                   brd_history=None):
    """Emission, load-shape and merit-order threshold indicators of one policy."""
    base_load = np.asarray(nf_load, dtype=float)
    load = total_load_mw(nf_load, profiles, fleet_scale)
    ev_load = load - base_load
    base_co2 = f_co2_total(base_load, ctx)
    total_co2 = f_co2_total(load, ctx)
    ev_energy_kwh = float(np.sum(ev_load) * SLOT_HOURS * 1000)
    nuclear_thr = ctx["cumulative_capacity_until"]("Nuclear")
    gas_thr = ctx["cumulative_capacity_until"]("Gas")
    row = {
        "year": year,
        "season": season,
        "policy": policy,
        "J": sim_fleet_size,
        "fleet_size": fleet_size,
        "total_co2_tons": total_co2 / 1e6,
        "ev_increment_co2_tons": (total_co2 - base_co2) / 1e6,
        "ev_energy_gwh": ev_energy_kwh / 1e6,
        "ev_intensity_g_per_kwh": (total_co2 - base_co2) / ev_energy_kwh if ev_energy_kwh > 0 else 0.0,
        "total_load_peak_mw": float(np.max(load)),
        "total_load_mean_mw": float(np.mean(load)),
        "total_load_std_mw": float(np.std(load)),
        "ev_load_peak_mw": float(np.max(ev_load)),
        "base_peak_mw": float(np.max(base_load)),
        "base_mean_mw": float(np.mean(base_load)),
        "base_std_mw": float(np.std(base_load)),
        "brd_iters": len(brd_history) if brd_history is not None else "",
        "brd_final_move_norm": float(brd_history[-1]) if brd_history else "",
        "nuclear_threshold_mw": nuclear_thr,
        "gas_threshold_mw": gas_thr,
        "hours_above_nuclear": float(np.sum(load > nuclear_thr) * SLOT_HOURS),
        "hours_above_gas": float(np.sum(load > gas_thr) * SLOT_HOURS),
        "mw_hours_above_nuclear": float(np.sum(np.maximum(load - nuclear_thr, 0.0)) * SLOT_HOURS),
        "mw_hours_above_gas": float(np.sum(np.maximum(load - gas_thr, 0.0)) * SLOT_HOURS),
    }
    return row

INTERMEDIATE_FIELDS = [
    "year", "season", "policy", "J", "fleet_size", "total_co2_tons", "ev_increment_co2_tons", "ev_energy_gwh",
    "ev_intensity_g_per_kwh", "total_load_peak_mw", "total_load_mean_mw", "total_load_std_mw", "ev_load_peak_mw",
    "base_peak_mw", "base_mean_mw", "base_std_mw", "brd_iters", "brd_final_move_norm",
]
THRESHOLD_FIELDS = [
    "year", "season", "policy", "nuclear_threshold_mw", "gas_threshold_mw", "hours_above_nuclear",
    "hours_above_gas", "mw_hours_above_nuclear", "mw_hours_above_gas",
]

def write_metrics_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def _available_slots(time_slot_arr, time_slot_dep):
    a_j, d_j = int(time_slot_arr), int(time_slot_dep)
    if a_j == d_j:
        return []
    if a_j < d_j:
        return list(range(a_j + 1, d_j))
    return list(range(a_j + 1, TIME_SLOTS_PER_DAY + 1)) + list(range(1, d_j))

def policy_plug_and_charge(ev_row):
    """Plug-and-Charge: Charge max power until energy need is met."""
    a_j, d_j, need_energy = int(ev_row['time_slot_arr']), int(ev_row['time_slot_dep']), float(ev_row['energy_need (kWh)'])
    profile = np.zeros(TIME_SLOTS_PER_DAY)
    rem = need_energy
    slots = _available_slots(a_j, d_j)
    for t in slots:
        charge = min(EV_MAX_POWER_KW, rem / SLOT_HOURS)
        profile[t-1] = charge
        rem -= charge * SLOT_HOURS
        if rem <= 1e-9:
            break
    return profile

def policy_off_peak(ev_row, rng=None):
    """Off-peak: randomized start in the off-peak window 22:00-06:00 (slots 45..60, taken modulo 48)."""
    a_j, d_j, need_energy = int(ev_row['time_slot_arr']), int(ev_row['time_slot_dep']), float(ev_row['energy_need (kWh)'])
    profile = np.zeros(TIME_SLOTS_PER_DAY)
    rem = need_energy
    avail = _available_slots(a_j, d_j)
    
    # Spread start times over 16 slots (8 hours) to minimize peak
    rng = np.random.default_rng(OFF_PEAK_SEED) if rng is None else rng
    start_slot = 45 + int(rng.integers(0, 16))
    
    for t in sorted(avail, key=lambda s: (s - start_slot) % TIME_SLOTS_PER_DAY):
        charge = min(EV_MAX_POWER_KW, rem / SLOT_HOURS)
        profile[t-1] = charge
        rem -= charge * SLOT_HOURS
        if rem <= 1e-9:
            break
    return profile

def policy_off_peak_22(ev_row):
    """Synchronized off-peak tariff: every EV starts charging at 22:00 (slot 45) at full power."""
    a_j, d_j, need_energy = int(ev_row['time_slot_arr']), int(ev_row['time_slot_dep']), float(ev_row['energy_need (kWh)'])
    profile = np.zeros(TIME_SLOTS_PER_DAY)
    rem = need_energy
    for t in sorted(_available_slots(a_j, d_j), key=lambda s: (s - 45) % TIME_SLOTS_PER_DAY):
        charge = min(EV_MAX_POWER_KW, rem / SLOT_HOURS)
        profile[t-1] = charge
        rem -= charge * SLOT_HOURS
        if rem <= 1e-9:
            break
    return profile

def water_filling(ev_row, baseline_total_load_mw, epsilon=0.1, kw_to_mw=1 / 1000):
    """Individual EV optimization: Water-Filling heuristic."""
    a_j, d_j, need_energy = int(ev_row['time_slot_arr']), int(ev_row['time_slot_dep']), float(ev_row['energy_need (kWh)'])
    target = need_energy / SLOT_HOURS
    avail = _available_slots(a_j, d_j)
    profile = np.zeros(TIME_SLOTS_PER_DAY)
    curr_sum = 0
    while curr_sum < target - 1e-9:
        non_sat = [t for t in avail if profile[t-1] < EV_MAX_POWER_KW]
        if not non_sat:
            break
        
        # Calculate marginal costs (proxy using load levels)
        min_l = min(baseline_total_load_mw[t-1] + profile[t-1] * kw_to_mw for t in non_sat)
        min_s = [t for t in non_sat if (baseline_total_load_mw[t-1] + profile[t-1] * kw_to_mw) <= min_l + 1e-9]
        inc = min(epsilon, (target - curr_sum) / len(min_s))
        for t in min_s:
            add = min(inc, EV_MAX_POWER_KW - profile[t-1])
            profile[t-1] += add
            curr_sum += add
    return profile

def plot_f_co2(ctx, output_path="results/f_co2_curve.png"):
    """Plots f_CO2(load): total CO2 emissions per hour as a function of load."""
    p_hat = ctx["p_hat"]
    assets = ctx["assets"]
    f = ctx["f_co2"]

    max_capacity = p_hat[-1] + 5000 
    loads = np.linspace(0, max_capacity, 600)
    co2_kt_per_h = np.array([f(L) for L in loads]) / 1e9  # g/h -> kt/h

    plt.figure(figsize=(10, 6))
    plt.plot(loads, co2_kt_per_h, color="#ef476f", linewidth=2)

    # Merit-order capacity thresholds
    for i, asset in enumerate(assets):
        plt.axvline(x=p_hat[i + 1], color="gray", linestyle="--", alpha=0.25)

    plt.xlabel("Total Demand (MW)")
    plt.ylabel("CO2 emissions (kt/h)")
    plt.title("French CO2 Emissions vs Total Demand (f_CO2)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path)
    print(f"Saved f_CO2 curve: {output_path}")
    plt.close()

def plot_marginal_co2_rate(ctx, output_path="results/co2_marginal_rate.png"):
    """Plots the (piecewise-constant) marginal CO2 rate (gCO2/kWh) as a function of load."""
    p_hat = ctx["p_hat"]
    cumulative_rates = ctx["cumulative_rates"]
    assets = ctx["assets"]

    max_capacity = p_hat[-1] + 5000 
    loads = np.linspace(0, max_capacity, 600)
    rates = []

    for L in loads:
        rate = cumulative_rates[-1]
        for i in range(len(assets)):
            if L <= p_hat[i + 1] + 1e-9:
                rate = cumulative_rates[i]
                break
        rates.append(rate)

    plt.figure(figsize=(10, 6))
    plt.step(loads, rates, where="post", color="#00a8e8", linewidth=2)
    plt.fill_between(loads, rates, step="post", alpha=0.2, color="#00a8e8")

    for i, asset in enumerate(assets):
        plt.axvline(x=p_hat[i + 1], color="gray", linestyle="--", alpha=0.25)

    plt.xlabel("Total Demand (MW)")
    plt.ylabel("Marginal CO2 rate (gCO2/kWh)")
    plt.title("Marginal CO2 Content vs Total Demand (Merit Order Model)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path)
    print(f"Saved marginal CO2 rate: {output_path}")
    plt.close()

def plot_valley_filling_demo(year=2019, season="winter", output_path="results/valley_filling_demo.png",
                             ev_limit=DEFAULT_SIM_FLEET_SIZE, nuclear_availability=0.5, max_k=10):
    """Generates a valley-filling illustration (NF load vs NF+BRD EV load)."""
    print(f"Generating Valley Filling Demo for {year} {season}...")

    ev_data, nf_load = load_data(year=year, season=season, ev_limit=ev_limit)
    if nf_load.size == 0:
        print("Error: No data found.")
        return

    test_evs = ev_data[:ev_limit]
    fleet_scale = FLEET_SIZE / max(len(test_evs), 1)

    ctx = get_model_context(nuclear_availability)
    brd_profiles, _ = brd_coordination(test_evs, nf_load, fleet_scale=fleet_scale, ctx=ctx, max_k=max_k)

    total_brd_load = nf_load + (np.sum(brd_profiles, axis=0) * fleet_scale) / 1000  # kW -> MW

    plt.figure(figsize=(12, 7))
    plt.plot(nf_load, label="Non-Flexible Load (Base)", color="black", linewidth=2)
    plt.plot(total_brd_load, label="Non-Flexible + Smart EV Load (Valley Filling)", color="#2a9d8f", linewidth=2.5)
    plt.fill_between(range(len(nf_load)), nf_load, total_brd_load, color="#2a9d8f", alpha=0.3, label="EV Charging Load")

    trough_idx = int(np.argmin(nf_load))
    y_span = float(np.max(total_brd_load) - np.min(nf_load))
    y_offset = 0.15 * y_span if y_span > 0 else 5000
    plt.annotate(
        'Valley Filling\n("Pouring Water")',
        xy=(trough_idx, float(total_brd_load[trough_idx])),
        xytext=(trough_idx, float(total_brd_load[trough_idx] + y_offset)),
        arrowprops=dict(facecolor="black", shrink=0.05),
        ha="center",
        fontsize=10,
        backgroundcolor="white",
    )

    plt.title(f"Demonstration of Valley Filling (Smart Charging)\n{year} {season.capitalize()}", fontsize=14)
    plt.xlabel("Time Slot (30 min)", fontsize=12)
    plt.ylabel("Power Load (MW)", fontsize=12)
    plt.legend(loc="upper right", fontsize=11)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Saved plot to {output_path}")
    plt.close()

def plot_water_filling(nf_load, brd_profiles, fleet_scale, year, season, output_path):
    """BRD equilibrium seen as water-filling: EV energy fills the valleys of the non-flexible load."""
    total = total_load_mw(nf_load, brd_profiles, fleet_scale)
    slots = np.arange(len(nf_load))
    plt.figure(figsize=(10, 6))
    plt.fill_between(slots, 0, nf_load, color="lightgray", alpha=0.35)
    plt.fill_between(slots, nf_load, total, color="#00b4d8", alpha=0.5, label='EV Charging Load ("Water")')
    plt.plot(slots, nf_load, color="black", linewidth=2, label="Non-flexible Load")
    plt.plot(slots, total, color="#0077b6", linewidth=2, label="Total Load (with Smart Charging)")
    plt.title(f"Water Filling Effect - {year} {season.capitalize()}")
    plt.xlabel("Time Slot (30 min)")
    plt.ylabel("Power (MW)")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved water-filling plot: {output_path}")
    plt.close()

def plot_convergence(history, output_path="results/brd_convergence.png"):
    """Plots the convergence history of the BRD algorithm."""
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(history) + 1), history, marker='o', color='#5e60ce', linewidth=2)
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Move Norm (Log Scale)')
    plt.title('BRD Algorithm Convergence History')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved convergence plot: {output_path}")
    plt.close()

def plot_co2_comparison(results, output_path="results/co2_comparison.png"):
    """Generates a bar chart comparing CO2 emissions across scenarios and policies."""
    scenarios = [f"{r['year']} {r['season'].capitalize()}" for r in results]
    pc = [r['pc_co2'] / 1e6 for r in results]
    op = [r['op_co2'] / 1e6 for r in results]
    brd = [r['brd_co2'] / 1e6 for r in results]

    x = np.arange(len(scenarios))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, pc, width, label='Plug-and-Charge', color='#ffca3a')
    ax.bar(x, op, width, label='Off-peak (22:00-06:00)', color='#ff595e')
    ax.bar(x + width, brd, width, label='BRD (Smart)', color='#8ac926')

    ax.set_ylabel('Total CO2 (Tons)')
    ax.set_title('CO2 Emissions Comparison Across Scenarios')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved plot: {output_path}")
    plt.close()

def plot_load_curves(nf_load, pc_profiles, op_profiles, brd_profiles, year, season, fleet_scale, fleet_size, output_path, ctx=None):
    """Visualizes the load profiles for a specific scenario."""
    total_pc = nf_load + (np.sum(pc_profiles, axis=0) * fleet_scale) / 1000
    total_op = nf_load + (np.sum(op_profiles, axis=0) * fleet_scale) / 1000
    total_brd = nf_load + (np.sum(brd_profiles, axis=0) * fleet_scale) / 1000

    plt.figure(figsize=(12, 6))
    plt.plot(nf_load, label='Base (Non-flexible)', color='black', linestyle='--', alpha=0.5)
    plt.plot(total_pc, label='Plug-and-Charge', color='#ffca3a', alpha=0.8)
    plt.plot(total_op, label='Off-peak', color='#ff595e', alpha=0.8)
    plt.plot(total_brd, label='BRD (Smart)', color='#8ac926', linewidth=2)

    # Merit order thresholds
    if ctx:
        nuclear_threshold = ctx["cumulative_capacity_until"]("Nuclear")
        gas_threshold = ctx["cumulative_capacity_until"]("Gas")
        if nuclear_threshold is not None:
            plt.axhline(y=nuclear_threshold, color='blue', linestyle=':', alpha=0.3, label='Nuclear Threshold')
        if gas_threshold is not None:
            plt.axhline(y=gas_threshold, color='red', linestyle=':', alpha=0.3, label='Fossil Threshold')

    plt.title(f'Total Grid Load - {year} {season.capitalize()} (Fleet: {fleet_size/1e6:.0f}M EVs)')
    plt.xlabel('Time Slot (30 min)')
    plt.ylabel('Load (MW)')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved load curve plot: {output_path}")
    plt.close()

def plot_ev_load_comparison(pc_profiles, op_profiles, brd_profiles, year, season, sim_fleet_size, output_path):
    """Visualizes only the EV charging profiles to compare policy behaviors."""
    pc_agg = np.sum(pc_profiles, axis=0)
    op_agg = np.sum(op_profiles, axis=0)
    brd_agg = np.sum(brd_profiles, axis=0)

    plt.figure(figsize=(12, 6))
    plt.plot(pc_agg, label='Plug-and-Charge', color='#ffca3a', alpha=0.8, linewidth=1.5)
    plt.plot(op_agg, label='Off-peak', color='#ff595e', alpha=0.8, linewidth=1.5)
    plt.plot(brd_agg, label='BRD (Smart)', color='#8ac926', linewidth=2.5)

    plt.title(f'Aggregated EV Charging Profile - {year} {season.capitalize()} (Sim Fleet: {sim_fleet_size} EVs)')
    plt.xlabel('Time Slot (30 min)')
    plt.ylabel('Total EV Charging Power (kW)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved EV comparison plot: {output_path}")
    plt.close()

def calculate_co2_scores(nf_load, profiles, fleet_scale, ctx):
    total_ev = np.sum(profiles, axis=0) if len(profiles) > 0 else np.zeros(TIME_SLOTS_PER_DAY)
    # Scale from sim fleet to total fleet size
    total_load_mw = nf_load + (total_ev * fleet_scale) / 1000
    return f_co2_total(total_load_mw, ctx)

def brd_coordination(ev_data, nf_load, fleet_scale, ctx, epsilon=0.1, eta=1.0, max_k=20):
    """Decentralized coordination using Best-Response-Dynamics."""
    kw_to_mw = fleet_scale / 1000
    J = len(ev_data)
    profiles = [np.zeros(TIME_SLOTS_PER_DAY) for _ in range(J)]
    agg_ev_load_kw = np.zeros(TIME_SLOTS_PER_DAY)
    history = []
    
    # Water-filling heuristic is independent of f_co2
    
    for k in range(1, max_k + 1):
        prev_profiles = [p.copy() for p in profiles]
        
        for j, row in enumerate(ev_data):
            other_agg_kw = agg_ev_load_kw - profiles[j]
            baseline_total_mw = nf_load + other_agg_kw * kw_to_mw
            new_p = water_filling(row, baseline_total_mw, epsilon=epsilon, kw_to_mw=kw_to_mw)
            agg_ev_load_kw = agg_ev_load_kw - profiles[j] + new_p
            profiles[j] = new_p
            
        diff = 0
        for j in range(J):
            diff += np.sum((profiles[j] - prev_profiles[j])**2)
            
        history.append(diff)
        print(f"Iteration {k}, Move norm: {diff:.2f}")
        if diff < eta:
            break
            
    return profiles, history

def run_sensitivity_analysis():
    print("\n" + "="*80)
    print("RUNNING NUCLEAR AVAILABILITY SENSITIVITY ANALYSIS")
    print("="*80)
    
    factors = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    year, season = 2019, "winter" # High impact scenario
    
    ev_count = DEFAULT_SIM_FLEET_SIZE
    ev_data, nf_load = load_data(year=year, season=season, ev_limit=ev_count)
    if not ev_data or nf_load.size == 0:
        print("Error: Could not load data for sensitivity analysis")
        return

    test_evs = ev_data[:ev_count]
    fleet_scale = FLEET_SIZE / max(len(test_evs), 1)
    
    print("Running BRD Coordination (Base Case)...")
    # Use a dummy context for coordination (water filling doesn't use it anyway)
    dummy_ctx = get_model_context() 
    brd, _ = brd_coordination(test_evs, nf_load, fleet_scale, ctx=dummy_ctx, max_k=10)
    pc = [policy_plug_and_charge(r) for r in test_evs]
    
    results = []
    
    print(f"{'Factor':<10} | {'P&C (t)':<15} | {'BRD (t)':<15} | {'Reduction %':<15}")
    print("-" * 65)
    
    for av in factors:
        ctx = get_model_context(av)
        
        pc_co2 = calculate_co2_scores(nf_load, pc, fleet_scale, ctx)
        brd_co2 = calculate_co2_scores(nf_load, brd, fleet_scale, ctx)
        
        red = (pc_co2 - brd_co2) / pc_co2 * 100
        results.append({
            "availability": av,
            "pc_co2": pc_co2,
            "brd_co2": brd_co2,
            "reduction": red
        })
        print(f"{av:<10.1f} | {pc_co2/1e6:<15.1f} | {brd_co2/1e6:<15.1f} | {red:<15.2f}")
        
    # Plotting
    avs = [r["availability"] for r in results]
    reds = [r["reduction"] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(avs, reds, marker='o', linestyle='-', color='#ef476f', linewidth=2)
    plt.xlabel('Nuclear Availability Factor')
    plt.ylabel('CO2 Reduction from Smart Charging (%)')
    plt.title(f'Impact of Nuclear Availability on Smart Charging Benefits ({year} {season.capitalize()})')
    plt.grid(True, alpha=0.3)
    plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Context 1.1 (0.5)')
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/nuclear_sensitivity.png")
    print("\nSaved sensitivity plot to results/nuclear_sensitivity.png")

if __name__ == "__main__":
    if not os.path.exists("results"):
        os.makedirs("results")

    scenarios = [
        (2019, "summer"),
        (2019, "winter"),
        (2020, "summer"),
        (2020, "winter")
    ]
    
    results = []
    for yr, ss in scenarios:
        res = run_scenario(yr, ss)
        if res:
            results.append(res)
            
    # Generate overall CO2 comparison plot
    plot_co2_comparison(results)
    
    # Generate f_CO2 curve plot (using default context)
    plot_f_co2(get_model_context())
    
    # Mean-field valley-filling illustration (2019 winter)
    plot_valley_filling_demo()

    # Run Sensitivity Analysis
    run_sensitivity_analysis()

    # Policy-level indicators (load shape, emissions, merit-order thresholds)
    all_metrics = [m for r in results for m in r["metrics"]]
    write_metrics_csv("results/intermediate_metrics.csv", all_metrics, INTERMEDIATE_FIELDS)
    write_metrics_csv("results/threshold_metrics.csv", all_metrics, THRESHOLD_FIELDS)

    # Save summary as CSV for organization
    with open("results/summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["year", "season", "sim_fleet_size", "fleet_size", "pc_co2", "op_co2", "brd_co2", "red_pct"],
            extrasaction="ignore",
        )
        writer.writeheader()
        for r in results:
            red = (r['pc_co2'] - r['brd_co2']) / r['pc_co2'] * 100
            writer.writerow({**r, "red_pct": red})

    print("\n" + "="*80)
    print("COMPARATIVE RESULTS (Total CO2 over 24h)")
    if results:
        sim_size = results[0].get("sim_fleet_size", DEFAULT_SIM_FLEET_SIZE)
        fleet_size = results[0].get("fleet_size", FLEET_SIZE)
    else:
        sim_size = DEFAULT_SIM_FLEET_SIZE
        fleet_size = FLEET_SIZE
    print(f"Scenario: {sim_size} EVs scaled to {fleet_size/1e6:.1f}M vehicles")
    print("-" * 80)
    print(f"{'Year':<6} | {'Season':<8} | {'P&C (t)':<10} | {'OffP (t)':<10} | {'BRD (t)':<10} | {'Red %':<10}")
    print("-" * 80)
    for r in results:
        red = (r['pc_co2'] - r['brd_co2']) / r['pc_co2'] * 100
        print(f"{r['year']:<6} | {r['season']:<8} | {r['pc_co2']/1e6:<10.1f} | {r['op_co2']/1e6:<10.1f} | {r['brd_co2']/1e6:<10.1f} | {red:<10.1f}")
    print("="*80)
    print("Detailed results saved to 'results/' folder.")
