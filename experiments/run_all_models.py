import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.scenario_generator import (
    n_fac, n_cust, scenarios, fac_df, cust_df
)
from src.solver_utils import get_solver
from src.deterministic_model import solve_deterministic
from src.robust_minmax_regret import solve_minmax_regret
from src.stochastic_model import solve_stochastic_expected

# ============================================================
# Output directories
# ============================================================
BASE_RESULTS = "results"
FIG_DIR = os.path.join(BASE_RESULTS, "figures")
DATA_DIR = os.path.join(BASE_RESULTS, "data")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================
# Solver
# ============================================================
solver = get_solver()

# ============================================================
# Deterministic scenario-wise solutions
# ============================================================
opt_costs = {}
opt_x = {}
opt_y = {}

print("Solving deterministic scenario problems...")

for s in scenarios:
    cost, x_sol, y_sol = solve_deterministic(
        n_fac,
        n_cust,
        scenarios[s]['fixed_costs'],
        scenarios[s]['cijs'],
        scenarios[s]['demand'],
        solver
    )
    opt_costs[s] = cost
    opt_x[s] = x_sol
    opt_y[s] = y_sol
    print(f"{s}: Optimal cost = {cost:.2f}")

# ============================================================
# Min–Max Regret optimisation
# ============================================================
print("Solving min–max regret model...")
R_val, x_sol_mm, y_sol_mm = solve_minmax_regret(
    n_fac, n_cust, scenarios, opt_costs, solver
)
print(f"Min–Max Regret R = {R_val:.2f}")

# ============================================================
# Stochastic expected-cost model
# ============================================================
print("Solving stochastic expected-cost model...")
x_sol_stoch = solve_stochastic_expected(
    n_fac, n_cust, scenarios, solver
)

# ============================================================
# Save numerical results
# ============================================================
np.savez(
    os.path.join(DATA_DIR, "mm_solution.npz"),
    x_sol_mm=np.array(x_sol_mm, dtype=int),
    y_sol_mm=y_sol_mm,
    opt_costs=opt_costs,
    x_sol_stoch=np.array(x_sol_stoch, dtype=int)
)

# ============================================================
# Scenario labels
# ============================================================
scenario_labels = {
    's0': 'Base scenario',
    's1': 'Demand spike (1st half)',
    's2': 'Demand spike (2nd half)',
    's3': 'Random demand variation',
    's4': 'High carbon price',
    's5': 'Facility failure',
    's6': 'Transport cost spike',
    's7': 'Combined disruption'
}
# ============================================================
# Load saved results
# ============================================================
sol = np.load(os.path.join(DATA_DIR, "mm_solution.npz"), allow_pickle=True)

x_sol_mm = sol["x_sol_mm"]
y_sol_mm = sol["y_sol_mm"].item()
opt_costs = sol["opt_costs"].item()
x_sol_stoch = sol["x_sol_stoch"]

# ============================================================
# Compute costs and regrets
# ============================================================
opened_mm = [i for i, v in enumerate(x_sol_mm) if v == 1]
opened_stoch = [i for i, v in enumerate(x_sol_stoch) if v == 1]

costs_mm = {}
regrets_mm = {}
costs_stoch = {}
regrets_stoch = {}

for s in scenarios:
    # Robust min–max regret solution
    fixed_mm = sum(scenarios[s]['fixed_costs'][i] * x_sol_mm[i] for i in range(n_fac))
    assign_mm = sum(
        min((scenarios[s]['cijs'][i, j], i) for i in opened_mm)[0]
        * scenarios[s]['demand'][j]
        for j in range(n_cust)
    )
    costs_mm[s] = fixed_mm + assign_mm
    regrets_mm[s] = costs_mm[s] - opt_costs[s]

    # Stochastic expected-cost solution
    fixed_stoch = sum(scenarios[s]['fixed_costs'][i] * x_sol_stoch[i] for i in range(n_fac))
    assign_stoch = sum(
        min((scenarios[s]['cijs'][i, j], i) for i in opened_stoch)[0]
        * scenarios[s]['demand'][j]
        for j in range(n_cust)
    )
    costs_stoch[s] = fixed_stoch + assign_stoch
    regrets_stoch[s] = costs_stoch[s] - opt_costs[s]

# ============================================================
# Facility map plotting function
# ============================================================
def plot_facility_map(
    solution_opened,
    y_sol=None,
    scenario_key='s0',
    title='Facility Map',
    filename=None,
    dashed=False
):
    fig, ax = plt.subplots(figsize=(12, 9))

    ax.scatter(fac_df['x'], fac_df['y'], s=120, color='lightblue', label='Candidate Facilities')
    ax.scatter(cust_df['x'], cust_df['y'], s=80, color='gray', label='Customers')

    colors = plt.cm.tab10.colors
    color_map = {i: colors[k % 10] for k, i in enumerate(solution_opened)}
    max_demand = max(scenarios[scenario_key]['demand'])

    for j in range(n_cust):
        if y_sol is not None:
            col = [y_sol[scenario_key][i][j] for i in solution_opened]
            if np.allclose(col, 0, atol=1e-9):
                facility = min(
                    (scenarios[scenario_key]['cijs'][i, j], i)
                    for i in solution_opened
                )[1]
            else:
                facility = solution_opened[int(np.argmax(col))]
        else:
            facility = min(
                (scenarios[scenario_key]['cijs'][i, j], i)
                for i in solution_opened
            )[1]

        fx, fy = fac_df.loc[facility, ['x', 'y']]
        cx, cy = cust_df.loc[j, ['x', 'y']]
        demand = scenarios[scenario_key]['demand'][j]

        lw = 0.5 + 2.5 * demand / max_demand
        ax.plot(
            [fx, cx], [fy, cy],
            '--' if dashed else '-',
            color=color_map[facility],
            alpha=0.7,
            linewidth=lw
        )

    for i in solution_opened:
        ax.scatter(
            fac_df.loc[i, 'x'],
            fac_df.loc[i, 'y'],
            s=250,
            marker='*',
            color=color_map[i],
            edgecolors='black',
            label=f'Facility {i + 1}'
        )

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(loc='best', fontsize=8)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300)
    plt.show()

# ============================================================
# Facility maps
# ============================================================
plot_facility_map(
    opened_mm,
    y_sol=y_sol_mm,
    scenario_key='s0',
    title="Robust Min–Max Regret Facility Map",
    filename=os.path.join(FIG_DIR, "facility_map_robust.png"),
    dashed=False
)

plot_facility_map(
    opened_stoch,
    y_sol=None,
    scenario_key='s0',
    title="Stochastic Expected-Cost Facility Map",
    filename=os.path.join(FIG_DIR, "facility_map_stoch.png"),
    dashed=True
)

# ============================================================
# Cost comparison plot
# ============================================================
plt.figure(figsize=(10, 5))

x = np.arange(len(scenarios))
width = 0.25

plt.bar(x - width, [costs_mm[s] for s in scenarios], width, label='Robust Min–Max Regret')
plt.bar(x, [costs_stoch[s] for s in scenarios], width, label='Stochastic Expected-Cost')
plt.bar(x + width, [opt_costs[s] for s in scenarios], width, label='Scenario Optimal')

for i in range(len(x)):
    plt.plot([x[i] - width, x[i] - width],
             [opt_costs[list(scenarios.keys())[i]], costs_mm[list(scenarios.keys())[i]]])
    plt.plot([x[i], x[i]],
             [opt_costs[list(scenarios.keys())[i]], costs_stoch[list(scenarios.keys())[i]]])

plt.xticks(x, [scenario_labels[s] for s in scenarios], rotation=30, ha='right')
plt.ylabel("Cost")
plt.title("Robust vs Stochastic vs Scenario-Optimal Costs")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "tradeoff_robust_vs_stoch_with_lines.png"), dpi=300)
plt.show()

# ============================================================
# Regret per scenario
# ============================================================
plt.figure(figsize=(10, 4))

plt.bar(
    np.arange(len(scenarios)) - 0.35 / 2,
    [regrets_mm[s] for s in scenarios],
    0.35,
    label='Robust Regret'
)
plt.bar(
    np.arange(len(scenarios)) + 0.35 / 2,
    [regrets_stoch[s] for s in scenarios],
    0.35,
    label='Stochastic Regret'
)

plt.ylabel("Regret")
plt.title("Regret per Scenario")
plt.xticks(
    np.arange(len(scenarios)),
    [scenario_labels[s] for s in scenarios],
    rotation=30,
    ha='right'
)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "regret_per_scenario_robust_vs_stoch.png"), dpi=300)
plt.show()

# ============================================================
# Cost breakdown (worst-case scenario)
# ============================================================
worst_s = max(regrets_mm, key=lambda s: regrets_mm[s])

def compute_cost_breakdown(opened, scenario_key):
    opened_indices = [i for i, v in enumerate(opened) if v]
    fixed_sum = sum(scenarios[scenario_key]['fixed_costs'][i] for i in opened_indices)

    assign_cost = sum(
        min((scenarios[scenario_key]['cijs'][i, j], i) for i in opened_indices)[0]
        * scenarios[scenario_key]['demand'][j]
        for j in range(n_cust)
    )

    carbon_sum = 0.0
    for j in range(n_cust):
        best_i = min(
            (scenarios[scenario_key]['cijs'][i, j], i)
            for i in opened_indices
        )[1]
        emission_cost = (
            0.01
            * np.sqrt(((fac_df.loc[best_i] - cust_df.loc[j]) ** 2).sum())
            * scenarios[scenario_key]['carbon_price']
            * scenarios[scenario_key]['demand'][j]
        )
        carbon_sum += emission_cost

    transport_sum = assign_cost - carbon_sum
    return [fixed_sum, transport_sum, carbon_sum]

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].pie(
    compute_cost_breakdown(x_sol_mm, worst_s),
    labels=["Fixed", "Transport", "Carbon"],
    autopct="%1.1f%%"
)
axs[0].set_title("Robust Min–Max Regret (Worst Scenario)")

axs[1].pie(
    compute_cost_breakdown(x_sol_stoch, worst_s),
    labels=["Fixed", "Transport", "Carbon"],
    autopct="%1.1f%%"
)
axs[1].set_title("Stochastic Expected-Cost (Worst Scenario)")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "cost_breakdown_worst_comparison.png"), dpi=300)
plt.show()
