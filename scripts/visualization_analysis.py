import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scenario_generator import fac_df, cust_df, scenarios, n_fac, n_cust
# ============================================================
# Descriptive names for scenarios
# ============================================================
scenario_labels = {
    's0': 'Base scenario',
    's1': 'Demand spike 1st half',
    's2': 'Demand spike 2nd half',
    's3': 'Random demand variation',
    's4': 'High carbon price',
    's5': 'Facility 3 down',
    's6': 'Transport multiplier spike',
    's7': 'Combined disruption'
}

# ============================================================
# Load solutions
# ============================================================
sol = np.load("mm_solution.npz", allow_pickle=True)
x_sol_mm = sol["x_sol_mm"]
y_sol_mm = sol["y_sol_mm"].item()
opt_costs = sol["opt_costs"].item()
x_sol_stoch = sol["x_sol_stoch"]

# ============================================================
# Prepare directories
# ============================================================
output_dir = "figures"
os.makedirs(output_dir, exist_ok=True)

# ============================================================
# Compute costs and regrets
# ============================================================
opened_mm = [i for i, v in enumerate(x_sol_mm) if v == 1]
opened_stoch = [i for i, v in enumerate(x_sol_stoch) if v == 1]

costs_mm = {}
regrets = {}
costs_stoch = {}
regrets_stoch = {}

for s in scenarios:
    # Robust min-max regret
    fixed_mm = sum(scenarios[s]['fixed_costs'][i] * x_sol_mm[i] for i in range(n_fac))
    assign_mm = sum(min((scenarios[s]['cijs'][i, j], i) for i in opened_mm)[0] * scenarios[s]['demand'][j] for j in range(n_cust))
    costs_mm[s] = fixed_mm + assign_mm
    regrets[s] = costs_mm[s] - opt_costs[s]

    # Stochastic expected-cost solution
    fixed_stoch = sum(scenarios[s]['fixed_costs'][i] * x_sol_stoch[i] for i in range(n_fac))
    assign_stoch = sum(min((scenarios[s]['cijs'][i, j], i) for i in opened_stoch)[0] * scenarios[s]['demand'][j] for j in range(n_cust))
    costs_stoch[s] = fixed_stoch + assign_stoch
    regrets_stoch[s] = costs_stoch[s] - opt_costs[s]

# ============================================================
# Facility map function
# ============================================================
def plot_facility_map(solution_opened, y_sol=None, scenario_key='s0', title='Facility Map', filename=None, dashed=False):
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
                facility = min((scenarios[scenario_key]['cijs'][i, j], i) for i in solution_opened)[1]
            else:
                facility = solution_opened[int(np.argmax(col))]
        else:
            facility = min((scenarios[scenario_key]['cijs'][i, j], i) for i in solution_opened)[1]

        fx, fy = fac_df.loc[facility, ['x', 'y']]
        cx, cy = cust_df.loc[j, ['x', 'y']]
        demand = scenarios[scenario_key]['demand'][j]
        lw = 0.5 + 2.5 * demand / max_demand
        ax.plot([fx, cx], [fy, cy], '--' if dashed else '-', color=color_map[facility], alpha=0.7, linewidth=lw)

    for i in solution_opened:
        ax.scatter(fac_df.loc[i, 'x'], fac_df.loc[i, 'y'], s=250, marker='*', color=color_map[i],
                   edgecolors='black', label=f'Facility {i + 1}')

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(loc='best', fontsize=8)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300)
    plt.show()

# ============================================================
# Plot robust and stochastic facility maps
# ============================================================
plot_facility_map(opened_mm, y_sol=y_sol_mm, scenario_key='s0',
                  title="Robust Min-Max Regret Facility Map",
                  filename=os.path.join(output_dir, "facility_map_robust.png"),
                  dashed=False)

plot_facility_map(opened_stoch, y_sol=None, scenario_key='s0',
                  title="Stochastic Expected-Cost Facility Map",
                  filename=os.path.join(output_dir, "facility_map_stoch.png"),
                  dashed=True)

# ============================================================
# Trade-off plot: Robust vs Stochastic vs Scenario Optima
# ============================================================
plt.figure(figsize=(10, 5))
x = np.arange(len(scenarios))
width = 0.25

robust_list = [costs_mm[s] for s in scenarios]
stoch_list = [costs_stoch[s] for s in scenarios]
opt_list = [opt_costs[s] for s in scenarios]

plt.bar(x - width, robust_list, width, label='Robust Min-Max Regret', color='skyblue')
plt.bar(x, stoch_list, width, label='Stochastic Expected-Cost', color='lightgreen')
plt.bar(x + width, opt_list, width, label='Scenario Optimal', color='lightgray')

# Draw vertical lines for regret
for i in range(len(x)):
    plt.plot([x[i] - width, x[i] - width], [opt_list[i], robust_list[i]], 'b-', alpha=0.5)
    plt.plot([x[i], x[i]], [opt_list[i], stoch_list[i]], 'g-', alpha=0.5)

plt.xticks(x, [scenario_labels[s] for s in scenarios], rotation=30, ha='right')
plt.ylabel("Cost")
plt.title("Candidate vs Stochastic vs Scenario Optimal Costs (with Regret Lines)")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "tradeoff_robust_vs_stoch_with_lines.png"), dpi=300)
plt.show()

# ============================================================
# Regret per scenario
# ============================================================
plt.figure(figsize=(10, 4))
width = 0.35
scenario_names = [scenario_labels[s] for s in scenarios]

plt.bar(np.arange(len(scenarios)) - width / 2, [regrets[s] for s in scenarios],
        width, label='Robust Regret', color='skyblue')
plt.bar(np.arange(len(scenarios)) + width / 2, [regrets_stoch[s] for s in scenarios],
        width, label='Expected Regret under Scenario Probabilities', color='lightgreen')

plt.ylabel("Regret (Candidate - Scenario Optimal)")
plt.title("Regret per Scenario")
plt.xticks(np.arange(len(scenarios)), scenario_names, rotation=30, ha='right')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "regret_per_scenario_robust_vs_stoch.png"), dpi=300)
plt.show()

# ============================================================
# Cost breakdown pie chart (worst-case scenario)
# ============================================================
worst_s = max(regrets, key=lambda s: regrets[s])
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

def compute_cost_breakdown(opened, scenario_key):
    opened_indices = [i for i, v in enumerate(opened) if v]
    fixed_sum = sum(scenarios[scenario_key]['fixed_costs'][i] for i in opened_indices)
    assign_cost = sum(min((scenarios[scenario_key]['cijs'][i, j], i) for i in opened_indices)[0] * scenarios[scenario_key]['demand'][j] for j in range(n_cust))
    carbon_sum = 0.0
    for j in range(n_cust):
        best_i = min((scenarios[scenario_key]['cijs'][i, j], i) for i in opened_indices)[1]
        emission_cost = 0.01 * np.sqrt(((fac_df.loc[best_i] - cust_df.loc[j])**2).sum()) * scenarios[scenario_key]['carbon_price'] * scenarios[scenario_key]['demand'][j]
        carbon_sum += emission_cost
    transport_sum = assign_cost - carbon_sum
    return [fixed_sum, transport_sum, carbon_sum]

# Robust pie
axs[0].pie(compute_cost_breakdown(x_sol_mm, worst_s),
           labels=["Fixed", "Transport", "Carbon"], autopct="%1.1f%%",
           colors=["orange", "lightgreen", "lightcoral"])
axs[0].set_title("Robust Min-Max Regret - Worst Scenario")

# Stochastic pie
axs[1].pie(compute_cost_breakdown(x_sol_stoch, worst_s),
           labels=["Fixed", "Transport", "Carbon"], autopct="%1.1f%%",
           colors=["orange", "lightgreen", "lightcoral"])
axs[1].set_title("Stochastic Expected-Cost - Worst Scenario")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "cost_breakdown_worst_comparison.png"), dpi=300)
plt.show()

# ============================================================
# Trade-off plot: Expected Cost vs Worst-Case Regret
# ============================================================

# Compute expected cost for each solution
expected_cost_mm = sum(
    scenarios[s]['prob'] * costs_mm[s]
    for s in scenarios
)

expected_cost_stoch = sum(
    scenarios[s]['prob'] * costs_stoch[s]
    for s in scenarios
)

# Worst-case regret
worst_regret_mm = max(regrets[s] for s in scenarios)
worst_regret_stoch = max(regrets_stoch[s] for s in scenarios)

plt.figure(figsize=(7, 6))

plt.scatter(expected_cost_mm, worst_regret_mm, s=120, marker='o', label='Robust Min–Max Regret')
plt.scatter(expected_cost_stoch, worst_regret_stoch, s=120, marker='^', label='Stochastic Expected-Cost')

# Annotate points
plt.annotate("Robust",
             (expected_cost_mm, worst_regret_mm),
             textcoords="offset points",
             xytext=(8, -10))

plt.annotate("Stochastic",
             (expected_cost_stoch, worst_regret_stoch),
             textcoords="offset points",
             xytext=(8, -10))

plt.xlabel("Expected Total Cost")
plt.ylabel("Worst-Case Regret")
plt.title("Efficiency–Robustness Trade-off")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "expected_cost_vs_worst_regret.png"), dpi=300)
plt.show()
