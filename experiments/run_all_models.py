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
