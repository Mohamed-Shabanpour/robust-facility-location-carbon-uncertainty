import pyomo.environ as pyo
import numpy as np
from scenario_generator import n_fac, n_cust, scenarios

# -------------------------
# Solver selection
# -------------------------
def get_solver():
    for s in ['highs', 'cbc', 'glpk']:
        try:
            solver = pyo.SolverFactory(s)
            if solver.available():
                print(f"Using solver: {s}")
                return solver
        except Exception:
            continue
    raise Exception("No suitable solver available")

solver = get_solver()

# -------------------------
# Deterministic scenario-wise solve
# -------------------------
def solve_deterministic(fac_count, cust_count, fixed_costs, cijs, demands, solver):
    I = range(fac_count)
    J = range(cust_count)
    model = pyo.ConcreteModel()
    model.x = pyo.Var(I, within=pyo.Binary)
    model.y = pyo.Var(I, J, within=pyo.NonNegativeReals)

    model.obj = pyo.Objective(
        expr=sum(fixed_costs[i]*model.x[i] for i in I) +
             sum(cijs[i,j]*model.y[i,j] for i in I for j in J),
        sense=pyo.minimize
    )

    model.demand = pyo.Constraint(J, rule=lambda m,j: sum(m.y[i,j] for i in I) == demands[j])
    model.assign_open = pyo.Constraint(I,J, rule=lambda m,i,j: m.y[i,j] <= demands[j]*m.x[i])

    res = solver.solve(model, tee=False)
    x_sol = [int(round(pyo.value(model.x[i]))) for i in I]
    y_sol = np.array([[pyo.value(model.y[i,j]) for j in J] for i in I])
    cost = float(pyo.value(model.obj))
    return cost, x_sol, y_sol

# -------------------------
# Compute scenario optima
# -------------------------
opt_costs = {}
opt_x = {}
opt_y = {}

print("Solving deterministic scenario problems...")
for s in scenarios:
    cijs = scenarios[s]['cijs']
    demands = scenarios[s]['demand']
    fixed_costs = scenarios[s]['fixed_costs']
    cost, x_sol, y_sol = solve_deterministic(n_fac, n_cust, fixed_costs, cijs, demands, solver)
    opt_costs[s] = cost
    opt_x[s] = x_sol
    opt_y[s] = y_sol
    print(f"{s}: Optimal cost={cost:.2f}")

# -------------------------
# Min-Max Regret Optimization
# -------------------------
def solve_minmax_regret(fac_count, cust_count, scenarios, opt_costs, solver):
    I = range(fac_count)
    J = range(cust_count)
    Slist = list(scenarios.keys())

    model = pyo.ConcreteModel()
    model.x = pyo.Var(I, within=pyo.Binary)
    model.y = pyo.Var(Slist, I, J, within=pyo.NonNegativeReals)
    model.R = pyo.Var(within=pyo.NonNegativeReals)

    model.obj = pyo.Objective(expr=model.R, sense=pyo.minimize)

    model.demand = pyo.Constraint(Slist, J, rule=lambda m,s,j: sum(m.y[s,i,j] for i in I) == scenarios[s]['demand'][j])
    model.assign_open = pyo.Constraint(Slist, I, J, rule=lambda m,s,i,j: m.y[s,i,j] <= scenarios[s]['demand'][j]*m.x[i])

    def regret_rule(m,s):
        fixed = sum(scenarios[s]['fixed_costs'][i]*m.x[i] for i in I)
        trans = sum(scenarios[s]['cijs'][i,j]*m.y[s,i,j] for i in I for j in J)
        return fixed + trans - opt_costs[s] <= m.R
    model.regret = pyo.Constraint(Slist, rule=regret_rule)

    res = solver.solve(model, tee=False)
    x_sol = [int(round(pyo.value(model.x[i]))) for i in I]
    y_sol = {s: np.array([[float(pyo.value(model.y[s,i,j])) for j in J] for i in I]) for s in Slist}
    R_val = float(pyo.value(model.R))
    return R_val, x_sol, y_sol

print("Solving min-max regret problem...")
R_val, x_sol_mm, y_sol_mm = solve_minmax_regret(n_fac, n_cust, scenarios, opt_costs, solver)
print(f"Min-Max Regret R={R_val:.2f}, Facilities opened={x_sol_mm}")

# -------------------------
# Stochastic expected-cost solution
# -------------------------
def solve_stochastic_expected(fac_count, cust_count, scenarios, solver):
    I = range(fac_count)
    J = range(cust_count)
    Slist = list(scenarios.keys())

    model = pyo.ConcreteModel()
    model.x = pyo.Var(I, within=pyo.Binary)
    model.y = pyo.Var(Slist, I, J, within=pyo.NonNegativeReals)

    model.obj = pyo.Objective(
        expr=sum(
            scenarios[s]['prob'] * (
                sum(scenarios[s]['fixed_costs'][i]*model.x[i] for i in I) +
                sum(scenarios[s]['cijs'][i,j]*model.y[s,i,j] for i in I for j in J)
            )
            for s in Slist
        ),
        sense=pyo.minimize
    )

    model.demand = pyo.Constraint(Slist, J, rule=lambda m,s,j: sum(m.y[s,i,j] for i in I) == scenarios[s]['demand'][j])
    model.assign_open = pyo.Constraint(Slist, I, J, rule=lambda m,s,i,j: m.y[s,i,j] <= scenarios[s]['demand'][j]*m.x[i])

    res = solver.solve(model, tee=False)
    x_sol = [int(round(pyo.value(model.x[i]))) for i in I]
    return x_sol

print("Solving stochastic expected-cost problem...")
x_sol_stoch = solve_stochastic_expected(n_fac, n_cust, scenarios, solver)
print(f"Stochastic solution facilities opened: {x_sol_stoch}")

# -------------------------
# Save results for visualization
# -------------------------
np.savez("mm_solution.npz",
         x_sol_mm=np.array(x_sol_mm, dtype=int),
         y_sol_mm=y_sol_mm,
         opt_costs=opt_costs,
         x_sol_stoch=np.array(x_sol_stoch, dtype=int))

print("Saved mm_solution.npz successfully.")
