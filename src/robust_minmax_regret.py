import pyomo.environ as pyo
import numpy as np

def solve_minmax_regret(fac_count, cust_count, scenarios, opt_costs, solver):
    I = range(fac_count)
    J = range(cust_count)
    Slist = list(scenarios.keys())

    model = pyo.ConcreteModel()

    model.x = pyo.Var(I, within=pyo.Binary)
    model.y = pyo.Var(Slist, I, J, within=pyo.NonNegativeReals)
    model.R = pyo.Var(within=pyo.NonNegativeReals)

    model.obj = pyo.Objective(expr=model.R, sense=pyo.minimize)

    model.demand = pyo.Constraint(
        Slist, J,
        rule=lambda m, s, j: sum(m.y[s, i, j] for i in I) == scenarios[s]['demand'][j]
    )

    model.assign_open = pyo.Constraint(
        Slist, I, J,
        rule=lambda m, s, i, j: m.y[s, i, j] <= scenarios[s]['demand'][j] * m.x[i]
    )

    def regret_rule(m, s):
        fixed = sum(scenarios[s]['fixed_costs'][i] * m.x[i] for i in I)
        trans = sum(scenarios[s]['cijs'][i, j] * m.y[s, i, j] for i in I for j in J)
        return fixed + trans - opt_costs[s] <= m.R

    model.regret = pyo.Constraint(Slist, rule=regret_rule)

    solver.solve(model, tee=False)

    x_sol = [int(round(pyo.value(model.x[i]))) for i in I]
    y_sol = {
        s: np.array([[float(pyo.value(model.y[s, i, j])) for j in J] for i in I])
        for s in Slist
    }

    R_val = float(pyo.value(model.R))

    return R_val, x_sol, y_sol
