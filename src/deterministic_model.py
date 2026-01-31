import pyomo.environ as pyo
import numpy as np

def solve_deterministic(fac_count, cust_count, fixed_costs, cijs, demands, solver):
    I = range(fac_count)
    J = range(cust_count)

    model = pyo.ConcreteModel()

    model.x = pyo.Var(I, within=pyo.Binary)
    model.y = pyo.Var(I, J, within=pyo.NonNegativeReals)

    model.obj = pyo.Objective(
        expr=sum(fixed_costs[i] * model.x[i] for i in I)
        + sum(cijs[i, j] * model.y[i, j] for i in I for j in J),
        sense=pyo.minimize
    )

    model.demand = pyo.Constraint(
        J, rule=lambda m, j: sum(m.y[i, j] for i in I) == demands[j]
    )

    model.assign_open = pyo.Constraint(
        I, J, rule=lambda m, i, j: m.y[i, j] <= demands[j] * m.x[i]
    )

    solver.solve(model, tee=False)

    x_sol = [int(round(pyo.value(model.x[i]))) for i in I]
    y_sol = np.array([[pyo.value(model.y[i, j]) for j in J] for i in I])
    cost = float(pyo.value(model.obj))

    return cost, x_sol, y_sol
