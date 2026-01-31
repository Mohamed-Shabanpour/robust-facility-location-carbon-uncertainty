import pyomo.environ as pyo

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
                sum(scenarios[s]['fixed_costs'][i] * model.x[i] for i in I)
                + sum(scenarios[s]['cijs'][i, j] * model.y[s, i, j] for i in I for j in J)
            )
            for s in Slist
        ),
        sense=pyo.minimize
    )

    model.demand = pyo.Constraint(
        Slist, J,
        rule=lambda m, s, j: sum(m.y[s, i, j] for i in I) == scenarios[s]['demand'][j]
    )

    model.assign_open = pyo.Constraint(
        Slist, I, J,
        rule=lambda m, s, i, j: m.y[s, i, j] <= scenarios[s]['demand'][j] * m.x[i]
    )

    solver.solve(model, tee=False)

    return [int(round(pyo.value(model.x[i]))) for i in I]
