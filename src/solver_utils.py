import pyomo.environ as pyo

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
