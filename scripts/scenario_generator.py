import numpy as np
import pandas as pd

np.random.seed(123)

# Problem size
n_fac = 10
n_cust = 15
n_scen = 8
region_scale = 100

# Facility coordinates
fac_coords = np.random.uniform(0, region_scale, size=(n_fac, 2))
fac_df = pd.DataFrame(fac_coords, columns=['x', 'y'])

# Customer coordinates
cust_coords = np.random.uniform(0, region_scale, size=(n_cust, 2))
cust_df = pd.DataFrame(cust_coords, columns=['x', 'y'])

def euclidean_matrix(A, B):
    A = np.asarray(A)
    B = np.asarray(B)
    return np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2))

dist_matrix = euclidean_matrix(fac_coords, cust_coords)

# Base scenario probabilities (sum to 1)
scenario_probs = np.array([0.15, 0.15, 0.1, 0.15, 0.15, 0.1, 0.1, 0.1])

# Generate scenarios
scenarios = {}
for s in range(n_scen):
    demand = np.ones(n_cust) * 10
    if s % 3 == 0:
        demand[:n_cust // 2] *= np.random.uniform(1.5, 3.0)
    elif s % 3 == 1:
        demand[n_cust // 2:] *= np.random.uniform(1.5, 3.0)
    else:
        demand *= np.random.uniform(0.5, 1.8, size=n_cust)

    transport_mult = np.random.uniform(0.8, 1.8)
    carbon_price = np.random.choice([0, 10, 30, 50, 80])
    fixed_costs = 1200 * np.random.uniform(0.8, 1.4, size=n_fac)
    base_unit = 1.0
    emission_factor = 0.01 * dist_matrix
    cijs = base_unit * transport_mult * dist_matrix + emission_factor * carbon_price

    # Introduce single-facility failure (facility unavailable)
    fail_facility = None
    if s % 4 == 3:
        fail_facility = np.random.choice(n_fac)
        fixed_costs[fail_facility] = 1e6
        cijs[fail_facility, :] = 1e6

    scenarios[f's{s}'] = {
        'demand': demand,
        'fixed_costs': fixed_costs,
        'cijs': cijs,
        'transport_mult': transport_mult,
        'carbon_price': carbon_price,
        'fail_facility': fail_facility,
        'prob': scenario_probs[s]
    }

if __name__ == "__main__":
    print(f"Generated {len(scenarios)} scenarios with {n_fac} facilities and {n_cust} customers.")

