# robust-facility-location-carbon-uncertainty
Robust and stochastic facility location under demand uncertainty and carbon pricing using min–max regret optimisation

## Overview
This project investigates a strategic supply chain network design problem under demand uncertainty, transportation cost variability, carbon pricing, and facility disruption risk. The objective is to determine facility location and customer–facility flow decisions that remain effective across multiple adverse scenarios.

The study compares three decision-making paradigms commonly used in operations research:
1. Deterministic scenario-wise optimisation
2. Stochastic expected-cost optimisation
3. Robust optimisation based on a Min–Max Regret criterion

The modelling framework is motivated by real-world supply chains operating under volatile demand, regulatory pressure for carbon reduction, and increasing exposure to disruption risks.

---

## Problem Description
We consider a two-echelon supply chain consisting of:
- A set of candidate facilities
- A set of customer demand locations

Uncertainty is represented through a finite set of discrete scenarios capturing:
- Demand surges and asymmetric demand shifts
- Transportation cost fluctuations
- Carbon pricing variability
- Single-facility failure disruptions

Strategic decisions include:
- Facility opening decisions (binary)
- Flow allocation from facilities to customers (continuous)

---

## Methodology

### Deterministic Benchmark
For each scenario, a deterministic mixed-integer linear program (MILP) is solved independently. These scenario-optimal solutions provide lower bounds and serve as benchmarks for regret computation.

### Robust Optimisation (Min–Max Regret)
A robust counterpart is formulated in which a single facility location decision must perform across all scenarios. The objective minimises the maximum regret, defined as the deviation from the scenario-optimal cost in each scenario.

### Stochastic Programming Benchmark
A stochastic expected-cost model is solved using exogenously specified scenario probabilities. This model highlights the trade-off between expected performance and worst-case robustness.

All models are implemented in **Pyomo** and solved using open-source MILP solvers.

---

## Key Features
- Scenario-based uncertainty modelling
- Explicit incorporation of carbon costs into transportation decisions
- Facility disruption and failure scenarios
- Comparison of robust and stochastic decision paradigms
- Network and cost visualisation for policy interpretation

---

## Key Insights
- Robust min–max regret solutions diversify facility selections to hedge against extreme scenarios.
- Stochastic solutions perform well in expectation but can incur large regrets under adverse conditions.
- Carbon pricing materially alters facility–customer assignments and network structure.
- Min–max regret optimisation provides balanced protection without excessive conservatism.

---


## Tools & Technologies
- Python
- Pyomo
- Mixed-Integer Linear Programming (MILP)
- NumPy, Pandas
- Matplotlib
- HiGHS / CBC / GLPK solvers

---

## Author
[Mohamed Shabanpour]
