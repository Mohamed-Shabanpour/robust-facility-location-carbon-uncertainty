# robust-facility-location-carbon-uncertainty

Robust and stochastic facility location under demand uncertainty, carbon pricing, and disruption risk using min–max regret optimisation.

## Overview
This project develops and analyses a strategic supply chain network design model under multiple interacting sources of uncertainty, including demand variability, transportation cost fluctuations, carbon pricing, and facility disruption risk. The objective is to determine facility location and customer–facility assignment decisions that remain effective across adverse operating conditions while maintaining cost efficiency.

The study compares three optimisation paradigms commonly employed in operations research:
1. Deterministic scenario-wise optimisation
2. Stochastic expected-cost optimisation
3. Robust optimisation based on a Min–Max Regret criterion

The modelling framework is motivated by real-world supply chains operating under volatile demand, tightening environmental regulations, and increasing exposure to operational disruptions.

---

## Problem Description
We consider a two-echelon supply chain consisting of:
- A set of candidate facility locations
- A set of geographically distributed customer demand points

Uncertainty is represented using a finite set of discrete scenarios capturing:
- Demand surges and asymmetric demand shifts across customer segments
- Transportation cost variability
- Carbon pricing uncertainty
- Single-facility disruption and unavailability events

Strategic decisions include:
- Binary facility opening decisions
- Continuous customer–facility flow allocation decisions

---

## Methodology

### Deterministic Benchmark
For each scenario, a deterministic mixed-integer linear program (MILP) is solved independently. These scenario-optimal solutions provide lower bounds on achievable costs and serve as benchmarks for regret evaluation.

### Robust Optimisation (Min–Max Regret)
A robust counterpart is formulated in which a single facility location decision must perform across all scenarios. The objective minimises the maximum regret, defined as the difference between the realised cost of a candidate decision and the scenario-optimal cost in each scenario.

This approach explicitly balances protection against adverse outcomes with avoidance of overly conservative decisions.

### Stochastic Programming Benchmark
A stochastic expected-cost model is solved using exogenously specified scenario probabilities. This benchmark highlights the trade-off between expected cost efficiency and vulnerability to unfavourable scenarios.

All models are implemented in **Pyomo** and solved using open-source MILP solvers.

---

## Key Insights
- Robust min–max regret solutions prioritise stability relative to scenario-optimal benchmarks rather than minimising expected cost.
- Stochastic solutions can achieve lower expected costs but exhibit higher exposure to worst-case regret.
- Carbon pricing materially alters customer–facility assignments and regret profiles.
- The efficiency–robustness trade-off shows that modest efficiency losses can yield substantial reductions in downside risk.


---

## Tools & Technologies
- Python
- Pyomo
- Mixed-Integer Linear Programming (MILP)
- NumPy, Pandas
- Matplotlib
- HiGHS / CBC / GLPK solvers
