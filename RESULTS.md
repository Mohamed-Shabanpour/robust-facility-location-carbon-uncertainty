# Results and Managerial Insights

## Overview
This section summarises the main computational results obtained from the deterministic, stochastic, and robust optimisation models. The focus is on understanding trade-offs between expected performance, robustness to adverse scenarios, and sustainability-related costs.

---

## Facility Location Decisions
The robust min–max regret solution selects a more diversified set of facilities compared to the stochastic expected-cost solution. This diversification reflects a hedging strategy against demand surges, transportation cost volatility, and facility disruption scenarios.

In contrast, the stochastic solution concentrates capacity in fewer facilities, prioritising expected performance under scenario probabilities.

---

## Cost and Regret Trade-offs
Across all scenarios, the stochastic expected-cost solution achieves lower average costs but exhibits substantial regret in adverse scenarios. The robust solution consistently limits worst-case regret, at the expense of slightly higher costs in benign scenarios.

These results highlight the classical robustness–efficiency trade-off in strategic supply chain design.

---

## Impact of Carbon Pricing
Carbon costs materially affect customer–facility assignments and, in some scenarios, alter facility opening decisions. Under high carbon price scenarios, both robust and stochastic solutions favour geographically closer assignments, even when transportation base costs are higher.

This demonstrates the importance of explicitly incorporating environmental pricing mechanisms into strategic network design models.

---

## Worst-Case Scenario Analysis
In the worst-case scenario, the robust solution shows a more balanced cost structure, with reduced exposure to extreme transportation and carbon costs. The stochastic solution, while efficient on average, concentrates risk and experiences significantly higher regret under this scenario.

---

Overall, the findings support the use of robust optimisation as a viable decision-making paradigm in uncertain and regulation-intensive supply chain environments.
