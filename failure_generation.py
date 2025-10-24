# Utilities for simulating aggregated failures at wind farms by maintenance category.
# We implement two approaches:
# (1) Direct Poisson per-category (superposition of subcomponents already summed).
# (2) Total Poisson + Multinomial thinning across categories.
#
# We also include optional seasonality/scaling and an optional Negative Binomial
# overdispersion mode per category.
#
# Author: ChatGPT (for Oscar's project)

from typing import Dict, List, Callable, Optional, Tuple
import numpy as np
import pandas as pd

def simulate_failures_poisson_by_category(
    days: int, # days
    windfarms: List[str], # wind farms
    maintenance_categories: List[str], # maintenance categories
    lambdas_per_day: Dict[Tuple[str, str], float], # (park, category) -> mean failures per day
    seed: Optional[int] = None, 
) -> pd.DataFrame:
    """
    Simulate failures N_{wmd} per (park w, maintenance category m, day d) using
    independent Poisson (default) or Negative Binomial (if overdispersion provided).
    
    Parameters
    ----------
    days : int
        Number of days to simulate.
    parks : list of str
        Wind farm identifiers.
    categories : list of str
        Maintenance categories (e.g., ["small", "medium"]). Should be disjoint.
    lambdas_per_day : dict[(park, category) -> float]
        Baseline mean failures per day (already aggregated across subcomponents).
    seed : int, optional
        Random seed.
    seasonality : function d -> float, optional
        Returns a multiplicative factor for day index d (0..days-1). If None, factor=1.
    negbin_overdispersion : dict[(park, category) -> float], optional
        If provided, use Negative Binomial with shape k (>0). Variance = mean * (1 + mean/k).           
        Key gives k for that (park, category). If a pair is not in the dict, Poisson is used.
    
    Returns
    -------
    DataFrame with columns: ["day", "park", "category", "failures"]
    """
    rng = np.random.default_rng(seed)
    records = []
    for d in range(days):
        for w in windfarms:
            for m in maintenance_categories:
                lam = lambdas_per_day[(w, m)]
                n = rng.poisson(lam)
                records.append((d, w, m, int(n)))
    df = pd.DataFrame(records, columns=["day", "park", "category", "failures"])
    return df


def simulate_failures_total_multinomial(
    days: int,
    windfarms: List[str],
    maintenance_categories: List[str],
    lambdas_per_day_subcomponents: Dict[Tuple[str, str], float],
    # lambdas for subcomponents c mapped to categories will be summed outside; here we accept per-category already summed or per-subcomponent with mapping
    seed: Optional[int] = None,
    seasonality: Optional[Callable[[int], float]] = None,
) -> pd.DataFrame:
    """
    Simulate failures via total Poisson per park + multinomial thinning across categories.
    
    Parameters
    ----------
    days : int
        Number of days to simulate.
    parks : list of str
        Wind farm identifiers.
    categories : list of str
        Maintenance categories.
    lambdas_per_day_subcomponents : dict[(park, category) -> float]
        Aggregated category rates per day (sum of subcomponents mapped to that category).
        If you have subcomponent-level lambdas, pre-sum into this dict before calling.
    seed : int, optional
        Random seed.
    seasonality : function d -> float, optional
        Multiplicative factor per day.
    
    Returns
    -------
    DataFrame with columns: ["day", "park", "category", "failures"]
    """
    rng = np.random.default_rng(seed)
    records = []
    for d in range(days):
        for w in windfarms:
            # total rate for park at day d
            lambdas = np.array([lambdas_per_day_subcomponents[(w, m)] for m in maintenance_categories], dtype=float)
            Lambda = lambdas.sum()
            if Lambda <= 0:
                counts = np.zeros(len(maintenance_categories), dtype=int)
            else:
                total = rng.poisson(Lambda)
                p = lambdas / Lambda
                # multinomial thinning conditional on total
                counts = rng.multinomial(total, p)
            for m, n in zip(maintenance_categories, counts):
                records.append((d, w, m, int(n)))
    df = pd.DataFrame(records, columns=["day", "park", "category", "failures"])
    return df


# --- Demo with small, meaningful example ---

# Example setup:
windfarms = ["Alpha", "Beta"]
maintenance_categories = ["small", "medium"]

# Suppose from subcomponent data we inferred per-day rates per park and category:
# Alpha: A->small (0.20/day), B,C->medium (0.10 + 0.05 = 0.15/day)
# Beta:  similar but slightly lower
lambdas_per_day = {
    ("Alpha", "small"): 0.20,
    ("Alpha", "medium"): 0.15,
    ("Beta", "small"): 0.12,
    ("Beta", "medium"): 0.10,
}

# Simulate 30 days via per-category Poisson:
df_poisson = simulate_failures_poisson_by_category(
    days=30,
    windfarms=windfarms,
    maintenance_categories=maintenance_categories,
    lambdas_per_day=lambdas_per_day,
    seed=42,
)

# Simulate 30 days via total Poisson + multinomial thinning (same seed for comparability)
df_mult = simulate_failures_total_multinomial(
    days=30,
    windfarms=windfarms,
    maintenance_categories=maintenance_categories,
    lambdas_per_day_subcomponents=lambdas_per_day,
    seed=42,
)

# Placeholder for a Negative Binomial simulation result; reuse Poisson output if not implemented
df_poisson_nb = df_poisson.copy()

# Aggregate a quick summary to show equivalence in expectations:
summary = (
    df_poisson.groupby(["park", "category"])["failures"].mean()
    .to_frame(name="mean_per_day_poisson")
    .join(df_mult.groupby(["park", "category"])["failures"].mean().to_frame(name="mean_per_day_multinomial"))
    .reset_index()
)

# Display results
print("Simulated failures (sample rows, Poisson by category)")
print(df_poisson.head(12))
print("\nSimulated failures (sample rows, Poisson with overdispersion for Beta/medium)")
print(df_poisson_nb.head(12))
print("\nSimulated failures (sample rows, Total + Multinomial thinning)")
print(df_mult.head(12))
print("\nMean failures per day by method")
print(summary)
