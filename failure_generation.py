import numpy as np
from classes import WindFarm
from classes import MaintenanceCategory
import config

def failures(scenarios, wind_farms, maintenance_categories):
    """
    Returns
    -------
    F : dict
        (wind farm, task, day, scenario): number of failures
    """
    np.random.seed(config.RANDOM_SEED)
    p = [m.failure_rate / 365 for m in maintenance_categories]

    if sum(p) > 1.0:
        raise ValueError("Sum of failure probabilities > 1.")
    p.append(1 - sum(p))

    F = {}

    for w in wind_farms:
        for d in range(1, config.DAYS * len(config.MONTHS) + 1):
            for s in scenarios:
                failures = np.random.multinomial(w.n_turbines, p)

                for i, m in enumerate(maintenance_categories):
                    F[(w.name, m.name, d, s)] = failures[i]

    return F
