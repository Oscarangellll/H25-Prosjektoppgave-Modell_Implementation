import gurobipy as gp
import random

from classes import VesselType, WindFarm, MaintenanceCategory
from weather_windows import find_weather_windows
from failure_generation import failures
from generate_patterns import generate_patterns
import config


def model(
    name: str, 
    vessels: list[VesselType], 
    wind_farms: list[WindFarm],
    days_per_month: int,
    months: list[str],
    maintenance_categories: list[MaintenanceCategory],
    pattern_indexes_for_v: dict[str, list[int]],
    scenarios: list[int],
    failures: dict[tuple[str, str, int, int], int],
    patterns: dict[tuple[str, int], int],
    pattern_lengths: dict[int, int],
    weather_windows: dict[tuple[str, str, int, int], int]
):
    model = gp.Model(name)
    
    ### Sets
    # First stage
    V = [v.name for v in vessels] 
    T = months 
    # Second stage
    W = [w.name for w in wind_farms]
    M = [m.name for m in maintenance_categories]
    K = pattern_indexes_for_v
    D = [d for d in range(1, len(months) * days_per_month + 1)]
    D_t = {month: D[i * days_per_month : (i + 1) * days_per_month] for i, month in enumerate(months)} 
    S = scenarios

    ### Parameters
    # First stage
    C_ST = {(v.name, t): v.calculate_ST(days_per_month) for v in vessels for t in T}
    C_LT = {v.name: v.calculate_LT(days_per_month, len(months)) for v in vessels}
    # Second stage
    F = failures
    N = {v.name: v.n_teams for v in vessels}
    P = patterns
    L = pattern_lengths
    L_RT = 1
    A = weather_windows
    C_D = {(w, m, d, s): 9100 for w in W for m in M for d in D for s in S}
    C_U = {(v, w): 2 for v in V for w in W}
    B0 = 0

    ### Variables
    # First stage
    gamma_ST = model.addVars(V, T, vtype=gp.GRB.INTEGER, name="gamma_ST")
    gamma_LT = model.addVars(V, vtype=gp.GRB.INTEGER, name="gamma_LT")
    # Second stage
    x = model.addVars(V, W, D, S, vtype=gp.GRB.INTEGER, name="x")
    lmbd = model.addVars(
        ((v, w, d, k, s) for v in V for w in W for d in D for k in K[v] for s in S),
        vtype=gp.GRB.INTEGER,
        name="lambda"
    )
    z = model.addVars(W, M, D, S, vtype=gp.GRB.INTEGER, name="z")
    b = model.addVars(W, M, D, S, vtype=gp.GRB.INTEGER, name="b")

    ### Objective 
    # First stage
    first_obj = gamma_ST.prod(C_ST) + gamma_LT.prod(C_LT)
    # Second stage
    second_obj = (
        b.prod(C_D) 
        + gp.quicksum(C_U[v, w] * x[v, w, d, s] for v in V for w in W for d in D for s in S)
    )/len(S)

    model.setObjective(first_obj + second_obj)

    ### Constraints
    model.addConstrs(
        (x.sum(v, '*', d, s) <= gamma_ST[v, t] + gamma_LT[v] 
        for v in V 
        for t in T 
        for d in D_t[t] 
        for s in S),
        name="vessels_available"
    )

    model.addConstrs(
        (lmbd.sum(v, w, d, '*', s) <= N[v] * x[v, w, d, s]
        for v in V
        for w in W
        for d in D
        for s in S),
        name="teams_available"
    )

    model.addConstrs(
        ((L[k] + L_RT - A[v, w, d, s]) * lmbd[v, w, d, k, s] <= 0
        for v in V
        for w in W
        for d in D
        for k in K[v]
        for s in S),
        name="weather_window"
    )
    
    model.addConstrs(
        (z[w, m, d, s] <= gp.quicksum(P[m, k] * lmbd[v, w, d, k, s] for v in V for k in K[v])
        for w in W
        for m in M
        for d in D
        for s in S),
        name="tasks_performed"
    )

    model.addConstrs(
        (b[w, m, s, D[0]] == B0
        for w in W
        for m in M
        for s in S),
        name="initial_backlog"
    )

    model.addConstrs(
        (b[w, m, d, s] == b[w, m, d-1, s,] + F[w, m, d, s] - z[w, m, d, s]
        for w in W 
        for m in M
        for d in D[1:]
        for s in S),
        name="backlog"
    )
    
    
    model.optimize()

def real_model(
    name,
    vessels,
    wind_farms,
    maintenance_categories,
    num_scenarios
):
    
    random.seed(config.SEED)
    scenarios = random.sample(
        range(1, config.SCENARIOS + 1), num_scenarios
    )

    # Generate weather window
    A = find_weather_windows(scenarios, wind_farms, vessels)
    
    # Generatue failures
    F = failures(scenarios, wind_farms, maintenance_categories)

    # Generate patterns
    K, L, P = generate_patterns(vessels, maintenance_categories)

    return model(
        name,
        vessels,
        wind_farms,
        days_per_month=config.DAYS,
        months=config.MONTHS,
        maintenance_categories=maintenance_categories,
        pattern_indexes_for_v=K,
        scenarios=scenarios,
        failures=F,
        patterns=P,
        pattern_lengths=L,
        weather_windows=A
    )



""" Test case
name = "Two Stage Test Model"

vessels = [
    VesselType("CTV", n_teams=3, max_wind=15, max_wave=2, shift_length=10, day_rate=20, mob_rate=300)
]

days_per_month = 5
months = ["January", "February"]

wind_farms = [
    WindFarm("Wind Farm A", n_turbines=120, location=None, distance_to_base=None, weather_data_file=None)
]

maintenance_categories = [
    MaintenanceCategory("Small", failure_rate=None, duration=2, vessel_types=["CTV"])
]

pattern_indexes_for_v = {"CTV": [0, 1]}

scenarios = [1, 2]

failures = {}
for m in maintenance_categories:
    for w in wind_farms:
        for d in range(1, len(months) * days_per_month + 1):
            for s in scenarios:
                if m.name == "Small":
                    failures[(w.name, m.name, d, s)] = random.randint(0, 10) 
                else:
                    failures[(w.name, m.name, d, s)] = random.randint(0, 2)  

patterns = {}
patterns[("Small", 0)] = 2
patterns[("Small", 1)] = 1

pattern_lengths = {}
pattern_lengths[0] = 5
pattern_lengths[1] = 2

weather_windows = {}
for v in vessels:
    for w in wind_farms:
        for d in range(1, len(months) * days_per_month + 1):
            for s in scenarios:
                weather_windows[(v.name, w.name, d, s)] = 10

model(
    name, 
    vessels, 
    wind_farms,
    days_per_month,
    months,
    maintenance_categories,
    pattern_indexes_for_v,
    scenarios,
    failures,
    patterns,
    pattern_lengths,
    weather_windows
)
"""

name = "Two Stage Model"

vessels = [
    VesselType("CTV", n_teams=3, max_wind=15, max_wave=2, shift_length=10, day_rate=20, mob_rate=300),
    VesselType("SOV", n_teams=5, max_wind=20, max_wave=2.5, shift_length=12, day_rate=200, mob_rate=300)
]

wind_farms = [
    WindFarm("Wind Farm A", n_turbines=120, location=None, distance_to_base=None, weather_data_file="Location 1.csv"),
    WindFarm("Wind Farm B", n_turbines=100, location=None, distance_to_base=None, weather_data_file="Location 1.csv")
]

maintenance_categories = [
    MaintenanceCategory("Small", failure_rate=5, duration=2, vessel_types=["CTV", "SOV"]),
    MaintenanceCategory("Large", failure_rate=3, duration=5, vessel_types=["SOV"])
]

real_model(
    name,
    vessels,
    wind_farms, 
    maintenance_categories,
    4
)






