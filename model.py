import gurobipy as gp
import random
from haversine import haversine, Unit

from classes import Vessel, VesselType, WindFarm, MaintenanceCategory, Base
from weather_windows import find_weather_windows
from failure_generation import failures
from generate_patterns import generate_patterns
from calculate_downtime_cost import calculate_downtime_cost
import config

def model(
    name: str, 
    vessel_types: list[VesselType], 
    vessels: list[Vessel],
    wind_farms: list[WindFarm],
    base: Base,
    days_per_month: int,
    months: list[str],
    maintenance_categories: list[MaintenanceCategory],
    pattern_indexes_for_hids: dict[tuple[str, str, int, int], int],
    scenarios: list[int],
    failures: dict[tuple[str, str, int, int], int],
    patterns: dict[tuple[str, int], int],
    weather_windows: dict[tuple[str, str, int, int], int],
    downtime_cost: dict[tuple[str, int, int]],
):
    model = gp.Model(name)
    
    ### Sets
    # First stage
    H = [h.name for h in vessel_types]
    H_M = [h.name for h in vessel_types if h.multiday] 
    H_S = [h.name for h in vessel_types if not h.multiday]
    V = {h: [v.name for v in vessels if v.vessel_type.name == h] for h in H_M}
    T = months 
    # Second stage
    W = [w.name for w in wind_farms]
    B = base.name
    L = [B] + W
    M = [m.name for m in maintenance_categories]
    K = pattern_indexes_for_hids
    D = [d for d in range(1, len(months) * days_per_month + 1)]
    D_t = {month: D[i * days_per_month : (i + 1) * days_per_month] for i, month in enumerate(months)}
    D_B = [d for d in D if d % days_per_month == 1 and d != 1]
    S = scenarios

    ### Parameters
    # First stage
    C_ST = {(h.name, t): h.calculate_ST(days_per_month) for h in vessel_types for t in T}
    C_LT = {h.name: h.calculate_LT(days_per_month, len(months)) for h in vessel_types}
    # Second stage
    F = failures
    N = {h.name: h.n_teams for h in vessel_types}
    P = patterns
    # L_k = pattern_lengths
    # L_RT = {(h.name, i.name): 0 if h.name in H_M else 2 * haversine(i.coordinates, base.coordinates, unit=Unit.KILOMETERS) / h.speed for i in wind_farms for h in vessel_types}
    A = weather_windows
    C_D = downtime_cost
    C_U = {(h.name): h.usage_cost_per_day for h in vessel_types}
    C_RT = {(h.name, i.name): 2 * haversine(i.coordinates, base.coordinates, unit=Unit.KILOMETERS) * h.cost_per_km for h in vessel_types if h.name in H_S for i in wind_farms}
    C_T = {(h.name, i.name, j.name): haversine(i.coordinates, j.coordinates, unit=Unit.KILOMETERS) * h.cost_per_km for h in vessel_types if h.name in H_M for i in wind_farms + [base] for j in wind_farms + [base] if i!= j}
    B_init = {(i.name, m): 0 for i in wind_farms for m in M}
    R_h = {h.name: h.periodic_return for h in vessel_types}

    ### Variables
    # First stage
    gamma_ST = model.addVars(H, T, vtype=gp.GRB.INTEGER, ub=5, name="gamma_ST")
    gamma_LT = model.addVars(H, vtype=gp.GRB.INTEGER, ub= 5, name="gamma_LT")
    alpha_ST = model.addVars(
        ((v, t) for h in H_M for v in V[h] for t in T),
        vtype=gp.GRB.BINARY,
        name="alpha_ST"
    )
    alpha_LT = model.addVars(
        (v for h in H_M for v in V[h]),
        vtype=gp.GRB.BINARY,
        name="alpha_LT"
    )
    # Second stage
    x = model.addVars(H, W, D, S, vtype=gp.GRB.INTEGER, name="x")
    lmbd = model.addVars(
        
        ((h, i, d, k, s) for h in H for i in W for d in D for s in S for k in K[h, i, d, s]),
        vtype=gp.GRB.INTEGER,
        name="lambda"
    )
    z = model.addVars(W, M, D, S, vtype=gp.GRB.INTEGER, name="z")
    b = model.addVars(W, M, [0] + D, S, vtype=gp.GRB.INTEGER, name="b")
    
    delta = model.addVars(
        ((v, i, d, s) for h in H_M for v in V[h] for i in L for d in D for s in S), 
        vtype=gp.GRB.BINARY, 
        name="delta"
        )

    f = model.addVars(
        ((v, i, j, d, s) for h in H_M for v in V[h] for i in L for j in L if i != j for d in D for s in S),
        vtype=gp.GRB.BINARY, 
        name="f"
    )
    
    r_START = model.addVars(
        ((v, i, d, s) for h in H_M for v in V[h] for i in L for d in D_B for s in S),
        vtype=gp.GRB.BINARY,
        name="r_START"
    )
    
    r_END = model.addVars(
        ((v, i, d, s) for h in H_M for v in V[h] for i in L for d in D_B for s in S),
        vtype=gp.GRB.BINARY,
        name="r_END"
    )
    
    ### Objective 
    # First stage
    first_obj = gamma_ST.prod(C_ST) + gamma_LT.prod(C_LT)
    # print objective function
    # print("First stage objective:", first_obj)
    # Second stage
    second_obj = (
        gp.quicksum(C_D[i, d, s] * b[i, m, d, s] for i in W for m in M for d in D for s in S) 
        + gp.quicksum(C_U[h] * x[h, i, d, s] for h in H for i in W for d in D for s in S)
        + gp.quicksum(C_RT[h, i] * x[h, i, d, s] for h in H_S for i in W for d in D for s in S)
        + gp.quicksum(C_T[h, i, j] * f[v, i, j, d, s] for h in H_M for v in V[h] for i in L for j in L if i != j for d in D for s in S)
    )/len(S)

    model.setObjective(first_obj + second_obj)

    ### Constraints
    # First stage
    model.addConstrs(
        (gamma_ST[h, t] == gp.quicksum(alpha_ST[v, t] for v in V[h])
        for h in H_M
        for t in T),
        name="binding_ST_vars"
    )
    model.addConstrs(
        (gamma_LT[h] == gp.quicksum(alpha_LT[v] for v in V[h])
        for h in H_M),
        name="binding_LT_vars"
    )
    model.addConstrs(
        (alpha_ST[v, t] + alpha_LT[v] <= 1
        for h in H_M
        for v in V[h]
        for t in T),
        name="either_ST_LT"
    )
    model.addConstrs(
        (alpha_ST[V[h][v], t] >= alpha_ST[V[h][v+1], t]
        for h in H_M
        for v in range(len(V[h]) - 1)
        for t in T),
        name="symmetry_break_ST"        
    )
    model.addConstrs(
    (alpha_LT[V[h][v]] >= alpha_LT[V[h][v+1]]
    for h in H_M
    for v in range(len(V[h]) - 1)),
    name="symmetry_break_LT"        
    )
    # Second stage
    ###### force variables #####
    # model.addConstrs(
    #     alpha_LT[v] == 0 for h in H_M for v in V[h]
    # )
    # model.addConstrs(
    #     alpha_ST[v, "January"] == 1 for h in H_M for v in V[h]
    # )
    # model.addConstrs(
    #     alpha_ST[v, "February"] == 1 for h in H_M for v in V[h]
    # )
    ##########
    model.addConstrs(
        (x.sum(h, '*', d, s) <= gamma_ST[h, t] + gamma_LT[h] 
        for h in H 
        for t in T 
        for d in D_t[t] 
        for s in S),
        name="vessels_available"
    )
    
    model.addConstrs(
        (
            gp.quicksum(delta[v, i, d, s] for i in L) <= alpha_ST[v, t] + alpha_LT[v]
            for h in H_M
            for v in V[h]
            for t in T
            for d in D_t[t]
            for s in S
        ),
        name="M_vessels_available"
    )
    
    model.addConstrs(
        (x[h, i, d, s] == gp.quicksum(delta[v, i, d, s] for v in V[h])
        for h in H_M
        for i in W
        for d in D
        for s in S),
        name="allocation_from_delta"
    )

    model.addConstrs(
        (delta[v, B, d, s] == alpha_ST[v, t] + alpha_LT[v]
        for h in H_M
        for v in V[h]
        for t in T
        for d in D_t[t] if d % R_h[h] == 0
        for s in S),
        name="base_visit"
    )
    model.addConstrs(
        (delta[v, B, 1, s] == alpha_ST[v, months[0]] + alpha_LT[v]
        for h in H_M
        for v in V[h]
        for s in S),
        name="base_visit_day_0"
    )
    model.addConstrs(
        (delta[v, B, D[-1], s] == alpha_ST[v, months[-1]] + alpha_LT[v]
        for h in H_M
        for v in V[h]
        for s in S),
        name="base_visit_day_0"
    )
    # model.addConstrs(
    #     (delta[v, B, d, s] == r_START[v, B, d, s]
    #     for h in H_M
    #     for v in V[h]
    #     for d in D_B
    #     for s in S)
    # )
    # model.addConstrs(
    #     (delta[v, B, d-1, s] == r_END[v, B, d, s]
    #     for h in H_M
    #     for v in V[h]
    #     for d in D_B
    #     for s in S)
    # )
    model.addConstrs(
        (gp.quicksum(lmbd[h, i, d, k, s] for k in K[h, i, d, s]) <= N[h] * x[h, i, d, s]
        for h in H
        for i in W
        for d in D
        for s in S),
        name="teams_available"
    )
    # model.addConstrs(
    #     ((L_k[k] + L_RT[h, i] - A[h, i, d, s]) * lmbd[h, i, d, k, s] <= 0
    #     for h in H
    #     for i in W
    #     for d in D
    #     for k in K[h]
    #     for s in S),
    #     name="weather_window"
    # )
    model.addConstrs(
        (z[i, m, d, s] <= gp.quicksum(P[m, k] * lmbd[h, i, d, k, s] for h in H for k in K[h, i, d, s])
        for i in W
        for m in M
        for d in D
        for s in S),
        name="tasks_performed"
    )
    model.addConstrs(
        (b[i, m, 0, s] == B_init[i, m]
        for i in W
        for m in M
        for s in S),
        name="initial_backlog"
    )
    model.addConstrs(
        (b[i, m, d, s] == b[i, m, d-1, s,] + F[i, m, d, s] - z[i, m, d, s]
        for i in W 
        for m in M
        for d in D
        for s in S),
        name="backlog"
    )
    model.addConstrs(
        (delta[v, i, d-1, s] + gp.quicksum(f[v, j, i, d-1, s] for j in L if j != i) - gp.quicksum(f[v, i, j, d-1, s] for j in L if j != i) == delta[v, i, d, s]
        for h in H_M
        for v in V[h]
        for i in L
        for t in T
        for d in D_t[t] if d != 1 and d not in D_B #if d % len(D_t[t]) != 1
        for s in S),
        name="flow"
    )
    model.addConstrs(
        (delta[v, i, d-1, s] + gp.quicksum(f[v, j, i, d-1, s] for j in L if j != i) - gp.quicksum(f[v, i, j, d-1, s] for j in L if j != i) == delta[v, i, d, s] - r_START[v, i, d, s] + r_END[v, i, d, s]
        for h in H_M
        for v in V[h]
        for i in L
        for t in T
        for d in D_t[t] if d in D_B #if d % len(D_t[t]) != 1
        for s in S),
        name="flow"
    )
    model.addConstrs(
        (gp.quicksum(r_START[v, i, d, s] for i in L) <= alpha_ST[v, T[t]]
        for h in H_M
        for v in V[h]
        for t in range(1, len(T[1:]) + 1)
        for d in D_t[T[t]] if d in D_B
        for s in S),
        name="ST_charter_transition_upper"
    )
    model.addConstrs(
        (gp.quicksum(r_START[v, i, d, s] for i in L) <= 1 - alpha_ST[v, T[t-1]]
        for h in H_M
        for v in V[h]
        for t in range(1, len(T[1:]) + 1)
        for d in D_t[T[t]] if d in D_B
        for s in S),
        name="ST_charter_transition_bound"
    )
    model.addConstrs(
        (r_START[v, B, d, s] <= delta[v, B, d, s]
        for h in H_M
        for v in V[h]
        for d in D_B
        for s in S)
    )
    model.addConstrs(
        (gp.quicksum(r_END[v, i, d, s] for i in L) <= alpha_ST[v, T[t-1]]
        for h in H_M
        for v in V[h]
        for t in range(1, len(T[1:]) + 1)
        for d in D_t[T[t]] if d in D_B
        for s in S),
        name="ST_charter_transition_upper"
    )
    model.addConstrs(
        (gp.quicksum(r_END[v, i, d, s] for i in L) <= 1 - alpha_ST[v, T[t]]
        for h in H_M
        for v in V[h]
        for t in range(1, len(T[1:]) + 1)
        for d in D_t[T[t]] if d in D_B
        for s in S),
        name="ST_charter_transition_bound"
    )
    model.addConstrs(
        (r_END[v, B, d, s] <= delta[v, B, d-1, s]
        for h in H_M
        for v in V[h]
        for d in D_B
        for s in S)
    )
    model.addConstrs(
        (gp.quicksum(r_START[v, i, d, s] + r_END[v, i, d, s] for i in W) == 0
        for h in H_M
        for v in V[h]
        for d in D_B
        for s in S)
    )
    
    return model
    
def real_model(
    name,
    vessel_types,
    vessels,
    wind_farms,
    maintenance_categories,
    num_scenarios,
    scenario,
    days,
    months,
    n_total_days,
    seed
):
    
    # random.seed(config.RANDOM_SEED)
    if scenario:
        scenarios = seed
    else:
        random.seed(seed)
        scenarios = random.sample(
            range(1, 101), num_scenarios
        )

    # Generate weather window
    A = find_weather_windows(scenarios, wind_farms, vessel_types)
    
    # Generatue failures
    F = failures(scenarios, wind_farms, maintenance_categories, seed)
    
    base = Base("Base A",  coordinates=(53.7, 7.4))
    
    # Generate patterns
    K, L, K_hids, P = generate_patterns(vessel_types, maintenance_categories, wind_farms, n_total_days, scenarios, A, base)
    
    downtime_cost = calculate_downtime_cost(wind_farms, scenarios, 0.17)

    return model(
        name,
        vessel_types,
        vessels,
        wind_farms,
        base,
        days_per_month=days,
        months=months,
        maintenance_categories=maintenance_categories,
        pattern_indexes_for_hids=K_hids,
        scenarios=scenarios,
        failures=F,
        patterns=P,
        weather_windows=A,
        downtime_cost=downtime_cost
    )

# #Test case
# name = "Two Stage Test Model"

# vessel_types = [
#     VesselType("CTV", multiday=False, n_teams=3, max_wind=15, max_wave=2, shift_length=10, day_rate=20, mob_rate=300, speed=30, cost_per_km=3, periodic_return=None, usage_cost_per_day=10),
#     VesselType("SOV", multiday=True, n_teams=5, max_wind=18, max_wave=2.5, shift_length=12, day_rate=5, mob_rate=300, speed=30, cost_per_km=3, periodic_return=7, usage_cost_per_day=10)
# ]

# vessels = [
#     Vessel("SOV1", vessel_type=vessel_types[1]),
#     Vessel("SOV2", vessel_type=vessel_types[1])
# ]

# days_per_month = 30
# months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

# wind_farms = [
#     WindFarm("Wind Farm A", coordinates=(54.4, 7.3), n_turbines=120, weather_data_file=None),
#     WindFarm("Wind Farm B", coordinates=(53.9, 6.9), n_turbines=100, weather_data_file=None)
# ]

# base = Base("Base", coordinates=(53.7, 7.4))

# maintenance_categories = [
#     MaintenanceCategory("Small", failure_rate=None, duration=2, vessel_types=["CTV", "SOV"])
# ]

# pattern_indexes_for_h = {
#     "CTV": [0, 1],
#     "SOV": [0, 1]
#     }

# scenarios = [1, 2, 3]

# failures = {}
# for m in maintenance_categories:
#     for w in wind_farms:
#         for d in range(1, len(months) * days_per_month + 1):
#             for s in scenarios:
#                 if m.name == "Small":
#                     failures[(w.name, m.name, d, s)] = random.randint(0, 10) 
#                 else:
#                     failures[(w.name, m.name, d, s)] = random.randint(0, 2)  

# patterns = {}
# patterns[("Small", 0)] = 2
# patterns[("Small", 1)] = 1

# pattern_lengths = {}
# pattern_lengths[0] = 5
# pattern_lengths[1] = 2

# weather_windows = {}
# for v in vessel_types:
#     for w in wind_farms:
#         for d in range(1, len(months) * days_per_month + 1):
#             for s in scenarios:
#                 weather_windows[(v.name, w.name, d, s)] = 10

# model = model(
#     name, 
#     vessel_types, 
#     vessels,
#     wind_farms,
#     base,
#     days_per_month,
#     months,
#     maintenance_categories,
#     pattern_indexes_for_h,
#     scenarios,
#     failures,
#     patterns,
#     pattern_lengths,
#     weather_windows,
#     downtime_cost_per_day=200000000,
    
# )

# model.optimize()

# #print some results
# # for v in model.getVars():
# #     if v.X >= 0:
# #         print(f"{v.VarName}: {v.X}")

# # name = "Two Stage Model"

# # vessels = [
# #     VesselType("CTV", n_teams=3, max_wind=15, max_wave=2, shift_length=10, day_rate=20, mob_rate=300),
# #     VesselType("SOV", n_teams=5, max_wind=20, max_wave=2.5, shift_length=12, day_rate=200, mob_rate=300)
# # ]

# # wind_farms = [
# #     WindFarm("Wind Farm A", n_turbines=120, location=None, distance_to_base=None, weather_data_file="Location 1.csv"),
# #     WindFarm("Wind Farm B", n_turbines=100, location=None, distance_to_base=None, weather_data_file="Location 1.csv")
# # ]

# # maintenance_categories = [
# #     MaintenanceCategory("Small", failure_rate=5, duration=2, vessel_types=["CTV", "SOV"]),
# #     MaintenanceCategory("Large", failure_rate=3, duration=5, vessel_types=["SOV"])
# # ]

# # real_model(
# #     name,
# #     vessels,
# #     wind_farms, 
# #     maintenance_categories,
# #     4
# # )






