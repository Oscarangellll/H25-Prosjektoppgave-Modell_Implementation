from classes import MaintenanceCategory, Vessel, VesselType, WindFarm, Base
import gurobipy as gp
from gurobipy import GRB
from haversine import haversine, Unit
from classes import WindFarm
from weather_windows import find_weather_windows

def generate_patterns(vessel_types, maintenance_categories, wind_farms, days, scenarios, L_RT, weather_windows, debug=False):
    model = gp.Model()

    l = model.addVars((m.name for m in maintenance_categories), vtype=GRB.INTEGER, name="l")
    d = model.addVar(name="d")

    model.setObjective(0)
    model.addConstr(gp.quicksum(l[m.name] * m.duration for m in maintenance_categories) <= 4)
    model.addConstr(d == gp.quicksum(l[m.name] * m.duration for m in maintenance_categories))
    
    if not debug:
        model.Params.OutputFlag = 0

    model.Params.PoolSolutions = 1000
    model.Params.PoolSearchMode = 2

    model.optimize()

    nSolutions = model.SolCount

    K = {v.name: [] for v in vessel_types}
    L = {}
    P = {}

    for k in range(nSolutions):
        model.Params.SolutionNumber = k
        L[k] = int(d.Xn)
        
        active_m = []
        for m in maintenance_categories:
            val = l[m.name].Xn
            P[(m.name, k)] = val
            if val > 1e-4:
                active_m.append(m)
        
        for v in vessel_types:
            if all([v.name in m.vessel_types for m in active_m]):
                K[v.name].append(k)
                
    K_hids = remove_infeasible_patterns(vessel_types, wind_farms, days, scenarios, K, L, L_RT, weather_windows)
    K_hids = remove_dominated_patterns(vessel_types, wind_farms, days, scenarios, K_hids, maintenance_categories, P)  
                    
    return K,L, K_hids, P

def remove_infeasible_patterns(vessel_types, wind_farms, days, scenarios, K, L_k, L_RT, weather_windows):
    
    K_hids = {(h.name, i.name, d, s): [] for h in vessel_types for i in wind_farms for d in range(1, days + 1) for s in scenarios}

    for h in vessel_types:
        for i in wind_farms:
            for d in range(1, days + 1):
                for s in scenarios:
                    for k in K[h.name]:
                        if L_k[k] + L_RT[h.name, i.name] <= weather_windows[h.name, i.name, d, s]:
                            K_hids[(h.name, i.name, d, s)].append(k)
    return K_hids

def remove_dominated_patterns(vessel_types, wind_farms, days, scenarios, K_hids, maintenance_categories, P_mk):
    for h in vessel_types:
        for i in wind_farms:
            for d in range(1, days + 1):
                for s in scenarios:
                    for k1_idx, k1_val in enumerate(K_hids[(h.name, i.name, d, s)]):
                        for k2_idx, k2_val in enumerate(K_hids[(h.name, i.name, d, s)]):
                            if k1_val != k2_val:
                                dominates = True
                                for m in maintenance_categories:
                                    if not P_mk[m.name, k1_val] >= P_mk[m.name, k2_val]:
                                        dominates = False
                                if dominates:
                                    K_hids[(h.name, i.name, d, s)][k2_idx]
                                    
    return K_hids

maintenance_categories = [
    MaintenanceCategory("Annual Service", failure_rate=5.0, duration=2, vessel_types=["CTV", "SOV"]),
    MaintenanceCategory("Manual Reset", failure_rate=7.5, duration=4, vessel_types=["CTV", "SOV"]),
    # MaintenanceCategory("Minor Repair", failure_rate=3, duration=3.75, vessel_types=["CTV", "SOV"]),
    # MaintenanceCategory("Medium Repair", failure_rate=0.825, duration=3.67, vessel_types=["CTV", "SOV"]),
    # MaintenanceCategory("Severe Repair", failure_rate=0.12, duration=4.33, vessel_types=["SOV"]),
]

vessel_types = [
    VesselType("CTV", multiday=False, n_teams=3, max_wind=25, max_wave=1.5, shift_length=10, day_rate=2_940, mob_rate=58_825, speed=35, cost_per_km=8, periodic_return=None, usage_cost_per_day=800),
    # VesselType("SOV", multiday=True, n_teams=7, max_wind=30, max_wave=2, shift_length=12, day_rate=11_765, mob_rate=235_295, speed=20, cost_per_km=10, periodic_return=14, usage_cost_per_day=5000)
]
scenarios = [1]
base = Base("Base A",  coordinates=(53.7, 7.4))
wind_farms = [
    WindFarm("Wind Farm A", coordinates=(54.0, 7.3), n_turbines=100, weather_data_file="Location 1.csv")
]
L_RT = {(h.name, i.name): 0 if h.multiday else 2 * haversine(i.coordinates, base.coordinates, unit=Unit.KILOMETERS) / h.speed for i in wind_farms for h in vessel_types}
days = 5
weather_windows = find_weather_windows(scenarios=scenarios, wind_farms=wind_farms, vessel_types=vessel_types)

K, L, K_hids, P = generate_patterns(vessel_types, maintenance_categories, wind_farms, days, scenarios, L_RT, weather_windows,)
print("Kv:", K)
print(" ")
print("Lk:", L)
print(" ")
print("K_hids:", K_hids)
print(" ")
print("Pmk:", P)
