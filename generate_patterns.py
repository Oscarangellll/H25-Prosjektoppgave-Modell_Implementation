from classes import MaintenanceCategory, VesselType
import gurobipy as gp
from gurobipy import GRB

def generate_patterns(vessels, maintenance_categories, debug=False):
    model = gp.Model()

    l = model.addVars((m.name for m in maintenance_categories), vtype=GRB.INTEGER, name="l")
    d = model.addVar(vtype=GRB.INTEGER, name="d")

    model.setObjective(0)
    model.addConstr(gp.quicksum(l[m.name] * m.duration for m in maintenance_categories) <= 7)
    model.addConstr(d == gp.quicksum(l[m.name] * m.duration for m in maintenance_categories))
    
    if not debug:
        model.Params.OutputFlag = 0

    model.Params.PoolSolutions = 100
    model.Params.PoolSearchMode = 2

    model.optimize()

    nSolutions = model.SolCount

    K = {v.name: [] for v in vessels}
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
        
        for v in vessels:
            if all([v.name in m.vessel_types for m in active_m]):
                K[v.name].append(k)
                    
    return K, L, P
"""
maintenance_categories = [
    MaintenanceCategory("Annual Service", failure_rate=5.0, duration=2, vessel_types=["CTV", "SOV"]),
    MaintenanceCategory("Manual Reset", failure_rate=7.5, duration=4, vessel_types=["SOV"]),
]

vessels = [
    VesselType("CTV", n_teams=2, max_wind=15, max_wave=1.5, shift_length=12, day_rate=3000, mob_rate=50_000),
    VesselType("SOV", n_teams=1, max_wind=20, max_wave=2.5, shift_length=24, day_rate=10_000, mob_rate=200_000),
]

K, L, P = generate_patterns(vessels, maintenance_categories)
print("Kv:", K)
print(" ")
print("Lk:", L)
print(" ")
print("Pmk:", P)
"""
