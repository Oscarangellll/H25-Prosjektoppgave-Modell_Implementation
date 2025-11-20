from classes import MaintenanceCategory, VesselType
import gurobipy as gp
from gurobipy import GRB

def generate_patterns(vessel_types, maintenance_categories, debug=False):
    model = gp.Model()

    l = model.addVars((m.name for m in maintenance_categories), vtype=GRB.INTEGER, name="l")
    d = model.addVar(name="d")

    model.setObjective(0)
    model.addConstr(gp.quicksum(l[m.name] * m.duration for m in maintenance_categories) <= 12)
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
                    
    return K, L, P

# maintenance_categories = [
#     MaintenanceCategory("Annual Service", failure_rate=5.0, duration=2, vessel_types=["CTV", "SOV"]),
#     MaintenanceCategory("Manual Reset", failure_rate=7.5, duration=4, vessel_types=["CTV", "SOV"]),
#     MaintenanceCategory("Minor Repair", failure_rate=3, duration=3.75, vessel_types=["CTV", "SOV"]),
#     MaintenanceCategory("Medium Repair", failure_rate=0.825, duration=3.67, vessel_types=["CTV", "SOV"]),
#     MaintenanceCategory("Severe Repair", failure_rate=0.12, duration=4.33, vessel_types=["SOV"]),
# ]

# vessel_types = [
#     VesselType("CTV", multiday=False, n_teams=3, max_wind=25, max_wave=1.5, shift_length=10, day_rate=2_940, mob_rate=58_825, speed=35, cost_per_km=8, periodic_return=None, usage_cost_per_day=800),
#     VesselType("SOV", multiday=True, n_teams=7, max_wind=30, max_wave=2, shift_length=12, day_rate=11_765, mob_rate=235_295, speed=20, cost_per_km=10, periodic_return=14, usage_cost_per_day=5000)
# ]

# K, L, P = generate_patterns(vessel_types, maintenance_categories)
# print("Kv:", K)
# print(" ")
# print("Lk:", L)
# print(" ")
# print("Pmk:", P)
# print(" ")
# print("Number of patterns generated:", len(L))