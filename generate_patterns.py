from classes import MaintenanceCategory, VesselType
import gurobipy as gp
from gurobipy import GRB

def generate_patterns(vessels, maintenance_categories):
    model = gp.Model()

    l = model.addVars((m.name for m in maintenance_categories), vtype=GRB.INTEGER, name="l")
    d = model.addVar(vtype=GRB.INTEGER, name="d")

    model.setObjective(0)
    model.addConstr(gp.quicksum(l[m.name] * m.duration for m in maintenance_categories) <= 4)
    model.addConstr(d == gp.quicksum(l[m.name] * m.duration for m in maintenance_categories))

    model.Params.PoolSolutions = 100
    model.Params.PoolSearchMode = 2

    model.optimize()

    nSolutions = model.SolCount

    Kv = {v.name: [] for v in vessels}
    Lk = {}
    Pmk = {}

    for k in range(nSolutions):
        model.Params.SolutionNumber = k
        Lk[k] = int(d.Xn)
        for m in maintenance_categories:
            Pmk[(m.name, k)] = l[m.name].Xn
            for v in vessels:
                if v.name in m.vessel_types:
                    Kv[v.name].append(k)
                    
    return Kv, Lk, Pmk

maintenance_categories = [
    MaintenanceCategory("Annual Service", failure_rate=5.0, duration=2, vessel_types=["CTV", "SOV"]),
    MaintenanceCategory("Manual Reset", failure_rate=7.5, duration=4, vessel_types=["SOV"]),
]

vessels = [
    VesselType("CTV", n_teams=2, max_wind=15, max_wave=1.5, shift_length=12, day_rate=3000, mob_rate=50_000),
    VesselType("SOV", n_teams=1, max_wind=20, max_wave=2.5, shift_length=24, day_rate=10_000, mob_rate=200_000),
]

Kv, Lk, Pmk = generate_patterns(vessels, maintenance_categories)
print("Kv:", Kv)
print(" ")
print("Lk:", Lk)
print(" ")
print("Pmk:", Pmk)