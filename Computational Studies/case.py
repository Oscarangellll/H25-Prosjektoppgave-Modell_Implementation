import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import config
from model import real_model

from classes import MaintenanceCategory, Vessel, VesselType, WindFarm

vessel_types = [
    VesselType("CTV", multiday=False, n_teams=3, max_wind=25, max_wave=1.5, shift_length=10, day_rate=2_940, mob_rate=58_825, speed=35, cost_per_km=8, periodic_return=None, usage_cost_per_day=800),
    VesselType("SOV", multiday=True, n_teams=7, max_wind=30, max_wave=2, shift_length=12, day_rate=11_765, mob_rate=235_295, speed=20, cost_per_km=10, periodic_return=14, usage_cost_per_day=5000)
]
vessels = [
    Vessel("SOV1", vessel_type=vessel_types[1]),
    Vessel("SOV2", vessel_type=vessel_types[1]),
    Vessel("SOV3", vessel_type=vessel_types[1]),
]
wind_farms = [
    WindFarm("Wind Farm A", coordinates=(54.0, 7.3), n_turbines=100, weather_data_file="Location 2.csv", turbine_model="Nordex_N90_2500"),
    WindFarm("Wind Farm B", coordinates=(53.94, 6.6), n_turbines=80, weather_data_file="Location 2.csv", turbine_model="Nordex_N90_2500"),
]
maintenance_categories = [
    MaintenanceCategory("Annual Service", failure_rate=5.0, duration=2, vessel_types=["CTV", "SOV"]),
    MaintenanceCategory("Manual Reset", failure_rate=7.5, duration=3, vessel_types=["CTV", "SOV"]),
    MaintenanceCategory("Minor Repair", failure_rate=3, duration=7.5, vessel_types=["CTV", "SOV"]),
    MaintenanceCategory("Medium Repair", failure_rate=0.825, duration=7.33, vessel_types=["CTV", "SOV"]),
    MaintenanceCategory("Severe Repair", failure_rate=0.12, duration=8.66, vessel_types=["CTV", "SOV"]),
]
days = 30
months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
n_total_days = days * len(months)

# Solve Case A
modelA = real_model(
    "Case A",
    vessel_types,
    vessels[:2],
    [wind_farms[0]],
    maintenance_categories,
    9,
    False,
    days,
    months,
    n_total_days,
    67
)
modelA.params.MIPGap = 0.002
modelA.params.TimeLimit = 14_400  # 4 hours
modelA.params.OutputFlag = 0

modelA.optimize()

objectiveA = modelA.ObjVal
gapA = modelA.MIPGap
chartered_vessels_A = []
for v in modelA.getVars():
    if "gamma" in v.VarName and v.X > 0:
        chartered_vessels_A.append(f"({v.VarName}: {v.X})")
runtimeA = modelA.Runtime

# Solve Case B

modelB = real_model(
    "Case B",
    vessel_types,
    vessels[:2],
    [wind_farms[1]],
    maintenance_categories,
    9,
    False,
    days,
    months,
    n_total_days,
    67
)
modelB.params.MIPGap = 0.002
modelB.params.TimeLimit = 14_400  # 4 hours
modelB.params.OutputFlag = 0

modelB.optimize()

objectiveB = modelB.ObjVal
gapB = modelB.MIPGap
chartered_vessels_B = []
for v in modelB.getVars():
    if "gamma" in v.VarName and v.X > 0:
        chartered_vessels_B.append(f"({v.VarName}: {v.X})")
runtimeB = modelB.Runtime

# Solve Case AB

modelAB = real_model(
    "Case A",
    vessel_types,
    vessels,
    wind_farms,
    maintenance_categories,
    9,
    False,
    days,
    months,
    n_total_days,
    67
)
modelAB.params.MIPGap = 0.002
modelAB.params.TimeLimit = 14_400  # 4 hours
modelAB.params.OutputFlag = 0

modelAB.optimize()

objectiveAB = modelAB.ObjVal
gapAB = modelAB.MIPGap
chartered_vessels_AB = []
for v in modelAB.getVars():
    if "gamma" in v.VarName and v.X > 0:
        chartered_vessels_AB.append(f"({v.VarName}: {v.X})")
runtimeAB = modelAB.Runtime
        
# write results to csv file
import csv
with open('case_results.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Case', 'Objective Value', 'MIP Gap', 'Chartered Vessels', 'Runtime'])
    writer.writerow(['Case A', objectiveA, gapA, '; '.join(chartered_vessels_A), runtimeA])
    writer.writerow(['Case B', objectiveB, gapB, '; '.join(chartered_vessels_B), runtimeB])
    writer.writerow(['Case AB', objectiveAB, gapAB, '; '.join(chartered_vessels_AB), runtimeAB])