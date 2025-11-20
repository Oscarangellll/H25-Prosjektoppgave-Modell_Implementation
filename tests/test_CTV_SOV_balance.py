import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from classes import MaintenanceCategory, Vessel, VesselType, WindFarm, Base
from model import model

name = "Two Stage Test Model"
vessel_types = [
    VesselType("CTV", multiday=False, n_teams=3, max_wind=15, max_wave=2, shift_length=10, day_rate=20, mob_rate=300, speed=30, cost_per_km=3, periodic_return=None, usage_cost_per_day=10),
    VesselType("SOV", multiday=True, n_teams=5, max_wind=20, max_wave=2.5, shift_length=12, day_rate=50, mob_rate=1000, speed=20, cost_per_km=5, periodic_return=5, usage_cost_per_day=20)
]
vessels = [
    Vessel("SOV1", vessel_type=vessel_types[1])
]
wind_farms = [
    WindFarm("Wind Farm A", coordinates=(54.0, 7.3), n_turbines=100, weather_data_file=None)
]
base = Base("Base A",  coordinates=(53.7, 7.4))
maintenance_categories = [
    MaintenanceCategory("Small", failure_rate=None, duration=2, vessel_types=["CTV", "SOV"]),
    MaintenanceCategory("Large", failure_rate=None, duration=5, vessel_types=["SOV"])
]
days_per_month = 5
months = ["January", "February"]
scenarios = [1]
failures = {
    ("Wind Farm A", "Small", 1, 1): 0, #swap here
    ("Wind Farm A", "Small", 2, 1): 0, 
    ("Wind Farm A", "Small", 3, 1): 0,
    ("Wind Farm A", "Small", 4, 1): 0,
    ("Wind Farm A", "Small", 5, 1): 0,
    ("Wind Farm A", "Small", 6, 1): 0,
    ("Wind Farm A", "Small", 7, 1): 0,
    ("Wind Farm A", "Small", 8, 1): 0,
    ("Wind Farm A", "Small", 9, 1): 0,
    ("Wind Farm A", "Small", 10, 1): 0,
    ("Wind Farm A", "Large", 1, 1): 1, #swap here
    ("Wind Farm A", "Large", 2, 1): 0, 
    ("Wind Farm A", "Large", 3, 1): 0,
    ("Wind Farm A", "Large", 4, 1): 0,
    ("Wind Farm A", "Large", 5, 1): 0,
    ("Wind Farm A", "Large", 6, 1): 0,
    ("Wind Farm A", "Large", 7, 1): 0,
    ("Wind Farm A", "Large", 8, 1): 0,
    ("Wind Farm A", "Large", 9, 1): 0,
    ("Wind Farm A", "Large", 10, 1): 0
}
patterns = {}
patterns[("Small", 0)] = 1
patterns[("Large", 0)] = 0
patterns[("Small", 1)] = 0
patterns[("Large", 1)] = 1

pattern_lengths = {}
pattern_lengths[0] = 2
pattern_lengths[1] = 5

pattern_indexes_for_v = {"CTV": [0], "SOV": [0, 1]}

weather_windows = {
    ("CTV", "Wind Farm A", 1, 1): 24,
    ("CTV", "Wind Farm A", 2, 1): 24,
    ("CTV", "Wind Farm A", 3, 1): 24,
    ("CTV", "Wind Farm A", 4, 1): 24,
    ("CTV", "Wind Farm A", 5, 1): 24,
    ("CTV", "Wind Farm A", 6, 1): 24,
    ("CTV", "Wind Farm A", 7, 1): 24,
    ("CTV", "Wind Farm A", 8, 1): 24,
    ("CTV", "Wind Farm A", 9, 1): 24,
    ("CTV", "Wind Farm A", 10, 1): 24,
    ("SOV", "Wind Farm A", 1, 1): 24,
    ("SOV", "Wind Farm A", 2, 1): 24,
    ("SOV", "Wind Farm A", 3, 1): 24,
    ("SOV", "Wind Farm A", 4, 1): 24,
    ("SOV", "Wind Farm A", 5, 1): 24,
    ("SOV", "Wind Farm A", 6, 1): 24,
    ("SOV", "Wind Farm A", 7, 1): 24,
    ("SOV", "Wind Farm A", 8, 1): 24,
    ("SOV", "Wind Farm A", 9, 1): 24,
    ("SOV", "Wind Farm A", 10, 1): 24
}
downtime_cost_per_day = 1000

model = model(
    name, 
    vessel_types,
    vessels, 
    wind_farms,
    base,
    days_per_month,
    months,
    maintenance_categories,
    pattern_indexes_for_v,
    scenarios,
    failures,
    patterns,
    pattern_lengths,
    weather_windows,
    downtime_cost_per_day
)

#turn off output
model.Params.OutputFlag = 0

model.optimize()

#print objective value
print("Objective value:", model.ObjVal)
#print variable values
for v in model.getVars():
    # print(v)
    if v.X > 0:
        print(f"{v.VarName}: {v.X}")

# Expected output when failure is small:
# Charter CTV short term in january
# Fix failure on day 2 in January
# Objective value: 402

# Expected output when failure is large
# Charter one SOV for January 
# Fix failure on day 2 in January
# Objective value: 1252

