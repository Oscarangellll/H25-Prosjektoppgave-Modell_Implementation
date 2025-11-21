import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import config
from model import model
from generate_patterns import generate_patterns
from failure_generation import failures
from weather_windows import find_weather_windows

from classes import MaintenanceCategory, Vessel, VesselType, WindFarm, Base
from calculate_downtime_cost import calculate_downtime_cost

name = "Two Stage Test Model"
vessel_types = [
    VesselType("CTV", multiday=False, n_teams=3, max_wind=25, max_wave=1.5, shift_length=10, day_rate=2_940, mob_rate=58_825, speed=35, cost_per_km=8, periodic_return=None, usage_cost_per_day=800),
    VesselType("SOV", multiday=True, n_teams=7, max_wind=30, max_wave=2, shift_length=12, day_rate=11_765, mob_rate=235_295, speed=20, cost_per_km=10, periodic_return=14, usage_cost_per_day=5000)
]
vessels = [
    Vessel("SOV1", vessel_type=vessel_types[1]),
    Vessel("SOV2", vessel_type=vessel_types[1]),
    # Vessel("SOV3", vessel_type=vessel_types[1]),
    # Vessel("SOV4", vessel_type=vessel_types[1]),
    # Vessel("SOV5", vessel_type=vessel_types[1])    
]
maintenance_categories = [
    MaintenanceCategory("Annual Service", failure_rate=5.0, duration=2, vessel_types=["CTV", "SOV"]),
    MaintenanceCategory("Manual Reset", failure_rate=7.5, duration=3, vessel_types=["CTV", "SOV"]), #d=4
    MaintenanceCategory("Minor Repair", failure_rate=3, duration=3.75, vessel_types=["CTV", "SOV"]),
    MaintenanceCategory("Medium Repair", failure_rate=0.825, duration=3.67, vessel_types=["CTV", "SOV"]),
    MaintenanceCategory("Severe Repair", failure_rate=0.12, duration=8.66, vessel_types=["CTV", "SOV"]),
]
wind_farms = [
    WindFarm("Wind Farm A", coordinates=(54.0, 7.3), n_turbines=100, weather_data_file="Location 1.csv", turbine_model="Nordex_N90_2500"),
    # WindFarm("Wind Farm B", coordinates=(54.0, 7.3), n_turbines=50, weather_data_file="Location 1.csv"),
    # WindFarm("Wind Farm C", coordinates=(54.0, 7.3), n_turbines=50, weather_data_file="Location 1.csv"),
    # WindFarm("Wind Farm D", coordinates=(54.0, 7.3), n_turbines=50, weather_data_file="Location 1.csv"),
    # WindFarm("Wind Farm E", coordinates=(54.0, 7.3), n_turbines=50, weather_data_file="Location 1.csv"),

]
base = Base("Base A",  coordinates=(53.7, 7.4))
days_per_month = 30
months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
ndays_in_year = days_per_month * len(months)
#create list of scenarios with n random numbers
num_s = 1
np.random.seed(config.RANDOM_SEED)
scenarios = [np.random.randint(1, 101) for _ in range(num_s)]
print("Scenarios:", scenarios)
failures = failures(scenarios=scenarios, wind_farms=wind_farms, maintenance_categories=maintenance_categories)
#lager patterns for et eller annet øvre vindu (må være større enn høyeste max shift length)

L_RT = {(h.name, i.name): 0 if h.multiday else 2 * haversine(i.coordinates, base.coordinates, unit=Unit.KILOMETERS) / h.speed for i in wind_farms for h in vessel_types}

weather_windows = find_weather_windows(scenarios=scenarios, wind_farms=wind_farms, vessel_types=vessel_types)

# Definere en K_hid 
# For hver h, i, d: hvis pattern k fra K er feasible, legge inn i K_hid
# For hver h, i, d: sammenlikn alle par av patterns i K_hid, og fjern de som er dominated
# Nå kan vi lage lambda variablene basert på K_hid og droppe feasibility sjekken i modellen (tror jeg)
downtime_cost = calculate_downtime_cost(wind_farms, scenarios, 0.1)
model = model(
    name, 
    vessel_types,
    vessels, 
    wind_farms,
    base,
    days_per_month,
    months,
    maintenance_categories,
    K_hids,
    scenarios,
    failures,
    P,
    L_RT,
    weather_windows,
    downtime_cost
)
model.Params.Presolve = 0
model.optimize()
print("Objective value:", model.ObjVal)

for v in model.getVars():
    if v.X > 0:
        if "gamma" in v.VarName:
            print(f"{v.VarName}: {v.X}")

# A = 5624545.68
# B = 5881784.97
# S = A + B + B 
# T = 12068700.96
# D = S - T
# print(S)
# print(D)
# print(D/S)
