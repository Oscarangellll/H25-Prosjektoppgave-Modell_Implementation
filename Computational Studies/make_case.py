import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import config
from model import real_model

from classes import MaintenanceCategory, Vessel, VesselType, WindFarm

def make_case(name, n_vessels, n_wind_farms, n_scenarios, seed, scenario):
    vessel_types = [
        VesselType("CTV", multiday=False, n_teams=3, max_wind=25, max_wave=1.5, shift_length=10, day_rate=2_940, mob_rate=58_825, speed=35, cost_per_km=8, periodic_return=None, usage_cost_per_day=800),
        VesselType("SOV", multiday=True, n_teams=7, max_wind=30, max_wave=2, shift_length=12, day_rate=11_765, mob_rate=235_295, speed=20, cost_per_km=10, periodic_return=14, usage_cost_per_day=5000)
    ]
    vessels = [
        Vessel("SOV1", vessel_type=vessel_types[1]),
        Vessel("SOV2", vessel_type=vessel_types[1]),
        Vessel("SOV3", vessel_type=vessel_types[1]),
        Vessel("SOV5", vessel_type=vessel_types[1]),  
        Vessel("SOV6", vessel_type=vessel_types[1]),
        Vessel("SOV7", vessel_type=vessel_types[1]),  
        Vessel("SOV8", vessel_type=vessel_types[1]),
        Vessel("SOV9", vessel_type=vessel_types[1]),  
        Vessel("SOV10", vessel_type=vessel_types[1]),
    ]
    vessels = vessels[:n_vessels]
    wind_farms = [
        WindFarm("Wind Farm A", coordinates=(54.0, 7.3), n_turbines=100, weather_data_file="Location 2.csv", turbine_model="DTU"),
        WindFarm("Wind Farm B", coordinates=(53.94, 6.6), n_turbines=80, weather_data_file="Location 2.csv", turbine_model="DTU"),
        WindFarm("Wind Farm C", coordinates=(54.4, 7.9), n_turbines=70, weather_data_file="Location 2.csv", turbine_model="DTU"),
    ]
    wind_farms = wind_farms[:n_wind_farms]
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

    return real_model(
        name,
        vessel_types,
        vessels,
        wind_farms,
        maintenance_categories,
        n_scenarios,
        scenario,
        days,
        months,
        n_total_days,
        seed
    )
    
# model = make_case("Test Case", n_vessels=2, n_wind_farms=1, n_scenarios=3)
# model.optimize()