import pandas as pd
import numpy as np

from classes import WindFarm

def calculate_downtime_cost(wind_farms, scenarios, price):
    """
    price: EUR/kWh

    returns: downtime cost per day per turbine
    """
    downtime_cost = {}

    for i in wind_farms:
        
        data = pd.read_csv(f"Synthetic Weather Data/{i.weather_data_file}")
        data = data[data["Scenario"].isin(scenarios)]
        
        for s, scenario_data in data.groupby("Scenario"):
            for d, day_data in scenario_data.groupby("Day"):
                
                avg_speed = day_data["Speed"].mean()
                power_loss = np.interp(   
                    avg_speed, 
                    i.power_curve["speed"],
                    i.power_curve["power"]
                )

                downtime_cost[(i.name, d, s)] = power_loss * 24 * price 
    return downtime_cost

"""
wind_farms = [
    WindFarm("Wind Farm A", coordinates=(54.0, 7.3), n_turbines=100, weather_data_file="Location 1.csv", turbine_model="Repower_MM82")
]

calculate_downtime_cost(wind_farms, [1], 0.1)
"""
