import pandas as pd
import os

import config

def _find_window(speed, height, vessel):
    """
    Find the longest operable weather window (in hours)
    for a given vessel and daily weather data.
    """

    max_speed = vessel.max_wind
    max_height = vessel.max_wave
    shift_limit = vessel.shift_length
    
    current_window = 0
    max_window = 0
    
    operable = (speed <= max_speed) & (height <= max_height)
    
    for hour in operable:
        if hour:
            current_window += 1
            max_window = max(current_window, max_window)
        else:
            current_window = 0

    max_window = min(max_window, shift_limit)
    return max_window

def find_weather_windows(scenarios, wind_farms, vessels):
    weather_windows = {}

    for w in wind_farms:
        data = pd.read_csv(os.path.join(config.SYN_WEATHER_DATA_FOLDER, w.weather_data_file))
        data = data[data["Scenario"].isin(scenarios)]
        for s, scenario_data in data.groupby("Scenario"):
            for d, daily_data in scenario_data.groupby("Day"):
                for v in vessels:
                    weather_windows[(v.name, w.name, d, s)] = _find_window(
                                    daily_data["Speed"],
                                    daily_data["Height"],
                                    v
                                )
    return weather_windows
"""
W = ["Wind Farm A", "Wind Farm B"]

V = ["CTV", "SOV"]

Limits = {
    "CTV": {"Speed": 10, "Height": 2, "Shift Length": 12},
    "SOV": {"Speed": 12, "Height": 2.5, "Shift Length": 16}
}

A = weather_windows([1])

# plot A for verification
import matplotlib.pyplot as plt
for v in V:
    for w in W:
        days = sorted(set(d for (_, ww, d, _) in A.keys() if ww == w))
        values = [A[(v, w, d, 1)] for d in days]
        plt.plot(days, values, label=f"{v} at {w}")
plt.xlabel("Day")
plt.ylabel("Max Operable Window (hours)")
plt.title("Max Operable Weather Windows")
plt.legend()
plt.show()
"""
