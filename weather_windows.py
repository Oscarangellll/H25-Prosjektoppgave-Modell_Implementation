import pandas as pd

def _find_window(speed, height, vessel_limits):
    """
    Find the longest operable weather window (in hours)
    for a given vessel and daily weather data.
    """

    max_speed = vessel_limits["Speed"]
    max_height = vessel_limits["Height"]
    shift_limit = vessel_limits["Shift Length"]
    
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

def weather_windows(S):
    A = {}

    for w in W:
        data = pd.read_csv(f"Synthetic Weather Data/Wind Farm 1 hourly_synthetic.csv")
        data = data[data["Scenario"].isin(S)]
        for s, scenario_data in data.groupby("Scenario"):
            for d, daily_data in scenario_data.groupby("Day"):
                for v in V:
                    A[(v, w, d, s)] = _find_window(
                                    daily_data["Speed"],
                                    daily_data["Height"],
                                    Limits[v]
                                )
    return A

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
