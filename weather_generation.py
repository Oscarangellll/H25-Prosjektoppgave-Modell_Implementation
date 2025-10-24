import gurobipy as gp
from gurobipy import GRB
import random
seed = 18
random.seed(seed)
from typing import Tuple, List, Dict
import numpy as np

def operability_tensor(wind: np.ndarray, wave: np.ndarray, weather_limits: Dict[str, Dict[str, float]]):
    WF, S, D, H = wind.shape
    vessels = list(weather_limits.keys())
    V = len(vessels)
    op = np.zeros((V, WF, S, D, H), dtype=bool)
    for v_idx, v in enumerate(vessels):
        mw = weather_limits[v]["max_wind"]
        mh = weather_limits[v]["max_wave"]
        op[v_idx] = (wind <= mw) & (wave <= mh)
    return vessels, op

def longest_resource_window(operational_hours: List[int], min_window: int = 5) -> int: 
    max_window = 0 
    current_window = 0 
    for operational_hour in operational_hours: 
        if operational_hour: 
            current_window += 1 
            max_window = max(max_window, current_window) 
        else: 
            current_window = 0 
    return max_window if max_window >= min_window else 0

def daily_weather_windows(
    operable: np.ndarray,
    min_window_hours: int,
    max_window_hours: int,
    vessels: List[str]
):
    V, WF, S, D, H = operable.shape
    Wwin = np.zeros((V, WF, S, D), dtype=int)

    for v in range(V):
        for f in range(WF):
            for s in range(S):
                for d in range(D):
                    L = longest_resource_window(operable[v, f, s, d, :])
                    Wwin[v, f, s, d] = L if L >= min_window_hours else 0
    return min(Wwin, max_window_hours)

def pack_windows_to_dict(
    vessels: List[str],
    wind_farms: List[str],
    days: List[int],
    scenarios: List[int],
    Wwin: np.ndarray,
) -> Dict[tuple, int]:
    A = {}
    V, WF, S, D = Wwin.shape 
    for v_idx, v in enumerate(vessels):
        for w_idx, w in enumerate(wind_farms):
            for s_idx, s in enumerate(scenarios):
                for d_idx, d in enumerate(days):
                    A[(v, w, d, s)] = int(Wwin[v_idx, w_idx, s_idx, d_idx])
    return A

# # test the functions
# # find what wind and wave are created 
# W = ["Wind Farm A"]
# S = range(1)
# D = [d for d in range(360)] # 360 days
# wind, wave = gen_weather(W, S, D, seed=18)
# print(f"wind: {wind}")
# print(f"wave: {wave}")
# # find operability tensor for CTV
# weather_limits = {
#     "Crew Transfer Vessel": {"max_wind": 25.0, "max_wave": 1.5},
# }
# vessels, op = operability_tensor(wind, wave, weather_limits)
# print(f"operability: {op}")
# Wwin = daily_weather_windows(op, min_window_hours=4, max_window_hours=12, vessels=vessels)
# A_vwd_s = pack_windows_to_dict(vessels, W, D, S, Wwin)
# # print the different weather windows values
# print(f"A_vwd_s: {A_vwd_s.values()}")

# #plot the weather for the first scenario
# import matplotlib.pyplot as plt
# plt.figure(figsize=(12, 6))
# plt.subplot(2, 1, 1)
# plt.plot(wind[0, 0, :, :].flatten(), label='Wind Speed (m/s)')
# plt.axhline(y=weather_limits["Crew Transfer Vessel"]["max_wind"], color='r', linestyle='--', label='CTV Max Wind Limit')
# plt.title('Wind Speed Over Time')
# plt.xlabel('Hours')
# plt.ylabel('Wind Speed (m/s)')
# plt.legend()
# plt.subplot(2, 1, 2)
# plt.plot(wave[0, 0, :, :].flatten(), label='Wave Height (m)', color='orange')
# plt.axhline(y=weather_limits["Crew Transfer Vessel"]["max_wave"], color='r', linestyle='--', label='CTV Max Wave Limit')
# plt.title('Wave Height Over Time')
# plt.xlabel('Hours')
# plt.ylabel('Wave Height (m)')
# plt.legend()
# plt.tight_layout()
# plt.show()

# #plot weather windows for the first scenario
# plt.figure(figsize=(12, 4))
# plt.plot(Wwin[0, 0, 0, :], marker='')
# plt.title('Daily Weather Windows for Crew Transfer Vessel at Wind Farm A')
# plt.xlabel('Days')
# plt.ylabel('Max Operable Hours')
# plt.grid()
# # include trend curve
# p = np.poly1d(np.polyfit(range(len(D)), Wwin[0, 0, 0, :], 4)) # 4th degree polynomial
# plt.plot(p(range(len(D))), linestyle='--', color='orange', label='Trend Curve')
# plt.show()