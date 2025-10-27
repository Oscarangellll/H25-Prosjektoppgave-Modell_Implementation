import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",   # specify font family here
    "font.sans-serif": ["Arial"],  # specify font here
    "mathtext.fontset": "cm",
    "font.size":9,
    "legend.fontsize": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.5,
    "patch.linewidth": 0.5}) 

def plot_real_weather_year(root, file):
    real_weather = pd.read_csv(f"{root}/{file}.csv", sep =",", parse_dates=["Time"])
    real_weather = real_weather.sort_values(by="Time")
    #select one year
    real_weather = real_weather[real_weather["Time"].dt.year == real_weather["Time"].dt.year.min()]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 3.4), sharex=True)
    ax1.plot(real_weather["Time"], real_weather["Speed"], linewidth=0.8)
    ax2.plot(real_weather["Time"], real_weather["Height"], linewidth=0.8)
    ax1.set_ylabel("Wind Speed [m/s]")
    ax2.set_ylabel("Significant Wave Height [m]")
    ax1.set_xlabel("")
    ax2.set_xlabel("")
    ax1.set_title("Wind Speed")
    ax2.set_title("Significant Wave Height")
    ax1.tick_params(direction = "in", which = "major", right = True, top = True)
    ax2.tick_params(direction = "in", which = "major", right = True, top = True)
    #set master header
    fig.suptitle(f"Observed Weather Data, {file}", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{root}/{file}.pdf", dpi = 600, bbox_inches = "tight")
    
def plot_synthetic_weather_year(root, file):
    synthetic_weather = pd.read_csv(f"{root}/{file}.csv", sep =",")
    scenario_to_plot = 2  # tilsvarer "Sample"/år i din gamle kode
    dfp = (
        synthetic_weather
        .loc[synthetic_weather["Scenario"] == scenario_to_plot]
        .sort_values(["Month", "Day", "Hour"])
        .reset_index(drop=True)
    )
    # Løpende timeindeks innen scenarioet
    
    dfp["t"] = np.arange(1, len(dfp) + 1)

    # (Valgfritt) markører for månedsskifte på x-aksen
    month_starts = dfp.groupby("Month")["t"].min().tolist()
    month_labels = [f"M{m}" for m in sorted(dfp["Month"].unique())]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 3.4), sharex=True)
    ax1.plot(dfp["t"], dfp["Speed"], linewidth=0.8)
    ax2.plot(dfp["t"], dfp["Height"], linewidth=0.8)
    ax1.set_ylabel("Wind Speed [m/s]")
    ax2.set_ylabel("Significant Wave Height [m]")
    ax1.set_xlabel("")
    ax2.set_xlabel("Time [hours]")
    ax1.set_title("Wind Speed")
    ax2.set_title("Significant Wave Height")
    ax1.tick_params(direction = "in", which = "major", right = True, top = True)
    ax2.tick_params(direction = "in", which = "major", right = True, top = True)
    ax2.set_xticks(month_starts)
    ax2.set_xticklabels(month_labels)
    #set master header
    fig.suptitle(f"Synthetic Weather Data, {file}, scenario {scenario_to_plot}", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{root}/{file}.pdf", dpi = 600, bbox_inches = "tight")

plot_real_weather_year("Weather Data", "Location 1")
plot_real_weather_year("Weather Data", "Location 2")
plot_synthetic_weather_year("Synthetic Weather Data", "Location 1")
plot_synthetic_weather_year("Synthetic Weather Data", "Location 2")