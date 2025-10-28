# File containing functions used to generate synthetic weather data and plot results.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA, sarimax
import statsmodels as sm
import scipy.stats
import sklearn
from statsmodels.tsa.vector_ar.var_model import VAR
import statsmodels.api as sma
from scipy.optimize import curve_fit
from statsmodels.graphics.gofplots import qqplot
from statsmodels.distributions.empirical_distribution import ECDF

from config import Config

config = Config()

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

# Load data
weather_data = pd.read_csv("Weather Data/Wind Farm 1.csv", sep=",")

# Convert Time column to datetime
weather_data['Time'] = pd.to_datetime(weather_data['Time'])

# Create Month column
weather_data['Month'] = weather_data['Time'].dt.month
# Create Year column
weather_data['Year'] = weather_data['Time'].dt.year

# Remove negative values using temporal linear interpolation
# First mark negative values as NaN
weather_data.loc[weather_data["Speed"] <= 0, "Speed"] = np.nan
weather_data.loc[weather_data["Height"] <= 0, "Height"] = np.nan
# Then, interpolate the NaN values using neighboring points, then fill any remaining NaNs at start/end
weather_data["Speed"] = weather_data["Speed"].interpolate(method='linear').bfill().ffill()
weather_data["Height"] = weather_data["Height"].interpolate(method='linear').bfill().ffill()

##################################
# Transform series to remove skew
##################################

# Perform Box-Cox transformation on 'Speed' column
weather_data['Speed_boxcox'], speed_lambda = scipy.stats.boxcox(weather_data['Speed']) 

# Perform Box-Cox transformation on 'Height' column
weather_data['Height_boxcox'], height_lambda = scipy.stats.boxcox(weather_data['Height'])

#####################
# Standardize series
#####################

# Group by month and calculate mean and standard deviation for 'Speed' and 'Height'
monthly_stats = weather_data.groupby(weather_data["Month"])[['Speed_boxcox', 'Height_boxcox']].agg(['mean', 'std'])

# Flatten the multi-index column names
monthly_stats.columns = [f'{col[0]}_{col[1]}' for col in monthly_stats.columns]

# Merge the calculated statistics back into the original DataFrame
weather_data = weather_data.merge(monthly_stats, on='Month', how='left')
weather_data = weather_data.sort_values(by='Time')
weather_data.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore') 

# Standardize wind speed
weather_data["Speed_standardized"] = (weather_data["Speed_boxcox"] - weather_data["Speed_boxcox_mean"])/weather_data["Speed_boxcox_std"]
weather_data["Height_standardized"] = (weather_data["Height_boxcox"] - weather_data["Height_boxcox_mean"])/weather_data["Height_boxcox_std"]

#weather_data = weather_data.sort_values(by='Time')

###############
# Fit VAR model
###############

model = VAR(endog = weather_data[["Speed_standardized", "Height_standardized"]])
results = model.fit(maxlags=20, ic='bic') 

# Print summary of the model
print(results.summary())

##############
# Simulate VAR
##############

H, D, M, S = config.HOURS, config.DAYS, config.MONTHS, config.SYNTHETIC_YEARS
periods = H * D * M * S

sim = results.simulate_var(seed = 18, steps = periods)

sim = np.asarray(sim)
sim_wind = sim[:, 0]
sim_wave = sim[:, 1]

idx = pd.MultiIndex.from_product(
    [range(1, S + 1), range(1, M + 1), range(1, D + 1), range(1, H + 1)],
    names=["Scenario", "Month", "Day", "Hour"]
)

sim_weather = pd.DataFrame(index=idx).reset_index()
sim_weather["Sim_speed"] = sim_wind
sim_weather["Sim_height"] = sim_wave

#make repeating dayID for every scenario
sim_weather["DayID"] = D * sim_weather["Month"] + sim_weather["Day"] - D

sim_weather = sim_weather.merge(monthly_stats, on='Month', how='left')

# De-standardize series
sim_weather["Sim_speed"] = sim_weather["Sim_speed"] * sim_weather["Speed_boxcox_std"] + sim_weather["Speed_boxcox_mean"]
sim_weather["Sim_height"] = sim_weather["Sim_height"] * sim_weather["Height_boxcox_std"] + sim_weather["Height_boxcox_mean"]

# Inverse Box-Cox transformation
sim_weather["Sim_speed"] = scipy.special.inv_boxcox(sim_weather["Sim_speed"], speed_lambda)
sim_weather["Sim_height"] = scipy.special.inv_boxcox(sim_weather["Sim_height"], height_lambda)

# Temporal linear interpolation to handle any NaNs
sim_weather["Sim_speed"] = sim_weather["Sim_speed"].interpolate(method='linear').bfill().ffill().clip(lower=0)
sim_weather["Sim_height"] = sim_weather["Sim_height"].interpolate(method='linear').bfill().ffill().clip(lower=0)

######################
# Clean and save data
######################

sim_weather = sim_weather[["Scenario", "Hour", "DayID", "Month", "Sim_speed", "Sim_height"]]
sim_weather = sim_weather.rename(columns={'Sim_height': "Height", 'Sim_speed': "Speed", "DayID": "Day"})

# save raw hourly weather data to results folder
sim_weather.to_csv("Synthetic Weather Data/Wind Farm 1 hourly_synthetic.csv", sep = ",", index = False) # save full synthetic weather data

#######################
# Calculate statistics 
#######################

sim_monthly_mean = sim_weather.groupby(["Month"]).mean()
observed_monthly_mean = weather_data.groupby(["Month"]).mean()

sim_monthly_std = sim_weather.groupby(["Month"]).std()
observed_monthly_std = weather_data.groupby(["Month"]).std()

############################
# Plot original time series
############################

weather_data = weather_data.sort_values(by = "Time")

fig, (ax1, ax2) = plt.subplots(nrows=2, figsize = (6.5, 3.7), sharex = True)
sns.lineplot(weather_data[0:365*24], ax = ax1, x = "Time", y = "Speed", color = "royalblue", label = "Wind Speed", linewidth = 0.8, legend=False)
sns.lineplot(weather_data[0:365*24], x = "Time", y = "Height", color = "navy", label = "Wave Height", linewidth = 0.8, legend = False)
ax1.set_ylabel("Wind Speed [m/s]")
ax2.set_ylabel("Significant Wave Height [m]")
ax1.set_xlabel("")
ax2.set_xlabel("")
ax1.set_title("Wind Speed, FINO1")
ax2.set_title("Significant Wave Height, FINO1")
ax1.tick_params(direction = "in", which = "major", right = True, top = True)
ax2.tick_params(direction = "in", which = "major", right = True, top = True)
plt.tight_layout()
plt.savefig("Synthetic Weather Data/Wind Farm 1 Observed_data.pdf", dpi = 600, bbox_inches = "tight")


###########################
# Plot synthetic time series
###########################
sim_weather = pd.read_csv("Synthetic Weather Data/Wind Farm 1 hourly_synthetic.csv", sep =",")
scenario_to_plot = 2  # tilsvarer "Sample"/år i din gamle kode
dfp = (
    sim_weather
    .loc[sim_weather["Scenario"] == scenario_to_plot]
    .sort_values(["Month", "Day", "Hour"])
    .reset_index(drop=True)
)

# Løpende timeindeks innen scenarioet
dfp["t"] = np.arange(1, len(dfp) + 1)

# (Valgfritt) markører for månedsskifte på x-aksen
month_starts = dfp.groupby("Month")["t"].min().tolist()
month_labels = [f"M{m}" for m in sorted(dfp["Month"].unique())]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 3.4), sharex=True)

# Bruk gjerne én av disse to variantene:

# Variant A: seaborn
sns.lineplot(data=dfp, x="t", y="Speed",  ax=ax1, linewidth=0.8)
sns.lineplot(data=dfp, x="t", y="Height", ax=ax2, linewidth=0.8)

# # Variant B: ren matplotlib
# ax1.plot(dfp["t"], dfp["Speed"], linewidth=0.8)
# ax2.plot(dfp["t"], dfp["Height"], linewidth=0.8)

ax1.set_title(f"Simulated Wind Speed (Scenario {scenario_to_plot})")
ax2.set_title(f"Simulated Significant Wave Height (Scenario {scenario_to_plot})")

# Pynt x-aksen med månedsskifter (valgfritt)
ax2.set_xticks(month_starts)
ax2.set_xticklabels(month_labels, rotation=0)
ax2.set_xlabel("Progressive hour (M/D/H order)")

plt.tight_layout()
plt.savefig("Synthetic Weather Data/Wind Farm 1 Simulated_weather.pdf", dpi=600, bbox_inches="tight")

# ###############################
# # Plot empirical distributions
# ###############################
# plt.rcParams.update({
#     "text.usetex": False,
#     "font.family": "sans-serif",   # specify font family here
#     "font.sans-serif": ["Arial"],  # specify font here
#     "mathtext.fontset": "cm",
#     "font.size":9,
#     "legend.fontsize": 9,
#     "axes.labelsize": 9,
#     "axes.titlesize": 10,
#     "xtick.labelsize": 9,
#     "ytick.labelsize": 9,
#     "axes.linewidth": 0.5,
#     "patch.linewidth": 0.5}) 

# fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize = (7, 3.2))
# fig.subplots_adjust(wspace = 0.2)
# ax1.hist(sim_weather["Sim_speed"], histtype = "step", bins = 100, density = True, label = "Synthetic", color = "red")
# ax1.hist(weather_data["Speed"], histtype = "step", bins = 100, density = True, label = "Observed", color = "royalblue")
# ax1.legend(edgecolor = "black")
# ax1.set_title("Wind Speed")
# ax1.set_ylabel("Density")
# ax1.set_xlabel("Wind Speed [m/s]")
# ax1.minorticks_on()
# ax1.tick_params(direction = "in", which = "major", right = True, top = True)
# ax1.tick_params(direction = "in", which = "minor", right = True, top = True)

# ax2.hist(sim_weather["Sim_height"], histtype = "step", bins = 100, density = True, label = "Synthetic", color = "red")
# ax2.hist(weather_data["Height"], histtype = "step", bins = 100, density = True, label = "Observed", color = "royalblue")
# ax2.legend(edgecolor = "black")
# ax2.set_title("Significant Wave Height")
# #ax2.set_xlim([0, 10])
# ax2.set_xlabel("Significant Wave Height [m]")
# ax2.minorticks_on()
# ax2.tick_params(direction = "in", which = "major", right = True, top = True)
# ax2.tick_params(direction = "in", which = "minor", right = True, top = True)

# plt.savefig("wave_wind_distributions_empirical.pdf", dpi = 600, bbox_inches="tight")

# ########################################
# # Calculate and plot weather window distribution 
# ########################################

# # Function for calculating number of weather windows of different lengths
# def calc_weather_window_dist(wind_speeds, wave_heights, wind_cap, wave_cap):
#     weather_windows = []
    
#     consecutives = 0
#     for speed, height in zip(wind_speeds, wave_heights):
#         if speed <= wind_cap and height <= wave_cap:
#             consecutives += 1
        
#         else:
#             if consecutives > 0:
#                 weather_windows.append(consecutives)
#                 consecutives = 0

#     return weather_windows

# # Hourly data weather window bins, long term
# bins = np.arange(24, 24 * 20, 24)

# window_sim = calc_weather_window_dist(sim_weather["Sim_speed"], sim_weather["Sim_height"], 18, 2)
# binned_windows_sim = pd.cut(window_sim, bins=bins)
# window_df_sim = pd.DataFrame({'Weather Window Lengths': window_sim, 'Bins': binned_windows_sim})
# # Group by bins and calculate the sum of values in each bin
# sums_by_bin_sim = window_df_sim.groupby('Bins')['Weather Window Lengths'].sum()/len(sim_weather)
# sums_by_bin_sim = pd.DataFrame(sums_by_bin_sim)
# sums_by_bin_sim.columns =["Proportion_sim"]

# window_observed = calc_weather_window_dist(weather_data["Speed"], weather_data["Height"], 18, 2)
# binned_windows = pd.cut(window_observed, bins = bins)
# window_df = pd.DataFrame({'Weather Window Lengths': window_observed, 'Bins': binned_windows})
# sums_by_bin = window_df.groupby('Bins')['Weather Window Lengths'].sum()/len(weather_data)
# sums_by_bin = pd.DataFrame(sums_by_bin)
# sums_by_bin.columns = ["Proportion"]

# sums_by_bin_total = pd.merge(sums_by_bin_sim, sums_by_bin, on = "Bins", how = "inner")

# bin_names = []
# for i in range(0, len(bins) - 1, 1):
#     bin_names.append(str(bins[i]))
# sums_by_bin_total["Bin_names"] = bin_names

# # rename columns
# sums_by_bin_total.rename(columns = {'Proportion_sim': 'Synthetic', 'Proportion': 'Observed'}, inplace = True) 

# # Melt the dataframe to long format
# sums_by_bin_total = sums_by_bin_total.melt(id_vars=['Bin_names'], value_vars=['Synthetic', 'Observed'], var_name='Variable', value_name='Proportion')

# # Plot the distribution of weather windows
# plt.rcParams.update({
#     "text.usetex": False,
#     "font.family": "sans-serif",   # specify font family here
#     "font.sans-serif": ["Arial"],  # specify font here
#     "mathtext.fontset": "cm",
#     "font.size":9,
#     "legend.fontsize": 9,
#     "axes.labelsize": 9,
#     "axes.titlesize": 10,
#     "xtick.labelsize": 9,
#     "ytick.labelsize": 9,
#     "axes.linewidth": 0.5,
#     "patch.linewidth": 0.5}) 

# fig, ax = plt.subplots(figsize = (6.5, 3.5))
# sns.barplot(data=sums_by_bin_total, x='Bin_names', y='Proportion', hue='Variable', palette= ["red", "royalblue"])
# # Adjust bottom margin to accommodate rotated x-labels
# #plt.subplots_adjust(bottom=0.2)
# ax.set_xlabel("Persistence [h]")
# ax.set_ylabel("Duration relative to toal duration")
# plt.legend(edgecolor = "black")

# plt.savefig("persistance_stats.pdf", dpi = 600)

# #########################################################
# # Cumulative distribution of simulated and observed data
# #########################################################

# ecdf_sim_speed = ECDF(sim_weather["Sim_speed"])
# ecdf_observed_speed = ECDF(weather_data["Speed"])

# ecdf_sim_height = ECDF(sim_weather["Sim_height"])
# ecdf_observed_height = ECDF(weather_data["Height"])

# fig, (ax1, ax2) = plt.subplots(ncols= 2)

# ax1.plot(ecdf_sim_speed.x, ecdf_sim_speed.y, label = "Simulated")
# ax1.plot(ecdf_observed_speed.x, ecdf_observed_speed.y, label = "Observed")
# ax1.set_xlabel("Wind speed [m/s]")
# ax1.set_ylabel("Probability")
# ax1.legend()

# ax2.plot(ecdf_sim_height.x, ecdf_sim_height.y, label = "Simulated")
# ax2.plot(ecdf_observed_height.x, ecdf_observed_height.y, label = "Observed")
# ax2.set_xlabel("Significant wave height [m]")
# ax2.set_ylabel("Probability")
# ax2.legend()

# #############################################################
# # Plot sample autocorrelation with simulated autocorrelation
# #############################################################

# acf_wind_data = sma.tsa.acf(weather_data["Speed"], nlags = 24 * 7)
# acf_wind_sim = sma.tsa.acf(sim_weather["Sim_speed"], nlags = 24 * 7)
# acf_wave_data = sma.tsa.acf(weather_data["Height"], nlags = 24 * 7)
# acf_wave_sim = sma.tsa.acf(sim_weather["Sim_height"], nlags = 24 * 7)

# plt.rcParams.update({
#     "text.usetex": False,
#     "font.family": "sans-serif",   # specify font family here
#     "font.sans-serif": ["Arial"],  # specify font here
#     "mathtext.fontset": "cm",
#     "font.size":9,
#     "legend.fontsize": 9,
#     "axes.labelsize": 9,
#     "axes.titlesize": 10,
#     "xtick.labelsize": 9,
#     "ytick.labelsize": 9,
#     "axes.linewidth": 0.5,
#     "patch.linewidth": 0.5}) 

# fig, (ax1, ax2) = plt.subplots(ncols=2, sharey = True, figsize = (7, 3.2))
# fig.subplots_adjust(wspace = 0.2)
# sns.lineplot(acf_wind_sim, color = "red", ax = ax1, label = "Synthetic")
# sns.lineplot(acf_wind_data, color = "royalblue", ax = ax1, label = "Observed")
# ax1.set_title("Wind Speed")
# ax1.set_ylabel("Autocorrelation")
# ax1.set_xlabel("Lag [h]")
# ax1.legend(edgecolor = "black")
# ax1.minorticks_on()
# ax1.tick_params(direction = "in", which = "major",  right = True, top = True)
# ax1.tick_params(direction = "in", which = "minor",  right = True, top = True)

# sns.lineplot(acf_wave_sim, color = "red", ax = ax2, label = "Synthetic")
# sns.lineplot(acf_wave_data, color = "royalblue", ax = ax2, label = "Observed")
# ax2.set_title("Significant Wave Height")
# ax2.set_xlabel("Lag [h]")
# ax2.legend(edgecolor = "black")
# ax2.minorticks_on()
# ax2.tick_params(direction = "in", which = "major", right = True, top = True)
# ax2.tick_params(direction = "in", which = "minor", right = True, top = True)
# plt.savefig("ACF_wind_wave.pdf", dpi = 600, bbox_inches = "tight")

# # =============================================================================
# # ###################################################
# # # Replace interpolated values with simulated values
# # ###################################################
# # 
# # interpolated_wind_indices = weather_data[weather_data["Interpolated_wind"] == True].index
# # interpolated_wave_indices = weather_data[weather_data["Interpolated_wave"] == True].index
# # 
# # for idx in interpolated_wind_indices:
# #     weather_data.at[idx, "Speed"] = sim_wind[idx]
# #     
# # for idx in interpolated_wave_indices:
# #     weather_data.at[idx, "Height"] = sim_wave[idx]
# #
# # weather_data_inputed = weather_data.iloc[:, range(1, 7)]
# # weather_data_inputed.to_csv("weather_hourly_inputed_04_17.csv")
# # 
# # =============================================================================

# ###################################################################
# # Plot original distribution and transformed (box-cox) distribution
# ###################################################################

# plt.rcParams.update({
#     "text.usetex": False,
#     "font.family": "sans-serif",   # specify font family here
#     "font.sans-serif": ["Arial"],  # specify font here
#     "mathtext.fontset": "cm",
#     "font.size":11,
#     "legend.fontsize": 9,
#     "axes.labelsize": 11,
#     "xtick.labelsize": 11,
#     "ytick.labelsize": 11,
#     "axes.linewidth": 0.8,
#     "patch.linewidth": 0.8})  

# #sns.set(style = "ticks", rc = {'figure.figsize': (6, 5)}, font_scale = 1)
# fig, ax1 = plt.subplots(figsize = (5,4))
# ax1 = sns.histplot(weather_data["Speed"], ax = ax1, color = "royalblue", edgecolor = "royalblue", bins = 80, stat = "density")
# ax1.set_xlabel("Wind speed [m/s]")
# ax1.set_ylabel("Density")
# ax1.set_title("FINO1 Wind Speed")
# plt.minorticks_on()
# plt.savefig("wind_speed_hist.pdf", dpi = 600)

# fig, ax1 = plt.subplots(figsize = (5,4))
# ax1 = sns.histplot(weather_data["Speed_standardized"], ax = ax1, color = "royalblue",  edgecolor = "royalblue", bins = 80, stat = "density")
# ax1.set_xlabel("Standardized Value")
# ax1.set_ylabel("Density")
# ax1.set_title("Standardized Wind Speed")
# # calculate the pdf
# x0, x1 = ax1.get_xlim()  # extract the endpoints for the x-axis
# x_pdf = np.linspace(x0, x1, 100)
# y_pdf = scipy.stats.norm.pdf(x_pdf) * np.std(weather_data["Speed_standardized"]) + np.mean(weather_data["Speed_standardized"])
# ax1.plot(x_pdf, y_pdf, 'r', lw=1.5, label='pdf')    
# plt.minorticks_on()                                               
# plt.savefig("wind_speed_hist_standardized.pdf", dpi = 600)

# fig, ax1 = plt.subplots(figsize = (5,4))
# ax1 = sns.histplot(weather_data["Height"], ax = ax1, color = "navy",  edgecolor = "navy", bins = 80, stat = "density")
# ax1.set_xlim([0, 10])
# ax1.set_xlabel("Significant Wave Height [m]")
# ax1.set_ylabel("Density")
# ax1.set_title("FINO1 Significant Wave Height")
# plt.minorticks_on()
# plt.savefig("wave_height_hist.pdf", dpi = 600)

# fig, ax1 = plt.subplots(figsize = (5,4))
# ax1 = sns.histplot(weather_data["Height_standardized"], ax = ax1, color = "navy",  edgecolor = "navy", bins = 80, stat = "density")
# ax1.set_xlabel("Standardized Value")
# ax1.set_ylabel("Density")
# ax1.set_title("Standardized Significant Wave Height")
# # calculate the pdf
# x0, x1 = ax1.get_xlim()  # extract the endpoints for the x-axis
# x_pdf = np.linspace(x0, x1, 100)
# y_pdf = scipy.stats.norm.pdf(x_pdf) * np.std(weather_data["Height_standardized"]) + np.mean(weather_data["Height_standardized"])
# ax1.plot(x_pdf, y_pdf, 'r', lw=1.5, label='pdf')   
# plt.minorticks_on()
# plt.savefig("wave_height_hist_standardized.pdf", dpi = 600)

# ########################################
# # distribution plot, showing correlation
# ########################################

# plt.rcParams.update({
#     "text.usetex": False,
#     "font.family": "sans-serif",   # specify font family here
#     "font.sans-serif": ["Arial"],  # specify font here
#     "mathtext.fontset": "cm",
#     "font.size":9,
#     "legend.fontsize": 9,
#     "axes.labelsize": 9,
#     "axes.titlesize": 10,
#     "xtick.labelsize": 9,
#     "ytick.labelsize": 9,
#     "axes.linewidth": 0.5,
#     "patch.linewidth": 0.5}) 

# fig, (ax1, ax2) = plt.subplots(ncols = 2, sharex= True, sharey=True, figsize = (6.5, 3.6))
# ax1 = sns.histplot(data = weather_data,
#             y = "Speed",
#             x = "Height",
#             color = "royalblue",
#             ax = ax1)
# ax1.annotate(r"Pearson's Correlation: 0.578",
#     xy=(0.25, 0.9), xycoords='axes fraction', fontsize = 9)
# ax2 = sns.histplot(data = sim_weather.sample(frac=0.05), 
#     y = "Speed", 
#     x = "Height",
#     color = "red",
#     ax = ax2)
# ax2.annotate(r"Pearson's Correlation: 0.523",
#     xy=(0.25, 0.9), xycoords='axes fraction', fontsize = 9)
# ax1.set_xlim([0, 12])
# ax1.set_ylim([0, 40])
# ax1.set_xlabel("Significant Wave Height [m]")
# ax2.set_xlabel("Significant Wave Height [m]")
# ax1.set_ylabel("Wind Speed [m/s]")
# ax1.set_title("Observed Data")
# ax2.set_title("Synthetic Data")
# plt.savefig("joint_distribution.png", dpi = 600)


# plt.rcParams.update({
#     "text.usetex": False,
#     "font.family": "sans-serif",   # specify font family here
#     "font.sans-serif": ["Arial"],  # specify font here
#     "mathtext.fontset": "cm",
#     "font.size":9,
#     "legend.fontsize": 9,
#     "axes.labelsize": 9,
#     "axes.titlesize": 10,
#     "xtick.labelsize": 9,
#     "ytick.labelsize": 9,
#     "axes.linewidth": 0.5,
#     "patch.linewidth": 0.5}) 

# fig, (ax1, ax2) = plt.subplots(ncols = 2, sharex= True, sharey=True, figsize = (6.5, 4))
# ax1 = sns.kdeplot(data = weather_data,
#             y = "Speed",
#             x = "Height",
#             color = "royalblue",
#             ax = ax1)
# ax1.annotate(r"Correlation: 0.578",
#             xy=(0.55, 0.9), xycoords='axes fraction', fontsize = 9)
# ax2 = sns.kdeplot(data = sim_weather, 
#                 y = "Speed", 
#                 x = "Height",
#                 color = "red",
#                 ax = ax2)
# ax2.annotate(r"Correlation: 0.523",
#             xy=(0.55, 0.9), xycoords='axes fraction', fontsize = 9)
# ax1.set_xlim([0, 12])
# ax1.set_ylim([0, 40])
# ax1.set_xlabel("Significant Wave Height [m]")
# ax2.set_xlabel("Significant Wave Height [m]")
# ax1.set_ylabel("Wind Speed [m/s]")
# ax1.set_title("Observed Data")
# ax2.set_title("Synthetic Data")
