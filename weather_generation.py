import os
import pandas as pd
import numpy as np
import scipy
from statsmodels.tsa.vector_ar.var_model import VAR

import config

INPUT_FOLDER = config.WEATHER_DATA_FOLDER
OUTPUT_FOLDER = config.SYN_WEATHER_DATA_FOLDER 

# Generate data for each location
files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".csv")]
for file in files:
    # Read real weather data
    wd = pd.read_csv(os.path.join(INPUT_FOLDER, file) , parse_dates=["Time"])

    # Ensure that data contains only positive nonzero values.
    # Requirement for the box-cox transformation.
    wd.loc[wd["Speed"] <= 0, "Speed"] = np.nan
    wd.loc[wd["Height"] <= 0, "Height"] = np.nan

    wd["Speed"] = wd["Speed"].interpolate(method='linear').bfill().ffill()
    wd["Height"] = wd["Height"].interpolate(method='linear').bfill().ffill()

    # Perform box-cox transormation
    wd["Speed_boxcox"], speed_lambda = scipy.stats.boxcox(wd["Speed"])
    wd["Height_boxcox"], height_lambda = scipy.stats.boxcox(wd["Height"])

    # Monthly statistics
    wd["Month"] = wd["Time"].dt.month

    monthly_stats = wd.groupby(wd["Month"])[['Speed_boxcox', 'Height_boxcox']].agg(['mean', 'std'])
    monthly_stats.columns = [f'{col[0]}_{col[1]}' for col in monthly_stats.columns]

    wd = wd.merge(monthly_stats, on='Month', how='left')
    
    # Standardize data
    wd["Speed_standardized"] = (wd["Speed_boxcox"] - wd["Speed_boxcox_mean"])/wd["Speed_boxcox_std"]
    wd["Height_standardized"] = (wd["Height_boxcox"] - wd["Height_boxcox_mean"])/wd["Height_boxcox_std"]

    ###############
    # Fit VAR model
    ###############

    model = VAR(wd[["Speed_standardized", "Height_standardized"]])
    results = model.fit(maxlags=20, ic='bic')

    ##############
    # Simulate VAR
    ##############

    H, D, M, S = config.HOURS, config.DAYS, len(config.MONTHS), config.SCENARIOS 
    seed = config.SEED
    num_to_simulate = H * D * M * S

    # Simulate one year of syntethic data in isolation.

    sim = results.simulate_var(steps=num_to_simulate, seed=seed)    
    sim_speed = sim[:, 0]
    sim_height = sim[:, 1]

    # Build the simulated data which will be concatenated
    idx = pd.MultiIndex.from_product(
        [range(1, S +1), range(1, M + 1), range(1, D + 1), range(0, H)],
            names=["Scenario", "Month", "Day", "Hour"]
    )
        
    sd = pd.DataFrame(index=idx).reset_index()
    sd["Speed_standardized"] = sim_speed
    sd["Height_standardized"] = sim_height
    sd["DayID"] = D * sd["Month"] + sd["Day"] - D
    sd = sd.merge(monthly_stats, on='Month', how='left')
        
    # De-standardize data
    sd["Speed_boxcox"] = sd["Speed_standardized"] * sd["Speed_boxcox_std"] + sd["Speed_boxcox_mean"]
    sd["Height_boxcox"] = sd["Height_standardized"] * sd["Height_boxcox_std"] + sd["Height_boxcox_mean"]

    # Inverse box-cox transformation
    sd["Speed"] = scipy.special.inv_boxcox(sd["Speed_boxcox"], speed_lambda)
    sd["Height"] = scipy.special.inv_boxcox(sd["Height_boxcox"], height_lambda)

    # Transformation can result in negative values, fix these
    sd["Speed"] = sd["Speed"].interpolate(method='linear').bfill().ffill().clip(lower=0)
    sd["Height"] = sd["Height"].interpolate(method='linear').bfill().ffill().clip(lower=0)

    # Only keep columns of interest
    sd = sd[["Scenario", "Hour", "DayID", "Month", "Speed", "Height"]]
    sd = sd.rename(columns={"DayID": "Day"})

    
    # Save to CSV
    sd.to_csv(
        os.path.join(OUTPUT_FOLDER, file), 
        index=False,
        float_format="%.4f"
    )
