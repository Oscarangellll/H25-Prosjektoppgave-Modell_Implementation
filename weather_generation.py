import os
import pandas as pd
import numpy as np
import scipy
from statsmodels.tsa.vector_ar.var_model import VAR

import conf

INPUT_FOLDER = "Weather Data"
OUTPUT_FOLDER = "Synthetic Weather Data"

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

    H, D, M, nS = conf.HOURS, conf.DAYS, conf.MONTHS, conf.SCENARIOS
    seed = conf.SEED
    num_to_simulate = H * D * M 

    all_scenarios = []

    # Simulate one year of syntethic data in isolation.
    for s in range(1, nS + 1):
        sim = results.simulate_var(steps=num_to_simulate, seed=seed)    

        # Build the simulated data which will be concatenated
        idx = pd.MultiIndex.from_product(
            [[s], range(1, M + 1), range(1, D + 1), range(0, H)],
                names=["Scenario", "Month", "Day", "Hour"]
            )
        
        sd = pd.DataFrame(index=idx).reset_index()
        sd = sd[["Scenario", "Hour", "Day", "Month"]] # Change order of columns
        sd["Speed_standardized"] = sim[:, 0]
        sd["Height_standardized"] = sim[:, 1]
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
        sd = sd[["Scenario", "Hour", "Day", "Month", "Speed", "Height"]]

        all_scenarios.append(sd)

    all_scenarios = pd.concat(all_scenarios, ignore_index=True)

    # Save to CSV
    all_scenarios.to_csv(
        os.path.join(OUTPUT_FOLDER, file), 
        index=False,
        float_format="%.4f"
    )
