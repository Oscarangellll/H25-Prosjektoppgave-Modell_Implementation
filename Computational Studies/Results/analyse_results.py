import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Les inn alle filene
df1 = pd.read_csv("Computational Studies/Results/runtime_analysis_results 2x1x1_10.csv")
df2 = pd.read_csv("Computational Studies/Results/runtime_analysis_results 3x2x1_7.csv")
df3 = pd.read_csv("Computational Studies/Results/runtime_analysis_results 3x2x8_10.csv")
df4 = pd.read_csv("Computational Studies/Results/runtime_analysis_results 3x3x1_7.csv")
df5 = pd.read_csv("Computational Studies/Results/runtime_analysis_results 3x3x8_10.csv")
df6 = pd.read_csv("Computational Studies/Results/runtime_analysis_results2 2x1x1_10.csv")
df7 = pd.read_csv("Computational Studies/Results/runtime_analysis_results2 3x2x1_7.csv")
df8 = pd.read_csv("Computational Studies/Results/runtime_analysis_results2 3x2x8_10.csv")
df9 = pd.read_csv("Computational Studies/Results/runtime_analysis_results2 3x3x1_7.csv")
df10 = pd.read_csv("Computational Studies/Results/runtime_analysis_results2 3x3x8_10.csv")

# Slå sammen til én DataFrame
df_combined = pd.concat(
    [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10],
    ignore_index=True
)

# Lagre kombinert fil (om ønskelig)
df_combined.to_csv("runtime_analysis_combined.csv", index=False)

# Gruppér etter kombinasjon av (n_vessels, n_wind_farms, n_scenarios)
grouped = df_combined.groupby(['n_vessels', 'n_wind_farms', 'n_scenarios'])

# Gjennomsnittlig objval og solution_gap
avg_results = grouped.agg(
    objval=('objval', 'mean'),
    solution_gap=('solution_gap', 'mean')
).reset_index()

# 95 % kvantiler (2.5 % og 97.5 %) for både objval og solution_gap
percentiles = grouped.agg(
    objval_lower=('objval', lambda x: x.quantile(0.025)),
    objval_upper=('objval', lambda x: x.quantile(0.975)),
    gap_lower=('solution_gap', lambda x: x.quantile(0.025)),
    gap_upper=('solution_gap', lambda x: x.quantile(0.975))
).reset_index()

# === Coefficient of Variation (CV) for objval ===
# CV = std / mean (her bruker vi absoluttverdi i nevner for sikkerhets skyld)
cv_results = grouped.agg(
    objval_mean=('objval', 'mean'),
    objval_std=('objval', 'std')
).reset_index()

cv_results['cv_objval'] = np.where(
    cv_results['objval_mean'] != 0,
    cv_results['objval_std'] / cv_results['objval_mean'].abs(),
    np.nan  # unngå deling på 0
)

# === Plot 1: Average Objective Value med 95 % kvantilbånd ===
plt.figure(figsize=(10, 6))

for key, grp in avg_results.groupby(['n_vessels', 'n_wind_farms']):
    n_vessels, n_wind_farms = key

    # Finn samme kombinasjon i percentiles
    mask = (
        (percentiles['n_vessels'] == n_vessels) &
        (percentiles['n_wind_farms'] == n_wind_farms)
    )
    perc_grp = percentiles[mask]

    # Slå sammen på n_scenarios for å garantere korrekt rekkefølge
    tmp = grp.merge(
        perc_grp,
        on=['n_vessels', 'n_wind_farms', 'n_scenarios'],
        how='inner'
    ).sort_values('n_scenarios')

    # Gjennomsnittslinje
    plt.plot(
        tmp['n_scenarios'],
        tmp['objval'],
        label=f'Vessels: {n_vessels}, Wind Farms: {n_wind_farms}'
    )

    # Shaded område mellom nedre og øvre kvantil
    plt.fill_between(
        tmp['n_scenarios'],
        tmp['objval_lower'],
        tmp['objval_upper'],
        alpha=0.2
    )

plt.xlabel('Number of Scenarios')
plt.ylabel('Average Objective Value')
plt.title('Average Objective Value vs Number of Scenarios')
plt.legend()
plt.show()

# === Plot 2: Average Solution Gap med 95 % kvantilbånd ===
plt.figure(figsize=(10, 6))

for key, grp in avg_results.groupby(['n_vessels', 'n_wind_farms']):
    n_vessels, n_wind_farms = key

    mask = (
        (percentiles['n_vessels'] == n_vessels) &
        (percentiles['n_wind_farms'] == n_wind_farms)
    )
    perc_grp = percentiles[mask]

    tmp = grp.merge(
        perc_grp,
        on=['n_vessels', 'n_wind_farms', 'n_scenarios'],
        how='inner'
    ).sort_values('n_scenarios')

    plt.plot(
        tmp['n_scenarios'],
        tmp['solution_gap'],
        label=f'Vessels: {n_vessels}, Wind Farms: {n_wind_farms}'
    )

    plt.fill_between(
        tmp['n_scenarios'],
        tmp['gap_lower'],
        tmp['gap_upper'],
        alpha=0.2
    )

plt.xlabel('Number of Scenarios')
plt.ylabel('Average Solution Gap')
plt.title('Average Solution Gap vs Number of Scenarios')
plt.legend()
plt.show()

# === Plot 3: Runtime vs Number of Scenarios (som i original kode) ===
plt.figure(figsize=(10, 6))

for n_wind_farms in df_combined['n_wind_farms'].unique():
    subset = df_combined[df_combined['n_wind_farms'] == n_wind_farms]
    grouped_runtime = subset.groupby(['n_scenarios']).agg(
        runtime=('runtime', 'mean')
    ).reset_index()

    plt.plot(
        grouped_runtime['n_scenarios'],
        grouped_runtime['runtime'],
        marker='o',
        label=f'Wind Farms: {n_wind_farms}'
    )

plt.xlabel('Number of Scenarios')
plt.ylabel('Average Runtime')
plt.title('Average Runtime vs Number of Scenarios')
plt.legend()
plt.show()

# === Plot 4: Coefficient of Variation for Objective Value ===
plt.figure(figsize=(10, 6))

for key, grp in cv_results.groupby(['n_vessels', 'n_wind_farms']):
    n_vessels, n_wind_farms = key
    grp_sorted = grp.sort_values('n_scenarios')

    plt.plot(
        grp_sorted['n_scenarios'],
        grp_sorted['cv_objval'],
        marker='o',
        label=f'Vessels: {n_vessels}, Wind Farms: {n_wind_farms}'
    )

plt.xlabel('Number of Scenarios')
plt.ylabel('Coefficient of Variation (Objective Value)')
plt.title('Coefficient of Variation of Objective Value vs Number of Scenarios')
plt.legend()
plt.show()
