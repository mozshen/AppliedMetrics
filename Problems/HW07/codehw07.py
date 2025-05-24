
#%%

import pandas as pd
import numpy as np
import cvxpy as cp
from itertools import product
import matplotlib.pyplot as plt
from tqdm import tqdm

#%%

# Load your data
d = pd.read_csv('Cleaned_Synthetic_Control_Data_edited.csv')

#%%

# Parameters
intervention_year = 1990
treated_unit = 'West Germany'
covariates = ['metric_inflation', 'trade', 'metric_industry_share']
outcome_var = 'indicator_score'
year_var = 'year_recorded'
unit_var = 'entity_id'
pre_treatment_years = list(range(1960, intervention_year))

#%%

# Clean and split data
treated_data = d[d[unit_var] == treated_unit]
donor_data = d[d[unit_var].isin(d[unit_var].unique().tolist())]
donor_data = donor_data[donor_data[unit_var] != treated_unit]

#%%

# Get Y1
Y1 = treated_data[treated_data[year_var].isin(pre_treatment_years)].sort_values(year_var)[outcome_var].values

# Get Y0
Y0, valid_donors = [], []
for country in donor_data[unit_var].unique():
    temp = donor_data[(donor_data[unit_var] == country) & (donor_data[year_var].isin(pre_treatment_years))]
    if len(temp) == len(pre_treatment_years):
        Y0.append(temp.sort_values(year_var)[outcome_var].values)
        valid_donors.append(country)

Y0 = np.column_stack(Y0)

#%%

# Covariate matrices
X1 = treated_data[treated_data[year_var] < intervention_year][covariates].mean().values
X0_df = donor_data[donor_data[year_var] < intervention_year].groupby(unit_var)[covariates].mean().dropna()
X0 = X0_df.T.values

#%%

# Grid search over V
v_grid = np.linspace(-5, 10, 10)
v_combinations = list(product(v_grid, repeat=len(covariates)))
results = []

for v_tuple in tqdm(v_combinations):
    
    V = np.diag(v_tuple)
    W = cp.Variable(Y0.shape[1])
    objective = cp.Minimize(cp.sum_squares(X1 @ V - (X0 @ W) @ V))
    constraints = [cp.sum(W) == 1, W >= 0]
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve()
        if W.value is not None:
            Y_synth = Y0 @ W.value
            mspe = np.mean((Y1 - Y_synth) ** 2)
            results.append((v_tuple, W.value, mspe))
    except:
        continue

# Best result
best_v, best_weights, best_mspe = min(results, key=lambda x: x[2])

#%%

# Plot
all_years = sorted(treated_data[year_var].unique())
Y0_full = []
for country in valid_donors:
    temp = donor_data[donor_data[unit_var] == country]
    if len(temp) == len(all_years):
        Y0_full.append(temp.sort_values(year_var)[outcome_var].values)
Y0_full = np.column_stack(Y0_full)
Y_synthetic = Y0_full @ best_weights
Y_actual = treated_data.sort_values(year_var)[outcome_var].values

plt.plot(all_years, Y_actual, label='West Germany')
plt.plot(all_years, Y_synthetic, label='Synthetic Control', linestyle='--')
plt.axvline(x=1990, color='grey', linestyle=':')
plt.xlabel("Year")
plt.ylabel("GDP per Capita")
plt.legend()
plt.title("Synthetic Control for West Germany (Optimal V)")
plt.show()

# Show weights
print("Best V:", best_v)
print("Best MSPE:", best_mspe)
for name, w in zip(valid_donors, best_weights):
    print(f"{name}: {w:.4f}")

#%%