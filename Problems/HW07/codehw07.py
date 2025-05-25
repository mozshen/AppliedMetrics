
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
        
    else:
        print(country)
        
Y0 = np.column_stack(Y0)

#%%

# Covariate matrices
X1 = treated_data[treated_data[year_var] < intervention_year][covariates].mean().values
X0_df = donor_data[donor_data[year_var] < intervention_year].groupby(unit_var)[covariates].mean().dropna()
X0 = X0_df.T.values

#%%

# Grid search over V
v_grid = np.linspace(10, 100, 10)
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

#%%

# Years
all_years = np.array(all_years)
post_treatment_years = all_years[all_years >= intervention_year]

# Original gap over all years
gap_all = Y_actual - Y_synthetic

# Post-treatment gaps
gap_post = gap_all[all_years >= intervention_year]

# Cumulative post-treatment gap
cum_gap_post = np.cumsum(gap_post)

# Plot all
plt.figure(figsize=(12, 8))

plt.subplot(3,1,1)
plt.plot(all_years, gap_all, label='Original Gap (Treated - Synthetic)')
plt.axvline(x=intervention_year, color='grey', linestyle='--')
plt.title('Original Gap Over All Years')
plt.xlabel('Year')
plt.ylabel('Gap')
plt.legend()

plt.subplot(3,1,2)
plt.plot(post_treatment_years, gap_post, label='Pointwise Post-treatment Gap', color='orange')
plt.axhline(0, color='black', linewidth=0.8)
plt.title('Pointwise Gap (Post-treatment)')
plt.xlabel('Year')
plt.ylabel('Gap')
plt.legend()

plt.subplot(3,1,3)
plt.plot(post_treatment_years, cum_gap_post, label='Cumulative Post-treatment Gap', color='green')
plt.axhline(0, color='black', linewidth=0.8)
plt.title('Cumulative Gap (Post-treatment)')
plt.xlabel('Year')
plt.ylabel('Cumulative Gap')
plt.legend()

plt.tight_layout()
plt.show()

#%%

# Compute synthetic covariate averages
synthetic_covariates = X0_df.T @ best_weights

cov_balance_df = pd.DataFrame({
    'Covariate': covariates,
    'Treated': X1,
    'Synthetic Control': synthetic_covariates,
    'Absolute Difference': np.abs(X1 - synthetic_covariates)
})

print(cov_balance_df)

cov_balance_df.to_excel('cov_balance_df.xlsx', index= False)

#%%

weights= pd.DataFrame(list(zip(valid_donors, best_weights)))
weights.columns= ['Country', 'Weight']
weights= weights.sort_values(['Weight'], ascending= False)    
weights.to_excel('weights.xlsx', index= False)
    
#%%

def run_scm(intervention_year):
    # Pre-treatment years for this placebo test
    pre_years = list(range(1960, intervention_year))
    
    # Filter Y1 and Y0 similarly but relative to placebo year
    Y1 = treated_data[treated_data[year_var].isin(pre_years)].sort_values(year_var)[outcome_var].values

    Y0, valid_donors = [], []
    for country in donor_data[unit_var].unique():
        temp = donor_data[(donor_data[unit_var] == country) & (donor_data[year_var].isin(pre_years))]
        if len(temp) == len(pre_years):
            Y0.append(temp.sort_values(year_var)[outcome_var].values)
        else:
            continue
    Y0 = np.column_stack(Y0)

    # Covariates averaged pre-treatment
    X1 = treated_data[treated_data[year_var] < intervention_year][covariates].mean().values
    X0_df = donor_data[donor_data[year_var] < intervention_year].groupby(unit_var)[covariates].mean().dropna()
    X0 = X0_df.T.values

    # Use the best V from original or grid search here or just identity
    # For simplicity, just use identity matrix:
    V = np.eye(len(covariates))

    # Solve optimization to find weights W
    W = cp.Variable(Y0.shape[1])
    objective = cp.Minimize(cp.sum_squares(X1 @ V - (X0 @ W) @ V))
    constraints = [cp.sum(W) == 1, W >= 0]
    prob = cp.Problem(objective, constraints)
    prob.solve()

    weights = W.value
    # Now get full donor data for all years available
    all_years = sorted(treated_data[year_var].unique())
    Y0_full = []
    for country in donor_data[unit_var].unique():
        temp = donor_data[donor_data[unit_var] == country]
        if len(temp) == len(all_years):
            Y0_full.append(temp.sort_values(year_var)[outcome_var].values)
    Y0_full = np.column_stack(Y0_full)

    # Synthetic control outcome
    Y_synthetic = Y0_full @ weights
    Y_actual = treated_data.sort_values(year_var)[outcome_var].values

    # Compute gap post placebo year
    gap = Y_actual - Y_synthetic

    return all_years, gap, intervention_year

#%%

# Run true SCM with intervention_year = 1990
true_years, true_gap, true_treat = run_scm(1990)

# Run placebo SCM with intervention_year = 1982
placebo_years, placebo_gap, placebo_treat = run_scm(1982)

# Plot both gaps

plt.figure(figsize=(10,6))
plt.plot(true_years, true_gap, label='True Treatment Gap (post-1990)', color='blue')
plt.plot(placebo_years, placebo_gap, label='Placebo Gap (post-1982)', color='red', linestyle='--')
plt.axvline(x=true_treat, color='blue', linestyle=':', label='True Treatment Year')
plt.axvline(x=placebo_treat, color='red', linestyle=':', label='Placebo Treatment Year')
plt.xlabel('Year')
plt.ylabel('Gap (Treated - Synthetic)')
plt.title('In-time Placebo Test: True vs Placebo Treatment Gaps')
plt.legend()
plt.show()

#%%


def compute_rmspe(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def run_scm_for_unit(treated, donors, intervention_year, covariates, outcome_var, year_var, unit_var, data):
    pre_years = list(range(1960, intervention_year))

    # Treated data
    treated_data = data[data[unit_var] == treated]
    Y1 = treated_data[treated_data[year_var].isin(pre_years)].sort_values(year_var)[outcome_var].values

    # Donor data (exclude treated from donors)
    donor_pool = [d for d in donors if d != treated]

    Y0_list = []
    valid_donors = []
    for donor in donor_pool:
        temp = data[(data[unit_var] == donor) & (data[year_var].isin(pre_years))]
        if len(temp) == len(pre_years):
            Y0_list.append(temp.sort_values(year_var)[outcome_var].values)
            valid_donors.append(donor)
    if len(Y0_list) == 0:
        return None  # no valid donors for this unit
    Y0 = np.column_stack(Y0_list)

    # Covariates for treated and donors
    X1 = treated_data[treated_data[year_var] < intervention_year][covariates].mean().values
    X0_df = data[(data[year_var] < intervention_year) & (data[unit_var].isin(valid_donors))].groupby(unit_var)[covariates].mean()
    X0 = X0_df.T.values

    # Weighting matrix V = identity for simplicity
    V = np.eye(len(covariates))

    # Optimization
    W = cp.Variable(Y0.shape[1])
    objective = cp.Minimize(cp.sum_squares(X1 @ V - (X0 @ W) @ V))
    constraints = [cp.sum(W) == 1, W >= 0]
    prob = cp.Problem(objective, constraints)
    prob.solve()

    if W.value is None:
        return None

    weights = W.value

    # Full outcome for all years
    all_years = sorted(treated_data[year_var].unique())
    Y0_full_list = []
    for donor in valid_donors:
        temp = data[data[unit_var] == donor]
        if len(temp) == len(all_years):
            Y0_full_list.append(temp.sort_values(year_var)[outcome_var].values)
    if len(Y0_full_list) == 0:
        return None
    Y0_full = np.column_stack(Y0_full_list)

    Y_synth = Y0_full @ weights
    Y_actual = treated_data.sort_values(year_var)[outcome_var].values

    pre_treated_idx = [i for i, y in enumerate(all_years) if y < intervention_year]
    pre_RMSPE = compute_rmspe(Y_actual[pre_treated_idx], Y_synth[pre_treated_idx])

    gap = Y_actual - Y_synth

    return {
        'unit': treated,
        'all_years': all_years,
        'gap': gap,
        'pre_RMSPE': pre_RMSPE
    }

#%%

# Get list of donors (all countries except West Germany)
all_units = d[unit_var].unique().tolist()
treated_unit = 'West Germany'
donor_units = [c for c in all_units if c != treated_unit]

# Run SCM for West Germany (treated)
west_germany_res = run_scm_for_unit(treated_unit, donor_units, 1990, covariates, outcome_var, year_var, unit_var, d)
if west_germany_res is None:
    raise Exception("Failed to run SCM for West Germany")

# Run SCM for all donors as placebo treated
placebo_results = []
for donor in tqdm(donor_units):
    res = run_scm_for_unit(donor, [c for c in donor_units if c != donor] + [treated_unit], 1990, covariates, outcome_var, year_var, unit_var, d)
    if res is not None:
        placebo_results.append(res)

# Filter placebos by pre_RMSPE
threshold = 20 * west_germany_res['pre_RMSPE']
filtered_placebos = [r for r in placebo_results if r['pre_RMSPE'] <= threshold]

# Plot
plt.figure(figsize=(12,8))

# Plot placebo gaps
for r in filtered_placebos:
    plt.plot(r['all_years'], r['gap'], color='gray', alpha=0.4)

# Plot West Germany gap in bold
plt.plot(west_germany_res['all_years'], west_germany_res['gap'], color='red', linewidth=2.5, label='West Germany')

plt.axvline(x=1990, color='black', linestyle='--', label='Treatment Year')

plt.xlabel('Year')
plt.ylabel('Gap (Treated - Synthetic)')
plt.title('In-Space Placebo Test: Gaps for West Germany and Placebo Units')
plt.legend()
plt.show()

#%%

# Calculate post-treatment RMSPE and ratio for West Germany
all_years = np.array(west_germany_res['all_years'])
post_idx = np.where(all_years >= 1990)[0]
pre_idx = np.where(all_years < 1990)[0]

def rmspe_for_gap(gap, idx):
    return np.sqrt(np.mean(gap[idx] ** 2))

wg_post_rmspe = rmspe_for_gap(west_germany_res['gap'], post_idx)
wg_pre_rmspe = rmspe_for_gap(west_germany_res['gap'], pre_idx)
wg_ratio = wg_post_rmspe / wg_pre_rmspe

# Collect ratios for placebos
ratios = []
units = []
for res in placebo_results:
    gap = res['gap']
    post_rmspe = rmspe_for_gap(gap, post_idx)
    pre_rmspe = rmspe_for_gap(gap, pre_idx)
    # Avoid division by zero
    if pre_rmspe == 0:
        ratio = np.inf
    else:
        ratio = post_rmspe / pre_rmspe
    ratios.append(ratio)
    units.append(res['unit'])

# Create DataFrame for plotting
df = pd.DataFrame({'unit': units, 'ratio': ratios})

# Include West Germany
df = pd.concat([df, pd.DataFrame({'unit': ['West Germany'], 'ratio': [wg_ratio]})], ignore_index=True)

# Sort by ratio descending
df_sorted = df.sort_values('ratio', ascending=False).reset_index(drop=True)

# Empirical p-value: fraction of placebo units with ratio > West Germany's ratio
placebo_higher = sum(df['ratio'][df['unit'] != 'West Germany'] > wg_ratio)
p_value = placebo_higher / len(df[df['unit'] != 'West Germany'])

# Plot
plt.figure(figsize=(12,6))
bars = plt.bar(df_sorted.index, df_sorted['ratio'], color='gray')
# Highlight West Germany bar
wg_index = df_sorted.index[df_sorted['unit'] == 'West Germany'][0]
bars[wg_index].set_color('red')

plt.xticks(df_sorted.index, df_sorted['unit'], rotation=90)
plt.ylabel('Post / Pre RMSPE Ratio')
plt.title('Post-treatment / Pre-treatment RMSPE Ratios (West Germany Highlighted)')

# Annotate p-value
plt.text(0.95, 0.95, f'Empirical p-value = {p_value:.3f}',
         horizontalalignment='right',
         verticalalignment='top',
         transform=plt.gca().transAxes,
         fontsize=12,
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

plt.tight_layout()
plt.show()

print(f"West Germany post/pre RMSPE ratio: {wg_ratio:.3f}")
print(f"Empirical p-value: {p_value:.3f} ({placebo_higher} out of {len(df) - 1} placebos had higher ratio)")

#%%

