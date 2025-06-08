
#%%

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from linearmodels.iv import IV2SLS
import numpy as np
import statsmodels.formula.api as smf
import seaborn as sns

#%%

d= pd.read_stata('maimonides.dta')
d= d.dropna()

#%%

# 12. sharp or fuzzy
a= d.groupby(['enrollment'], as_index= False).agg({'classize': 'mean'})


# Create figure
plt.figure(figsize=(10, 6))
plt.plot(a['enrollment'], a['classize'], color='steelblue', linewidth=2, marker='o', markersize=4)

# Add Maimonides rule lines (cutoffs at 40, 80, 120, ...)
for threshold in range(40, a['enrollment'].max(), 40):
    plt.axvline(x=threshold, color='red', linestyle='--', linewidth=1)
    plt.text(threshold + 1, a['classize'].min(), f'{threshold}', color='red', fontsize=8)

# Titles and labels
plt.title('Average Class Size by Enrollment (Grouped)', fontsize=14, fontweight='bold')
plt.xlabel('Enrollment', fontsize=12)
plt.ylabel('Average Class Size', fontsize=12)

# Show plot
plt.tight_layout()
plt.show()


#%% 11. Generate descriptive statistics

# Select relevant variables
vars_to_describe = ['classize', 'avgmath', 'avgverb', 'enrollment', 'perc_disadvantaged']

# Compute descriptive statistics
desc_stats = d[vars_to_describe].describe().T

# Rename and format
desc_stats = desc_stats[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
desc_stats.columns = ['Mean', 'Std. Dev.', 'Min', '25%', 'Median', '75%', 'Max']
desc_stats = desc_stats.round(2)

desc_stats.to_excel('desc_stats.xlsx')

#%%
# 12. replicating table 2
# Define variables
y = d['avgmath']

# Column (4): only class size
X4 = sm.add_constant(d[['classize']])
model4 = sm.OLS(y, X4).fit()

# Column (5): add enrollment
X5 = sm.add_constant(d[['classize', 'enrollment']])
model5 = sm.OLS(y, X5).fit()

# Column (6): add enrollment and disadvantaged
X6 = sm.add_constant(d[['classize', 'enrollment', 'perc_disadvantaged']])
model6 = sm.OLS(y, X6).fit()

# Collect results
results_df = pd.DataFrame({
    'Column 4': model4.params,
    'Column 4 SE': model4.bse,
    'Column 5': model5.params,
    'Column 5 SE': model5.bse,
    'Column 6': model6.params,
    'Column 6 SE': model6.bse
}).round(3)


results_df.to_excel('table2rep.xlsx')

#%%

# 13. a. sharp RD
# Step 1: Subset to enrollment between 20 and 60
df_sub = d[(d['enrollment'] >= 20) & (d['enrollment'] <= 60)].copy()

# Step 2: Create treatment dummy for crossing 41-student threshold
df_sub['large_class'] = (df_sub['enrollment'] >= 41).astype(int)

X_a = df_sub[['large_class', 'enrollment', 'perc_disadvantaged']]
X_a = sm.add_constant(X_a)
y_a = df_sub['avgmath']

model_a = sm.OLS(y_a, X_a).fit()
results_a = model_a.summary()

#%%

# 13. b. IV for column 6
# Step 1: Calculate predicted class size
d['predicted_class_size'] = d['enrollment'] / (np.floor(d['enrollment'] / 40).replace(0, np.nan))

# Step 2: Drop any missing predictions
df_iv = d.dropna(subset=['predicted_class_size', 'classize', 'avgmath', 'perc_disadvantaged'])

# Step 3: Define variables
iv_formula = 'avgmath ~ 1 + enrollment + perc_disadvantaged + [classize ~ predicted_class_size]'

# Step 4: Run IV regression
iv_model = IV2SLS.from_formula(iv_formula, data=df_iv).fit()
results_b = iv_model.summary

#%%

# Save sharp RDD coefficients
model_a_df = model_a.summary2().tables[1]  # Table with coefficients and stats
model_a_df.to_excel("sharp_rdd_coefficients.xlsx")

# Save IV regression coefficients
iv_df = pd.DataFrame(iv_model.summary.tables[1])  # This is a pandas DataFrame in linearmodels
iv_df.to_excel("iv_rdd_coefficients.xlsx")

#%%

# 15. a

# Subset sample: restrict to enrollment between 20 and 60
df_fuzzy = d[(d['enrollment'] >= 20) & (d['enrollment'] <= 60)].copy()

# Instrument: first threshold dummy
df_fuzzy['above_41'] = (df_fuzzy['enrollment'] >= 41).astype(int)

# Define IV regression formula
formula_a = 'avgmath ~ 1 + enrollment + perc_disadvantaged + [classize ~ above_41]'

# Estimate 2SLS
fuzzy_model_a = IV2SLS.from_formula(formula_a, data=df_fuzzy).fit()

# Save result
with open("fuzzy_rdd_single_threshold.txt", "w") as f:
    f.write(str(fuzzy_model_a.summary))

#%%

# 15. b

# Prepare full sample and compute predicted class size
d['predicted_classize'] = d['enrollment'] / np.floor(d['enrollment'] / 40)
d = d.replace([np.inf], np.nan)
df_all = d.dropna(subset=['avgmath', 'classize', 'predicted_classize', 'enrollment', 'perc_disadvantaged']).copy()

# IV formula using predicted class size as instrument
formula_b = 'avgmath ~ 1 + enrollment + perc_disadvantaged + [classize ~ predicted_classize]'

# Estimate 2SLS
fuzzy_model_b = IV2SLS.from_formula(formula_b, data=df_all).fit()

# Save result
with open("fuzzy_rdd_all_thresholds.txt", "w") as f:
    f.write(str(fuzzy_model_b.summary))

#%%

# 17. a. Manupulation Test
# Plot histogram of enrollment
plt.figure(figsize=(10, 6))
sns.histplot(d['enrollment'], bins=105, kde=False, color='gray', edgecolor='black')

# Add threshold line at 41
plt.axvline(x=41, color='red', linestyle='--', label='Threshold: 41')

plt.title('Distribution of School Enrollment (Forcing Variable)', fontsize=14)
plt.xlabel('Enrollment')
plt.ylabel('Number of Schools')
plt.legend()
plt.tight_layout()
plt.show()

#%%

# 17b. binscatter plot
# Subset data around the cutoff for clarity
df_plot = d[(d['enrollment'] >= 20) & (d['enrollment'] <= 60)].copy()

# Bin enrollment into 2-student buckets
df_plot['enroll_bin'] = pd.cut(df_plot['enrollment'], bins=range(20, 61, 2))
bin_means = df_plot.groupby('enroll_bin').agg({
    'enrollment': 'mean',
    'avgmath': 'mean'
}).dropna()

# Plot binscatter
plt.figure(figsize=(10, 6))
plt.scatter(bin_means['enrollment'], bin_means['avgmath'], color='steelblue', label='Binned Averages')
plt.axvline(x=41, color='red', linestyle='--', label='Threshold: 41')

plt.title('Math Scores vs. Enrollment (Binned Averages)', fontsize=14)
plt.xlabel('Enrollment')
plt.ylabel('Average Math Score')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%

# 17c. binscatter with fits

# Subset to window around the cutoff
df_local = d[(d['enrollment'] >= 20) & (d['enrollment'] <= 60)].copy()

# Create running variable relative to cutoff
df_local['enroll_centered'] = df_local['enrollment'] - 41
df_local['above_41'] = (df_local['enroll_centered'] >= 0).astype(int)

# Separate sides of the cutoff
left = df_local[df_local['enroll_centered'] < 0]
right = df_local[df_local['enroll_centered'] >= 0]

# Create polynomial terms
for df_ in [left, right]:
    df_['poly1'] = df_['enroll_centered']
    df_['poly2'] = df_['enroll_centered'] ** 2

# Fit linear and quadratic regressions on both sides
lin_left = sm.OLS(left['avgmath'], sm.add_constant(left['poly1'])).fit()
lin_right = sm.OLS(right['avgmath'], sm.add_constant(right['poly1'])).fit()

quad_left = sm.OLS(left['avgmath'], sm.add_constant(left[['poly1', 'poly2']])).fit()
quad_right = sm.OLS(right['avgmath'], sm.add_constant(right[['poly1', 'poly2']])).fit()

# Generate x-axis for plots
x_vals = np.linspace(-21, 19, 100)
x_poly = pd.DataFrame({'poly1': x_vals, 'poly2': x_vals**2})
x_vals_full = x_vals + 41  # to revert to actual enrollment scale


# For linear prediction: use only 'poly1'
x_poly_linear = pd.DataFrame({'poly1': x_vals})

# Predict linear and quadratic fits
lin_pred_left = lin_left.predict(sm.add_constant(x_poly_linear[x_vals < 0]))
lin_pred_right = lin_right.predict(sm.add_constant(x_poly_linear[x_vals >= 0]))

quad_pred_left = quad_left.predict(sm.add_constant(x_poly[x_vals < 0]))
quad_pred_right = quad_right.predict(sm.add_constant(x_poly[x_vals >= 0]))

# Plot binscatter
df_local['enroll_bin'] = pd.cut(df_local['enrollment'], bins=range(20, 61, 2))
bin_means = df_local.groupby('enroll_bin').agg({
    'enrollment': 'mean',
    'avgmath': 'mean'
}).dropna()

plt.figure(figsize=(10, 6))
plt.scatter(bin_means['enrollment'], bin_means['avgmath'], color='steelblue', label='Binned Averages')

# Plot fitted lines
plt.plot(x_vals_full[x_vals < 0], lin_pred_left, color='green', linestyle='--', label='Linear Fit (Left)')
plt.plot(x_vals_full[x_vals >= 0], lin_pred_right, color='green', linestyle='--', label='Linear Fit (Right)')

plt.plot(x_vals_full[x_vals < 0], quad_pred_left, color='darkorange', label='Quadratic Fit (Left)')
plt.plot(x_vals_full[x_vals >= 0], quad_pred_right, color='darkorange', label='Quadratic Fit (Right)')

# Add vertical cutoff line
plt.axvline(41, color='red', linestyle='-', label='Cutoff at 41')

# Labels
plt.title('Math Scores vs Enrollment: Binned with Linear & Quadratic Fits', fontsize=14)
plt.xlabel('Enrollment')
plt.ylabel('Average Math Score')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%


# Define grid
bandwidths = [5, 10, 15, 20]
orders = [1, 2]

# Store results
sensitivity_results = []

for bw in bandwidths:
    df_bw = d[(d['enrollment'] >= 41 - bw) & (d['enrollment'] <= 41 + bw)].copy()
    df_bw['treatment'] = (df_bw['enrollment'] >= 41).astype(int)
    df_bw['running'] = df_bw['enrollment'] - 41  # center around cutoff

    for order in orders:
        # Add polynomial terms
        df_bw['running_sq'] = df_bw['running'] ** 2 if order == 2 else 0

        # Specify formula
        if order == 1:
            formula = "avgmath ~ treatment + running + perc_disadvantaged"
        else:
            formula = "avgmath ~ treatment + running + running_sq + perc_disadvantaged"

        # Fit model
        model = smf.ols(formula=formula, data=df_bw).fit()
        coef = model.params['treatment']
        se = model.bse['treatment']

        # Save result
        sensitivity_results.append({
            'Bandwidth': bw,
            'Order': f'Poly {order}',
            'Estimate': round(coef, 3),
            'Std. Error': round(se, 3),
            '95% CI Lower': round(coef - 1.96 * se, 3),
            '95% CI Upper': round(coef + 1.96 * se, 3)
        })

# Convert to DataFrame
sensitivity_df = pd.DataFrame(sensitivity_results)
sensitivity_df.to_excel("sensitivity.xlsx")

#%%

# 17. placebo

# Subset to a symmetric window around the cutoff (e.g., ±20)
df_placebo = d[(d['enrollment'] >= 21) & (d['enrollment'] <= 61)].copy()

# Define treatment indicator
df_placebo['treatment'] = (df_placebo['enrollment'] >= 41).astype(int)
df_placebo['running'] = df_placebo['enrollment'] - 41

# RD specification with linear trend
placebo_model = smf.ols(
    formula='perc_disadvantaged ~ treatment + running',
    data=df_placebo
).fit()

# Show result
print(placebo_model.summary())

with open("placebo_rd_disadvantaged.txt", "w") as f:
    f.write(placebo_model.summary().as_text())

#%%


# Bin for scatter plot
df_placebo['bin'] = pd.cut(df_placebo['enrollment'], bins=range(20, 62, 2))
bin_means = df_placebo.groupby('bin').agg({
    'enrollment': 'mean',
    'perc_disadvantaged': 'mean'
}).dropna()

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(bin_means['enrollment'], bin_means['perc_disadvantaged'], color='gray')
plt.axvline(41, color='red', linestyle='--', label='Cutoff at 41')
plt.title('Placebo RD: Proportion Disadvantaged vs. Enrollment')
plt.xlabel('Enrollment')
plt.ylabel('Proportion Disadvantaged')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#%%











