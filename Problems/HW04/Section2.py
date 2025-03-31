


#%%

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import numpy as np
from tqdm import tqdm
import statsmodels.api as sm

#%%

d= pd.read_csv('Employee_Performance.csv')

#%%

# a. treatment distribution
# Count the number of treated (1) and untreated (0)
sns.countplot(x=d['treatment'], palette=['skyblue', 'salmon'])

# Labels and title
plt.xlabel("Treatment Group")
plt.ylabel("Count")
plt.xticks(ticks=[0, 1], labels=["Control (0)", "Treated (1)"])
plt.title("Distribution of Treatment Assignment")

# Show the plot
plt.show()

#%%

# a. logistic regression
covariates = ['age', 'years_at_company', 'projects_handled', 'remote_work_frequency']
X = d[covariates]  # Covariates
y = d['treatment']  # Binary treatment assignment

# Fit logistic regression model
logit = LogisticRegression()
logit.fit(X, y)

# Compute propensity scores
d['propensity_score'] = logit.predict_proba(X)[:, 1]

# Summary statistics of propensity scores
a= (d[['treatment', 'propensity_score']].groupby('treatment').describe())

#%%

#  plotting summary
fig, ax = plt.subplots(figsize=(10, 0.5))  # Adjust figure size
ax.axis('tight')
ax.axis('off')
ax.table(cellText=a.round(3).values, 
         colLabels=a.columns, 
         rowLabels=a.index, 
         cellLoc='center', 
         loc='center')

# plt.savefig("summary_table.png", dpi=300, bbox_inches='tight')
plt.show()

#%%

# a. distribution chart
# Plot distribution of propensity scores for treated and control groups
plt.figure(figsize=(8, 5))
sns.kdeplot(d.loc[d['treatment'] == 1, 'propensity_score'], fill=True, label="Treated", color='salmon')
sns.kdeplot(d.loc[d['treatment'] == 0, 'propensity_score'], fill=True, label="Control", color='skyblue')

# Labels and formatting
plt.xlabel("Estimated Propensity Score")
plt.ylabel("Density")
plt.title("Distribution of Propensity Scores by Treatment Group")
plt.legend()
plt.show()

#%%
del a, ax, fig

#%%

# b. matching
def psm_noreplace(d, caliper, precise= 1000):
    # Define treated and control groups
    treated = d[d['treatment'] == 1].copy()
    control = d[d['treatment'] == 0].copy()
    
    # Fit nearest neighbors model on control group
    # for having no replacement, as we dont have library in python
    # for each point we  get 1000 closest points
    # then we iterate through these 1000 points and check whether they are matched before or not
    # increasing 1000 would increase the  number of matchings a bit but it is still quite large now!
    nn = NearestNeighbors(n_neighbors=precise, metric='euclidean')  # Get 800 closest
    nn.fit(control[['propensity_score']])
    
    # Find nearest neighbors for treated group
    distances, indices = nn.kneighbors(treated[['propensity_score']], return_distance=True)
    
    # Track used control units
    matched_control_indices = set()
    matched_pairs = []
    
    # Iterate through treated units
    for treat_idx, dists, ctrl_idxs in tqdm(zip(treated.index, distances, indices)):
        for dist, ctrl_idx in zip(dists, ctrl_idxs):  # Try closest first
            if np.sqrt(dist) <= caliper and ctrl_idx not in matched_control_indices:  # Check caliper & availability
                matched_pairs.append((treat_idx, control.index[ctrl_idx]))
                matched_control_indices.add(ctrl_idx)  # Mark as used
                break  # Stop after finding the first available match
    
    # Extract matched samples
    matched_treated = treated.loc[[pair[0] for pair in matched_pairs]]
    matched_control = control.loc[[pair[1] for pair in matched_pairs]]
    
    # Combine matched data
    matched_data = pd.concat([matched_treated, matched_control]).reset_index(drop=True)

    return matched_data

#%%

matched_data= psm_noreplace(d, caliper= 0.2, precise= 1000)

#%%

# b. SMD calculation
def compute_smd(df, covariates):
    smd_values = {}
    for cov in covariates:
        mean_treated = df[df['treatment'] == 1][cov].mean()
        mean_control = df[df['treatment'] == 0][cov].mean()
        var_treated = df[df['treatment'] == 1][cov].var()
        var_control = df[df['treatment'] == 0][cov].var()

        smd = (mean_treated - mean_control) / np.sqrt((var_treated + var_control) / 2)
        smd_values[cov] = smd
    return smd_values

#%%

# b. balance check
# Define covariates to check balance
covariates = ['age', 'years_at_company', 'projects_handled', 'remote_work_frequency']

# Compute SMD before matching
smd_before = compute_smd(d, covariates)

# Compute SMD after matching
smd_after = compute_smd(matched_data, covariates)

# Convert to DataFrame for visualization
smd_df = pd.DataFrame({'Before Matching': smd_before, 'After Matching': smd_after})

#%%

# b. smd check
# Plot SMD before and after matching
ax = smd_df.plot(kind='bar', figsize=(8, 5), color=['skyblue', 'salmon'])
plt.axhline(y=0.1, color='gray', linestyle='--', label="SMD = 0.1 Threshold")
plt.ylabel("Standardized Mean Difference (SMD)")
plt.title("Covariate Balance Before and After Matching")
plt.legend()

# Show the plot
plt.show()

#%%

# b. distribution check
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# Before Matching
sns.histplot(d, x="propensity_score", hue="treatment", bins=30, kde=True, ax=axes[0], palette=['skyblue', 'salmon'], alpha=0.5)
axes[0].set_title("Propensity Score Distribution (Before Matching)")
axes[0].set_xlabel("Propensity Score")
axes[0].set_ylabel("Count")

# After Matching
sns.histplot(matched_data, x="propensity_score", hue="treatment", bins=30, kde=True, ax=axes[1], palette=['skyblue', 'salmon'], alpha=0.5)
axes[1].set_title("Propensity Score Distribution (After Matching)")
axes[1].set_xlabel("Propensity Score")

plt.show()

#%%

# c. ATT Effect
# Compute mean salary for treated and matched control groups
mean_salary_treated = matched_data[matched_data['treatment'] == 1]['monthly_salary'].mean()
mean_salary_control = matched_data[matched_data['treatment'] == 0]['monthly_salary'].mean()

# Compute ATT
ATT = mean_salary_treated - mean_salary_control

print(f"Average Monthly Salary (Treated): {mean_salary_treated:.2f}")
print(f"Average Monthly Salary (Control): {mean_salary_control:.2f}")
print(f"Estimated ATT: {ATT:.2f}")

# Create a bar plot for ATT
plt.figure(figsize=(6, 5))
plt.bar(["Treated", "Control"], [mean_salary_treated, mean_salary_control], color=['salmon', 'skyblue'])

# Add labels and title
plt.ylabel("Average Monthly Salary")
plt.title("Average Monthly Salary: Treated vs. Control (Matched Sample)")

# Annotate bars with values
plt.text(0, mean_salary_treated + 100, f"{mean_salary_treated:.2f}", ha='center', fontsize=12)
plt.text(1, mean_salary_control + 100, f"{mean_salary_control:.2f}", ha='center', fontsize=12)

# Save the figure
plt.savefig("att_bar_chart.png", dpi=300, bbox_inches='tight')

plt.show()

#%%

# c. OLS regression
# Define dependent variable (Y) and independent variables (X)
X = matched_data[['age', 'years_at_company', 'projects_handled', 'remote_work_frequency']]
y = matched_data['monthly_salary']

# Add constant for intercept
X = sm.add_constant(X)

# Fit OLS model
ols_model = sm.OLS(y, X).fit()

# Print summary
print(ols_model.summary())

#%%

# Extract regression coefficients, standard errors, and p-values
ols_results = pd.DataFrame({
    "Coefficient": ols_model.params,
    "Std. Error": ols_model.bse,
    "P-Value": ols_model.pvalues
})

# Add R-squared value
r_squared = ols_model.rsquared
ols_results.loc["R-squared", "Coefficient"] = r_squared

# Display table
print(ols_results)

# Save as an image using Matplotlib
fig, ax = plt.subplots(figsize=(8, 1.5))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=ols_results.round(6).values, 
                 colLabels=ols_results.columns, 
                 rowLabels=ols_results.index, 
                 cellLoc='center', 
                 loc='center')

plt.show()

#%%

# d. OLS with full data and treatment
# Define dependent variable (Y) and independent variables (X)
X = d[['age', 'years_at_company', 'projects_handled', 'remote_work_frequency', 'treatment']]
y = d['monthly_salary']

# Add constant for intercept
X = sm.add_constant(X)

# Fit OLS model
ols_model = sm.OLS(y, X).fit()

# Print summary
print(ols_model.summary())

#%%

# Extract regression coefficients, standard errors, and p-values
ols_results = pd.DataFrame({
    "Coefficient": ols_model.params,
    "Std. Error": ols_model.bse,
    "P-Value": ols_model.pvalues
})

# Add R-squared value
r_squared = ols_model.rsquared
ols_results.loc["R-squared", "Coefficient"] = r_squared

# Display table
print(ols_results)

# Save as an image using Matplotlib
fig, ax = plt.subplots(figsize=(8, 1.5))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=ols_results.round(6).values, 
                 colLabels=ols_results.columns, 
                 rowLabels=ols_results.index, 
                 cellLoc='center', 
                 loc='center')

plt.show()

#%%

# e. OLS with full data and training hours
# Define dependent variable (Y) and independent variables (X)
X = d[['age', 'years_at_company', 'projects_handled', 'remote_work_frequency', 'training_hours']]
y = d['monthly_salary']

# Add constant for intercept
X = sm.add_constant(X)

# Fit OLS model
ols_model = sm.OLS(y, X).fit()

# Print summary
print(ols_model.summary())

#%%

# Extract regression coefficients, standard errors, and p-values
ols_results = pd.DataFrame({
    "Coefficient": ols_model.params,
    "Std. Error": ols_model.bse,
    "P-Value": ols_model.pvalues
})

# Add R-squared value
r_squared = ols_model.rsquared
ols_results.loc["R-squared", "Coefficient"] = r_squared

# Display table
print(ols_results)

# Save as an image using Matplotlib
fig, ax = plt.subplots(figsize=(8, 1.5))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=ols_results.round(6).values, 
                 colLabels=ols_results.columns, 
                 rowLabels=ols_results.index, 
                 cellLoc='center', 
                 loc='center')

plt.show()

#%%

