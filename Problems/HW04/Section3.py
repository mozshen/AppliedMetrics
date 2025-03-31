
#%%

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import numpy as np
from tqdm import tqdm
import statsmodels.api as sm
from scipy.spatial.distance import cdist

#%%

d= pd.read_csv('Employee_Performance.csv')

#%%

# a. creating new treatment
d['remote_work_treatment']= d['remote_work_frequency'].map({
    0: 0,
    25:0,
    75:1,
    100:1
    })

d= d[d['remote_work_treatment'].notna()]

#%%

# a. logistic regression
covariates = ['age', 'years_at_company', 'overtime_hours']

# Convert education_level to dummy variables (one-hot encoding)
education_dummies = pd.get_dummies(d['education_level'], prefix='edu', drop_first=True)

# Combine numeric covariates with dummies
X = pd.concat([d[covariates], education_dummies], axis=1)

# Treatment variable
y = d['treatment']

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

matched_data_psm= psm_noreplace(d, caliper= 0.2, precise= 1000)

#%%

# SMD check
def compute_smd(df, covariates):
    smd_results = {}
    for cov in covariates:
        mean_treated = df[df['treatment'] == 1][cov].mean()
        mean_control = df[df['treatment'] == 0][cov].mean()
        std_treated = df[df['treatment'] == 1][cov].std()
        std_control = df[df['treatment'] == 0][cov].std()

        smd = (mean_treated - mean_control) / np.sqrt((std_treated**2 + std_control**2) / 2)
        smd_results[cov] = smd

    return pd.DataFrame.from_dict(smd_results, orient='index', columns=['SMD'])

#%%

# Compute SMD before and after matching
smd_before = compute_smd(d, covariates)
smd_after = compute_smd(matched_data_psm, covariates)

# Display results
smd_table = pd.concat([smd_before.rename(columns={'SMD': 'Before Matching'}), 
                       smd_after.rename(columns={'SMD': 'After Matching'})], axis=1)
print(smd_table)

#%%

# b. smd check
# Plot SMD before and after matching
ax = smd_table.plot(kind='bar', figsize=(8, 5), color=['skyblue', 'salmon'])
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
sns.histplot(matched_data_psm, x="propensity_score", hue="treatment", bins=30, kde=True, ax=axes[1], palette=['skyblue', 'salmon'], alpha=0.5)
axes[1].set_title("Propensity Score Distribution (After Matching)")
axes[1].set_xlabel("Propensity Score")

plt.show()

#%%

# c. Mahalanobis matching
# Define covariates for Mahalanobis matching
covariates = ['age', 'years_at_company', 'projects_handled']


# Separate treated and control groups
treated = d[d['treatment'] == 1].copy()
control = d[d['treatment'] == 0].copy()

# Compute Mahalanobis distance
V_inv = np.linalg.inv(np.cov(control[covariates].T))  # Inverse of covariance matrix
dist_matrix = cdist(treated[covariates], control[covariates], metric='mahalanobis', VI=V_inv)

# Find nearest neighbor (smallest Mahalanobis distance)
matched_indices = np.argmin(dist_matrix, axis=1)
matched_control = control.iloc[matched_indices]

# Create matched dataset
matched_data_dis = pd.concat([treated.reset_index(drop=True), matched_control.reset_index(drop=True)])

# Reset index
matched_data_dis.reset_index(drop=True, inplace=True)

del dist_matrix

#%%

# Compute SMD before and after matching
smd_before = compute_smd(d, covariates)
smd_after = compute_smd(matched_data_dis, covariates)

# Display results
smd_table = pd.concat([smd_before.rename(columns={'SMD': 'Before Matching'}), 
                       smd_after.rename(columns={'SMD': 'After Matching'})], axis=1)
print(smd_table)

#%%

# Plot SMD before and after matching
ax = smd_table.plot(kind='bar', figsize=(8, 5), color=['skyblue', 'salmon'])
plt.axhline(y=0.1, color='gray', linestyle='--', label="SMD = 0.1 Threshold")
plt.ylabel("Standardized Mean Difference (SMD)")
plt.title("Covariate Balance Before and After Matching")
plt.legend()

# Show the plot
plt.show()

#%%

# d. ATT effect

def compute_att(df, outcome_var):
    treated = df[df['remote_work_treatment'] == 1][outcome_var]
    control = df[df['remote_work_treatment'] == 0][outcome_var]
    
    att = treated.mean() - control.mean()  # ATT = Mean(Treated) - Mean(Control)
    return att

#%%

def compute_att_se(df, outcome_var, n_bootstrap=1000):
    treated = df[df['remote_work_treatment'] == 1][outcome_var]
    control = df[df['remote_work_treatment'] == 0][outcome_var]
    
    att = treated.mean() - control.mean()  # ATT = Mean(Treated) - Mean(Control)
    
    # Bootstrap standard error
    boot_diffs = []
    for _ in range(n_bootstrap):
        boot_treated = treated.sample(frac=1, replace=True)  # Resample with replacement
        boot_control = control.sample(frac=1, replace=True)
        boot_diffs.append(boot_treated.mean() - boot_control.mean())

    se = np.std(boot_diffs)  # Standard deviation of bootstrap estimates
    return att, se

#%%

# Compute ATT & SE for performance score
att_perf_dis, se_perf_dis = compute_att_se(matched_data_dis, 'performance_score')
att_perf_psm, se_perf_psm = compute_att_se(matched_data_psm, 'performance_score')

# Compute ATT & SE for resignation probability
att_resign_dis, se_resign_dis = compute_att_se(matched_data_dis, 'resigned')
att_resign_psm, se_resign_psm = compute_att_se(matched_data_psm, 'resigned')

# Create summary table
att_table = pd.DataFrame({
    "Matching Method": ["Mahalanobis", "Propensity Score"],
    "ATT (Performance Score)": [att_perf_dis, att_perf_psm],
    "SE (Performance Score)": [se_perf_dis, se_perf_psm],
    "ATT (Resigned Probability)": [att_resign_dis, att_resign_psm],
    "SE (Resigned Probability)": [se_resign_dis, se_resign_psm]
})

# Display table
print(att_table)

#%%

fig, ax = plt.subplots(figsize=(10, 0.5))  # Adjust figure size
ax.axis('tight')
ax.axis('off')
ax.table(cellText=att_table.round(4).values, 
         colLabels=att_table.columns, 
         rowLabels=att_table.index, 
         cellLoc='center', 
         loc='center')

plt.show()

#%%

