
#%%

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

#%%

# Load data
d = pd.read_csv('lowbirth.csv')

#%%

# adding id
ids= []
for i in range(1, int(len(d)/ 2+ 1)):
    ids+= [i, i]

d['id']= ids

#%%

# Define outcome and regressors
y = d['lowbrth']
X = d[['d90', 'lphypc', 'lbedspc', 'afdcprc', 'lpcinc', 'lpopul']]
X = sm.add_constant(X)

#%%

# Pooled OLS (usual SEs)
model_ols = sm.OLS(y, X).fit()

# Robust SEs (heteroskedasticity-consistent)
model_robust = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': d['id']})


print("=== Pooled OLS with usual SEs ===")
print(model_ols.summary())

print("\n=== Pooled OLS with robust (White) SEs ===")
print(model_robust.summary())

#%%

results_df = pd.DataFrame({
    'Coefficient': model_ols.params.round(4),
    'Usual SE': model_ols.bse.round(4),
    'Robust SE': model_robust.bse.round(4)
})

print(results_df)

#%%

# Plot DataFrame as a table
fig, ax = plt.subplots(figsize=(10, 3))
ax.axis('off')  # Remove axes

# Create the table
table = ax.table(
    cellText=results_df.round(3).values,
    rowLabels=results_df.index,
    colLabels=results_df.columns,
    cellLoc='center',
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.5)  # Adjust table size

plt.tight_layout()
plt.show()

#%%

# part c
y = d['lowbrth']
X = d[['d90', 'clphypc', 'clbedspc', 'cafdcprc', 'clpcinc', 'lpopul']]
X = sm.add_constant(X)
X= X[X['d90']== 1]
y= y[y.index.isin(X.index)]

#%%

# Pooled OLS (usual SEs)
model_ols = sm.OLS(y, X).fit()

print("=== Pooled OLS with usual SEs ===")
print(model_ols.summary())

#%%

results_df = pd.DataFrame({
    'Coefficient': model_ols.params.round(4),
    'Usual SE': model_ols.bse.round(4)
})

print(results_df)

#%%

# Plot DataFrame as a table
fig, ax = plt.subplots(figsize=(10, 3))
ax.axis('off')  # Remove axes

# Create the table
table = ax.table(
    cellText=results_df.round(3).values,
    rowLabels=results_df.index,
    colLabels=results_df.columns,
    cellLoc='center',
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.5)  # Adjust table size

plt.tight_layout()
plt.show()

#%%

# part d
d['afdcprc2']= d['afdcprc']**2
d['cafdcprc2']= d.groupby('id')['afdcprc2'].diff()

#%%


y = d['lowbrth']
X = d[['d90', 'clphypc', 'clbedspc', 'cafdcprc', 'clpcinc', 'cafdcprc2', 'lpopul']]
X = sm.add_constant(X)
X= X[X['d90']== 1]
y= y[y.index.isin(X.index)]

#%%

# Pooled OLS (usual SEs)
model_ols = sm.OLS(y, X).fit()

print("=== Pooled OLS with usual SEs ===")
print(model_ols.summary())

#%%

results_df = pd.DataFrame({
    'Coefficient': model_ols.params.round(4),
    'Usual SE': model_ols.bse.round(4)
})

print(results_df)

#%%

# Plot DataFrame as a table
fig, ax = plt.subplots(figsize=(10, 3))
ax.axis('off')  # Remove axes

# Create the table
table = ax.table(
    cellText=results_df.round(3).values,
    rowLabels=results_df.index,
    colLabels=results_df.columns,
    cellLoc='center',
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.5)  # Adjust table size

plt.tight_layout()
plt.show()

#%%

turning_point = -model_ols.params['cafdcprc'] / (2 * model_ols.params['cafdcprc2'])
print("Turning point of AFDC effect:", turning_point)

#%%





