
#%%

import pandas as pd
from linearmodels.panel import PooledOLS, PanelOLS, FirstDifferenceOLS, RandomEffects
import statsmodels.api as sm
import matplotlib.pyplot as plt

#%%

d= pd.read_csv('wagepan.csv')
d['const'] = 1

#%%

d = d.set_index(['nr', 'year'])
y = d['lwage']
X = d[['const', 'd81', 'd82', 'd83', 'd84', 'd85', 'd86', 'd87']]


#%%

pooled = PooledOLS(y, X).fit()
re_gls = RandomEffects(y, X).fit()
within = PanelOLS(y, X, entity_effects=True).fit()

#%%

# Drop constant
X_fd = d[['lwage', 'd81', 'd82', 'd83', 'd84', 'd85', 'd86', 'd87']]
# .diff().dropna()

y_fd = X_fd['lwage']
X_fd = X_fd.drop(columns=['lwage'])

fd = FirstDifferenceOLS(y_fd, X_fd).fit()

#%%

# Extract year coefficients from each model
rows = ['d81', 'd82', 'd83', 'd84', 'd85', 'd86', 'd87']
coefs = {
    'Pooled OLS': [pooled.params.get(var, None) for var in rows],
    'Random Effects': [re_gls.params.get(var, None) for var in rows],
    'Fixed Effects': [within.params.get(var, None) for var in rows],
    'First Differences': [fd.params.get(var, None) for var in rows]
}

# Create summary table
coef_df = pd.DataFrame(coefs, index=rows).round(4)

print(coef_df)

#%%

fig, ax = plt.subplots(figsize=(10, 3))
ax.axis('off')  # Remove axes

# Create the table
table = ax.table(
    cellText=coef_df.round(3).values,
    rowLabels=coef_df.index,
    colLabels=coef_df.columns,
    cellLoc='center',
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.5)  # Adjust table size

plt.tight_layout()
plt.show()

#%%

# b. adding other variables
X= d[['const', 'd81', 'd82', 'd83', 'd84', 'd85', 'd86', 'd87', 'educ', 'black', 'hisp']]
y = d['lwage']

pooled= PooledOLS(y, X).fit()
re_gls = RandomEffects(y, X).fit()
fe= PanelOLS(y, X.drop('const', axis= 1), entity_effects=True, drop_absorbed=True).fit()

#%%

rows = ['d81', 'd82', 'd83', 'd84', 'd85', 'd86', 'd87', 'educ', 'black', 'hisp']

# Function to format coef (se) string
def coef_se(model, var):
    coef = model.params.get(var, None)
    se = model.std_errors.get(var, None)
    if coef is not None and se is not None:
        return f"{coef:.4f} ({se:.4f})"
    else:
        return ""

# Build table
table_data = {
    'Pooled OLS': [coef_se(pooled, var) for var in rows],
    'Random Effects': [coef_se(re_gls, var) for var in rows],
    'Fixed Effects': [coef_se(within, var) for var in rows]
}

coef_df = pd.DataFrame(table_data, index=rows)

# Print result
print(coef_df)

#%%

fig, ax = plt.subplots(figsize=(10, 3))
ax.axis('off')  # Remove axes

# Create the table
table = ax.table(
    cellText=coef_df.round(3).values,
    rowLabels=coef_df.index,
    colLabels=coef_df.columns,
    cellLoc='center',
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.5)  # Adjust table size

plt.tight_layout()
plt.show()

#%%

pooled= PooledOLS(y, X).fit(cov_type='robust')
re_gls = RandomEffects(y, X).fit(cov_type='robust')
fe= PanelOLS(y, X.drop('const', axis= 1), entity_effects=True, drop_absorbed=True).fit(cov_type='robust')

#%%

rows = ['d81', 'd82', 'd83', 'd84', 'd85', 'd86', 'd87', 'educ', 'black', 'hisp']

# Function to format coef (se) string
def coef_se(model, var):
    coef = model.params.get(var, None)
    se = model.std_errors.get(var, None)
    if coef is not None and se is not None:
        return f"{coef:.4f} ({se:.4f})"
    else:
        return ""

# Build table
table_data = {
    'Pooled OLS': [coef_se(pooled, var) for var in rows],
    'Random Effects': [coef_se(re_gls, var) for var in rows],
    'Fixed Effects': [coef_se(within, var) for var in rows]
}

coef_df = pd.DataFrame(table_data, index=rows)

# Print result
print(coef_df)

#%%

fig, ax = plt.subplots(figsize=(10, 3))
ax.axis('off')  # Remove axes

# Create the table
table = ax.table(
    cellText=coef_df.round(3).values,
    rowLabels=coef_df.index,
    colLabels=coef_df.columns,
    cellLoc='center',
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.5)  # Adjust table size

plt.tight_layout()
plt.show()

