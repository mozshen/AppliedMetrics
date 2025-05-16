


#%%

import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS, PooledOLS, FirstDifferenceOLS, BetweenOLS, RandomEffects
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

#%%

d = pd.read_csv('MOM.dta', sep='\s+', header=None)
d.columns = ['lnhr', 'lnwg', 'kids', 'ageh', 'agesq', 'disab', 'id', 'year']

#%%

# a. fixed effect model
# clustering at id level
model = smf.ols('lnhr ~ lnwg + C(id)', data=d).fit(
    cov_type='cluster',
    cov_kwds={'groups': d['id']}
)

# Extract β (coefficient on lnwg)
print("β estimate:", model.params['lnwg'])
print("Std. error of β:", model.bse['lnwg'])

#%%

# b. different models

d = d.set_index(['id', 'year'])

# Add constant
d['const'] = 1

# Define dependent and independent variables
y = d['lnhr']
X = d[['const', 'lnwg']]

#%%

pooled = PooledOLS(y, X).fit()
between = BetweenOLS(y, X).fit()
within = PanelOLS(y, X, entity_effects=True).fit()
fd = FirstDifferenceOLS(y, X.drop(columns='const')).fit()
re_gls = RandomEffects(y, X).fit()

#%%

def bootstrap_se(model_class, y, X, index, reps=200):

    coefs = []

    
    for _ in range(reps):

        resampled_ids = np.random.choice(index, size=len(index), replace=True)
        resampled_index = index.isin(resampled_ids)
        y_boot = y[resampled_index]
        X_boot = X[resampled_index]
        
        try:
            if model_class == FirstDifferenceOLS:
                result = model_class(y_boot, X_boot.drop(columns='const')).fit()
                
            elif model_class == PanelOLS:
                result = model_class(y_boot, X_boot, entity_effects=True).fit()
                
            else:
                result = model_class(y_boot, X_boot).fit()
            coefs.append(result.params['lnwg'])
        
        except:
            continue

    return np.std(coefs, ddof=1)

#%%

results = {
    'pooled': pooled,
    'between': between,
    'within': within,
    'fd': fd,
    're_gls': re_gls
}

del pooled, between, within, fd, re_gls

#%%

bootstrap_ses = {}

for name, model in tqdm(results.items()):
    model_class = type(model.model)
    
    se = bootstrap_se(
        model_class,
        y,
        X,
        d.index
    )

    bootstrap_ses[name] = se

#%%

summary = pd.DataFrame({
    'β': [model.params['lnwg'] for model in results.values()],
    'Default SE': [model.std_errors['lnwg'] for model in results.values()],
    'Bootstrap SE': [bootstrap_ses[name] for name in results.keys()]
    }, 
    index=['Pooled OLS', 'Between', 'Within', 'First Diff', 'RE GLS'])

#%%

# Plot DataFrame as a table
fig, ax = plt.subplots(figsize=(10, 3))
ax.axis('off')  # Remove axes

# Create the table
table = ax.table(
    cellText=summary.round(3).values,
    rowLabels=summary.index,
    colLabels=summary.columns,
    cellLoc='center',
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.5)  # Adjust table size

plt.tight_layout()
plt.show()

#%%













