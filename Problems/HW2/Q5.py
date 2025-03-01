#%%

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np

#%%

d= pd.read_excel('DataQ5.xlsx')

#%%

a= d.describe()
a.to_excel('Q05a.xlsx')

#%%

# Create histograms for all variables in the dataframe
d.hist(figsize=(16, 10), bins=20, edgecolor='black')
plt.suptitle('Histograms of All Variables')
plt.show()

#%%

# c. OLS regression
# Define the independent variables: 
d_X = d[['atndrte', 'frosh', 'soph']]
d_X = sm.add_constant(d_X)

# Define the dependent variable
d_Y = d['stndfnl']

# Run the OLS regression
model = sm.OLS(d_Y, d_X).fit()

# Print the regression results
print(model.summary())

#%%

# residuals histogram
plt.figure(figsize=(8, 5))
plt.hist(model.resid, bins=30, edgecolor='black', alpha=0.7, color='skyblue', density=True)
plt.axvline(model.resid.mean(), color='red', linestyle='dashed', linewidth=2, label='Mean')
plt.xlabel('Residuals')
plt.ylabel('Density')
plt.title('Distribution of Residuals')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()
plt.show()

#%%

# resuduals and target
plt.figure(figsize=(8, 5))
plt.scatter(d['stndfnl'], model.resid, alpha=0.5, color='blue', edgecolor='black')
plt.axhline(0, color='red', linestyle='dashed', linewidth=2)
plt.xlabel('stndfnl')
plt.ylabel('Residuals')
plt.title('Residuals vs. stndfnl')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


#%%
