
#%%

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np

#%%

d= pd.read_excel('data.xlsx')

#%%

models = {}

#%%

# Model 1: Controlling all variables
X1 = d[['female', 'age', 'exercise', 'healthy_diet', 'fast_food', 'bmi', 'blood_pressure', 'genetic_risk']]
X1 = sm.add_constant(X1)
models["Full Model"] = sm.OLS(d['heart_disease'], X1).fit()

#%%

# Model 2: Controlling based on DAG
X2 = d[['female', 'age', 'exercise', 'healthy_diet', 'fast_food', 'genetic_risk']]
X2 = sm.add_constant(X2)
models["DAG Model"] = sm.OLS(d['heart_disease'], X2).fit()

#%%

# Create an empty DataFrame
variables = list(set(X1.columns).union(set(X2.columns)))  # Get all variables across models
summary_df = pd.DataFrame(index=variables, columns=[col for model in models.keys() for col in (model, model+ '_std', model+ '_tstat', model+ '_pvalue')])

# Fill the DataFrame with coefficients and standard errors
for model_name, model in models.items():
    for var in model.params.index:
        summary_df.loc[var, model_name] = model.params[var]
        summary_df.loc[var, model_name+ '_std'] = model.bse[var]
        summary_df.loc[var, model_name+ '_tstat'] = model.tvalues[var]
        summary_df.loc[var, model_name+ '_pvalue'] = model.pvalues[var]
        
# Add R-squared at the bottom
summary_df.loc["R-squared", :] = ""
for model_name, model in models.items():
    summary_df.loc["R-squared", model_name] = round(model.rsquared, 3)

# Replace NaNs with empty strings
summary_df = summary_df.fillna("")

# Display the table
print(summary_df)

summary_df.to_excel('result.xlsx')

#%%

