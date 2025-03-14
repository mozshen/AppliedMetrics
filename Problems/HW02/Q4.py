

#%%

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np

#%%

d1= pd.read_excel('data_1.xlsx')
d2= pd.read_excel('data_2.xlsx')

d= pd.concat([d1, d2])

del d1 , d2

#%%

# a. summary stat
a= d.describe()

#%%

# b. OLS regression
# Define the independent variables (education and gender indicator)
d_X = d[['educ']]
d_X = sm.add_constant(d_X)

# Define the dependent variable (wage)
d_Y = d['wage']

# Run the OLS regression
model = sm.OLS(d_Y, d_X).fit()

# Print the regression results
print(model.summary())


#%%

# Compute predicted wages for educ = 12 and educ = 16
educ_values = np.array([12, 16])
predicted_wages = model.predict(sm.add_constant(pd.DataFrame({'educ': educ_values})))

# Plot the regression function
educ_range = np.linspace(d_X['educ'].min(), d_X['educ'].max(), 100)
predicted_wage_range = model.predict(sm.add_constant(pd.DataFrame({'educ': educ_range})))

#%%

plt.figure(figsize=(8, 5))
plt.scatter(d_X['educ'], d_Y, alpha=0.5, label='Observed Data')
plt.plot(educ_range, predicted_wage_range, color='red', label='Estimated Regression Line')
plt.scatter(educ_values, predicted_wages, color='yellow', marker='o', label='Predicted Points')
plt.text(12, predicted_wages.iloc[0], f'({12}, {predicted_wages.iloc[0]:.2f})', fontsize=10, verticalalignment='bottom')
plt.text(16, predicted_wages.iloc[1], f'({16}, {predicted_wages.iloc[1]:.2f})', fontsize=10, verticalalignment='bottom')
plt.xlabel('Years of Education')
plt.ylabel('Wage')
plt.legend(loc= 'upper left')
plt.title('Estimated Sample Regression Function')
plt.show()

#%%









