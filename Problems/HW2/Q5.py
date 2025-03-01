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



