
#%%

import pandas as pd

#%%

d1= pd.read_excel('SumR1401.xlsx')
d2= pd.read_excel('SumU1401.xlsx')

d= pd.concat([d1, d2])
cols= d.columns
#%%

# 5 - Summary Statistics for Household Income and Education
# assuming daramad column is income
# no weighting

d= d.rename({
     'A05': 'HeadEducationLevel',
     'Daramad': 'Income',
     'C01': 'HeadCount' 
     },
    axis= 1)

d_q= d[['HeadEducationLevel', 'Income', 'HeadCount']].describe()
d_q.to_excel('Q5.xlsx', index= False)

#%%














