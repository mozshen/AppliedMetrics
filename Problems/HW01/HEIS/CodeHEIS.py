
#%%

import pandas as pd

#%%

d1= pd.read_excel('SumR1401.xlsx')
d1['Type']= 'Rural'
d2= pd.read_excel('SumU1401.xlsx')
d2['Type']= 'Urban'

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

#%%

# add province
d['ProvinceCode']= d['ADDRESS'].astype('str').str[1:3].astype(int)
prov= pd.read_excel('ProvinceMap.xlsx')

d= d.merge(prov, on= 'ProvinceCode')

#%%

d_q= d.groupby(
    ['Type', 'Province'], 
    as_index= False)\
    .agg({
        'HeadEducationLevel': ['min', 'max', 'mean', 'std', 'count'],
        'Income': ['min', 'max', 'mean', 'std', 'count'],
        'HeadCount': ['min', 'max', 'mean', 'std', 'count']
        })


d_q.to_excel('Q5.xlsx')

#%%














