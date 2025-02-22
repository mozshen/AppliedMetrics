
#%%

import pandas as pd
from statsmodels.stats.weightstats import DescrStatsW
import numpy as np

#%%

def weighted_mean(series, weights):
    return np.average(series, weights=weights) if not series.isna().all() else np.nan

def weighted_std(series, weights):
    if len(series) > 1:
        return DescrStatsW(series, weights=weights).std
    return np.nan

#%%

d1= pd.read_excel('SumR1401.xlsx')
d1['Type']= 'Rural'
d2= pd.read_excel('SumU1401.xlsx')
d2['Type']= 'Urban'

#%%

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

d['HeadEducationLevel']= d['HeadEducationLevel'].map({
    0: 'Non-Literate',
    9: 'Primary School',
    1: 'Primary School',
    2: 'Secondary School',
    3: 'High School',
    4: 'High School Diploma',
    5: 'High School Diploma',
    6: 'Bachelor',
    7: 'Master and Higher',
    8: 'Master and Higher'
    }
    )

#%%

# add province
d['ProvinceCode']= d['ADDRESS'].astype('str').str[1:3].astype(int)
prov= pd.read_excel('ProvinceMap.xlsx')

d= d.merge(prov, on= 'ProvinceCode')

#%%

# part 1: analyzing head count and income
d_q = d.groupby(['Type', 'Province'], as_index=False).agg(
    Income_min=('Income', 'min'),
    Income_max=('Income', 'max'),
    Income_mean=('Income', lambda x: weighted_mean(x, d.loc[x.index, 'weight'])),
    Income_std=('Income', lambda x: weighted_std(x, d.loc[x.index, 'weight'])),
    Income_count=('Income', 'count'),
    HeadCount_min=('HeadCount', 'min'),
    HeadCount_max=('HeadCount', 'max'),
    HeadCount_mean=('HeadCount', lambda x: weighted_mean(x, d.loc[x.index, 'weight'])),
    HeadCount_std=('HeadCount', lambda x: weighted_std(x, d.loc[x.index, 'weight'])),
    HeadCount_count=('HeadCount', 'count')
)

#%%

d_q.to_excel('Q5-part1.xlsx')


#%%

# part 2: head education level
# we get share of each group
d_q = d.groupby(['Type', 'Province', 'HeadEducationLevel'], as_index=False)\
    .agg({
        'weight': 'sum'
        })

d_q['weight']= d_q['weight']/ d_q.groupby(['Type', 'Province'])['weight'].transform('sum')
d_q= d_q.pivot(index= ['Type', 'Province'], columns= 'HeadEducationLevel', values= 'weight')
d_q.columns= ['Non-Literate', 'Primary School', 'Secondary School', 
              'High School', 'High School Diploma', 'Bachelor',
              'Master and Higher'
              ]

d_q= d_q.fillna(0)

#%%

d_q.to_excel('Q5-part2.xlsx')

#%%

# 6 - Expenditure Deciles for Each Household











#%%

