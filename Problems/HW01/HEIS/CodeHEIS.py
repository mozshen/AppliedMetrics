
#%%

import pandas as pd
from statsmodels.stats.weightstats import DescrStatsW
import numpy as np
import hbsir
import matplotlib.pyplot as plt
import seaborn as sns

#%%

def weighted_mean(series, weights):
    return np.average(series, weights=weights) if not series.isna().all() else np.nan

def weighted_std(series, weights):
    if len(series) > 1:
        return DescrStatsW(series, weights=weights).std
    return np.nan

#%%

def weighted_deciles(values, weights, num_groups=10):
    """Assigns weighted decile ranks (1 to 10) based on GHazineh values."""
    sorted_indices = np.argsort(values)
    sorted_weights = weights[sorted_indices]

    # Compute cumulative sum of weights
    cumulative_weights = np.cumsum(sorted_weights)
    total_weight = cumulative_weights[-1]

    # Compute decile thresholds
    thresholds = np.linspace(0, total_weight, num_groups + 1)

    # Assign decile groups
    deciles = np.searchsorted(cumulative_weights, thresholds[1:], side='right') + 1
    ranks = np.zeros_like(values)
    for i in range(num_groups):
        ranks[sorted_indices[deciles[i - 1] if i > 0 else 0 : deciles[i]]] = i + 1

    return ranks

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
# we create deciles based on withtax income. because tax data may be distorted.
d['Decile'] = weighted_deciles(d['GHazineh'].values, d['weight'].values)

# since it is said the we use summary data for getting decile
# and summary data doesnt have food and non-food expenditures
# we assume that the column GHazineh has all the income

#%%

# 7 - Histogram of Average Educational Expenditure by Expenditure Deciles

# we get raw data
hbsir.setup(
    years= 1401,
    method="create_from_raw",
    download_source="mirror",
    replace=False,
)

d_raw= hbsir.load_table("durable", 1401)

# filtering for education related costs
# education
a1= d_raw[d_raw['Commodity_Code'].astype(str).str[:3].isin(['101', '102', '103', '104', '105'])]
# education books
a2= d_raw[d_raw['Commodity_Code'].astype(str).str[:5].isin(['95111', '95112', '95113', '95114', '95115'])]

d_educ= pd.concat([a1, a2])
del a1, a2

#%%

# now we get total education expend
d_educ= d_educ[d_educ['Provision_Method']== 'Purchase']
d_educ= d_educ.groupby('ID', as_index= False).agg({'Expenditure': 'sum'})
d_educ.columns= ['ADDRESS', 'EducationExpenditure']

#%%

# now we merge both
d= d.merge(d_educ, on= 'ADDRESS', how= 'left')
d['EducationExpenditure']= d['EducationExpenditure'].fillna(0)

#%%

# now we get mean by decile
d_q = d.groupby(['Decile'], as_index=False).agg(
    Education_mean=('EducationExpenditure', lambda x: weighted_mean(x, d.loc[x.index, 'weight'])),
)

#%%

# Set style
sns.set_style("whitegrid")

# Create figure
plt.figure(figsize=(8, 5))

# Bar plot
sns.barplot(x=d_q['Decile'], y=d_q['Education_mean'], color='#4C72B0', edgecolor='black')

# Labels and title
plt.xlabel("Decile", fontsize=12, labelpad=10)
plt.ylabel("Average Education Expenduture", fontsize=12, labelpad=10)
plt.title("Education Level by Decile", fontsize=14, pad=15)

# Remove top and right spines for a clean look
sns.despine()

# Show plot
plt.show()

#%%

# 8 - Educational Expenditure per Child by Household Expenditure Decile
# first we want to know how many child members are educating
d_member= hbsir.load_table("members_properties", 1401)

d_member= d_member[d_member['Is_Student']== True]
d_member= d_member[d_member['Relationship']== 'Child']
d_member= d_member.groupby('ID', as_index= False).agg({'Relationship': 'count'})
d_member.columns= ['ADDRESS', 'StudentChildCount']

# merging
d= d.merge(d_member, on= 'ADDRESS', how= 'left')
d['StudentChildCount']= d['StudentChildCount'].fillna(0)

#%%

# we also need to know the total member count
d_member= hbsir.load_table("members_properties", 1401)
d_member= d_member.groupby('ID', as_index= False).agg({'Relationship': 'count'})
d_member.columns= ['ADDRESS', 'MemberCount']

d= d.merge(d_member, on= 'ADDRESS', how= 'left')
d['MemberCount']= d['MemberCount'].fillna(0)

#%%

# Assumptions of the calculation:
#  we assume expenditures are equaly distributed metween members
# so we get expend by student child with this formula: ((d['StudentChildCount']/ d['MemberCount'])* d['GHazineh'])
# and then divide the education expenditure to this value

d['EducationExpenditureperChildShare']= d['EducationExpenditure'] / ((d['StudentChildCount']/ d['MemberCount'])* d['GHazineh'])
d_q= d[d['EducationExpenditureperChildShare']<1]

d_q = d_q.groupby(['Decile'], as_index=False).agg(
    Education_mean=('EducationExpenditureperChildShare', lambda x: weighted_mean(x, d.loc[x.index, 'weight'])),
)

#%%

# Set style
sns.set_style("whitegrid")

# Create figure
plt.figure(figsize=(8, 5))

# Bar plot
sns.barplot(x=d_q['Decile'], y=d_q['Education_mean']* 100, color='#4C72B0', edgecolor='black')

# Labels and title
plt.xlabel("Decile", fontsize=12, labelpad=10)
plt.ylabel("Share of per Child Education Expenditure, %", fontsize=12, labelpad=10)
plt.title("Share of per Child Education Expenditure by Decile", fontsize=14, pad=15)

# Remove top and right spines for a clean look
sns.despine()

# Show plot
plt.show()

#%%





