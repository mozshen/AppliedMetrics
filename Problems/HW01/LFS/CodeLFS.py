

#%%

import lfsir
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.weightstats import DescrStatsW
from scipy.stats import ttest_ind_from_stats

#%%

def weighted_mean(series, weights):
    return np.average(series, weights=weights) if not series.isna().all() else np.nan

def weighted_std(series, weights):
    if len(series) > 1:
        return DescrStatsW(series, weights=weights).std
    return np.nan

def effective_sample_size(weights):
    return (weights.sum() ** 2) / (weights**2).sum()


#%%


lfsir.setup(
    years="1401",
    method="create_from_raw",
    download_source="mirror",
    replace=False,
)

#%%

d= lfsir.load_table(
    table_name= "data", 
    years= 1401,
    form= 'raw'
    )

# cleaned data is used just for the checking
d_cleaned= lfsir.load_table(
    table_name= "data", 
    years= 1401,
    form= 'normalized'
    )

#%%

# 1 - Labor Force Participation and Unemployment Rates

# editing gender
d['F2_D04']= d['F2_D04'].map({1: 'Male', 2: 'Female'})
d= d.rename({'F2_D04': 'Gender'}, axis= 1)

#%%

# editing age
d['F2_D07'] = pd.to_numeric(d['F2_D07'], errors='coerce')
d= d.rename({'F2_D07': 'Age'}, axis= 1)

#%%

d['ActivityStatus']= d['ActivityStatus'].map({
    1: 'Employed',
    2: 'Unemployed',
    3: 'Inactive'
    }).fillna('Below10')

#%%

# editing education
d['F2_D17']= d['F2_D17'].map({
    '1': 'Below High School and High School',
    '2': 'Below High School and High School',
    '3': 'Below High School and High School',
    '4': 'Below High School and High School',
    '5': 'Below High School and High School',
    '6': 'Bachelor’s Degree',
    '7': 'Master’s and Higher',
    '8': 'Master’s and Higher',
    '9': 'Other'
    }).fillna('Non-Literate')

d= d.rename({'F2_D17': 'EducationLevel'}, axis= 1)

#%%

# we consider those above 65 to be out of working age
d['ActivityStatusNew'] = np.where(d['Age'] >= 65, 'Over65', d['ActivityStatus'])
d['ActivityStatusNew'] = np.where(d['Age'] < 15, 'Below15', d['ActivityStatusNew'])


d['Active']= np.where(
    d['ActivityStatus'].isin(['Employed', 'Unemployed']), 
    d['IW_Yearly'], 
    0)

d['Employed']= np.where(
    d['ActivityStatus'].isin(['Employed']), 
    d['IW_Yearly'], 
    0)

#%%

# filtering to those in working age
d_q= d[d['ActivityStatusNew'].isin(['Inactive', 'Employed', 'Unemployed'])]

d_q= d_q.groupby(
    ['Gender', 'EducationLevel'], 
    as_index= False)\
    .agg({'IW_Yearly': 'sum',
        'Active': 'sum',
        'Employed': 'sum'
        })

d_q['ParticipationRate']= d_q['Active']/ d_q['IW_Yearly']
d_q['UnemploymentRate']= 1- d_q['Employed']/ d_q['Active']

d_q= d_q[d_q['EducationLevel']!= 'Other']
d_q= d_q[['Gender', 'EducationLevel', 'ParticipationRate', 'UnemploymentRate']]

d_q= d_q.pivot(
    columns= 'Gender', 
    index= 'EducationLevel',
    values= ['ParticipationRate', 'UnemploymentRate']
    ).reset_index()

#%%

d_q.to_excel('Q1.xlsx')

#%%

# 2 - Distribution of Unemployment Rate and Employmentto-Population Ratio by Age
d_q= d[d['ActivityStatusNew']!= 'Below15']

d_q= d_q.groupby(
    ['Age'], 
    as_index= False)\
    .agg({
        'IW_Yearly': 'sum',
        'Active': 'sum',
        'Employed': 'sum'
        })
    
d_q['UnemploymentRate']= 100* (1- d_q['Employed']/ d_q['Active']).fillna(0)
d_q['EmploymentRate']= 100* (d_q['Employed']/ d_q['IW_Yearly']).fillna(0)


#%%

plt.figure(figsize=(8, 5))

# Plot lines with colors and line styles
plt.plot(d_q['Age'], d_q['UnemploymentRate'], label='Unemployment Rate', color='red', linestyle='--', linewidth=2)
plt.plot(d_q['Age'], d_q['EmploymentRate'], label='Employment-Population Ratio', color='blue', linewidth=2)

plt.legend(frameon=False)
plt.title('Unemployment & Employment Ratio by Age', fontsize=12, fontweight='bold')
plt.xlabel('Age', fontsize=10)
plt.ylabel('Rate (%)', fontsize=10)
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.grid(True, linestyle=':', alpha=0.5)

plt.show()

#%%

# 3 - Employment in Three Sectors by Firm Size
d['ISICCode']= d['F3_D10'].str[:2]

isic= pd.read_excel('ISICRev4.xlsx')
isic['ISICCode']= isic['ISICCode'].str[:2]
isic= isic[['ISICCode', 'Level0']].drop_duplicates()

d= d.merge(isic, on= 'ISICCode', how= 'left')

#%%

# firm size
d['F3_D12'] = pd.to_numeric(d['F3_D12'], errors='coerce')
d= d.rename({'F3_D12': 'FirmSize'}, axis= 1)

#%%

d_q= d.groupby(['Level0', 'FirmSize'], as_index= False).agg({'Employed': 'sum'})

d_q['FirmSize']= d_q['FirmSize'].map({
    1: '01 to 04',
    2: '05 to 09',
    3: '10 to 19',
    4: '20 to 49',
    5: 'Over 50'
    })

d_q= d_q.pivot(index="FirmSize", columns="Level0", values="Employed")

#%%

d_q.plot(kind="bar", figsize=(10, 6), colormap="viridis", edgecolor="black", alpha=0.9)

plt.title("Employed Individuals by Firm Size", fontsize=14, fontweight="bold")
plt.xlabel("Firm Size", fontsize=12)
plt.ylabel("Number of Employees, Million", fontsize=12)
plt.legend(title="Sector", frameon=False)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.xticks(rotation= 0)
plt.show()

#%%

# 4 - Hours Worked by Insured and Uninsured Workers

# insurance flag
d['F3_D13']= d['F3_D13'].map({
    '1': 'Insured',
    '2': 'Uninsured'
    })

d= d.rename({'F3_D13': 'IsInsured'}, axis= 1)

#%%

# we consider only the main job working hours
d['F3_D16SHASLIS'] = pd.to_numeric(d['F3_D16SHASLIS'], errors='coerce')
d= d.rename({'F3_D16SHASLIS': 'HoursWorkedinMainJob'}, axis= 1)

#%%

# adding isic
d['ISICCode']= d['F3_D10'].str[:2]

isic= pd.read_excel('ISICRev4.xlsx')
isic['ISICCode']= isic['ISICCode'].str[:2]
isic= isic[['ISICCode', 'Level1']].drop_duplicates()

d= d.merge(isic, on= 'ISICCode', how= 'left')

#%%

# average hours in occupations
d_q = d.groupby(['Level1', 'IsInsured'], as_index=False).agg(
    HoursWorkedinMainJob=(
        'HoursWorkedinMainJob', 
        lambda x: np.average(x, weights=d.loc[x.index, 'IW_Yearly'])
    )
)

d_q= d_q.pivot(index="Level1", columns="IsInsured", values="HoursWorkedinMainJob")
d_q= d_q.reset_index()

d_q.to_excel('Q4-avghourinsuranceoccupation.xlsx', index= False)

#%%

# checking for difference in avg hours
d_q = d.groupby(['Level1'], as_index=False).agg(
    hour_mean=('HoursWorkedinMainJob', lambda x: weighted_mean(x, d.loc[x.index, 'IW_Yearly'])),
    hour_std=('HoursWorkedinMainJob', lambda x: weighted_std(x, d.loc[x.index, 'IW_Yearly'])),
    hour_n_eff=('IW_Yearly', lambda w: effective_sample_size(w))
)

#%%

# t test
row_A = d_q[d_q['Level1'] == 'Education'].iloc[0]
row_B = d_q[d_q['Level1'] == 'Mining and quarrying'].iloc[0]

# Extract values
mean_A, std_A, n_A = row_A['hour_mean'], row_A['hour_std'], row_A['hour_n_eff']
mean_B, std_B, n_B = row_B['hour_mean'], row_B['hour_std'], row_B['hour_n_eff']

# Perform t-test
t_stat, p_value = ttest_ind_from_stats(mean_A, std_A, n_A, mean_B, std_B, n_B, equal_var=False)

print(f"T-statistic: {t_stat}, P-value: {p_value}")

# so the values are different!

#%%

d_q= d_q.sort_values('HoursWorkedinMainJob')
d_q.plot(kind="barh", figsize=(10, 6), colormap="viridis", edgecolor="black", alpha=0.9)

plt.title("Average Working Hours in Occupations", fontsize=14, fontweight="bold")
plt.xlabel("Average Working Hours", fontsize=12)
plt.show()


#%%









