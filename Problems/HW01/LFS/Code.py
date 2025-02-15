
#%%

import lfsir
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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

#%%

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

d['ISICCode']= d['F3_D10'].str[:1]

isic= pd.read_excel('ISICRev4.xlsx')
isic['ISICCode']= isic['ISICCode'].str[:1]
isic= isic[['ISICCode', 'Level1']].drop_duplicates()

a= set(d['ISICCode'].unique())- set(isic['ISICCode'])





