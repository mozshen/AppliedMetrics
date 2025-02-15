
#%%

import lfsir
import pandas as pd
import numpy as np

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

d['ActivityStatus']= d['ActivityStatus'].map({
    1: 'Employed',
    2: 'Unemployed',
    3: 'Inactive'
    }).fillna('Below15')

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

#%%

# filtering to those in working age
d_q= d[d['ActivityStatusNew'].isin(['Inactive', 'Employed', 'Unemployed'])]

d_q['Active']= np.where(
    d_q['ActivityStatusNew'].isin(['Employed', 'Unemployed']), 
    d_q['IW_Yearly'], 
    0)

d_q['Employed']= np.where(
    d_q['ActivityStatusNew'].isin(['Employed']), 
    d_q['IW_Yearly'], 
    0)

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

d_q.to_excel('Q1.xlsx')


#%%













