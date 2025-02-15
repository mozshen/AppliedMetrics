#%%

import pandas as pd
import numpy as np

#%%

isic= pd.read_csv("ISIC_Rev_4_english_structure.txt", delimiter=",", quotechar='"')
isic['Main']= np.where(isic['Code'].str.len() == 1, isic['Code'], None)
isic['Main']= isic['Main'].fillna(method= 'ffill')

#%%

df= isic.copy()

def get_parents(code, df):
    parents = []
    for i in range(len(code), 0, -1):  # Stepwise reduction
        parent_code = code[:i]
        if parent_code in df["Code"].values and parent_code != code:  # Avoid self-matching
            parent_desc = df[df["Code"] == parent_code]["Description"].values[0]
            parents.append((parent_code, parent_desc))
        if len(parents) == 3:  # Stop after finding 3 parents
            break
    while len(parents) < 3:  # Fill missing levels with NaN if less than 3 parents exist
        parents.append((None, None))
    return parents

# Create expanded DataFrame
expanded_data = []

for index, row in df.iterrows():
    code = row["Code"]
    main = row["Main"]
    parents = get_parents(code, df)
    
    expanded_data.append([
        code, main, row["Description"],
        parents[0][0], parents[0][1],  # Parent
        parents[1][0], parents[1][1],  # Grandparent
        parents[2][0], parents[2][1]   # Great-grandparent
    ])

#%%

# Convert to DataFrame
expanded_df = pd.DataFrame(expanded_data, columns=[
    "Code","Main", "Description",
    "Parent Code", "Parent Description",
    "Grandparent Code", "Grandparent Description",
    "Great-grandparent Code", "Great-grandparent Description"
])


#%%

a= expanded_df[["Code","Main", "Description", "Parent Description", "Grandparent Description"]]

a.columns= ["Code", "Main", "Level4", "Level2", "Level3"]
level1= a[a['Code'].str.len()== 1]
level1= level1[['Main', 'Level4']]
level1.columns= ['Main', 'Level1']

a= a.merge(level1, on= 'Main')
a= a[a['Level3'].notna()]
a= a.drop('Main', axis= 1)

a.to_excel('ISICRev4.xlsx', index= False)
