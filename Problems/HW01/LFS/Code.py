
#%%

import lfsir
import pandas as pd

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











