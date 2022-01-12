#%%
import pandas as pd
df_test = pd.read_csv("corafull copy.csv")
# %%
import seaborn as sns


sns.lineplot(x='Epoch', y=' TestAcc', 
             hue=' Dropout Type', data=df_test)
#%%

sns.lineplot(x='Epoch', y=' Loss', 
             hue=' Dropout Type', data=df_test)
             
# %%