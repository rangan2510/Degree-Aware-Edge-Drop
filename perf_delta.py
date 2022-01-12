#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%
df = pd.read_csv("perf_delta copy.csv")
df

#%%
s = sns.barplot(x ="Dataset", y = 'Delta', data = df, hue = "Layers")
# %%
