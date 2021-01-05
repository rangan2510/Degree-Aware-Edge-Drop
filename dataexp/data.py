#%%
# explore datasets
from dgl.data import CoraFull
data = CoraFull()
g = data[0]
print(g)
print(max(g.ndata['label']))
# %%
