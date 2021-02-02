#%%
import torch
from dgl.data import RedditDataset

#%%
# return schema
# g, feat, label, train_mask, test_mask, in_feat, out_feat
def load_data():
    data = RedditDataset()
    graph = data[0]
    train_mask = graph.ndata['train_mask']
    val_mask = graph.ndata['val_mask']
    test_mask = graph.ndata['test_mask']
    label = graph.ndata['label']
    feat = graph.ndata['feat']
    in_feat = graph.ndata['feat'].shape[1]
    out_feat = int(max(label)+1)
    return graph, feat, label, train_mask, test_mask, in_feat, out_feat

# %%
