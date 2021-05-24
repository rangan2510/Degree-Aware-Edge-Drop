#%%
import torch
from dgl.data import CoraFullDataset

#%%
def load_data(split = 0.8):
    data = CoraFullDataset()
    g = data[0]
    num_class = data.num_classes
    feat = g.ndata['feat']  # get node feature
    label = g.ndata['label']
    mask = torch.BoolTensor(g.num_nodes())
    mask[:] = False
    split_idx = int((g.num_nodes()*split))
    mask[:split_idx] = True
    train_mask = mask
    test_mask = torch.logical_not(train_mask)
    in_feat = g.ndata['feat'].shape[1]
    out_feat = int(max(label) + 1)
    return g, feat, label, train_mask, test_mask, in_feat, out_feat

# %%
g, feat, label, train_mask, test_mask, in_feat, out_feat = load_data()
# %%
