#%%
import torch
from dgl.data import CoauthorCSDataset

#%%
# return schema
# g, feat, label, train_mask, test_mask, in_feat, out_feat
def load_data(split = 0.8):
    data = CoauthorCSDataset()
    graph = data[0]
    label = graph.ndata['label']
    feat = graph.ndata['feat']
    in_feat = graph.ndata['feat'].shape[1]
    out_feat = int(max(label)+1)
    mask = torch.BoolTensor(graph.num_nodes())
    mask[:] = False
    split_idx = int((graph.num_nodes()*split))
    mask[:split_idx] = True
    train_mask = mask
    test_mask = torch.logical_not(train_mask)
    return graph, feat, label, train_mask, test_mask, in_feat, out_feat

# %%
