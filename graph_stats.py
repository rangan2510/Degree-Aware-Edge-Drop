#%%
#%%
#loader
import dgl
import torch
from dgl import save_graphs, load_graphs

def load_graph(name, split):
    k_path = ""
    graph = load_graphs(k_path + name)[0][0]
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

#%%
graph_binaries = ["CoraFull.bin", "Pubmed.bin", "Citeseer.bin", "CoauthorCS.bin", "CoauthorPhysics.bin"]

#%%
dat = []
for bin in graph_binaries:
    row = []
    row.append(bin[:-4])
    g, features, labels, train_mask, test_mask, in_feat, out_feat = load_graph(bin,0.8)
    row.append(g.num_nodes())
    row.append(g.num_edges())
    row.append(in_feat)
    row.append(out_feat)
    print("-"*20)
    dat.append(row)

# %%
import pandas as pd
df = pd.DataFrame(data=dat, columns=["Dataset","#nodes", "#edges", "infeat", "classes"])
# %%
df.to_csv("graph_specs.csv",index=False)
# %%
