#%%
#loader
import dgl
import torch
from dgl import save_graphs, load_graphs
from random import randrange
from statistics import mean, stdev
import networkx as nx
import pandas as pd
from tqdm import tqdm

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
dropouts = [x/10 for x in range(1,10)]
results_ccs = []
results_iso = []
#%%
plot_source = []

for bin in graph_binaries:
    print(bin)
    
    row_ccs = []
    row_iso = []

    for dropout in tqdm(dropouts):
        all_ccs = all_iso = []    
        for run in range(3):
            g, features, labels, train_mask, test_mask, in_feat, out_feat = load_graph(bin,0.8)
            with g.local_scope():
                num_edges2drop = int(g.num_edges()*dropout)
                edges2drop = [randrange(g.num_edges()) for _ in range(num_edges2drop)]
                g.remove_edges(torch.tensor(edges2drop))

                nx_g = dgl.to_networkx(g, node_attrs=['feat'], edge_attrs=['droppable', 'deg'])
                gx = nx.Graph(nx_g)
                ccs = nx.number_connected_components(gx)
                iso = len(list(nx.isolates(gx)))

            all_ccs.append(ccs)
            all_iso.append(iso)

        ccs_mean = mean(all_ccs)
        ccs_stdv = stdev(all_ccs)
        row_ccs.append(ccs_mean)

        iso_mean = mean(all_iso)
        iso_stdv = stdev(all_iso)
        row_iso.append(iso_mean)

        plot_source.append([bin, dropout, "connected components", ccs_mean, ccs_stdv])
        plot_source.append([bin, dropout, "isolated nodes", iso_mean, iso_stdv])

    
    results_ccs.append([bin, *row_ccs])
    results_iso.append([bin, *row_iso])
#%%
df_iso = pd.DataFrame(data=results_iso, columns=["dataset",*[str(a) for a in dropouts]])
df_ccs = pd.DataFrame(data=results_ccs, columns=["dataset",*[str(a) for a in dropouts]])
# %%
# Forward Pass
# num_edges2drop = int(g.num_edges()*dropout)
# edges2drop = [randrange(g.num_edges()) for _ in range(num_edges2drop)]

# droppable = [idx for idx,i in enumerate(g.edata['droppable'].tolist()) if i==1]
# num_edges2drop = int(len(droppable)*dropout)
# edges2drop = random.sample(droppable, num_edges2drop)

# # %%
# g.remove_edges(torch.tensor(edges2drop))
#%%
df = pd.DataFrame(data=plot_source, columns=["dataset", "dropout", "type", "mean", "stdev"])
df.to_csv("isolate_stats.csv", index=False)

#%%
df = pd.read_csv("isolate_stats.csv")
df.head()

#%%
import seaborn as sns

sns.lineplot('dropout', 'mean', 
             hue='Dataset',marker='o', data=df)
# %%
