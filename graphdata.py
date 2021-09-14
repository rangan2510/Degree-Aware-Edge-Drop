#%%
import torch
from tqdm import tqdm
from dgl import save_graphs, load_graphs
#%%
def markEdges(graph, hi_cutoff = 0.9, lo_cutoff = 0.2):
    g = graph
    d = g.in_degrees(g.nodes()).float()
    d = (d - min(d))/(max(d) - min(d))
    g.ndata['deg'] = d.unsqueeze(1)

    g.apply_edges(lambda edges: {'deg' : torch.min(edges.src['deg'],edges.dst['deg'])})
    g = g.remove_self_loop()
    hi_cutoff = max(g.edata['deg']) * hi_cutoff
    lo_cutoff = lo_cutoff * min(g.edata['deg']) if (min(g.edata['deg']) > 0) else lo_cutoff
    
    droppable_idx = []
    print("Checking graph structure for droppable edges")
    for i in tqdm(range(g.num_edges())):
        if (g.edata['deg'][i] > lo_cutoff) and ((g.edata['deg'][i] <= hi_cutoff)):
            droppable_idx.append(True)
        else:
            droppable_idx.append(False)
    print("Marked",len(droppable_idx),"edges.")
    g.edata['droppable'] = torch.tensor(droppable_idx).bool()
    return g

def processGraphs(graph_info):
    for t in graph_info:
        filename = t[0]
        print("Processing", filename)
        graph = t[1]
        graph = markEdges(graph)
        save_graphs(filename, graph)

#%%
graph_info = []

#%%
from dgl.data import CiteseerGraphDataset
data = CiteseerGraphDataset()
graph_info.append(('Citeseer.bin',data[0]))
#%%
from dgl.data import CoraFullDataset
data = CoraFullDataset()
graph_info.append(('CoraFull.bin',data[0]))

# %%
from dgl.data import CoauthorCSDataset
data = CoauthorCSDataset()
graph_info.append(('CoauthorCS.bin',data[0]))

#%%
from dgl.data import PubmedGraphDataset
data = PubmedGraphDataset()
graph_info.append(('Pubmed.bin',data[0]))

#%%
from dgl.data import CoauthorPhysicsDataset
data = CoauthorPhysicsDataset()
graph_info.append(('CoauthorPhysics.bin',data[0]))

#%%
processGraphs(graph_info)