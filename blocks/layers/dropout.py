import dgl
import torch
import torch.nn as nn
from tqdm import tqdm
import random
from random import randrange

class DA_EdgeDropout(nn.Module):
    def __init__(self, graph, hi_cutoff = 0.9, lo_cutoff = 0.02, dropout=0.2):
        super(DA_EdgeDropout, self).__init__()
        self.lo_cutoff = lo_cutoff
        self.dropout = dropout


        # get the graph
        self.g = graph
        
        # set the number of edges to drop
        self.num_to_drop = int(self.g.num_edges()* self.dropout)

        # get and store degrees as node features
        d = self.g.in_degrees(self.g.nodes()).float()
        d = (d - min(d))/(max(d) - min(d))
        self.g.ndata['deg'] = d.unsqueeze(1)

        # apply node features to edges
        self.g.apply_edges(lambda edges: {'deg' : torch.min(edges.src['deg'],edges.dst['deg'])})
        self.g = self.g.remove_self_loop()
        self.hi_cutoff = max(self.g.edata['deg']) * hi_cutoff
        self.lo_cutoff = lo_cutoff * min(self.g.edata['deg']) if (min(self.g.edata['deg']) > 0) else lo_cutoff
        
        # get a list of edges that may be dropped
        self.droppable_idx = []
        print("Checking graph structure for droppable edges")
        for i in tqdm(range(self.g.num_edges())):
            if (self.g.edata['deg'][i] > self.lo_cutoff) and ((self.g.edata['deg'][i] <= self.hi_cutoff)):
                self.droppable_idx.append(i)
        print("Marked",len(self.droppable_idx),"edges.")
        print("Will drop", self.num_to_drop,"edges")
    
    def forward(self, g):
        to_drop_idx = random.choices(self.droppable_idx, k=self.num_to_drop)

        # create a subgraph
        g = g.remove_self_loop()
        sg_mask = torch.BoolTensor(g.num_edges())
        #print(g.num_edges())
        sg_mask[:] = True
        for idx in to_drop_idx:
            sg_mask[idx] = not sg_mask[idx]
        

        sg = dgl.edge_subgraph(g,sg_mask,preserve_nodes=True)

        g = g.add_self_loop()
        sg = sg.add_self_loop()
        #print(sg.num_edges())
        return sg

###############################################################################

class RandomEdgeDropout(nn.Module):
    def __init__(self, dropout = 0.2):
        super(RandomEdgeDropout, self).__init__()
        self.dropout = dropout

    def forward(self, g):
        g = g.local_var()
        g = g.remove_self_loop()
        num_edges2drop = int(g.num_edges()*self.dropout)
        edges2drop = [randrange(g.num_edges()) for _ in range(num_edges2drop)]
        g.remove_edges(torch.tensor(edges2drop))
        g = dgl.add_self_loop(g)
        return g

###############################################################################

class Edropout(nn.Module):
    def __init__(self, graph, edropout_type, edropout_lo, edropout_hi, dropout=0.2):
        super(Edropout, self).__init__()
        self.graph = graph
        self.edropout_type = edropout_type
        self.edropout = None
        if (edropout_type == 0):
            # do nothing
            pass
        elif (edropout_type == 1):
            self.edropout = RandomEdgeDropout(dropout=dropout)
        else: #edropout_type == 2
            self.edropout = DA_EdgeDropout(graph, hi_cutoff=edropout_hi, lo_cutoff=edropout_lo, dropout=dropout)
        
    def forward(self, g):
        if (self.edropout_type == 0):
            return g
        else:
            return self.edropout(g)

