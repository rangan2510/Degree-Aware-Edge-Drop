###############################################################################
# Configuration
RUNS = 1
DROPTYPE = 2
DROPOUT = 0.1
EPOCHS = 50
# DATASET
from dgl.data import CoraFull
DAT = CoraFull()
IN_FEAT = 8710
H_FEAT = 256
OUT_FEAT = 70
N_BLOCKS = 2

###############################################################################
#Imports

import random
import dgl
import dgl.function as fn
import torch as th
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.nn import GraphConv
import time
import numpy as np
import psutil
from statistics import mean
from random import randrange
import random
from tqdm import tqdm
###############################################################################
# For reproducibility.

dgl.seed(0)
dgl.random.seed(0)

###############################################################################
# We load the dataset using DGL's built-in data module.

def load_data(ds, split = 0.8):
    from dgl.data import CoraFullDataset
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
    return g, feat, label, train_mask, test_mask

g, features, labels, train_mask, test_mask = load_data(DAT)
###############################################################################


class DA_EdgeDropout(nn.Module):
    def __init__(self, graph, dropout = 0.2, hi_cutoff = 0.9, lo_cutoff = 0.02):
        super(DA_EdgeDropout, self).__init__()
        self.dropout = dropout
        self.lo_cutoff = lo_cutoff

        # get the graph
        self.g = graph.local_var()
        
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
    
    def forward(self, g):
        g = g.local_var()
        to_drop_idx = random.choices(self.droppable_idx, k=self.num_to_drop)
        g.remove_edges(torch.tensor(to_drop_idx))
        g = dgl.add_self_loop(g)
        return g

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


class Net(nn.Module):
    def __init__(self, graph, in_feat, h_feat, out_feat, n_blocks, ed=0, edge_dropout = 0.2):
        super(Net, self).__init__()
        self.in_feat = in_feat
        self.h_feat = h_feat
        self.out_feat = out_feat
        self.ed  = None
        self.n_blocks = n_blocks
        self.g = graph
        self.edrop = edge_dropout
        self.layers = nn.ModuleList()
        self.bn = torch.nn.BatchNorm1d(self.h_feat)
        for idx in range(self.n_blocks):
            if (idx==0):
                self.layers.append(GraphConv(self.in_feat, self.h_feat)) # first lyr
            elif (idx==self.n_blocks-1):
                self.layers.append(GraphConv(self.h_feat, self.h_feat)) # last lyr
            else:
                self.layers.append(GraphConv(self.h_feat, self.h_feat)) # mid lyrs
        if (ed==0):
            print("Using No Edge Dropout.")
            #no edgedrop
        elif (ed==1):
            print("Using Naive Random Edge Dropout.")
            self.ed = RandomEdgeDropout(dropout = self.edrop)
        else:
            print("Using Degree Aware Random Edge Dropout.")
            self.ed = DA_EdgeDropout(self.g, dropout = 0.2)
        self.fc = nn.Linear(self.h_feat, self.out_feat)
    
    def forward(self, g, features):
        _g = g # make backup
        x = features
        for lyr in self.layers:
            x = lyr(_g, x)
            x = F.relu(x)
            if (self.ed!=None):
                _g = self.ed(_g)
        # x = F.dropout(x, 0.2, training=self.training)
        # x = self.bn(x)
        x = self.fc(x)
        return x
###############################################################################
# Eval

def evaluate(model, g, features, labels, mask):
    model.eval()
    with th.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = th.max(logits, dim=1)
        correct = th.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

###############################################################################
# Train
print("CPU cores:", psutil.cpu_count())

g = dgl.add_self_loop(g)

for run in range(RUNS):
    RUN_ID = run+1
    print('-' ,RUN_ID, '-'*50)
    
    net = Net(g, IN_FEAT, H_FEAT, OUT_FEAT, N_BLOCKS, DROPTYPE, DROPOUT)
    optimizer = th.optim.Adam(net.parameters(), lr=1e-2)
    dur = []
        
    best_score = 0
    best_epoch = 0
    for epoch in range(EPOCHS):
        if epoch >=3:
            t0 = time.time()

        net.train()
        logits = net(g, features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[train_mask], labels[train_mask])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch >=3:
            dur.append(time.time() - t0)
        
        acc = evaluate(net, g, features, labels, test_mask)
        print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(
                epoch, loss.item(), acc, np.mean(dur)))
        if best_score < acc:
            best_score = acc
            best_epoch = epoch

    print("Best Test Acc {:.4f} at Epoch {:05d}".format(best_score, best_epoch))
    savefile = str(DROPTYPE) + "_" + str(RUN_ID) + "_" + str(N_BLOCKS) + "_" + str(int(best_score*10**5)) + '.weights'
    torch.save(net.state_dict(),savefile)
    del net
