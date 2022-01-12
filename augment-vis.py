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
# %%
g, features, labels, train_mask, test_mask, in_feat, out_feat = load_graph('CoraFull.bin',0.8)

# %%
def evaluate(model, g, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        # logits = logits[mask]
        # labels = labels[mask]
        logp = F.log_softmax(logits, 1)
        # test_loss = F.nll_loss(logp[mask], labels[mask])
        test_loss = F.nll_loss(logits[mask], labels[mask])
        # _, indices = torch.max(logits, dim=1)
        # correct = torch.sum(indices == labels)
        # return correct.item() * 1.0 / len(labels)
        return test_loss.item()

# %%
import dgl
import torch
import torch.nn.functional as F
import time
import numpy as np
import psutil
dgl.seed(0)
torch.manual_seed(0)
dgl.random.seed(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

IN_FEAT = in_feat
H_FEAT = 128
OUT_FEAT = out_feat

BACKBONE = 'gcn' #for logging only
# %%
CONFIG = 2
# set config
if CONFIG == 0: #no ed
    EDROP_DA = False #dont care
    EDROPOUT = 0
elif CONFIG == 1: #random ed
    EDROP_DA = False
    EDROPOUT = 0.2
elif CONFIG == 2: #DAw ed
    EDROP_DA = True
    EDROPOUT = 0.2
else: #default config
    EDROP_DA = True
    EDROPOUT = 0.2
    
NDROPOUT = 0.5
EPOCHS = 200
NORM = False
PRINT_EVERY = 10
# %%
import torch.nn as nn
import random
from random import randrange

class RandomEdgeDropout(nn.Module):
    def __init__(self, da=False, dropout = 0.2, device='cpu'):
        super(RandomEdgeDropout, self).__init__()
        self.dropout = dropout
        self.da = da

    def forward(self, g):
        g = g.local_var()
        g = g.remove_self_loop()
        
        if not self.da:
            num_edges2drop = int(g.num_edges()*self.dropout)
            edges2drop = [randrange(g.num_edges()) for _ in range(num_edges2drop)]
        else:
            droppable = [idx for idx,i in enumerate(g.edata['droppable'].tolist()) if i==1]
            num_edges2drop = int(len(droppable)*self.dropout)
            edges2drop = random.sample(droppable, num_edges2drop)

        
        g.remove_edges(torch.tensor(edges2drop, dtype=torch.int64).to(device))
        g = dgl.add_self_loop(g)
        return g
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv, SAGEConv

class Net(nn.Module):
    def __init__(
        self, 
        graph, 
        in_feat, 
        h_feat, 
        out_feat, 
        n_blocks,
        edropout = 0.2,
        edropout_degAw = True, 
        ndropout = 0.2,
        norm = True, 
        device = 'cuda:0'
        ):
        super(Net, self).__init__()
        self.in_feat = in_feat
        self.h_feat = h_feat
        self.out_feat = out_feat
        
        self.edropout = edropout
        self.edropout_degAw = edropout_degAw
        self.ndropout = ndropout
        
        self.norm = norm
        self.n_blocks = n_blocks
        self.g = graph
        
        self.device = device
        self.layers = nn.ModuleList()
        
        self.ed = RandomEdgeDropout(da=self.edropout_degAw, dropout = self.edropout, device=self.device)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=False)

        
        for idx in range(self.n_blocks-1):
            if (idx==0):
                self.layers.append(GraphConv(self.in_feat, self.h_feat, norm='both', weight=True, bias=True))
                # self.layers.append(SAGEConv(self.in_feat, self.h_feat,aggregator_type='gcn'))                
            else:
                self.layers.append(GraphConv(self.h_feat, self.h_feat, norm='both', weight=True, bias=True))
                # self.layers.append(SAGEConv(self.h_feat, self.h_feat,aggregator_type='gcn'))
        
        self.layers.append(GraphConv(self.h_feat, self.out_feat, norm='both', weight=True, bias=True))
        # self.layers.append(SAGEConv(self.h_feat, self.out_feat,aggregator_type='gcn'))
        
        
        self.fc = nn.Linear(self.out_feat, self.out_feat)
        self.bn = torch.nn.BatchNorm1d(self.out_feat)
    
    def forward(self, g, features):
        x = features
        for lyr in self.layers:
            with g.local_scope():
                if self.edropout != 0:
                    g = self.ed(g)
                x = self.lrelu(lyr(g, x))
                if self.ndropout != None:
                    x = F.dropout(x, self.ndropout, training=self.training)
        x = self.fc(x)
        if self.norm:
            x = self.bn(x)
        return x
# %%
g = dgl.add_self_loop(g)
g = g.to(device)
features = features.to(device)
labels = labels.to(device)

N_BLOCK = 8
best_scores =[]

net = Net(g, IN_FEAT, H_FEAT, OUT_FEAT, N_BLOCK, EDROPOUT, EDROP_DA, NDROPOUT, NORM, device).to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
dur = []



#%%
log = []
for epoch in range(EPOCHS):
    if epoch >=3:
        t0 = time.time()

    net.train()

    logits = net(g, features)

    logp = F.log_softmax(logits, 1)
    loss = F.nll_loss(logp[train_mask], labels[train_mask])
    

    optimizer.zero_grad()
    
    # acc = evaluate(net, g, features, labels, test_mask)
    # loss += acc

    loss.backward()
    optimizer.step()

    if epoch >=3:
        dur.append(time.time() - t0)


    acc = evaluate(net, g, features, labels, test_mask)
    if epoch%PRINT_EVERY==0:
        print("Epoch {:05d} | Train Loss {:.4f} | Test Loss {:.4f} | Time(s) {:.4f}".format(epoch, loss.item(), acc, np.mean(dur)))
        row = [epoch,loss.item(), acc]
        log.append(row)
    



# %%
import pandas as pd
log_df = pd.DataFrame(data=log, columns=["epoch", "train", "test"])

#%%
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
scaled_array = min_max_scaler.fit_transform(log_df[['train']])
log_df['train'] = scaled_array

min_max_scaler = preprocessing.MinMaxScaler()
scaled_array = min_max_scaler.fit_transform(log_df[['test']])
log_df['test'] = scaled_array

#%%
import seaborn as sns
import matplotlib.pyplot as plt

sns.lineplot(x='epoch', y='train', 
             marker='o', data=log_df)

sns.lineplot(x='epoch', y='test', 
             marker='o', data=log_df,)

plt.legend(labels=["train", "test"])
# %%
