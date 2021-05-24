
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
from blocks.GCN import GCNBlock

class Net(nn.Module):
    def __init__(
        self, 
        graph, 
        in_feat, 
        h_feat, 
        out_feat, 
        n_blocks,
        edropout = 0.2,
        edropout_type = 0, 
        edropout_lo = 0.002, 
        edropout_hi = 0.09, 
        ndropout = 0.2,
        norm = True
        ):
        super(Net, self).__init__()
        self.in_feat = in_feat
        self.h_feat = h_feat
        self.out_feat = out_feat
        self.edropout = edropout
        self.edropout_type = edropout_type
        self.edropout_hi = edropout_hi
        self.edropout_lo = edropout_lo
        self.ndropout = ndropout
        self.norm = norm

        self.n_blocks = n_blocks
        self.g = graph
        self.layers = nn.ModuleList()

        for idx in range(self.n_blocks-1):
            if (idx==0):
                self.layers.append(GCNBlock(graph=self.g, in_feat=self.in_feat, out_feat=self.h_feat, edropout=self.edropout ,edropout_type=self.edropout_type, edropout_lo=self.edropout_lo, edropout_hi=self.edropout_hi, ndropout=self.ndropout, norm=self.norm))
            else:
                self.layers.append(GCNBlock(graph=self.g, in_feat=self.h_feat, out_feat=self.h_feat, edropout=self.edropout, edropout_type=self.edropout_type, edropout_lo=self.edropout_lo, edropout_hi=self.edropout_hi, ndropout=self.ndropout, norm=self.norm))
        self.layers.append(GCNBlock(graph=self.g, in_feat=self.h_feat, out_feat=self.out_feat, edropout=self.edropout, edropout_type=self.edropout_type, edropout_lo=self.edropout_lo, edropout_hi=self.edropout_hi, ndropout=self.ndropout, norm=self.norm))
        self.fc = nn.Linear(self.out_feat, self.out_feat)
    
    def forward(self, g, features):
        x = features
        for lyr in self.layers:
            with g.local_scope():
                x = lyr(g, x)
        x = self.fc(x)
        return x