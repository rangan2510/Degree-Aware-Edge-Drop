#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
from blocks.layers.dropout import Edropout
from blocks.layers.GCNLayer import GCNLayer

#%%
class GCNBlock(nn.Module):
    def __init__(self, graph, in_feat, out_feat, edropout, edropout_type, edropout_lo, edropout_hi, ndropout, norm):
        super(GCNBlock, self).__init__()
        self.graph = graph
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.edropout = edropout
        self.edropout_type = edropout_type
        self.edropout_lo = edropout_lo
        self.edropout_hi = edropout_hi
        self.ndropout = ndropout
        self.norm = norm
        self.conv = GCNLayer(self.in_feat, self.out_feat)
        self.edropout = Edropout(self.graph, edropout_type = self.edropout_type, edropout_lo = self.edropout_lo, edropout_hi = self.edropout_hi, dropout=self.edropout)
        self.relu = nn.ReLU()
        self.bn = torch.nn.BatchNorm1d(self.out_feat)
    
    def forward(self, graph, features):
        x = features
        g = graph
        g = self.edropout(g)
        x = self.conv(g, x)
        if self.norm:
            x = self.bn(x)
        x = F.dropout(x, self.ndropout, training=self.training)
        
        return x