import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv

class Residual(nn.Module):
    def __init__(self, in_feat, out_feat, aggr_mode="linear"):
        super(Residual, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.aggr_mode = aggr_mode
        self.conv = GraphConv(self.in_feat, self.out_feat, norm='both', weight=True, bias=True)

        if aggr_mode=="linear":
            self.aggr = nn.Linear(self.out_feat*2, self.out_feat)
        if aggr_mode=="gcn":
            self.aggr = GraphConv(self.out_feat*2, self.out_feat, norm='both', weight=True, bias=True)
        if aggr_mode=="sum":
            pass #do nothing here, handle in forward pass
        if aggr_mode=="max":
            pass #just let it pass bro

    def forward(self, graph, features):
        x = x_skip = features
        x = self.conv(graph, x)
        if self.aggr_mode=="linear":
            x = torch.cat([x,x_skip],dim=1)
            x = self.aggr(x)
        if self.aggr_mode=="gcn":
            x = torch.cat([x,x_skip],dim=1)
            x = self.aggr(graph,x)
        if self.aggr_mode=="sum":
            x += x_skip
        if self.aggr_mode=="max":
            x = torch.max(x,x_skip)
        return x


class JumpingKnowledge(nn.Module): 
    def __init__(self, in_feat, out_feat):
        super(JumpingKnowledge, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.conv = GraphConv(self.in_feat, self.out_feat, norm='both', weight=True, bias=True)

    def forward(self, graph, features):
        features_transformed = self.conv(graph, features)
        return features_transformed, features


class JumpingKnowledgeAggregator(nn.Module):
    def __init__(self, in_feat, num_layers, out_feat, aggr_mode="linear"):
        super(JumpingKnowledgeAggregator, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_layers = num_layers
        self.aggr_mode = aggr_mode

        if aggr_mode=="linear":
            self.aggr = nn.Linear(self.in_feat*self.num_layers, self.out_feat)

    def forward(self, feat_list):
        if self.aggr_mode=="linear":
            feat = torch.cat(feat_list,dim=1)
            feat = self.aggr(feat)
        if self.aggr_mode=="max":
            feat = torch.max(feat_list)
        if self.aggr_mode=="sum":
            feat = torch.sum(torch.stack(feat_list, dim=0),dim=0)
        return feat










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
        
        self.ed = RandomEdgeDropout(da=True, dropout = self.edropout, device=self.device)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=False)

        for idx in range(self.n_blocks-1):
            if (idx==0):
                self.layers.append(GraphConv(self.in_feat, self.h_feat, norm='both', weight=True, bias=True))
            else:
                self.layers.append(GraphConv(self.h_feat, self.h_feat, norm='both', weight=True, bias=True))
        self.layers.append(GraphConv(self.h_feat, self.out_feat, norm='both', weight=True, bias=True))
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