import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
torch.set_printoptions(precision=3, sci_mode=False)


class Model(nn.Module): 
  def __init__(self, input_size, hidden_size, output_size): 
    super().__init__()
    self.gcn= GCN (input_size, hidden_size, output_size)
    self.edge_predictor= EdgePredictor()
  
  def forward(self, graphdata): 
    node_representations= self.gcn(graphdata.x, graphdata.edge_index)
    edge_prediction= self.edge_predictor(node_representations, graphdata.edge_index)
    return edge_prediction


  
class EdgePredictor(nn.Module): 
  def forward(self, node_representations, edge_indices):
    edges1, edges2= edge_indices
    score = (node_representations[edges1] * node_representations[edges2]).sum(dim=-1 ) # -> (num node edges, 1)
    return score  
    


class GCN(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super().__init__()
    self.conv1 = GCNConv(input_size, hidden_size)
    self.conv2 = GCNConv(hidden_size, output_size)

  def forward(self, node_features, edge_indices):
    # print('forward')
    # x, edge_index = data.x, data.edge_index

    # x=node_features.to(torch.float) # -> (instances, features) # can do same to model= model.to(float)
    x= node_features
    
    # print(1, x.shape)
    x = self.conv1(x, edge_indices) # -> (instances, hidden_size)
    # print(2, x.shape)
    x = F.relu(x) # -> (instances, hidden_size)  
    # print(3, x.shape)
    x = F.dropout(x, training=self.training) # -> (instances, hidden_size) 
    # print(4, x.shape)
    x = self.conv2(x, edge_indices) # -> (hidden_size, output_size)
    # print(5, x.shape)

    # for node representaton in edge regression 
    return x

    # for node classification 
    # output= F.log_softmax(x, dim=1) # -> (hidden_size, output_size) # log softmax = log(softmax [0,1] sum to 1)
    # return F.log_softmax(x, dim=1)
    