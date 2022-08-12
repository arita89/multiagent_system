#--------------------------------
## IMPORTS
#--------------------------------
import sys
import os
#from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn.norm import LayerNorm, BatchNorm, InstanceNorm, GraphNorm, GraphSizeNorm, PairNorm, MessageNorm, DiffGroupNorm

# current GCN
this_GCN = os.path.split(os.path.split(os.path.realpath(__file__))[0])[1]
edges_attr = False

# set the path to find the modules
sys.path.insert(0, '../005_src/') #use relative path
#os.chdir("../005_src")

from config import *

class GCN(torch.nn.Module):
    def __init__(self,
                 num_input_features,
                 hc_1,
                 num_output_features,
                 hc_2 = 256,
                 #hc_3 = 128,
                 random_seed =42,
                 
                ):
        super(GCN, self).__init__()
        torch.manual_seed(random_seed)
        self.conv1 = GCNConv(num_input_features, hc_1)
        self.conv2 = GCNConv(hc_1, hc_2)
        self.conv3 = GCNConv(hc_2, num_output_features)
        self.bn1 = torch.nn.BatchNorm1d(hc_1)
         
        self.layernorm1 = LayerNorm(256) 
        self.layernorm2 = LayerNorm(256)
        #self.layernorm1 = GraphSizeNorm() 
        #self.layernorm2 = GraphSizeNorm() 
        #print('test batch')

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        #print('after conv1')
        #print(x)
        x = self.layernorm1(x)
        #x = self.layernorm1(x, 32)
        #print('after norm layer')
        #print(x)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr)
        x = self.layernorm2(x)
        #x = self.layernorm2(x, 32)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_attr)
        return x

def check_import():
    print (f"imported model: {this_GCN} at {get_timestamp()}")
    print (f"{edges_attr=}")