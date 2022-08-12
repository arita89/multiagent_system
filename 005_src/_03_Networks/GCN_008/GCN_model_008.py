#--------------------------------
## IMPORTS
#--------------------------------
import sys
import os

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
                 random_seed =42,
                 
                ):
        super(GCN, self).__init__()
        torch.manual_seed(random_seed)
        self.conv1 = GCNConv(num_input_features, hc_1)
        self.conv2 = GCNConv(hc_1, num_output_features)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        return x

def check_import():
    print (f"imported model: {this_GCN} at {get_timestamp()}")
    print (f"{edges_attr=}")