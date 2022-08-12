#--------------------------------
## IMPORTS
#--------------------------------
import sys
import os

# current GCN
this_GCN = os.path.split(os.path.split(os.path.realpath(__file__))[0])[1]
edges_attr = True

# set the path to find the modules
sys.path.insert(0, '../005_src/') #use relative path
#os.chdir("../005_src")

from config import *
import inspect
#print (inspect.getmembers(_03_Networks.GCN_011.GCN_model_011))

class GCN_HL01(torch.nn.Module):
    def __init__(self,
                 num_input_features,
                 num_output_features,
                 hc_1 = 128,
                 random_seed =42,
                 
                ):
        super(GCN_HL01, self).__init__()
        torch.manual_seed(random_seed)
        self.conv1 = GCNConv(num_input_features, hc_1)
        self.conv2 = GCNConv(hc_1, num_output_features)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr)
        return x

class GCN_HL02(torch.nn.Module):
    def __init__(self,
                 num_input_features,                 
                 num_output_features,
                 hc_1 = 128,
                 hc_2 = 128,
                 random_seed =42,
                 
                ):
        super(GCN_HL02, self).__init__()
        torch.manual_seed(random_seed)
        self.conv1 = GCNConv(num_input_features, hc_1)
        self.conv2 = GCNConv(hc_1, hc_2)
        self.conv3 = GCNConv(hc_2, num_output_features)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_attr)
        return x
    
class GCN_HL03(torch.nn.Module):
    def __init__(self,
                 num_input_features,
                 num_output_features,
                 hc_1 = 128,
                 hc_2 = 256,
                 hc_3 = 64,
                 random_seed =42,
                 
                ):
        super(GCN_HL03, self).__init__()
        torch.manual_seed(random_seed)
        self.conv1 = GCNConv(num_input_features, hc_1)
        self.conv2 = GCNConv(hc_1, hc_2)
        self.conv3 = GCNConv(hc_2, hc_3)
        self.conv4 = GCNConv(hc_3, num_output_features)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv4(x, edge_index, edge_attr)
        return x    

def check_import():
    print (f"imported models: {this_GCN} at {get_timestamp()}")
    print (f"{edges_attr=}")    
    
all_classes_inthismodule = inspect.getmembers(sys.modules[__name__], 
                   lambda member: inspect.isclass(member) and member.__module__ == __name__ )

print (f"at {get_timestamp()} imported models:")
print ([a for a,b in all_classes_inthismodule], sep = "\n")
print (f"{edges_attr=}") 
