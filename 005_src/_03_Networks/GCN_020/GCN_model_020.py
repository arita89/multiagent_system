#--------------------------------
## IMPORTS
#--------------------------------
import sys
import os

#from torch_geometric.nn import DiffusionConv
from torch_geometric.nn import BatchNorm,DiffGroupNorm
from torch_geometric.nn import GraphNorm,GraphSizeNorm,InstanceNorm

# current GCN
this_GCN = os.path.split(os.path.split(os.path.realpath(__file__))[0])[1]

# set the path to find the modules
sys.path.insert(0, '../005_src/') #use relative path
#os.chdir("../005_src")

from config import *
import inspect 

#--------------------------------
## CLASSES
#--------------------------------
class HL01_MLP(torch.nn.Module):
    def __init__(self,
                 num_input_features,
                 num_output_features,
                 hc_1 = 512,
                 activation_function = "relu",
                 random_seed =42,
                 printstat = True,
                ):
        super(HL01_MLP, self).__init__()
        torch.manual_seed(random_seed)
        self.printstat = printstat
        self.actfun = choose_activation_function(activation_function)
        
        self.lin1 = Linear(num_input_features, hc_1)
        self.lin2 = Linear(hc_1, num_output_features)        


    def forward(self, x, edge_index, edge_attr):
        printstat = self.printstat
        #if edge_attr is not None:
        if x is not None: 
            x = self.lin1(x)
            x = self.actfun(x)
            x = self.lin2(x)
            
        return x
    
class HL03_MLP(torch.nn.Module):
    def __init__(self,
                 num_input_features,
                 num_output_features,
                 hc_1 = 128,
                 hc_2 = 256,
                 hc_3 = 64,
                 activation_function = "relu",
                 random_seed =42,
                 printstat = True,
                ):
        super(HL03_MLP, self).__init__()
        torch.manual_seed(random_seed)
        self.printstat = printstat
        self.actfun = choose_activation_function(activation_function)
        
        self.lin1 = Linear(num_input_features, hc_1)
        self.lin2 = Linear(hc_1, hc_2)
        self.lin3 = Linear(hc_2, hc_3)
        self.lin4 = Linear(hc_3, num_output_features)


    def forward(self, x, edge_index, edge_attr):
        printstat = self.printstat
        #if edge_attr is not None:
        if x is not None: 
            x = self.lin1(x)
            x = self.actfun(x)
            x = self.lin2(x)
            x = self.actfun(x)
            x = self.lin3(x)
            x = self.actfun(x)
            x = self.lin4(x)
            
        return x

    
    
def check_import():
    print (f"imported models: {this_GCN} at {get_timestamp()}")
    print (f"{edges_attr=}")    
    
all_classes_inthismodule = inspect.getmembers(sys.modules[__name__], 
                   lambda member: inspect.isclass(member) and member.__module__ == __name__ )

print (f"at {get_timestamp()} imported models:")
print ([a for a,b in all_classes_inthismodule], sep = "\n")

