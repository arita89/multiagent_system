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
    
class HL01_bn_class(torch.nn.Module):
    def __init__(self,
                 num_input_features,
                 num_output_features,
                 hc_1 = 512,
                 activation_function = "relu",
                 layer_type = "GraphConv",
                 normalization = "GraphNorm",
                 random_seed =42,
                 printstat = True,
                ):
        super(HL01_bn_class, self).__init__()
        torch.manual_seed(random_seed)
        self.printstat = printstat
        self.actfun = choose_activation_function(activation_function)
        self.outputfun = Softmax()
        
        if layer_type == "GCN":
            self.conv1 = GCNConv(num_input_features, hc_1)
            self.conv_out = GCNConv(hc_1, num_output_features)
        elif layer_type == "GraphConv":
            self.conv1 = GraphConv(num_input_features, hc_1)
            self.conv_out = GraphConv(hc_1, num_output_features)
        elif layer_type == "GATConv":
            self.conv1 = GATConv(num_input_features, hc_1)
            self.conv_out  = GATConv(hc_1, num_output_features)      
        if normalization == "GraphNorm":
            self.bn1 = GraphNorm(hc_1)
        else:
            print (f"{normalization} not implemented")

    def forward(self, x, edge_index, edge_attr):
        printstat = self.printstat
        #if edge_attr is not None:
        if x is not None: 
            x = self.conv1(x, edge_index, edge_attr)
            x = self.bn1(x)
            x = self.actfun(x)
            x = self.conv_out(x, edge_index, edge_attr)
            x = self.outputfun(x)
            
        return x
    
class HL01_bn_regress(torch.nn.Module):
    def __init__(self,
                 num_input_features,
                 num_output_features,
                 hc_1 = 512,
                 activation_function = "relu",
                 layer_type = "GraphConv",
                 normalization = "GraphNorm",
                 random_seed =42,
                 printstat = True,
                ):
        super(HL01_bn_regress, self).__init__()
        torch.manual_seed(random_seed)
        self.printstat = printstat
        self.actfun = choose_activation_function(activation_function)
        
        if layer_type == "GCN":
            self.conv1 = GCNConv(num_input_features, hc_1)
            self.conv_out = GCNConv(hc_1, num_output_features)
        elif layer_type == "GraphConv":
            self.conv1 = GraphConv(num_input_features, hc_1)
            self.conv_out = GraphConv(hc_1, num_output_features)
        elif layer_type == "GATConv":
            self.conv1 = GATConv(num_input_features, hc_1)
            self.conv_out  = GATConv(hc_1, num_output_features)      
        if normalization == "GraphNorm":
            self.bn1 = GraphNorm(hc_1)
        else:
            print (f"{normalization} not implemented")

    def forward(self, x, edge_index, edge_attr):
        printstat = self.printstat
        #if edge_attr is not None:
        if x is not None: 
            x = self.conv1(x, edge_index, edge_attr)
            x = self.bn1(x)
            x = self.actfun(x)
            x = self.conv_out(x, edge_index, edge_attr)
            
        return x    
    
def check_import():
    print (f"imported models: {this_GCN} at {get_timestamp()}")
    print (f"{edges_attr=}")    
    
all_classes_inthismodule = inspect.getmembers(sys.modules[__name__], 
                   lambda member: inspect.isclass(member) and member.__module__ == __name__ )

print (f"at {get_timestamp()} imported models:")
print ([a for a,b in all_classes_inthismodule], sep = "\n")

