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

class GCN_HL01_bn_relu(torch.nn.Module):
    def __init__(self,
                 num_input_features,
                 num_output_features,
                 hc_1 = 128,
                 random_seed =42,
                 printstat = True, 
                ):
        super(GCN_HL01_bn_relu, self).__init__()
        torch.manual_seed(random_seed)
        self.conv1 = GraphConv(num_input_features, hc_1)
        self.conv2 = GraphConv(hc_1, num_output_features)
        self.bn1 = torch.nn.BatchNorm1d(hc_1)

    def forward(self, x, edge_index, edge_attr):
        if edge_attr is not None: 
            x = self.conv1(x, edge_index, edge_attr)
            x = self.bn1(x)
            x = x.relu()
            x = self.conv2(x, edge_index, edge_attr)
        else:
            x = self.conv1(x, edge_index)
            x = self.bn1(x)
            x = x.relu()
            x = self.conv2(x, edge_index)
        return x
    
class GCN_HL02_bn_relu(torch.nn.Module):
    def __init__(self,
                 num_input_features,
                 num_output_features,
                 hc_1 = 128,
                 hc_2 = 256,
                 random_seed =42,
                 printstat = True, 
                ):
        super(GCN_HL02_bn_relu, self).__init__()
        torch.manual_seed(random_seed)
        self.conv1 = GraphConv(num_input_features, hc_1)
        self.conv2 = GraphConv(hc_1, hc_2)
        self.conv3 = GraphConv(hc_2, num_output_features)
        self.bn1 = torch.nn.BatchNorm1d(hc_1)
        self.bn2 = torch.nn.BatchNorm1d(hc_2)

    def forward(self, x, edge_index, edge_attr):
        if edge_attr is not None: 
            x = self.conv1(x, edge_index, edge_attr)
            x = self.bn1(x)
            x = x.relu()
            x = self.conv2(x, edge_index, edge_attr)
            x = self.bn2(x)
            x = x.relu()
            x = self.conv3(x, edge_index, edge_attr)  
        else:
            x = self.conv1(x, edge_index)
            x = self.bn1(x)
            x = x.relu()
            x = self.conv2(x, edge_index)
            x = self.bn2(x)
            x = x.relu()
            x = self.conv3(x, edge_index)
        return x 
    
class GCN_HL03_bn_relu(torch.nn.Module):
    def __init__(self,
                 num_input_features,
                 num_output_features,
                 hc_1 = 128,
                 hc_2 = 256,
                 hc_3 = 64,
                 random_seed =42,
                 printstat = True,
                ):
        super(GCN_HL03_bn_relu, self).__init__()
        torch.manual_seed(random_seed)
        self.printstat = printstat
        self.conv1 = GraphConv(num_input_features, hc_1)
        self.conv2 = GraphConv(hc_1, hc_2)
        self.conv3 = GraphConv(hc_2, hc_3)
        self.conv4 = GraphConv(hc_3, num_output_features)
        self.bn1 = GraphNorm(hc_1)
        self.bn2 = GraphNorm(hc_2)
        self.bn3 = GraphNorm(hc_3)

    def forward(self, x, edge_index, edge_attr):
        printstat = self.printstat
        if edge_attr is not None:
            
            printif(f"\n INPUT {x}",printstat)
            x = self.conv1(x, edge_index, edge_attr)
            
            printif(f"\n AFTER CONV1 {x}",printstat)
            x = self.bn1(x)
            
            printif(f"\n AFTER BN1 {x}",printstat)
            x = x.relu()
            
            printif(f"\n AFTER RELU {x}",printstat)
            x = self.conv2(x, edge_index, edge_attr)
            
            printif(f"\n AFTER CONV2 {x}",printstat)
            x = self.bn2(x)
            
            printif(f"\n AFTER BN2 {x}",printstat)
            x = x.relu()
            
            printif(f"\n AFTER RELU {x}",printstat)
            x = self.conv3(x, edge_index, edge_attr)
            
            printif(f"\n AFTER CONV3 {x}",printstat)
            x = self.bn3(x)
            
            printif(f"\n AFTER BN3 {x}",printstat)
            x = x.relu()
            
            printif(f"\n AFTER RELU {x}",printstat)
            x = self.conv4(x, edge_index, edge_attr)   
        else:
            x = self.conv1(x, edge_index)
            x = self.bn1(x)
            x = x.relu()
            x = self.conv2(x, edge_index)
            x = self.bn2(x)
            x = x.relu()
            x = self.conv3(x, edge_index)
            x = self.bn3(x)
            x = x.relu()
            x = self.conv4(x, edge_index)
        
        printif(f"\n OUTPUT {x}",printstat)
        #pdb.set_trace()
        return x
    
class GCN_HL01_bn_tanh(torch.nn.Module):
    def __init__(self,
                 num_input_features,
                 num_output_features,
                 hc_1 = 128,
                 random_seed = 42,
                 printstat = True, 
                ):
        super(GCN_HL01_bn_tanh, self).__init__()
        torch.manual_seed(random_seed)
        self.conv1 = GraphConv(num_input_features, hc_1)
        self.conv2 = GraphConv(hc_1, num_output_features)
        self.bn1 = torch.nn.BatchNorm1d(hc_1)

    def forward(self, x, edge_index, edge_attr):
        if edge_attr is not None: 
            x = self.conv1(x, edge_index, edge_attr)
            x = self.bn1(x)
            x = x.tanh()
            x = self.conv2(x, edge_index, edge_attr)
        else:
            x = self.conv1(x, edge_index)
            x = self.bn1(x)
            x = x.tanh()
            x = self.conv2(x, edge_index)
        return x
    
class GCN_HL02_bn_tanh(torch.nn.Module):
    def __init__(self,
                 num_input_features,
                 num_output_features,
                 hc_1 = 128,
                 hc_2 = 256,
                 random_seed =42,
                 printstat = True,                  
                ):
        super(GCN_HL02_bn_tanh, self).__init__()
        torch.manual_seed(random_seed)
        self.conv1 = GraphConv(num_input_features, hc_1)
        self.conv2 = GraphConv(hc_1, hc_2)
        self.conv3 = GraphConv(hc_2, num_output_features)
        self.bn1 = torch.nn.BatchNorm1d(hc_1)
        self.bn2 = torch.nn.BatchNorm1d(hc_2)

    def forward(self, x, edge_index, edge_attr):
        if edge_attr is not None: 
            x = self.conv1(x, edge_index, edge_attr)
            x = self.bn1(x)
            x = x.tanh()
            x = self.conv2(x, edge_index, edge_attr)
            x = self.bn2(x)
            x = x.tanh()
            x = self.conv3(x, edge_index, edge_attr)  
        else:
            x = self.conv1(x, edge_index)
            x = self.bn1(x)
            x = x.tanh()
            x = self.conv2(x, edge_index)
            x = self.bn2(x)
            x = x.tanh()
            x = self.conv3(x, edge_index)
        return x 
    
class GCN_HL03_bn_tanh(torch.nn.Module):
    def __init__(self,
                 num_input_features,
                 num_output_features,
                 hc_1 = 128,
                 hc_2 = 256,
                 hc_3 = 64,
                 random_seed =42,
                 printstat = True, 
                ):
        super(GCN_HL03_bn_tanh, self).__init__()
        torch.manual_seed(random_seed)
        self.conv1 = GraphConv(num_input_features, hc_1)
        self.conv2 = GraphConv(hc_1, hc_2)
        self.conv3 = GraphConv(hc_2, hc_3)
        self.conv4 = GraphConv(hc_3, num_output_features)
        self.bn1 = GraphNorm(hc_1)
        self.bn2 = GraphNorm(hc_2)
        self.bn3 = GraphNorm(hc_3)

    def forward(self, x, edge_index, edge_attr):
        if edge_attr is not None: 
            x = self.conv1(x, edge_index, edge_attr)
            x = self.bn1(x)
            x = x.tanh()
            x = self.conv2(x, edge_index, edge_attr)
            x = self.bn2(x)
            x = x.tanh()
            x = self.conv3(x, edge_index, edge_attr)
            x = self.bn3(x)
            x = x.tanh()
            x = self.conv4(x, edge_index, edge_attr)   
        else:
            x = self.conv1(x, edge_index)
            x = self.bn1(x)
            x = x.tanh()
            x = self.conv2(x, edge_index)
            x = self.bn2(x)
            x = x.tanh()
            x = self.conv3(x, edge_index)
            x = self.bn3(x)
            x = x.tanh()
            x = self.conv4(x, edge_index)   
        return x
    
def check_import():
    print (f"imported models: {this_GCN} at {get_timestamp()}")
    print (f"{edges_attr=}")    
    
all_classes_inthismodule = inspect.getmembers(sys.modules[__name__], 
                   lambda member: inspect.isclass(member) and member.__module__ == __name__ )

print (f"at {get_timestamp()} imported models:")
print ([a for a,b in all_classes_inthismodule], sep = "\n")

