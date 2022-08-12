#--------------------------------
## IMPORTS
#--------------------------------
import sys
import os

#from torch_geometric.nn import DiffusionConv
from torch_geometric.nn import BatchNorm,DiffGroupNorm
from torch_geometric.nn import GraphNorm,GraphSizeNorm,InstanceNorm

#activation functions
from torch.nn import LeakyReLU, Sigmoid, SELU, CELU, SiLU, ReLU, Softmax, ELU, RReLU, Tanh, ReLU6

#convolutional layers
from torch_geometric.nn.conv import AGNNConv, ARMAConv, ChebConv
#from dgl.nn.pytorch.conv import APPNPConv

# current GCN
this_GCN = os.path.split(os.path.split(os.path.realpath(__file__))[0])[1]

# set the path to find the modules
sys.path.insert(0, '../005_src/') #use relative path

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
        #self.conv1 = APPNPConv(num_input_features, hc_1)
        
        self.conv2 = GraphConv(hc_1, num_output_features)
        #self.conv2 = APPNPConv(hc_1, num_output_features)
        
        self.bn1 = torch.nn.BatchNorm1d(hc_1)
        
        #self.leakyrelu = LeakyReLU()
        self.leakyrelu = RReLU()

    def forward(self, x, edge_index, edge_attr):
        if edge_attr is not None: 
            x = self.conv1(x, edge_index, edge_attr)
            x = self.bn1(x)
            x = self.leakyrelu(x)
            
            
            x = self.conv2(x, edge_index, edge_attr)
        else:
            x = self.conv1(x, edge_index)
            x = self.bn1(x)
            x = x.leakyrelu()
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
        
        #for gcn
        #self.conv1 = GraphConv(num_input_features, hc_1)
        #self.conv2 = GraphConv(hc_1, hc_2)
        #self.conv3 = GraphConv(hc_2, num_output_features)
        
        #self.conv1 = ChebConv(num_input_features, hc_1, 5)
        #self.conv2 = ChebConv(hc_1, hc_2, 5)
        #self.conv3 = ChebConv(hc_2, num_output_features, 5)
        
        #for mlp
        self.conv1 = Linear(num_input_features, hc_1)
        self.conv2 = Linear(hc_1, hc_2)
        self.conv3 = Linear(hc_2, num_output_features)
        
        self.bn1 = torch.nn.BatchNorm1d(hc_1)
        self.bn2 = torch.nn.BatchNorm1d(hc_2)
        
        #change for testing different activation functions
        self.relu = LeakyReLU()
        #self.relu = SELU()

    def forward(self, x, edge_index, edge_attr):
        if edge_attr is not None: 
            
            #for gcn
            #x = self.conv1(x, edge_index, edge_attr)
            #x = self.bn1(x)
            #x = x.relu()
            #x = self.conv2(x, edge_index, edge_attr)
            #x = self.bn2(x)
            #x = x.relu()
            #x = self.conv3(x, edge_index, edge_attr)  
            
            #for linear
            x = self.conv1(x)
            x = self.bn1(x)
            x = x.relu()
            x = self.conv2(x)
            x = self.bn2(x)
            x = x.relu()
            x = self.conv3(x)
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
                 hc_1 = 4,
                 hc_2 = 8,
                 hc_3 = 2,
                 random_seed =42,
                 printstat = True,
                ):
        super(GCN_HL03_bn_relu, self).__init__()
        torch.manual_seed(random_seed)
        self.printstat = printstat
        
        #for GCN
        #self.conv1 = GraphConv(num_input_features, hc_1)
        #self.conv2 = GraphConv(hc_1, hc_2)
        #self.conv3 = GraphConv(hc_2, hc_3)
        #self.conv4 = GraphConv(hc_3, num_output_features)
        
        #for Linear
        self.conv1 = Linear(num_input_features, hc_1)
        self.conv2 = Linear(hc_1, hc_2)
        self.conv3 = Linear(hc_2, hc_3)
        self.conv4 = Linear(hc_3, num_output_features)
           
        self.bn1 = torch.nn.BatchNorm1d(hc_1)
        self.bn2 = torch.nn.BatchNorm1d(hc_2)
        self.bn3 = torch.nn.BatchNorm1d(hc_3)
        
        self.relu = LeakyReLU()

    def forward(self, x, edge_index, edge_attr):
        printstat = self.printstat
        if edge_attr is not None:
            
            #for GCN
            #x = self.conv1(x, edge_index, edge_attr)
            #x = self.bn1(x)
            #x = x.relu()
            #x = self.conv2(x, edge_index, edge_attr)
            #x = self.bn2(x)
            #x = x.relu()
            #x = self.conv3(x, edge_index, edge_attr)
            #x = self.bn3(x)
            #x = x.relu()
            #x = self.conv4(x, edge_index, edge_attr) 
            
            #for Linear
            x = self.conv1(x)
            x = self.bn1(x)
            x = x.relu()
            x = self.conv2(x)
            x = self.bn2(x)
            x = x.relu()
            x = self.conv3(x)
            x = self.bn3(x)
            x = x.relu()
            x = self.conv4(x) 
            
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
        self.bn1 = torch.nn.BatchNorm1d(hc_1)
        self.bn2 = torch.nn.BatchNorm1d(hc_2)
        self.bn3 = torch.nn.BatchNorm1d(hc_3)

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

