#--------------------------------
## IMPORTS
#--------------------------------
import sys
import os

# current GCN
this_GCN = os.path.split(os.path.split(os.path.realpath(__file__))[0])[1]

# set the path to find the modules
sys.path.insert(0, '../005_src/') #use relative path
#os.chdir("../005_src")

from config import *
import inspect 

leaky_relu = torch.nn.LeakyReLU
    
class GCN_HL01_tanh(torch.nn.Module):
    def __init__(self,
                 num_input_features,
                 num_output_features,
                 hc_1 = 128,
                 random_seed =42,
                 
                ):
        super(GCN_HL01_tanh, self).__init__()
        torch.manual_seed(random_seed)
        self.conv1 = GCNConv(num_input_features, hc_1)
        self.conv2 = GCNConv(hc_1, num_output_features)

    def forward(self, x, edge_index, edge_attr):
        if edge_attr is not None: 
            x = self.conv1(x, edge_index, edge_attr)
            x = x.tanh()
            x = self.conv2(x, edge_index, edge_attr)
        else:
            x = self.conv1(x, edge_index)
            x = x.tanh()
            x = self.conv2(x, edge_index)
        return x

class GCN_HL02_tanh(torch.nn.Module):
    def __init__(self,
                 num_input_features,                 
                 num_output_features,
                 hc_1 = 128,
                 hc_2 = 128,
                 random_seed =42,
                 
                ):
        super(GCN_HL02_tanh, self).__init__()
        torch.manual_seed(random_seed)
        self.conv1 = GCNConv(num_input_features, hc_1)
        self.conv2 = GCNConv(hc_1, hc_2)
        self.conv3 = GCNConv(hc_2, num_output_features)

    def forward(self, x, edge_index, edge_attr):
        if edge_attr is not None: 
            x = self.conv1(x, edge_index, edge_attr)
            x = x.tanh()
            x = self.conv2(x, edge_index, edge_attr)
            x = x.tanh()
            x = self.conv3(x, edge_index, edge_attr)
        else:
            x = self.conv1(x, edge_index)
            x = x.tanh()
            x = self.conv2(x, edge_index)
            x = x.tanh()
            x = self.conv3(x, edge_index)
        return x
    
class GCN_HL03_tanh(torch.nn.Module):
    def __init__(self,
                 num_input_features,
                 num_output_features,
                 hc_1 = 128,
                 hc_2 = 256,
                 hc_3 = 64,
                 random_seed =42,
                 
                ):
        super(GCN_HL03_tanh, self).__init__()
        torch.manual_seed(random_seed)
        self.conv1 = GCNConv(num_input_features, hc_1)
        self.conv2 = GCNConv(hc_1, hc_2)
        self.conv3 = GCNConv(hc_2, hc_3)
        self.conv4 = GCNConv(hc_3, num_output_features)

    def forward(self, x, edge_index, edge_attr):
        if edge_attr is not None: 
            x = self.conv1(x, edge_index, edge_attr)
            x = x.tanh()
            x = self.conv2(x, edge_index, edge_attr)
            x = x.tanh()
            x = self.conv3(x, edge_index, edge_attr)
            x = x.tanh()
            x = self.conv4(x, edge_index, edge_attr)
        else:
            x = self.conv1(x, edge_index)
            x = x.tanh()
            x = self.conv2(x, edge_index)
            x = x.tanh()
            x = self.conv3(x, edge_index)
            x = x.tanh()
            x = self.conv4(x, edge_index)
        return x     
    
class GCN_HL01_relu(torch.nn.Module):
    def __init__(self,
                 num_input_features,
                 num_output_features,
                 hc_1 = 128,
                 random_seed =42,
                 
                ):
        super(GCN_HL01_relu, self).__init__()
        torch.manual_seed(random_seed)
        self.conv1 = GCNConv(num_input_features, hc_1)
        self.conv2 = GCNConv(hc_1, num_output_features)

    def forward(self, x, edge_index, edge_attr):
        if edge_attr is not None: 
            x = self.conv1(x, edge_index, edge_attr)
            x = x.relu()
            x = self.conv2(x, edge_index, edge_attr)
        else:
            x = self.conv1(x, edge_index)
            x = x.relu()
            x = self.conv2(x, edge_index)
        return x

class GCN_HL02_relu(torch.nn.Module):
    def __init__(self,
                 num_input_features,                 
                 num_output_features,
                 hc_1 = 128,
                 hc_2 = 128,
                 random_seed =42,
                 
                ):
        super(GCN_HL02_relu, self).__init__()
        torch.manual_seed(random_seed)
        self.conv1 = GCNConv(num_input_features, hc_1)
        self.conv2 = GCNConv(hc_1, hc_2)
        self.conv3 = GCNConv(hc_2, num_output_features)

    def forward(self, x, edge_index, edge_attr):
        if edge_attr is not None: 
            x = self.conv1(x, edge_index, edge_attr)
            x = x.relu()
            x = self.conv2(x, edge_index, edge_attr)
            x = x.relu()
            x = self.conv3(x, edge_index, edge_attr)
        else:
            x = self.conv1(x, edge_index)
            x = x.relu()
            x = self.conv2(x, edge_index)
            x = x.relu()
            x = self.conv3(x, edge_index)
        return x
    
class GCN_HL03_relu(torch.nn.Module):
    def __init__(self,
                 num_input_features,
                 num_output_features,
                 hc_1 = 128,
                 hc_2 = 256,
                 hc_3 = 64,
                 random_seed =42,
                 
                ):
        super(GCN_HL03_relu, self).__init__()
        torch.manual_seed(random_seed)
        self.conv1 = GCNConv(num_input_features, hc_1)
        self.conv2 = GCNConv(hc_1, hc_2)
        self.conv3 = GCNConv(hc_2, hc_3)
        self.conv4 = GCNConv(hc_3, num_output_features)

    def forward(self, x, edge_index, edge_attr):
        if edge_attr is not None: 
            x = self.conv1(x, edge_index, edge_attr)
            x = x.relu()
            x = self.conv2(x, edge_index, edge_attr)
            x = x.relu()
            x = self.conv3(x, edge_index, edge_attr)
            x = x.relu()
            x = self.conv4(x, edge_index, edge_attr)
        else: 
            x = self.conv1(x, edge_index)
            x = x.relu()
            x = self.conv2(x, edge_index)
            x = x.relu()
            x = self.conv3(x, edge_index)
            x = x.relu()
            x = self.conv4(x, edge_index)
        return x


class GCN_HL01_leaky_relu(torch.nn.Module):
    def __init__(self,
                 num_input_features,
                 num_output_features,
                 hc_1 = 128,
                 random_seed =42,
                 
                ):
        super(GCN_HL01_leaky_relu, self).__init__()
        torch.manual_seed(random_seed)
        self.conv1 = GCNConv(num_input_features, hc_1)
        self.conv2 = GCNConv(hc_1, num_output_features)

    def forward(self, x, edge_index, edge_attr):
        if edge_attr is not None: 
            x = self.conv1(x, edge_index, edge_attr)
            x = x.leaky_relu()
            x = self.conv2(x, edge_index, edge_attr)
        else:
            x = self.conv1(x, edge_index)
            x = x.leaky_relu()
            x = self.conv2(x, edge_index)
        return x

class GCN_HL02_leaky_relu(torch.nn.Module):
    def __init__(self,
                 num_input_features,                 
                 num_output_features,
                 hc_1 = 128,
                 hc_2 = 128,
                 random_seed =42,
                 
                ):
        super(GCN_HL02_leaky_relu, self).__init__()
        torch.manual_seed(random_seed)
        self.conv1 = GCNConv(num_input_features, hc_1)
        self.conv2 = GCNConv(hc_1, hc_2)
        self.conv3 = GCNConv(hc_2, num_output_features)

    def forward(self, x, edge_index, edge_attr):
        if edge_attr is not None: 
            x = self.conv1(x, edge_index, edge_attr)
            x = x.leaky_relu()
            x = self.conv2(x, edge_index, edge_attr)
            x = x.leaky_relu()
            x = self.conv3(x, edge_index, edge_attr)
        else:
            x = self.conv1(x, edge_index)
            x = x.leaky_relu()
            x = self.conv2(x, edge_index)
            x = x.leaky_relu()
            x = self.conv3(x, edge_index)
        return x
    
class GCN_HL03_leaky_relu(torch.nn.Module):
    def __init__(self,
                 num_input_features,
                 num_output_features,
                 hc_1 = 128,
                 hc_2 = 256,
                 hc_3 = 64,
                 random_seed =42,
                 
                ):
        super(GCN_HL03_leaky_relu, self).__init__()
        torch.manual_seed(random_seed)
        self.conv1 = GCNConv(num_input_features, hc_1)
        self.conv2 = GCNConv(hc_1, hc_2)
        self.conv3 = GCNConv(hc_2, hc_3)
        self.conv4 = GCNConv(hc_3, num_output_features)

    def forward(self, x, edge_index, edge_attr):
        if edge_attr is not None: 
            x = self.conv1(x, edge_index, edge_attr)
            x = x.leaky_relu()
            x = self.conv2(x, edge_index, edge_attr)
            x = x.leaky_relu()
            x = self.conv3(x, edge_index, edge_attr)
            x = x.leaky_relu()
            x = self.conv4(x, edge_index, edge_attr)
        else: 
            x = self.conv1(x, edge_index)
            x = x.leaky_relu()
            x = self.conv2(x, edge_index)
            x = x.leaky_relu()
            x = self.conv3(x, edge_index)
            x = x.leaky_relu()
            x = self.conv4(x, edge_index)
        return x       
    
def check_import():
    print (f"imported models: {this_GCN} at {get_timestamp()}")
    print (f"{edges_attr=}")    
    
all_classes_inthismodule = inspect.getmembers(sys.modules[__name__], 
                   lambda member: inspect.isclass(member) and member.__module__ == __name__ )

print (f"at {get_timestamp()} imported models:")
print ([a for a,b in all_classes_inthismodule], sep = "\n")

