# NOTE run from cd <pathtodir>/project/
import os

# EDIT THIS 
ROOT = "/storage/remote/atcremers50/ss21_multiagentcontrol/"


# sklearn
from sklearn.externals._pilutil import bytescale
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage import io
from skimage.transform import resize

# PyTorch libraries and modules
import torch
from torch.nn import Linear
import torch.nn.functional as F

from torch_geometric.utils import to_networkx,add_self_loops, degree
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GraphConv # layer with neighborhood normalization, good for graph classification
from torch_geometric.nn import global_mean_pool, MessagePassing

from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.datasets import KarateClub,Planetoid,TUDataset
from torch_geometric.data import DataLoader, ClusterData, ClusterLoader
from torch_geometric.transforms import SamplePoints
from torch_geometric.data import Data,DataLoader
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing

from torch_cluster import knn_graph
from torch_geometric.nn import global_max_pool

from sklearn.manifold import TSNE


from pathlib import Path
import pickle as pkl
print ("-"*40)
print ("Packages import successful")

