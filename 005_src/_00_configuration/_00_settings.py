# NOTE run from cd <pathtodir>/project/
import os

ROOT = "/storage/remote/atcremers50/ss21_multiagentcontrol/"

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pdb
import networkx as nx
from matplotlib.pyplot import figure
import time

import numpy as np
from numpy.matlib import repmat
from skimage import io
import json
import pdb
import datetime
import open3d as o3d
# import time
import copy
from scipy import ndimage
#import cv2
from tqdm import tqdm
import string
import re
import os, glob, subprocess, sys, socket, struct, random, math, argparse, logging, time
import numpy as np
import pandas as pd
import seaborn as sns

from math import hypot
import itertools
from itertools import combinations, product
from statistics import mean
import pdb

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

#from torch_geometric.utils import to_networkx,add_self_loops, degree

# https://graphneural.network/layers/convolution/
#shortly commented out
#from torch_geometric.nn import AGNNConv,APPNPConv,ARMAConv,ChebConv,CrystalConv
from torch_geometric.nn import GATConv#DiffusionConv,GATConv
from torch_geometric.nn import GCNConv,GraphConv

from torch.nn import Linear,LazyLinear, Conv2d, BatchNorm2d, MaxPool2d, Dropout2d
#from torch.nn.functional import relu, elu, relu6,selu,celu, sigmoid, tanh, softmax, leaky_relu, rrelu,silu, linear
#from torch_geometric.nn import DiffusionConv
from torch_geometric.nn import BatchNorm,DiffGroupNorm
from torch_geometric.nn import GraphNorm,GraphSizeNorm,InstanceNorm

#activation functions
from torch.nn import LeakyReLU, Sigmoid, SELU, CELU, SiLU, ReLU, Softmax, ELU, RReLU, Tanh, ReLU6
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

from IPython.display import Javascript  # Restrict height of output cell.
from IPython.display import Image
from IPython.core.display import HTML 

from pathlib import Path
import pickle as pkl
print ("-"*40)
print ("Packages import successful")

