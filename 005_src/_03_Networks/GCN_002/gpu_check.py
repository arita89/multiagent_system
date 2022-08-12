#--------------------------------
## IMPORTS
#--------------------------------
import sys
import os

# set the path to find the modules
sys.path.insert(0, '../../../../005_src') #use relative path
os.chdir("../../../../005_src")
print (os.listdir())
from config import *

device = cudaOverview()