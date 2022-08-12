import os
original_wd = os.getcwd()

os.chdir("/storage/remote/atcremers50/ss21_multiagentcontrol/005_src/")
#print (os.getcwd())


#--------------------------------
# functions
#--------------------------------
from _01_functions.step1_select_input_data import *
from _01_functions.step2_add_edges_information import *
from _01_functions.step3_build_graph import *
from _01_functions.step4_get_gcn_input_with_TL import * 

os.chdir(original_wd)

