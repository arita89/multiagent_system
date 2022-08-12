import os
original_wd = os.getcwd()

os.chdir("/storage/remote/atcremers50/ss21_multiagentcontrol/005_src/")
#print (os.getcwd())

#--------------------------------
# configuration
#--------------------------------

from _00_configuration._00_settings import *
from _00_configuration._01_variables import *
#from _00_configuration._02_GCN_parameters import *

#--------------------------------
# functions
#--------------------------------

from _01_functions._00_helper_functions import *
from _01_functions._01_functions_dataframes import *
from _01_functions._02_functions_xml import *
from _01_functions._03_functions_graph import *
from _01_functions._04_functions_GCN import *
from _01_functions._06_functions_data_adjustments import *

from _01_functions._08_functions_plots_GCN_018 import *

from _01_functions.step1_select_input_data import *
from _01_functions.step2_add_edges_information import *
from _01_functions.step3_build_graph import *
from _01_functions.step4_get_gcn_input_mod import * 
#from _01_functions.step4_01_get_gcn_input_with_TL import * 

os.chdir(original_wd)

## print statements
print ("-"*40)
print(f"root directory: {ROOT}")
print(f"input directory: {DATA_FOLDER}")
print(f"output directory: {OUTPUT_DIR}")
print ("-"*40)
print ("")
