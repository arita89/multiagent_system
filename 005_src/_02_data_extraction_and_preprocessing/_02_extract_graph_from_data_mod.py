"""
Name: _01_extract_graph_from_data.py
Time: ------
Desc: loads the data from the sumo simulation, creates the graphs and the correspondent tensors for the network training 

NOTE: requires p37_GCN environment
"""


#--------------------------------
## IMPORTS
#--------------------------------
import sys
import os

# set the path to find the modules
sys.path.insert(0, '../005_src/') #use relative path
os.chdir("/storage/remote/atcremers50/ss21_multiagentcontrol/005_src")

from config import *

#--------------------------------
## DEBUGGING OPTIONS
#--------------------------------
# mark True during dubugging to trigger the prints
printstat = False 

#--------------------------------
## INPUT 
#--------------------------------
# read zipped pandas df 
dataframes_path = os.path.join(DATA_FOLDER ,"dataframes/")

#load_df = 15
print ("which zip df fo you want to load? eg 15")
list_all_zip_files = sorted(glob.glob(os.path.join(dataframes_path ,f"*zip")))
list_all_zip_files_short = [os.path.split(file)[1] for file in list_all_zip_files]
printif(list_all_zip_files, True, 10)
    
load_df = int(input())
l = '%06d' % load_df

## read info from simulation that you want to pass over
data_path_info = os.path.join(dataframes_path,f"df_sim_{l}.txt")
d = read_txt_data(data_path_info)
max_num_veh = int(d['max_num_veh'])

# read data
data_path = os.path.join(dataframes_path,f"df_sim_{l}.zip")
print (f"loading {data_path}")

# env variables   
date = get_date()
ts = get_timestamp()
running_usr = os.getenv('HOME')
running_env = os.getenv('CONDA_PREFIX')

# sim variables 
# ====================================================================================
# !!!! NOTE: to have a fully connected graph, set the edges radii >> 100 !!!
# ====================================================================================
edge_creation_radius = 3500 # if the distance between two vehicles is less than this, and edge is created
edge_maintenance_radius = 7000 # if the distance between two vehicles is less than this, and edge is maintened
print ("how many timesteps do you want to use from the simulation? eg 200,2000,15000")
simulation_max_duration = int(input())
#simulation_max_duration = 5000 # Int or None, 
opt = 3 # edges weights options, for now 1 or 2 or 3
predict_after_timesteps = 2

# saving variables
savestat = True
delete_tempFiles = True
SAVE_TEMP = create_subfolder_with_timestamp(GRAPH_FOLDER) 
minl = 6

#Decide to build the graph for visual inspection
plotgraphstat = True


#--------------------------------
## Step 1 - Get Input Data in Right Format
#--------------------------------
print ("\n>step 1: formatting input data")
df, limit_duration = step1_select_input_data(data_path, 
                                             printstat, 
                                             simulation_max_duration)
printif(df.sort_values(by=['timestep','vehID']),printstat)

#--------------------------------
## Step 2 - Add Edge Information
#--------------------------------
print (">step 2: add edges information")
df_with_edges = step2_add_edges_information(df, 
                                            edge_creation_radius, 
                                            edge_maintenance_radius, 
                                            printstat
                                           )


#--------------------------------
## Step 3 - Build Graph
#--------------------------------
print (">step 3: build graph")
dict_edges_per_frame,titleGif = step3_build_graph(df_with_edges, 
                                                  edge_creation_radius,
                                                  edge_maintenance_radius, 
                                                  SAVE_TEMP, 
                                                  date, 
                                                  ts, 
                                                  delete_tempFiles, 
                                                  minl, 
                                                  printstat, 
                                                  savestat,
                                                  opt = opt,
                                                  plotgraphstat = plotgraphstat,
                                                  color = "blue"
                                                 )

#--------------------------------
## Step 4 - Get GCN Input
#--------------------------------
print (">step 4: Get GCN Input")
data_dict = step4_get_gcn_input_mod(df, 
                                    df_with_edges,
                                    dict_edges_per_frame,
                                    predict_after_timesteps = predict_after_timesteps,
                                    printstat = printstat,                               
                                    )

# store pkl file with GCN input 
gcn_input_file_name = f"{date}{ts}_timesteps{limit_duration}_ec{edge_creation_radius}_em{edge_maintenance_radius}"
file_to_write = open(f"../004_data/GCN_input/{gcn_input_file_name}.pkl", "wb")
pkl.dump(data_dict, file_to_write)

file_to_write.close()

# store GCN input info as txt file
## NOTE: dont leave spaces between words, only when you want to id a new element (key_text: v)
GCN_INPUT_INFO = os.path.join("../004_data/GCN_input/",f"{gcn_input_file_name}.txt")
with open(GCN_INPUT_INFO, 'w') as filehandle:
    filehandle.write(f'max_num_veh: {max_num_veh}\n')
    filehandle.write(f'date: {date}\n')
    filehandle.write(f'time: {ts}\n')
    filehandle.write(f'usr: {running_usr}\n') 
    filehandle.write(f'env: {running_env}\n')
    filehandle.write(f'\n')
    filehandle.write(f'path_input_df: {data_path}\n')
    filehandle.write(f'info_input_df: {data_path_info}\n')    
    filehandle.write(f'\n')
    filehandle.write(f'sim_duration_timesteps: {simulation_max_duration}\n')
    filehandle.write(f'edge_creation_radius: {edge_creation_radius}\n')
    filehandle.write(f'edge_maintenance_radius: {edge_maintenance_radius}\n')
    filehandle.write(f'edge_weights_option: {opt}\n')
    filehandle.write(f'predict_after_timesteps: {predict_after_timesteps}\n')   
    filehandle.write(f'\n')
    filehandle.write(f'savestat: {savestat}\n')
    filehandle.write(f'delete_temporary_files: {delete_tempFiles}\n')
    filehandle.write(f'plotstat: {plotgraphstat}\n')
    filehandle.write(f'\n')
    filehandle.write(f'path_GCN_input: {file_to_write}\n')
    filehandle.write(f'path_GIF: {titleGif}\n')
    filehandle.close()

print (f"GCN input data info stored in {GCN_INPUT_INFO}")

# update the df with the new training set
df_overview = update_df_overview()
print ("updated df_overview")
