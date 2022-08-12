#--------------------------------
## IMPORTS
#--------------------------------
import sys
import os

printstat = True

# set the path to find the modules
sys.path.insert(0, '../005_src/') #use relative path
os.chdir("/storage/remote/atcremers50/ss21_multiagentcontrol/005_src")

#from config import *
#from _01_functions._00_helper_functions import *
from _50_config import *
#--------------------------------
## Step 2 - Add Edge Information
#--------------------------------

def step2_add_edges_information(df, edge_creation_radius, edge_maintenance_radius, printstat):

    # get one df with distances per each time step
    #printif(df.head(5), printstat)
    printif(f"computing mutual distances between vehicle per time step" , printstat) 
    df_list = get_df_per_timestep(df, 
                                  edge_creation_radius = edge_creation_radius,
                                  edge_maintenance_radius = edge_maintenance_radius
                                )
    df_alltimestep_all_distances = pd.concat(df_list)

    # one big dataframe out of the list of dataframes
    df_complete = df_alltimestep_all_distances.copy(deep=True)
    
    # in this case it is just fully connected... 
    if edge_creation_radius > 100 and edge_maintenance_radius > 100:
        edges_list = [True for e in list(df_complete.veh_a)]
        df_with_edges = df_complete.copy(deep=True)
        df_with_edges["edge"] = edges_list
    
    # this one takes long long long time (O(n**n)... avoid)
    else: 
        df_with_edges = add_edge_info_per_df(df_complete)
    print (df_with_edges.head(5))
    #pdb.set_trace()

    printif(f"added edge information given ec={edge_creation_radius} and em= {edge_maintenance_radius}" , printstat) 
    
    return df_with_edges
