#--------------------------------
## IMPORTS
#--------------------------------
import sys
import os

# set the path to find the modules
sys.path.insert(0, '../src/') #use relative path
os.chdir("/storage/remote/atcremers50/ss21_multiagentcontrol/src")

from config import *

#--------------------------------
## Step 2 - Add Edge Information
#--------------------------------

def step2_add_edges_information(df_test, edge_creation_radius, edge_maintenance_radius, printstat):

    # get one df with distances per each time step
    printif(f"computing mutual distances between vehicle per time step" , printstat) 
    df_list = get_df_per_timestep(df_test, 
                                edge_creation_radius = edge_creation_radius,
                                edge_maintenance_radius = edge_maintenance_radius
                                )

    df_alltimestep_all_distances = pd.concat(df_list)

    # one big dataframe out of the list of dataframes
    df_complete = df_alltimestep_all_distances.copy(deep=True)
    df_with_edges = add_edge_info_per_df(df_complete)

    printif(f"added edge information given ec={edge_creation_radius} and em= {edge_maintenance_radius}" , printstat) 
    
    return df_with_edges