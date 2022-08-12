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
## Step 3 - Build Graph
#--------------------------------

def step3_build_graph(df_with_edges, edge_creation_radius, edge_maintenance_radius, SAVE_TEMP, date, ts, delete_tempFiles, minl, printstat, savestat):
    
    #edit the timestep
    df_with_edges["timestep"] = pd.to_numeric(df_with_edges["timestep"])
    df_with_edges.sort_values("timestep")


    # build the graph, store the pictures, 
    # return dictionary per timestep with edge list = [(from,to),(),..(from, to)]
    dict_edges_per_frame,titleGif = build_graph(df_with_edges,
                                        ec = edge_creation_radius,
                                        em = edge_maintenance_radius,
                                        SAVE_TEMP= SAVE_TEMP,
                                        date = date,
                                        ts= ts,
                                        savestat = False, 
                                        delete_tempFiles= delete_tempFiles,
                                        minl= minl)

    printif("building the graph and returing edges", printstat)
    if savestat:
        print (f"graph visualization saved in {titleGif}")
        
