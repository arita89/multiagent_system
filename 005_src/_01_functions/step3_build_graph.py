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
from _50_config import *
#--------------------------------
## Step 3 - Build Graph
#--------------------------------

def step3_build_graph(df_with_edges, 
                      edge_creation_radius, 
                      edge_maintenance_radius, 
                      SAVE_TEMP, 
                      date, 
                      ts, 
                      delete_tempFiles, 
                      minl, 
                      printstat, 
                      savestat,
                      plotgraphstat,
                      opt,
                      color = "blue"
                     ):
    
    #edit the timestep
    df_with_edges["timestep"] = pd.to_numeric(df_with_edges["timestep"])
    df_with_edges.sort_values("timestep")


    # build the graph, store the pictures, 
    # return dictionary per timestep with edge list = [(from,to),(),..(from, to)]
    dict_edges_per_frame,titleGif = build_graph(df_with_edges,
                                                ec = edge_creation_radius,
                                                em = edge_maintenance_radius,
                                                plotgraphstat = plotgraphstat,
                                                SAVE_TEMP= SAVE_TEMP,
                                                date = date,
                                                ts= ts,
                                                savestat = savestat, 
                                                delete_tempFiles= delete_tempFiles,
                                                minl= minl,
                                                opt = opt,
                                                color = color,
                                               )

    printif("building the graph and returing edges", printstat)
    
    if savestat and printstat:
        print (f"graph visualization saved in {titleGif}")
        
    return dict_edges_per_frame,titleGif