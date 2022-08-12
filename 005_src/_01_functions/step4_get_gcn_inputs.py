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
## Step 4 - Get GCN Input
#--------------------------------
# ref https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html

def step4_get_gcn_input(df, 
                        df_with_edge_info, 
                        dict_edges_per_frame,
                        predict_after_timesteps = 5,
                        printstat = True):
    """
    - df contains the features extracted from the sumo sim, per timestep, per vehicle
    - df_with_edges contains the edge information per timestep, per pair of vehicles, is there an edge (True or False)
    - dict_edges_per_frame contains the edges weights, for all edges in the timestep
    """
    def unique_veha_vehb(df_edges):
            unique_veh_a = np.unique(df_edges[['veh_a']].values).tolist()
            unique_veh_b = np.unique(df_edges[['veh_b']].values).tolist()
            unique_veh = list(set(unique_veh_a+unique_veh_b))
            return unique_veh
        
    # correct time format 
    df = change_columns_format_to_numeric(df,"timestep")
    df_with_edge_info = change_columns_format_to_numeric(df_with_edge_info,"veh_a")
    df_with_edge_info = change_columns_format_to_numeric(df_with_edge_info,"veh_b")

    # select only rows where edge == True
    mask = (df_with_edge_info.edge == True)
    df_with_edges = df_with_edge_info[mask]
    
    # this is true if the above mask doesnt eliminate any row
    printif(f"all vehicles have edges: {len(df_with_edges) == len(df_with_edge_info)}",printstat)

    # dictionary of data
    data_dict = {}

    # overview 
    unique_ts = sorted(df_with_edges.timestep.unique())
    reduced_ts = unique_ts[:-predict_after_timesteps]
    printif (f"there are {len(unique_ts)} unique timesteps in which there are edges", printstat)
    printif (f"starting with {unique_ts[0]} and finishing with {unique_ts[-1]}", printstat)
    printif (f"we are trying to predict {predict_after_timesteps} ahead, therefore we will use only {len(reduced_ts)} timesteps", printstat)
    printif (f"starting with {reduced_ts[0]} and finishing with {reduced_ts[-1]}", printstat)
    
    #for timestep in tqdm(sorted(df_with_edges.timestep.unique())[:-predict_after_timesteps]):
    # per each time step
    for timestep in tqdm(reduced_ts):

        # filter the big datasets, taking only the timestep of interest
        df_features = df[df.timestep ==  timestep]
        df_edges = df_with_edges[df_with_edges.timestep == timestep]
        df_features_target = df[df.timestep ==  timestep+predict_after_timesteps]
        
        ## overview for debuggin
        if printstat and timestep <=25 :
            print ()
            print (f"{timestep} =====================")
            print (f"DF_FEATURES\n")
            print(df_features.head(10))
            print()
            print (f"DF_EDGES\n")
            print(df_edges.head(10))
            print()
            print (f"DF_FEATURES_TARGET\n")
            print(df_features_target.head(10))
            print()
        
        # for the edges weights we have a dictionary timestep: list of edges with weights
        # https://stackoverflow.com/questions/62936573/pytorch-geometric-data-object-edge-attr-for-undirected-graphs
        dict_edges_weights = dict_edges_per_frame[timestep]
        
        # get the vehicles with edges in this timestep
        unique_veh_a = np.unique(df_edges[['veh_a']].values).tolist()
        unique_veh_b = np.unique(df_edges[['veh_b']].values).tolist()
        unique_veh = list(set(unique_veh_a+unique_veh_b))

        printif(f"there are {len(unique_veh)} unique vehicles in timestep {timestep}" , printstat) 
       
        # list of strings "30","31" etc
        unique_veh_future = list(df_features_target.vehID.unique())
        # list of integers
        unique_veh_future = [int(veh) for veh in unique_veh_future]
        
        # vehicles that are in both timeframes
        # unique_veh = [veh for veh in unique_veh if veh in unique_veh_future]
        printif(f"{len(unique_veh_future)} are in timestep {timestep}+{predict_after_timesteps}" , printstat) 

        # for each unique vehicle extract the features and the positions
        data_x = []
        data_pos = []
        data_y = []
        for veh in unique_veh:
            #printif(veh,printstat and timestep <=25)
            df_veh = mask_veh(df_features,veh)
            df_future_veh = mask_veh(df_features_target,veh)
            
            ## append to data_x
            try:
                data_x.append(df_veh[['yaw','speed','intention']].values.tolist()[0])
            except Exception as e:
                # this in reality shouldnt happen
                printif (f"vehicle {veh} is not in the sim",printstat)
                continue
                
            ## append to data_pos
            data_pos.append(df_veh[['X','Y']].values.tolist()[0])
            
            ## append to data_y
            try:
                data_y.append(df_future_veh[['X','Y','yaw']].values.tolist()[0])
            except Exception as e:
                # maybe n timesteps later the car is not there anymore!
                printif (f"vehicle {veh} is not in the sim anymore",printstat)
                continue

        # get all the edges from the edges df
        data_edges = [df_edges.veh_a.tolist(),df_edges.veh_b.tolist()]
        data_edges_attr =  [dict_edges_weights[(f"{veh_pair[0]}",f"{veh_pair[1]}")] 
                           for veh_pair in list(zip(data_edges[0],data_edges[1]))]

        data_dict[timestep]= {"data_x":data_x,
                              "data_pos":data_pos,
                              "data_edges":data_edges,
                              "data_y":data_y,
                              "data_edges_attr": data_edges_attr,
                             }

    return data_dict
    
