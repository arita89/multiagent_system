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
## Step 4 - Get GCN Input
#--------------------------------

def step4_get_gcn_input(df_test, df_with_edges, printstat):
    # correct time format 
    df_test = change_columns_format_to_numeric(df_test,"time")
    df_with_edges = change_columns_format_to_numeric(df_with_edges,"veh_a")
    df_with_edges = change_columns_format_to_numeric(df_with_edges,"veh_b")

    # select only rows where edge == True
    mask = (df_with_edges.edge == True)
    df_with_edges = df_with_edges[mask]

    # dictionary of data
    data_dict = {}

    # per each time step
    for timestep in tqdm(sorted(df_with_edges.timestep.unique())[:-5]):

        # filter the big datasets, taking only the timestep of interest
        df_features = df_test[df_test.time ==  timestep]
        df_edges = df_with_edges[df_with_edges.timestep == timestep]
        df_features_target = df_test[df_test.time ==  timestep+5]

        #print (df_features)

        # get the vehicles with edges in this timestep
        unique_veh_a = np.unique(df_edges[['veh_a']].values).tolist()
        unique_veh_b = np.unique(df_edges[['veh_b']].values).tolist()
        unique_veh = list(set(unique_veh_a+unique_veh_b))
        printif(f"there are {len(unique_veh)} unique vehicles in timestep {timestep}" , printstat) 

        # for each unique vehicle extract the features and the positions
        data_x = []
        data_pos = []
        data_y = []
        for veh in unique_veh:
            df_veh = mask_veh(df_features,veh)
            df_future_veh = mask_veh(df_features_target,veh)
            data_x.append(df_veh[['yaw','speed']].values.tolist()[0])
            data_pos.append(df_veh[['X','Y']].values.tolist()[0])
            try:
                data_y.append(df_future_veh[['X','Y','yaw']].values.tolist()[0])
            except Exception as e:
                # maybe 5 timesteps later the car is not there anymore!
                break

            #print (df_features["(xa,ya)"].values)

        # get all the edges from the edges df
        data_edges = [df_edges.veh_a.tolist(),df_edges.veh_b.tolist()]
        #print (data_edges)

        # convert to tensor!
        data_x = torch.FloatTensor(data_x)
        data_pos = torch.FloatTensor(data_pos)
        data_edges = torch.FloatTensor(data_edges)
        data_y = torch.FloatTensor(data_y) #is target

        data_dict[timestep]= {"data_x":data_x,
                              "data_pos":data_pos,
                              "data_edges":data_edges,
                              "data_y":data_y,
                             }

    return data_dict
    
