#--------------------------------
## IMPORTS
#--------------------------------
import sys
import os

# set the path to find the modules
sys.path.insert(0, '../005_src/') #use relative path
os.chdir("/storage/remote/atcremers50/ss21_multiagentcontrol/005_src")

#from config_with_TL import *
from _50_config import *
#--------------------------------
## Step 4 - Get GCN Input
#--------------------------------
# ref https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html

#def step4_get_gcn_input_mod(df, 
def step4_get_gcn_input_with_TL(df, 
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
            
    def structure_columns(df):
    """
    bringing the traffic light information into the right shape    
    """
        
        #get TL_str column
        status = []
        for value in df["nextTrafficLight"]:
            f_rep=value.replace("),)", "")
            s_rep=f_rep.replace("(", "")
            t_rep=s_rep.replace(")", "")
            v_rep=t_rep.replace("'", "")
            split_list=v_rep.split(", ")
            if len(split_list)==1:
                status.append('None')
            else:
                extracted=split_list[3]
                status.append(extracted)

        df["TL_str"] = status 
        
        #get TL int column
        status_number = []
        for value in df["TL_str"]:
            if value=='None':
                status_number.append(3)

            elif value == 'G':
                status_number.append(2)

            elif value == 'y':
                status_number.append(1)

            elif value == 'r':
                status_number.append(0)

        df["TL_int"] = status_number 
        
        #get duration column 
        duration_until_next_status_change = []
        amount_rows=df.shape[0]
        dummy_list=[1] * amount_rows
        df["total_duration_until_next_status"] = dummy_list 

        grouped = df.groupby('vehID')
        for idx, one_veh_df in grouped:

            one_veh_df= one_veh_df.sort_values(by=['vehID','timestep'])
            amount_of_status_changes=(one_veh_df['time_until_next_switch'] == 0).sum()
            indexes_with_a_change=one_veh_df.time_until_next_switch[one_veh_df.time_until_next_switch == 0].index.tolist()
            list_duration_for_red=[]
            list_phase_duration=[]
            state=False
            for value in one_veh_df['index']:
                if one_veh_df['TL_str'][value]=='G':
                    dur_green=one_veh_df['time_until_next_switch'][value]
                    df["total_duration_until_next_status"][value]= dur_green

                elif one_veh_df['TL_str'][value]=='y':
                    dur_yellow=one_veh_df['time_until_next_switch'][value]
                    df["total_duration_until_next_status"][value]= dur_yellow

                elif one_veh_df['TL_str'][value]=='None':
                    df["total_duration_until_next_status"][value]= 40    

                elif one_veh_df['TL_str'][value]=='r':
                    list_duration_for_red.append(one_veh_df['time_until_next_switch'][value])
                    list_phase_duration.append(one_veh_df['phase_duration'][value])
                    state= True

            if state: 
                ind_zero=[]
                length_of = len(list_phase_duration)
                try:        
                    ind_zero = [i for i, x in enumerate(list_duration_for_red) if x == 0]
                except:
                    pass

                #for 2 zeros in array 
                if len(ind_zero)==2:
                    #ending with it
                    if ind_zero[1]==length_of-1:
                        first_val=list_duration_for_red[0]
                        second_val=list_phase_duration[ind_zero[1]]
                        total=first_val+second_val
                    #not ending with it
                    else:
                        first_val=list_duration_for_red[0]
                        second_val=list_phase_duration[ind_zero[1]]
                        third_val=list_duration_for_red[ind_zero[1]+1]+1
                        #third_val=len(list_phase_duration)-ind_zero[1]
                        total=first_val+second_val+third_val

                #for 3 zeros in  array 
                if len(ind_zero)==3:
                    #ending with it
                    if ind_zero[2]==length_of-1:
                        first_val=list_duration_for_red[0]
                        second_val=list_phase_duration[ind_zero[1]]
                        third_val=list_phase_duration[ind_zero[2]]
                        total=first_val+second_val+third_val
                    #not ending with it
                    else:
                        first_val=list_duration_for_red[0]
                        second_val=list_phase_duration[ind_zero[1]]
                        third_val=list_phase_duration[ind_zero[2]]
                        forth_val=list_duration_for_red[ind_zero[2]+1]+1
                        #print(forth_val)
                        #forth_val=len(list_phase_duration)-ind_zero[2]
                        total=first_val+second_val+third_val+forth_val

                #for 4 zeros in array --> ending after that  
                if len(ind_zero)==4: 
                    first_val=list_duration_for_red[0]
                    second_val=list_phase_duration[ind_zero[1]]
                    third_val=list_phase_duration[ind_zero[2]]
                    forth_val=list_phase_duration[ind_zero[3]]
                    total=first_val+second_val+third_val+forth_val    

                #for 1 oder 2 zeros in array
                if len(ind_zero)==1 or len(ind_zero)==0:
                    for value2 in one_veh_df['index']:
                        if one_veh_df['TL_str'][value2]=='r':
                            dur_red=one_veh_df['time_until_next_switch'][value2]
                            df["total_duration_until_next_status"][value2]= dur_red

                else:
                    what_to=[]
                    k=0
                    for k in range(int(total)):
                        what_to.append(total-k)

                    what_to.append(0.0)
                    counter=0
                    for value2 in one_veh_df['index']:
                        if one_veh_df['TL_str'][value2]=='r':
                            df["total_duration_until_next_status"][value2]=what_to[counter]
                            counter=counter+1
           
        return df
    
    
    index_col=[]
    m=0
    for m in range(int(df.shape[0])):
        index_col.append(m)
    df["index"] = index_col 
    
    df=structure_columns(df)
    
    
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
        
        printif(timestep, printstat)
        
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
        #print (df_features)

        # get the vehicles with edges in this timestep
        unique_veh_a = np.unique(df_edges[['veh_a']].values).tolist()
        unique_veh_b = np.unique(df_edges[['veh_b']].values).tolist()
        unique_veh = list(set(unique_veh_a+unique_veh_b))

        printif(f"there are {len(unique_veh)} unique vehicles in timestep {timestep}" , printstat and timestep <= 15) 
        printif(f" {unique_veh=} " , printstat and timestep <= 15) 
        
        # list of strings "30","31" etc
        unique_veh_future = list(df_features_target.vehID.unique())
        # list of integers
        unique_veh_future = [int(veh) for veh in unique_veh_future]
        
        # vehicles that are in both timeframes
        # unique_veh = [veh for veh in unique_veh if veh in unique_veh_future]
        printif(f"{len(unique_veh_future)} are in timestep {timestep}+{predict_after_timesteps}" , printstat and timestep <= 15) 

        # initialize lists 
        data_x = []
        data_pos = []
        data_y = []
        #data_tl = []
        
        # for each unique vehicle  in the present timestep extract the features and the positions
        # we have a known max number of vehicles per each timestep (M) 
        # we need to have fixed 1 to M naming for the vehicles at each time step
        # eg in timestep 75 there are vehicles 34,35
        # 34,35 --> 0,1
        # so that the edges [[34][35]] --> [[0][1]] can refer to the right tensor line later on!
        # note the edges indexes can start from zero
        # the input to the Network will be a matrix of fixed size [Mxfeatures]
        # vehicles not present have zero features
        rename_veh_dict = {}
        
        for i,veh in enumerate(sorted(unique_veh)):
            #printif(veh,printstat and timestep <=25)
            
            # creating the dict to rename the edges in a ordered manner
            rename_veh_dict[veh]= i
           
            df_veh = mask_veh(df_features,veh)
            df_future_veh = mask_veh(df_features_target,veh)
            try:
                data_x.append(df_veh[['yaw','speed','intention', 'TL_int', 'total_duration_until_next_status']].values.tolist()[0])
                
                
            except Exception as e:
                # this in reality shouldnt happen
                printif (f"vehicle {veh} is not in the sim",printstat and timestep <= 15)
                #continue
                
            ## append to data_pos
            print(df_veh.head(5))
            
            data_pos.append(df_veh[['X','Y']].values.tolist()[0])
            
            ## append to data_y
            try:
                #data_y_pos.append(df_future_veh[['X','Y']].values.tolist()[0])
                #data_y_yaw.append(df_future_veh[['yaw']].values.tolist()[0])
                #data_y.append(df_future_veh[['X','Y']].values.tolist()[0])
                data_y.append(df_future_veh[['X','Y', 'yaw']].values.tolist()[0])
            except Exception as e:
                printif (f"vehicle {veh} is not in the sim anymore",printstat and timestep <= 15)
                continue

            
        printif (f"{rename_veh_dict=}", printstat and timestep <= 15)

        # get all the edges from the edges df
        data_edges = [df_edges.veh_a.tolist(),df_edges.veh_b.tolist()]
        printif(data_edges, printstat and timestep <= 15)
        unique_veh_from_edges = list(set(np.unique(df_edges.veh_a.tolist())+np.unique(df_edges.veh_b.tolist())))
        printif(unique_veh_from_edges, printstat and timestep <= 15)
       
        # to get the weights we need to use the original veh_a and veh_b names
        # eg 34,35 
        data_edges_attr =  [dict_edges_weights[(f"{veh_pair[0]}",f"{veh_pair[1]}")] 
                           for veh_pair in list(zip(data_edges[0],data_edges[1]))]
        
        # then we edit the data_edges, following the previously build dictionary
        # eg rename_veh_dict[34] = 0
        renamed_starting_nodes = [rename_veh_dict[veh] for veh in df_edges.veh_a.tolist()]
        renamed_ending_nodes = [rename_veh_dict[veh] for veh in df_edges.veh_b.tolist()]
        data_edges_renamed = [renamed_starting_nodes,renamed_ending_nodes]


        data_dict[timestep]= {"data_x":data_x,
                              #"data_tl":data_tl,
                              "data_pos":data_pos,
                              "data_edges":data_edges,
                              "data_edges_renamed":data_edges_renamed,# keeping both for now for visual check!
                              "data_y":data_y,
                              "data_edges_attr": data_edges_attr,
                             }

    return data_dict
    
