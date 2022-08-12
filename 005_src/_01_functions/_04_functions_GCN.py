from _00_configuration._00_settings import *
from _00_configuration._01_variables import *
from _01_functions._00_helper_functions import *
from _01_functions._01_functions_dataframes import *


#--------------------------------
# PARSE THE DATA INPUT TO GCN
#--------------------------------

def check_df_2(df):
    """
    function to check rows coherence
    exact list of vehicles (is it needed?) 
    """
    df["nodes_from_data_x"] = df.apply(lambda row: row.data_x, axis=1)
    df["nodes_from_data_pos"] = df.apply(lambda row: row.data_pos, axis=1)
    df["nodes_from_data_edges"] = df.apply(lambda row: np.unique(row["data_edges"]), axis=1)
    df["num_nodes_from_data_y"] = df.apply(lambda row: row.data_y, axis=1)
    df["training_row"] = df.apply(lambda row: True 
                                  if row['num_nodes_from_data_x']== row['num_nodes_from_data_pos']
                                  and row['num_nodes_from_data_x']== row['num_nodes_from_data_edges']
                                  and row['num_nodes_from_data_x']== row['num_nodes_from_data_y']              
                                  else False, axis=1)
    
    return df



def check_df(df):
    """
    function to check rows coherence
    we want that the nodes in data_x and data_y are all the ones in the edges lists
    per construction we predict in t+deltatt only the state of vehicles present at time t
    so checking the number of veh is enough: new vehicles in t+deltat are not appended to data_y
    """
    df["num_nodes_from_data_x"] = df.apply(lambda row: len(row.data_x), axis=1)
    #df["num_nodes_from_data_x_rad"] = df.apply(lambda row: len(row.data_x_rad), axis=1)
    df["num_nodes_from_data_pos"] = df.apply(lambda row: len(row.data_pos), axis=1)
    df["num_nodes_from_data_edges"] = df.apply(lambda row: len(np.unique(row["data_edges"])), axis=1)
    df["num_nodes_from_data_y"] = df.apply(lambda row: len(row.data_y), axis=1)
    df["training_row"] = df.apply(lambda row: True 
                                  if row['num_nodes_from_data_x']== row['num_nodes_from_data_pos']
                                  and row['num_nodes_from_data_x']== row['num_nodes_from_data_edges']
                                  and row['num_nodes_from_data_x']== row['num_nodes_from_data_y']              
                                  else False, axis=1)
    
    return df


def get_model_size_input_output(df_input,input_columns,output_columns):
    """
    define the input and the output size of the model
    dependin on which columns are chosen as input(can change a lot) and output(2 or 3)
    """
    
    size_input = sum([len(df_input[select_column].iloc[0][0]) for select_column in input_columns])
    size_output = sum([len(df_input[select_column].iloc[0][0]) for select_column in output_columns])
    
    return size_input, size_output

def get_row_for_training(data, drop_col = False, tensor_input = False):
    """
    here casting from tensor first to cleanup
    
    """
    if tensor_input:
        data_numpy = {k:
                    {
                        kk:vv.numpy() 
                        for kk,vv in data[k].items()
                     }
                     for k,v in data.items()
                 }
    else: 
        data_numpy = data
    df_overview = pd.DataFrame(data_numpy).T
    print('---------------------------')
    print(type(df_overview))
    print('---------------------------')
    df = check_df(df_overview)
    if drop_col: 
        df = df.drop(columns=['num_nodes_from_data_x', 
                              'num_nodes_from_data_pos',
                              "num_nodes_from_data_edges",
                              "num_nodes_from_data_y"
                             ])
    return df

def select_rows_from_data(file_name,drop_col = True):
    """
    takes input like "20210703-16h59m22s_timesteps199_ec3500_em7000"
    selects only consistent rows from the df, in which nodes at t1,t2 and in the edges are the same
    returns the shorter df
    """
    file_to_read = open(f"../004_data/GCN_input/{file_name}.pkl", "rb")
    data = pkl.load(file_to_read)
    df = get_row_for_training(data, drop_col)
    mask = (df.training_row == True)
    return df, df[mask]


def choose_activation_function(activation_function, 
                               default = None
                              ):
        
    """
    takes the activation function from pytorch
    """
    if default is None:
        default = ReLU()
    if activation_function == "leakyrelu":
        actfun = LeakyReLU()
    elif activation_function == "relu":
        actfun = ReLU()
    elif activation_function == "elu":
        actfun = ELU()       
    elif activation_function == "rrelu":
        actfun = RReLU()   
    elif activation_function == "sigmoid":
        actfun = Sigmoid()
    elif activation_function == "tanh":
        actfun = Tanh()       
    elif activation_function == "softmax":
        actfun = Softmax()
    else:
        print (f"{activation_function} not implemented, using {default}")
        actfun = default
    return actfun


def choose_optimizer(select_optimizer,
                     model,
                     lr,
                     weight_decay,
                     momentum,
                    ):
    
    """
    optmizer = choose_optimizer(select_optimizer,
                     model,
                     lr,
                     weight_decay,
                     momentum,
                    )
    """
    if select_optimizer == "Adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), lr=lr, weight_decay=weight_decay)
        
    elif select_optimizer == "Adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)
        
    elif select_optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
    elif select_optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), 
                                      lr=lr, 
                                      weight_decay=weight_decay,
                                      amsgrad=False)
        
    elif select_optimizer == "SparseAdam":
        optimizer = torch.optim.SparseAdam(model.parameters(), lr=lr)
        
    elif select_optimizer == "Adamax":
        optimizer = torch.optim.Adamax(model.parameters(), lr=lr, weight_decay=weight_decay)
        
    elif select_optimizer == "ASGD":
        optimizer = torch.optim.ASGD(model.parameters(), 
                                     lr=lr,
                                     lambd=0.0001, alpha=0.75, t0=1000000.0, 
                                     weight_decay=weight_decay)
    elif select_optimizer == "LBFGS":
        print ("NOTE: LBFGS is a very memory intensive optimizer (it requires additional param_bytes * (history_size + 1) bytes). If it doesn?t fit in memory try reducing the history size, or use a different algorithm.")
        optimizer = torch.optim.LBFGS(model.parameters(), 
                                      lr=1, 
                                      max_iter=20, 
                                      max_eval=None, 
                                      tolerance_grad=1e-07, 
                                      tolerance_change=1e-09, 
                                      history_size=100, 
                                      line_search_fn=None)
    elif select_optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        
    elif select_optimizer == "Rprop":
        optimizer = torch.optim.Rprop(model.parameters(), 
                                        lr=lr, 
                                        etas=(0.5, 1.2), 
                                        step_sizes=(1e-06, 50))
        
    elif select_optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        
    else:
        print (f" {select_optimizer} not implemented, using SGD instead")
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        
    return optimizer

## Moving optimizer from CPU to GPU
#https://discuss.pytorch.org/t/moving-optimizer-from-cpu-to-gpu/96068/3
def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)




def choose_scheduler(select_scheduler,
                     optimizer,
                     lr,
                     Nepochs,
                     verbose = False,
                     ):
    
    """
    scheduler = choose_scheduler(select_scheduler,
                                 optimizer,
                                  )
    """
    #https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
    if select_scheduler == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                               mode='min', 
                                                               factor=lr/10, 
                                                               patience=10, 
                                                               threshold=0.0001, 
                                                               threshold_mode='rel', 
                                                               cooldown=0, 
                                                               min_lr=0, 
                                                               eps=1e-08, 
                                                               verbose=verbose)
        
    #https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiStepLR.html#torch.optim.lr_scheduler.MultiStepLR  
    elif select_scheduler == "MultiStepLR" and Nepochs >=10 :
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                         milestones = list(range(int(Nepochs/10),Nepochs,
                                                                                 int(Nepochs/10))), # epochs
                                                         gamma = 0.1, 
                                                         last_epoch=-1, 
                                                         verbose=False)
    #https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
    elif select_scheduler == "CosineAnnealingWarmRestarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
                                                                         T_0 = 10, 
                                                                         T_mult=1, 
                                                                         eta_min=0, 
                                                                         last_epoch=-1, 
                                                                         verbose=verbose)
        
    elif select_scheduler == "None":
        scheduler = None  
    else:
        print (f" {select_scheduler} not implemented, using None")
        scheduler = None
        
    return scheduler



##================================
## CREATING THE DATASET
##================================

def creating_storing_DATASETS(
                                df_input,
                                train_frames,
                                valid_frames,
                                test_frames,
                                shuttle_train,
                                shuttle_val,
                                input_columns,
                                output_columns,
                                MODEL_OUTPUT_PATH,
                                input_file_name,
                                use_edges_renamed = True,
                                printstat = False
                                ):
    
    
    print (f"> CREATING THE DATASET")
    

    dataset_train = Dataset_GCN_3(
                                     df_input,
                                     train_frames,
                                     printstat = printstat,
                                     input_columns = input_columns,
                                     output_columns = output_columns,  
                                     use_edges_renamed = use_edges_renamed,
                                     )
    
    dataset_val = Dataset_GCN_3(
                                 df_input,
                                 valid_frames,
                                 input_columns = input_columns,
                                 output_columns = output_columns,
                                 use_edges_renamed = use_edges_renamed,       
                                 )
    
    dataset_test = Dataset_GCN_3(
                                 df_input,
                                 test_frames,
                                 input_columns = input_columns,
                                 output_columns = output_columns,
                                 use_edges_renamed = use_edges_renamed,
                                 )
    
    dataset_shuttle_train = Dataset_GCN_3(
                                         df_input,
                                         [shuttle_train],
                                         input_columns = input_columns,
                                         output_columns = output_columns,
                                         use_edges_renamed = use_edges_renamed,   
                                         )
    
    dataset_shuttle_val = Dataset_GCN_3(
                                         df_input,
                                         [shuttle_val],
                                         input_columns = input_columns,
                                         output_columns = output_columns,
                                         use_edges_renamed = use_edges_renamed,  
                                         )
    
    # save datasets
    dataset_train_path = os.path.join(MODEL_OUTPUT_PATH,   f'{input_file_name}_{len(train_frames)}_IN{list2string(input_columns)}OUT{list2string(output_columns)}_dataset_01_train.pt')
    torch.save(dataset_train,dataset_train_path)

    dataset_val_path = os.path.join(MODEL_OUTPUT_PATH,   f'{input_file_name}_{len(train_frames)}_IN{list2string(input_columns)}OUT{list2string(output_columns)}_dataset_02_validation.pt')
    torch.save(dataset_val,dataset_val_path)
        
    dataset_test_path = os.path.join(MODEL_OUTPUT_PATH,
    f'{input_file_name}_{len(train_frames)}_IN{list2string(input_columns)}OUT{list2string(output_columns)}_dataset_03_test.pt')
    torch.save(dataset_test,dataset_test_path)
    
    dataset_shuttle_train_path = os.path.join(MODEL_OUTPUT_PATH, f'{input_file_name}_{len(train_frames)}_IN{list2string(input_columns)}OUT{list2string(output_columns)}_dataset_04_shuttle_train.pt')
    torch.save(dataset_shuttle_train,dataset_shuttle_train_path)
    
    dataset_shuttle_val_path = os.path.join(MODEL_OUTPUT_PATH,    f'{input_file_name}_{len(train_frames)}_IN{list2string(input_columns)}OUT{list2string(output_columns)}_dataset_05_shuttle_val.pt')
    torch.save(dataset_shuttle_val,dataset_shuttle_val_path)
        
    
    print (f"\n> DATASETS STORED:")
    print (f"{dataset_train_path=}")
    print (f"{dataset_val_path=}")
    print (f"{dataset_test_path=}")
    print (f"{dataset_shuttle_train_path=}")
    print (f"{dataset_shuttle_train_path=}")
        
    return [dataset_train,dataset_val,dataset_test,dataset_shuttle_train,dataset_shuttle_val], [dataset_train_path,dataset_val_path,dataset_test_path,dataset_shuttle_train_path,dataset_shuttle_val_path]
    
def Dataset_GCN_3(df_input,
                  all_indexes,
                  printstat = False,
                     
                  input_columns = ['data_pos',#x,y 
                                   'data_x', # yaw, speed, intention
                                   'Still_vehicle', # is the vehicle moving or not
                                   ],
                  output_columns = [
                                    #'data_y',
                                    'data_y_zc',
                                    'data_y_yaw'
                                   ],

                  use_edges_renamed = True,    
                  ):
    """
    df_input: input df
    all_indexes: row indexes of the train/test data
    """
    # check that all the input columns are actually in the df_input
    printif (f"requested input columns: {input_columns}",printstat)
    input_columns = [col for col in input_columns if col in df_input.columns]
    printif (f"available input columns: {input_columns}",printstat)
    
    # check that all the output columns are actually ni the df_input
    printif (f"requested output columns: {output_columns}",printstat)
    output_columns = [col for col in output_columns if col in df_input.columns]
    printif (f"available output columns: {output_columns}",printstat)  
    #pdb.set_trace()
    
    data_list = []
        
    for idx in tqdm(all_indexes):
        # take one row
        row = df_input.loc[idx]
        
        dash = "="*20
        printif(f"\n{dash} {idx} {dash}", printstat) 
        printif(f"\n{row}", printstat) 
    
        
        ##---------------------
        ## GET INPUT in NP--> CONCAT --> TENSOR
        ##---------------------
        printif (f"input given by concatenation of : {input_columns}", printstat)       
        input_to_concatenate = [np.asarray(row[selected_input]) for selected_input in input_columns]
        try:
            x = torch.from_numpy(np.concatenate(input_to_concatenate, axis=1)).float()
        except Exception as e:
            print (*input_to_concatenate, sep = "\n")
            pdb.set_trace()
        
        ##---------------------
        ## GET EDGE INFORMATION
        ##---------------------
        # we put both in the data object, then it depends on the model to use it or not
        if use_edges_renamed:
            # only integers 0-(max num vehicles per frame -1)
            edge_index = torch.FloatTensor(row.data_edges_renamed).long() # format long => integers
        else:
            # the actual cardinal (up to very high numbers if the sim is long)
            edge_index = torch.FloatTensor(row.data_edges).long()
        edge_attr = torch.FloatTensor(row.data_edges_attr).float()
        
        ##---------------------
        ## GET OUTPUT
        ##--------------------- 
        printif (f"output given by concatenation of : {output_columns}", printstat)
        output_to_concatenate = [np.asarray(row[selected_output]) for selected_output in output_columns]
        if "data_classes" in output_columns:
            y = torch.from_numpy(np.concatenate(output_to_concatenate, axis=1)).long()
        else: 
            y = torch.from_numpy(np.concatenate(output_to_concatenate, axis=1)).float()
        
        printif ("TENSORS", printstat)
        printif (f"{x=}", printstat)
        printif (f"{edge_index=}", printstat)
        printif (f"{edge_attr=}", printstat)
        printif (f"{y=}", printstat)


        # create a data object to append to the list 
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html
        data_list.append(Data
                         (
                             x = x,
                             edge_index = edge_index,
                             edge_attr = edge_attr,
                             y = y
                         )
                        )
            
    return data_list


def Dataset_GCN_CORRECTED(df_input, 
                    all_indexes,
                    printstat = False,
                          
                    input_columns = ['data_pos',#x,y 
                                     'data_x', # yaw, speed, intention
                                     'Still_vehicle', # is the vehicle moving or not
                                    ],
                    output_columns = [
                                      #'data_y',
                                      'data_y_zc',
                                      'data_y_yaw'
                                     ],
                          
                          
                   ):
    """
    df_input: input df
    all_indexes: row indexes of the train/test data
    """

    data_list = []
        
    for idx in tqdm(all_indexes):
        # take one row
        row = df_input.loc[idx]
        
        dash = "="*20
        printif(f"\n{dash} {idx} {dash}", printstat) 
        printif(f"\n{row}", printstat) 
    
        
        ##---------------------
        ## GET INPUT in NP--> CONCAT --> TENSOR
        ##---------------------
        printif (f"input given by concatenation of : {input_columns}", printstat)
        
        input_to_concatenate = [np.asarray(row[selected_input]) for selected_input in input_columns]
        
        x = torch.from_numpy(np.concatenate(input_to_concatenate, axis=1)).float()
        
        ##---------------------
        ## GET EDGE INFORMATION
        ##---------------------
        # we put both in the data object, then it depends on the model to use it or not
        edge_index = torch.FloatTensor(row.data_edges_renamed).long() # format long => integers
        edge_attr = torch.FloatTensor(row.data_edges_attr).float()
        
        ##---------------------
        ## GET OUTPUT
        ##--------------------- 
        printif (f"output given by concatenation of : {output_columns}", printstat)
        output_to_concatenate = [np.asarray(row[selected_output]) for selected_output in output_columns]
        y = torch.from_numpy(np.concatenate(output_to_concatenate, axis=1)).float()
        
        printif ("TENSORS", printstat)
        printif (f"{x=}", printstat)
        printif (f"{edge_index=}", printstat)
        printif (f"{edge_attr=}", printstat)
        printif (f"{y=}", printstat)


        # create a data object to append to the list 
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html
        data_list.append(Data
                         (
                             x = x,
                             edge_index = edge_index,
                             edge_attr = edge_attr,
                             y = y
                         )
                        )
            
    return data_list



def Dataset_GCN_CORRECTED_lanes(df_input, 
                    all_indexes,
                    printstat = False,
                          
                    input_columns = ['data_pos', # yaw, speed, intention, lane
                                     'data_x', #x,y
                                     'Still_vehicle', # is the vehicle moving or not
                                    ],
                    output_columns = [
                                      #'data_y',
                                      'data_y_zc',
                                      'data_y_yaw'
                                     ],
                          
                          
                   ):
    """
    df_input: input df
    all_indexes: row indexes of the train/test data
    """

    data_list = []
        
    for idx in tqdm(all_indexes):
        # take one row
        row = df_input.loc[idx]
        
        dash = "="*20
        printif(f"\n{dash} {idx} {dash}", printstat) 
        printif(f"\n{row}", printstat) 
    
        
        ##---------------------
        ## GET INPUT in NP--> CONCAT --> TENSOR
        ##---------------------
        printif (f"input given by concatenation of : {input_columns}", printstat)
        
        input_to_concatenate = [np.asarray(row[selected_input]) for selected_input in input_columns]
        
        test=input_to_concatenate[1]
        #lanes description form SUMO to int
        test[test == ':0_3_0'] = 19
        test[test == ':0_9_0'] = 18
        test[test == ':0_0_0'] = 17
        test[test == ':0_6_0'] = 16
        test[test == ':0_7_0'] = 15
        test[test == ':0_8_0'] = 14
        test[test == ':0_1_0'] = 13
        test[test == ':0_5_0'] = 12
        test[test == ':0_2_0'] = 11
        test[test == ':0_10_0'] = 9
        test[test == ':0_4_0'] = 10
        
        test[test == '4o_0'] = 8
        test[test == '3o_0'] = 7
        test[test == '2o_0'] = 6
        test[test == '1o_0'] = 5
        
        test[test == '4i_0'] = 4
        test[test == '3i_0'] = 3
        test[test == '2i_0'] = 2
        test[test == '1i_0'] = 1
       
        test=test.astype(np.float)
        
        input_to_concatenate[1]= test
        
        x = torch.from_numpy(np.concatenate(input_to_concatenate, axis=1)).float()
        
        ##---------------------
        ## GET EDGE INFORMATION
        ##---------------------
        # we put both in the data object, then it depends on the model to use it or not
        edge_index = torch.FloatTensor(row.data_edges_renamed).long() # format long => integers
        edge_attr = torch.FloatTensor(row.data_edges_attr).float()
        
        ##---------------------
        ## GET OUTPUT
        ##--------------------- 
        printif (f"output given by concatenation of : {output_columns}", printstat)
        output_to_concatenate = [np.asarray(row[selected_output]) for selected_output in output_columns]
        y = torch.from_numpy(np.concatenate(output_to_concatenate, axis=1)).float()
        
        printif ("TENSORS", printstat)
        printif (f"{x=}", printstat)
        printif (f"{edge_index=}", printstat)
        printif (f"{edge_attr=}", printstat)
        printif (f"{y=}", printstat)


        # create a data object to append to the list 
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html
        data_list.append(Data
                         (
                             x = x,
                             edge_index = edge_index,
                             edge_attr = edge_attr,
                             y = y
                         )
                        )
            
    return data_list

def Dataset_GCN_002_bis(df_input, 
                    all_indexes,
                    M,
                    printstat = False,
                    size_input = 5,
                    exclude_yaw = True
                   ):
    """
    df_input: input df
    all_indexes: row indexes of the train/test data
    M: max number of vehicles per each timestep, and dimension of input and output of GCN
    """
    
    data_list = []
        
    for idx in tqdm(all_indexes):
        row = df_input.loc[idx]
        
        dash = "="*20
        printif(f"\n{dash} {idx} {dash}", printstat) 
        printif(f"\n{row}", printstat) 
        
        ##---------------------
        ## TO NUMPY ARRAYS
        ##---------------------
        data_x = np.asarray(row.data_x)
        data_pos = np.asarray(row.data_pos)
        if exclude_yaw:
            data_y = np.asarray([[x,y] for x,y,yaw in row.data_y]) # excluding yaw for the moment
        else:
            data_y = np.asarray(row.data_y)
        #print (data_y)
        #pdb.set_trace()
        
        
        # concatenate data_x and data_pos horizontally
        if size_input == 5 :
            x = torch.from_numpy(np.concatenate((data_x, data_pos), axis=1)).float()
        elif size_input == 3: 
            x = torch.from_numpy(data_x).float()
        elif size_input == 2:
            x = torch.from_numpy(data_pos).float()
        else:
            print ("size input possible: 2,3,5")
        pos = torch.from_numpy(data_pos).float()
        edge_index = torch.FloatTensor(row.data_edges_renamed).long() # edge index in format long => integers
        edge_attr = torch.FloatTensor(row.data_edges_attr).float()
        y = torch.from_numpy(data_y).float()
        
        printif ("INPUT TENSORS", printstat)
        printif (f"{x=}", printstat)
        printif (f"{pos=}", printstat)
        printif (f"{edge_index=}", printstat)
        printif (f"{edge_attr=}", printstat)
        printif (f"{y=}", printstat)


        # create a data object to append to the list 
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html
        data_list.append(Data
                         (
                             x = x,
                             pos = pos, 
                             edge_index = edge_index,
                             edge_attr = edge_attr,
                             y = y
                         )
                        )
            
    return data_list

def Dataset_GCN_002(df_input, 
                    all_indexes,
                    M,
                    printstat = False,
                    concatenatestat = True,
                    paddingstat = False,
                    exclude_yaw = True
                   ):
    """
    df_input: input df
    all_indexes: row indexes of the train/test data
    M: max number of vehicles per each timestep, and dimension of input and output of GCN
    """
    
    data_list = []
        
    for idx in tqdm(all_indexes):
        row = df_input.loc[idx]
        
        dash = "="*20
        printif(f"\n{dash} {idx} {dash}", printstat) 
        printif(f"\n{row}", printstat) 
        
        ##---------------------
        ## TO NUMPY ARRAYS
        ##---------------------
        data_x = np.asarray(row.data_x)
        data_pos = np.asarray(row.data_pos)
        if exclude_yaw:
            data_y = np.asarray([[x,y] for x,y,yaw in row.data_y]) # excluding yaw for the moment
        else:
            data_y = np.asarray(row.data_y)
        
        
        ##---------------------
        ## PADDING
        ##---------------------
        if paddingstat: 
            padded_data_x = np.zeros((M,data_x.shape[1]))
            padded_data_x[:data_x.shape[0],:data_x.shape[1]] = data_x
            data_x = padded_data_x
            
            padded_data_pos = np.zeros((M,data_pos.shape[1]))
            padded_data_pos[:data_pos.shape[0],:data_pos.shape[1]] = data_pos
            data_pos = padded_data_pos

            padded_data_y = np.zeros((M,data_y.shape[1]))
            padded_data_y[:data_y.shape[0],:data_y.shape[1]] = data_y
            data_y = padded_data_y

            printif(f"{data_x.shape=}",printstat)
            printif(f"{data_pos.shape=}",printstat)
            printif(f"{data_y.shape=}",printstat)
        
        # concatenate data_x and data_pos horizontally
        if concatenatestat:
            x = torch.from_numpy(np.concatenate((data_x, data_pos), axis=1)).float()
        else: 
            x = torch.from_numpy(data_x).float()
        pos = torch.from_numpy(data_pos).float()
        edge_index = torch.FloatTensor(row.data_edges_renamed).long() # edge index in format long => integers
        edge_attr = torch.FloatTensor(row.data_edges_attr).float()
        y = torch.from_numpy(data_y).float()
        
        printif ("INPUT TENSORS", printstat)
        printif (f"{x=}", printstat)
        printif (f"{pos=}", printstat)
        printif (f"{edge_index=}", printstat)
        printif (f"{edge_attr=}", printstat)
        printif (f"{y=}", printstat)


        # create a data object to append to the list 
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html
        data_list.append(Data
                         (
                             x = x,
                             pos = pos, 
                             edge_index = edge_index,
                             edge_attr = edge_attr,
                             y = y
                         )
                        )
            
    return data_list


def Dataset_GCN(df_input, 
                    all_indexes,
                    M,
                    printstat = False,
                    concatenatestat = True,
                    paddingstat = False,
                    exclude_yaw = True
                   ):
    """
    df_input: input df
    all_indexes: row indexes of the train/test data
    M: max number of vehicles per each timestep, and dimension of input and output of GCN
    """
    
    data_list = []
        
    for idx in tqdm(all_indexes):
        row = df_input.loc[idx]
        
        dash = "="*20
        printif(f"\n{dash} {idx} {dash}", printstat) 
        printif(f"\n{row}", printstat) 
        
        ##---------------------
        ## TO NUMPY ARRAYS
        ##---------------------
        data_x = np.asarray(row.data_x)
        data_pos = np.asarray(row.data_pos)
        if exclude_yaw:
            data_y = np.asarray([[x,y] for x,y,yaw in row.data_y]) # excluding yaw for the moment
        else:
            data_y = np.asarray(row.data_y)
        
        ##---------------------
        ## PADDING
        ##---------------------
        if paddingstat: 
            padded_data_x = np.zeros((M,data_x.shape[1]))
            padded_data_x[:data_x.shape[0],:data_x.shape[1]] = data_x
            data_x = padded_data_x
            
            padded_data_pos = np.zeros((M,data_pos.shape[1]))
            padded_data_pos[:data_pos.shape[0],:data_pos.shape[1]] = data_pos
            data_pos = padded_data_pos

            padded_data_y = np.zeros((M,data_y.shape[1]))
            padded_data_y[:data_y.shape[0],:data_y.shape[1]] = data_y
            data_y = padded_data_y

            printif(f"{data_x.shape=}",printstat)
            printif(f"{data_pos.shape=}",printstat)
            printif(f"{data_y.shape=}",printstat)
        
        # concatenate data_x and data_pos horizontally
        if concatenatestat:
            x = torch.from_numpy(np.concatenate((data_pos, data_x), axis=1)).float()
        else: 
            x = torch.from_numpy(data_x).float()
        pos = torch.from_numpy(data_pos).float()
        edge_index = torch.FloatTensor(row.data_edges_renamed).long() # edge index in format long => integers
        edge_attr = torch.FloatTensor(row.data_edges_attr).float()
        y = torch.from_numpy(data_y).float()
        
        printif ("INPUT TENSORS", printstat)
        printif (f"{x=}", printstat)
        printif (f"{pos=}", printstat)
        printif (f"{edge_index=}", printstat)
        printif (f"{edge_attr=}", printstat)
        printif (f"{y=}", printstat)


        # create a data object to append to the list 
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html
        data_list.append(Data
                         (
                             x = x,
                             pos = pos, 
                             edge_index = edge_index,
                             edge_attr = edge_attr,
                             y = y
                         )
                        )
            
    return data_list

def Dataset_1(df_input, 
                    all_indexes,
                    M,
                    printstat = False,
                    concatenatestat = True,
                    paddingstat = False,
                    exclude_yaw = True
                   ):
    """
    df_input: input df
    all_indexes: row indexes of the train/test data
    M: max number of vehicles per each timestep, and dimension of input and output of GCN
    """
    
    data_list = []
        
    for idx in tqdm(all_indexes):
        row = df_input.loc[idx]
        
        dash = "="*20
        printif(f"\n{dash} {idx} {dash}", printstat) 
        printif(f"\n{row}", printstat) 
        
        ##---------------------
        ## TO NUMPY ARRAYS
        ##---------------------
        data_x = np.asarray([[a,b] for a,b,c in row.data_x])
        
        data_pos = np.asarray(row.data_pos)
        if exclude_yaw:
            data_y = np.asarray([[x,y] for x,y,yaw in row.data_y]) # excluding yaw for the moment
        else:
            data_y = np.asarray(row.data_y)

        
        # concatenate data_x and data_pos horizontally
        if concatenatestat:
            x = torch.from_numpy(np.concatenate((data_pos, data_x), axis=1)).float()
        else: 
            x = torch.from_numpy(data_x).float()
        pos = torch.from_numpy(data_pos).float()
        edge_index = torch.FloatTensor(row.data_edges_renamed).long() # edge index in format long => integers
        edge_attr = torch.FloatTensor(row.data_edges_attr).float()
        y = torch.from_numpy(data_y).float()
        
        printif ("INPUT TENSORS", printstat)
        printif (f"{x=}", printstat)
        printif (f"{pos=}", printstat)
        printif (f"{edge_index=}", printstat)
        printif (f"{edge_attr=}", printstat)
        printif (f"{y=}", printstat)


        # create a data object to append to the list 
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html
        data_list.append(Data
                         (
                             x = x,
                             pos = pos, 
                             edge_index = edge_index,
                             edge_attr = edge_attr,
                             y = y
                         )
                        )
            
    return data_list

print (f"Functions GCN import successful")
