#--------------------------------
## IMPORTS
#--------------------------------
import sys
import os
#import codecs

# set the path to find the modules
sys.path.insert(0, "/storage/remote/atcremers50/ss21_multiagentcontrol/005_src/") 
os.chdir("/storage/remote/atcremers50/ss21_multiagentcontrol/005_src/")

from config import *
from _03_Networks.GCN_011.GCN_model_011 import *
#check_import()
from _03_Networks.GCN_011.GCN_trainer_011 import *
this_GCN, ts = check_import()

this_date = get_date()
print(f"{this_date}-{ts}")
print ("-"*40)

device = cudaOverview()
print ("-"*40)


#--------------------------------
## FULL PRINTING
#--------------------------------
printstat = False


#--------------------------------
## WARNINGS MUTED
#--------------------------------
pd.options.mode.chained_assignment = None
plt.rcParams.update({'figure.max_open_warning': 0})

#--------------------------------
## MAIN
#--------------------------------
def main(
                 input_file_name,
                 random_seed,
                 train_size, 
                 batch_size,
                 Nepochs,
    
                 savestat,
                 save_every,
                 plot_every ,
                 transformstat,
                 plotstat,
    
                 shuttle_train, 
                 shuttle_val, # to plot one specific frame during the training/validation
    
                 hidden_layers_sizes, # list as min of size one
                 input_columns,
                 output_columns,

                 lr,
                 momentum,
                 weight_decay,
                 select_criterion,
                 reduction, # 'mean,'sum', 'none' 
                 select_optimizer,

        ):
    
    
    ##================================
    ## INIT SAVING PATHS
    ##================================

    MODEL_OUTPUT_PATH = os.path.join(OUTPUT_DIR,f"{this_GCN}/")
    MODEL_OUTPUT_PATH_TODAY = os.path.join(MODEL_OUTPUT_PATH,f"{this_date}{ts}/")
    print (MODEL_OUTPUT_PATH_TODAY)
    if not os.path.exists(MODEL_OUTPUT_PATH_TODAY):
        Path(MODEL_OUTPUT_PATH_TODAY).mkdir(parents=True, exist_ok=True)
    
    MODEL_OUTPUT_PATH_DATASETS = os.path.join(MODEL_OUTPUT_PATH,f"DATASETS/")
    print (MODEL_OUTPUT_PATH_DATASETS)
    if not os.path.exists(MODEL_OUTPUT_PATH_DATASETS):
        Path(MODEL_OUTPUT_PATH_DATASETS).mkdir(parents=True, exist_ok=True)

    dict_text_output = {}
    
    ##================================
    ## LOAD DATA
    ##================================
    
    dict_text_output["input_file_name"] = input_file_name
   
    print (f"\n> SELECTING INPUT ROWS")
    df_all,df_selected = select_rows_from_data(input_file_name,drop_col = True)
    print (f"\n> CORRECTING COLUMNS DATA_POS, DATA_Y_POS, DATA_Y_YAW")
    #df_selected = adjust_columns(df_selected)
    
    # for now we take only the coherent rows
    df_input = adjust_columns(df_selected)
    printif(f"selected {len(df_input)} coherent rows over {len(df_all)} total",printstat)
    
    ## add information to the dict
    txt_data = os.path.join(GCN_INPUT_FOLDER,f"{input_file_name}.txt")
    dict_text_data = read_txt_data(txt_data)
    M = int(dict_text_data['max_num_veh'])
    sim_duration_timesteps = dict_text_data['sim_duration_timesteps']
    printif(f"max_num_veh: {M}",printstat)
    printif(f"sim_duration_timesteps: {sim_duration_timesteps}",printstat)
    dict_text_output['max_num_veh'] = M
    dict_text_output['sim_duration_timesteps'] = sim_duration_timesteps
    
    ##================================
    ## TRAIN VAR
    ##================================
    training_losses = []
    validation_losses = []
    lr_rates = []

    ## add information to the dict
    dict_text_output.update({
                        'run_time':ts,
                        'model': this_GCN,
                        'random_seed': random_seed,
                        'train_size':train_size,
                        'batch_size': batch_size,
                        'Nepochs': Nepochs,
                        'save_every':save_every, 
                        'transformstat': transformstat,
                        'plotstat': plotstat,
                        'printstat' : printstat,
                        'intentionstat':"obsoleted"
                        })
    

    
    dict_text_output['shuttle_train_frame'] = shuttle_train
    dict_text_output['shuttle_val_frame'] = shuttle_val
    

    ##================================
    ## CREATING THE DATASET
    ##================================
    # search for pre-existing datasets
    datasets_list = sorted(glob.glob(f"{MODEL_OUTPUT_PATH_DATASETS}"+"/**/*"+f'{input_file_name}_dataset*',
                                     recursive=True))
    if len( datasets_list) == 5:                               
        print ("\n> DATASETS FOUND")
        printif(datasets_list, printstat)
        
        dataset_train = torch.load(datasets_list[0], map_location=torch.device('cpu') )
        # map_location=lambda storage, loc: storage.cuda(0))
        dataset_val = torch.load(datasets_list[1], map_location=torch.device('cpu') )
        dataset_test = torch.load(datasets_list[2], map_location=torch.device('cpu') )
        dataset_shuttle_train = torch.load(datasets_list[3], map_location=torch.device('cpu') )
        dataset_shuttle_val = torch.load(datasets_list[4], map_location=torch.device('cpu') )
    
        ## add information to the dict
        dict_text_output['num_rows_training'] = len(dataset_train)
        dict_text_output['num_rows_validation'] = len(dataset_val)
        dict_text_output['num_rows_test'] = len(dataset_test)
    else:
              
        ##--------------------------------
        ## TRAIN AND VALIDATION 
        ##--------------------------------
        print (f"\n> SPLIT TRAIN AND VALIDATION")
        c = df_input.index.tolist()

        train_frames, valid_frames = train_test_split(
                                        c,
                                        random_state=random_seed,
                                        train_size=train_size,
                                        shuffle=True)

        valid_frames, test_frames = train_test_split(
                                        valid_frames,
                                        random_state=random_seed,
                                        train_size=0.9,
                                        shuffle=True)


        ##--------------------------------
        ## SELECT POSSIBLE SHUTTLES FRAMES
        ##--------------------------------
        mask = df_input.all_veh_moving == True
        frames_all_veh_moving = df_input[mask].index

        if shuttle_train is None:
            shuttle_train_frames = [frame for frame in train_frames if frame in frames_all_veh_moving]
            shuttle_train = random.choice(shuttle_train_frames)

        if shuttle_val is None: 
            shuttle_val_frames = [frame for frame in valid_frames if frame in frames_all_veh_moving]
            shuttle_val = random.choice(shuttle_val_frames)

        printif (f"\n{shuttle_train=}", printstat)
        printif (f"\n{shuttle_val=}", printstat)

        ## add information to the dict
        dict_text_output['num_rows_training'] = len(train_frames)
        dict_text_output['num_rows_validation'] = len(valid_frames)
        dict_text_output['num_rows_test'] = len(test_frames)
        
        
        all_datasets = creating_storing_DATASETS(
                                                 df_input,
                                                 train_frames,
                                                 valid_frames,
                                                 test_frames,
                                                 shuttle_train,
                                                 shuttle_val,
                                                 input_columns,
                                                 output_columns,
                                                 MODEL_OUTPUT_PATH_DATASETS,
                                                 input_file_name,
                                                )
        dataset_train = all_datasets[0]
        dataset_val = all_datasets[1]
        dataset_test = all_datasets[2]
        dataset_shuttle_train = all_datasets[3]
        dataset_shuttle_val = all_datasets[4]
             
    printif(dataset_train,printstat, n = 10)
    printif(f"\nexample of data from train: \n{dataset_train[0]}",printstat)
    
    
    
    ##================================
    ## MODEL VAR
    ##================================
    #hc_1 =hl_1
     
    size_input = 0
    size_output = 0
    for select_column in input_columns:
        if select_column == "data_x_rad":
            size_input += 3
        elif select_column == "data_pos_zc":
            size_input += 2
        elif select_column == 'Still_vehicle':
            size_input +=1
        else: 
            print (f"{select_column} not a foreseen input")
            
    for select_column in output_columns:
        if select_column == "data_y_zc":
            size_output += 2
        elif select_column == "data_y_yaw":
            size_output += 1
        else: 
            print (f"{select_column} not a foreseen output")
            
    print(f"\nInput: {input_columns} \n [{batch_size=},{size_input=}]")#,printstat)
    print(f"\nPredicting: {output_columns} \n [{batch_size=},{size_output=}]")#,printstat)
                    
    ## add information to the dict
    dict_text_output['exclude_yaw'] = "obsoleted"
    dict_text_output['concatenatestat'] = "obsoleted"
    dict_text_output['paddingstat'] = "obsoleted"
    dict_text_output['size_input'] = size_input
    dict_text_output['size_output'] = size_output
    
    ##================================
    # DATALOADER
    ##================================
    print (f"\n> DATALOADER")
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)       
       
    printif("")
    printif(f"{len(train_loader.dataset)=}",printstat)
    printif(f"{len(val_loader.dataset)=}",printstat)
    printif(f"{len(test_loader.dataset)=}",printstat)
    
    ##================================
    ## TRANSFORMATIONS
    ##================================
    if transformstat: 
        # not implemented, not needed i think
        transforms_training = None
        transforms_validation = None
    else:
        transforms_training = None
        transforms_validation = None
        
    ##================================
    ## MODEL INIT
    ##================================
    num_hidden_layers = len(hidden_layers_sizes)
    
    if num_hidden_layers == 1:
        
        hc_1 = hidden_layers_sizes[0]
        hc_2 = None
        hc_3 = None
        
        model = GCN_HL01(
                            num_input_features=size_input,
                            num_output_features =size_output,
                            random_seed = random_seed,
                            hc_1 = hc_1,
                           )
    elif num_hidden_layers == 2:
        
        hc_1 = hidden_layers_sizes[0]
        hc_2 = hidden_layers_sizes[1]
        hc_3 = None
        
        model = GCN_HL02(
                            num_input_features=size_input,
                            num_output_features =size_output,
                            random_seed = random_seed,
                            hc_1 = hc_1,
                            hc_2 = hc_2,
                           )
    elif num_hidden_layers == 3:
        
        hc_1 = hidden_layers_sizes[0]
        hc_2 = hidden_layers_sizes[1]
        hc_3 = hidden_layers_sizes[2]
        
        model = GCN_HL03(
                            num_input_features=size_input,
                            num_output_features =size_output,
                            random_seed = random_seed,
                            hc_1 = hc_1,
                            hc_2 = hc_2,
                            hc_3 = hc_3,
                           )
    else:
        print (f"ERROR: model with {num_hidden_layers=} not implemented")
        print (f"using first three layers")
        
        hc_1 = hidden_layers_sizes[0]
        hc_2 = hidden_layers_sizes[1]
        hc_3 = hidden_layers_sizes[2]
        
        model = GCN_HL03(num_input_features=size_input,
                    num_output_features =size_output,
                    random_seed = random_seed,
                    hc_1 = hc_1,
                    hc_2 = hc_2,
                    hc_3 = hc_3,
                   )
        
    
    ## add information to the dict
    printif(f"\n{model=}",printstat)
    dict_text_output['model_architecture'] = model
    dict_text_output['hidden_layers_sizes'] = hidden_layers_sizes
        
    ##================================
    ## CRITERION
    ##================================   
    
    if select_criterion == "MSE":
        criterion = torch.nn.MSELoss()  # Define loss criterion for regression
    elif select_criterion == "L1":
        criterion = torch.nn.L1Loss(reduction = reduction)
    else:
        print (f" {select_criterion} not implemented, using L1 loss")
        criterion = torch.nn.L1Loss(reduction = reduction)

    ## add information to the dict
    dict_text_output['criterion']= criterion
    
    ##================================
    ## OPTIMIZER
    ##================================   
 
    optimizer = choose_optimizer(select_optimizer,
                     model,
                     lr,
                     weight_decay,
                     momentum,
                    )
    
    ## add information to the dict
    dict_text_output['optimizer']= optimizer
    dict_text_output['reduction']= reduction
    
    ##================================
    # Init Trainer Class
    ##================================
    trainer = Trainer(
                     # Model parameters
                     model = model,
                     device = device,
                     criterion = criterion,
                     optimizer = optimizer,

                     # Data 
                     training_DataLoader = train_loader,
                     validation_DataLoader = val_loader,
                     lr_scheduler = None,

                     # trigger statements
                     printstat= printstat,
                     savestat = savestat,
                     plotstat = plotstat,

                     # store intermediate model as TEMP 
                     epochs = Nepochs,
                     epoch = 0, # starting epoch 
                     save_every = save_every,
                     plot_every = plot_every, 

                     # get intermediate results for each epoch
                     shuttle_train = dataset_shuttle_train[0],
                     shuttle_val = dataset_shuttle_val[0],

                     # saving directory and description
                     save_dir = MODEL_OUTPUT_PATH_TODAY,
                     model_name = this_GCN,
                     date = this_date,
                     ts = ts,
                     notebook = False,
                     )
    
    ##================================
    # Train
    ##================================
    t_losses, v_losses, lr_r , load_paths, fig_paths = trainer.run_trainer()
    print(f"> Training completed")
    
    ##================================
    # DELETE TEMP FILES
    ##================================
    print ("temporary files stored: ")
    tempfiles = glob.glob(MODEL_OUTPUT_PATH_TODAY+'/*TEMP_*')
    print (*tempfiles, sep= "\n")
    print ("delete temporary files? y/n")
    deletestat = "y"
    if deletestat in list_yes:
        for i, file_path in tqdm(enumerate(tempfiles)):
            try:
                os.remove(file_path)
            except OSError as e:
                print("Error: %s : %s" % (file_path, e.strerror))
        print ("temporary files deleted")
    else:
        print ("temporary files not deleted")
    
    ##================================
    # STORE PARAMETERS
    ##================================
    ## add information to the dict
    for k,v in load_paths.items():
        dict_text_output[k]= v
    dict_text_output["figure_paths"] = fig_paths
        
    ## add information to the dict
    dict_text_output['final_train_loss']= t_losses[-1]
    dict_text_output['final_val_loss']= v_losses[-1]
    
    description = f"{this_date}{ts}"
    dict_text_output_descr = f"{description}_training_parameters"
    dict_text_output_path = os.path.join(MODEL_OUTPUT_PATH_TODAY,dict_text_output_descr)
    print (f"\n> Parameters stored in: \n{dict_text_output_path}.pkl")

    # store parameters as txt file
    TXT_OUTPUT = f'{dict_text_output_path}.txt'
    with open(TXT_OUTPUT, 'w') as filehandle:
        for k,v in dict_text_output.items():
            filehandle.write(f'{k}: {v}\n')
        filehandle.close()

    # store parameters as a pickle
    with open(f'{dict_text_output_path}.pkl', 'wb') as handle:
        pkl.dump(dict_text_output, handle, protocol=pkl.HIGHEST_PROTOCOL)
      
    ##================================
    # FINAL GLIMPSE
    ##================================
    final_print = False
    if final_print:
        np.set_printoptions(suppress=True)
        for data in dataset_val[:1]:
            print ("--------------")
            print ("\nINPUT DATA_POS,DATA_X")
            print (np.around(data.x.detach().numpy(),2))
            print ("\nTARGET DATA_Y")
            print (data.y.detach().numpy())
            print ("\nPREDICTIONS")
            model.to('cpu')
            if edges_attr:
                print (model(data.x, data.edge_index, data.edge_attr)).detach().detach().numpy()
            else:
                print (model(data.x, data.edge_index).detach().detach().numpy())
                
if __name__ == '__main__':
    #input_file_name = "20210710-20h38m27s_timesteps14930_ec3500_em7000" #15000
    #input_file_name = "20210711-17h59m44s_timesteps30000_ec3500_em7000" #30000
    
    input_file_name = "20210710-11h46m35s_timesteps200_ec3500_em7000"
    
    # OPTIONS
    reduction_options = [
                         'mean', 
                         #'sum', 
                         #'none' 
                        ]
    batch_sizes = [
                   #1,
                   #16,
                   #32,
                   #64,
                   #128,
                   #256,
                   512
                  ]
    
    optimizers = [
                  #"Adadelta",
                  #"Adagrad",
                  "Adam",
                  #"AdamW",
                  #"SparseAdam",
                  #"Adamax",
                  #"ASGD",
                  #"LBFGS",
                  #"RMSprop",
                  #"Rprop",
                  #"SGD"
                 ]
    
    criterion = [
                #"MSE",
                 "L1"
                ]
    
    hl_sizes = [
                #[64],
                #[64,128],
                #[64,128,256],
                [128,256,64]
                ]
    
    lr_sizes = [
                0.001,
                #0.01
                ]
    
    momentum_sizes = [
                      #0.1,
                      #0.6,
                      0.9
                    ]
    
    weight_decays = [
                      0,
                     #1e-4
                    ]
    
    

    list_parameters_space = [
                             reduction_options,
                             batch_sizes,
                             
                             optimizers,
                             criterion,
                             
                             hl_sizes,
                             lr_sizes,
                             
                             momentum_sizes,
                             weight_decays,                          
    ]

    all_combo = list(itertools.product(*list_parameters_space))
    # save the list
    all_combo_path = os.path.join(TRAIN_PARAMS_FOLDER,f'001_all_combos_params'+'.pkl')
    with open(all_combo_path, 'wb') as f:
        pkl.dump(all_combo, f)
    
    
    print(f"> TRAINING OVER {len(all_combo)} COMBINATIONS")
    
    
    
    tried_combo_path = os.path.join(TRAIN_PARAMS_FOLDER,f'001_tried_combos_params'+'.pkl')
    if os.path.exists(tried_combo_path):
        tried_combos = pkl.load(open(tried_combo_path, 'rb'))
    else: 
        tried_combos = []   
    
    for i,combo in enumerate(all_combo):
        ts = get_timestamp()
        
        print (f"\nCOMBINATION {pad(i,minl = 4)}")
        print (combo)
        print ("")
        
        ##--------------------------------
        # PARAMETERS HARDCODED
        ##--------------------------------   
        
        random_seed = 4562
        train_size = 0.9  
        Nepochs = 100000      
        save_every = int(Nepochs/4) # temp pkl, pt, png, can delete after final is stored
        #plot_every = int(Nepochs/10)
        
        plot_every = flatten([list(range(0,101,10)),
                              list(range(100,1001,100)),
                              list(range(1000,10001,1000)),
                              list(range(10000,100001,10000))
                             ])
        savestat = True
        transformstat = False
        plotstat = True
        
        input_columns = [
                         'data_pos_zc',  #x,y, size 2
                         'data_x_rad',#  yaw in rad, speed, intention size 3
                         'Still_vehicle', # is the vehicle moving or not (similar to traffi light info), size 1
                         ]
        
        output_columns = [
                           #'data_y',
                           'data_y_zc',
                           'data_y_yaw' 
                         ]
        shuttle_train = None # if none is set manually an appropriate one will be randomly chosen
        shuttle_val = None
        
        ##--------------------------------
        # PARAMETERS FROM COMBINATIONS
        ##--------------------------------       
        
        reduction = combo[0]
        batch_size = combo[1]
        
        select_optimizer = combo[2]
        select_criterion = combo[3]
        
        hidden_layers_sizes = combo[4]
        lr = combo[5]
        momentum = combo[6]
        weight_decay=combo[7]
       
        
        main(
                 input_file_name,
                 random_seed,
                 train_size, 
                 batch_size,
                 Nepochs,
    
                 savestat,
                 save_every,
                 plot_every ,
                 transformstat,
                 plotstat,
    
                 shuttle_train, 
                 shuttle_val, # to plot one specific frame during the training/validation
    
                 hidden_layers_sizes, # list as min of size one
                 input_columns,
                 output_columns,

                 lr,
                 momentum,
                 weight_decay,
                 select_criterion,
                 reduction, 
                 select_optimizer,
                )
        tried_combos.append(combo)
        with open(tried_combo_path, 'wb') as f:
            pkl.dump(tried_combos, f)
        
        #except Exception as e:
            #print (combo)
            #print ("")
            
            
    
    