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
from _03_Networks.GCN_014.GCN_model_014 import *
from _03_Networks.GCN_014.GCN_trainer_014 import *
from _03_Networks.GCN_014.GCN_parameters_014 import *
this_GCN, ts_import = check_import()

new_date = get_date()
print(f"starting: {new_date}-{ts_import}")
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
                 combo,
    
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
                 select_scheduler, 
                 activation_fun, 

                 run_unattended,
                 final_print,
                 use_edges_attr,
    
                 reload_model_path,
                 #t_losses_load,
                 #v_losses_load,
                 #lr_load
                 training_losses = [],
                 validation_losses = [],
                 lr_rates = [],
                 
                 dict_text_output = {}
        ):
    
    
    ##================================
    ## INIT SAVING PATHS
    ##================================

    #MODEL_OUTPUT_PATH = os.path.join(OUTPUT_DIR,f"{this_GCN}/")
    #MODEL_OUTPUT_PATH_TODAY = os.path.join(MODEL_OUTPUT_PATH,f"{this_date}{ts}/")
    print (MODEL_OUTPUT_PATH_TODAY)
    #if not os.path.exists(MODEL_OUTPUT_PATH_TODAY):
        #Path(MODEL_OUTPUT_PATH_TODAY).mkdir(parents=True, exist_ok=True)
    
    MODEL_OUTPUT_PATH_DATASETS = os.path.join(MODEL_OUTPUT_PATH,f"DATASETS/")
    print (MODEL_OUTPUT_PATH_DATASETS)
    if not os.path.exists(MODEL_OUTPUT_PATH_DATASETS):
        Path(MODEL_OUTPUT_PATH_DATASETS).mkdir(parents=True, exist_ok=True)

    
    
    ##================================
    ## LOAD DATA
    ##================================
    dict_text_output = update_dict_with_key("input_file_name",input_file_name, dict_text_output)
    #dict_text_output["input_file_name"] = input_file_name
   
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
    

    ## update 
    dict_updates = {
                    'train_size':train_size,
                    'batch_size': batch_size,
                    'Nepochs': Nepochs,
                        }
    dict_text_output = update_dict_many_keys(dict_updates, dict_text_output)
    
    ## add information to the dict
    dict_text_output.update({
                        'run_date': this_date,
                        'run_time':ts,
                        'model': this_GCN,
                        'combo': combo, 
                        'random_seed': random_seed,
                        #'train_size':train_size,
                        #'batch_size': batch_size,
                        #'Nepochs': Nepochs,
                        'save_every':save_every, 
                        'transformstat': transformstat,
                        'plotstat': plotstat,
                        'printstat' : printstat,
                        'intentionstat':"obsoleted",
                        'use_edges_attr':use_edges_attr,
                        'activation_function': activation_fun,
                        })
    

    
    dict_text_output['shuttle_train_frame'] = shuttle_train
    dict_text_output['shuttle_val_frame'] = shuttle_val
    

    ##================================
    ## CREATING THE DATASET
    ##================================
    # search for pre-existing datasets
    datasets_list = sorted(glob.glob(f"{MODEL_OUTPUT_PATH_DATASETS}"
                                     +"/**/*"
                                     + f'{input_file_name}_{train_size}_{input_columns}_{output_columns}_dataset*',
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

    #print (printstat)
    #pdb.set_trace()

    ## TANH

    if activation_fun == "tanh":
        if num_hidden_layers == 1:

            hc_1 = hidden_layers_sizes[0]
            hc_2 = None
            hc_3 = None

            model = GCN_HL01_bn_tanh(
                                num_input_features=size_input,
                                num_output_features =size_output,
                                random_seed = random_seed,
                                hc_1 = hc_1,
                                printstat = printstat, 
                                )
        elif num_hidden_layers == 2:

            hc_1 = hidden_layers_sizes[0]
            hc_2 = hidden_layers_sizes[1]
            hc_3 = None

            model = GCN_HL02_bn_tanh(
                                num_input_features=size_input,
                                num_output_features =size_output,
                                random_seed = random_seed,
                                hc_1 = hc_1,
                                hc_2 = hc_2,
                                printstat = printstat,   
                                )
        elif num_hidden_layers == 3:

            hc_1 = hidden_layers_sizes[0]
            hc_2 = hidden_layers_sizes[1]
            hc_3 = hidden_layers_sizes[2]

            model = GCN_HL03_bn_tanh(
                                num_input_features=size_input,
                                num_output_features =size_output,
                                random_seed = random_seed,
                                hc_1 = hc_1,
                                hc_2 = hc_2,
                                hc_3 = hc_3,
                                printstat = printstat,                
                                )
        else:
            print (f"ERROR: model with {num_hidden_layers=} not implemented")
            print (f"using first three layers")

            hc_1 = hidden_layers_sizes[0]
            hc_2 = hidden_layers_sizes[1]
            hc_3 = hidden_layers_sizes[2]

            model = GCN_HL03_bn_tanh(num_input_features=size_input,
                        num_output_features =size_output,
                        random_seed = random_seed,
                        hc_1 = hc_1,
                        hc_2 = hc_2,
                        hc_3 = hc_3,
                        printstat = printstat, 
                        )

    #### RELU       

    elif activation_fun == "relu": 
        if num_hidden_layers == 1:

            hc_1 = hidden_layers_sizes[0]
            hc_2 = None
            hc_3 = None

            model = GCN_HL01_bn_relu(
                                num_input_features=size_input,
                                num_output_features =size_output,
                                random_seed = random_seed,
                                hc_1 = hc_1,
                                printstat = printstat,                 
                                )
        elif num_hidden_layers == 2:

            hc_1 = hidden_layers_sizes[0]
            hc_2 = hidden_layers_sizes[1]
            hc_3 = None

            model = GCN_HL02_bn_relu(
                                num_input_features=size_input,
                                num_output_features =size_output,
                                random_seed = random_seed,
                                hc_1 = hc_1,
                                hc_2 = hc_2,
                                printstat = printstat,                 
                                )
        elif num_hidden_layers == 3:

            hc_1 = hidden_layers_sizes[0]
            hc_2 = hidden_layers_sizes[1]
            hc_3 = hidden_layers_sizes[2]

            model = GCN_HL03_bn_relu(
                                num_input_features=size_input,
                                num_output_features =size_output,
                                random_seed = random_seed,
                                hc_1 = hc_1,
                                hc_2 = hc_2,
                                hc_3 = hc_3,
                                printstat = printstat,               
                                )
        else:
            print (f"ERROR: model with {num_hidden_layers=} not implemented")
            print (f"using first three layers")

            hc_1 = hidden_layers_sizes[0]
            hc_2 = hidden_layers_sizes[1]
            hc_3 = hidden_layers_sizes[2]

            model = GCN_HL03_bn_relu(num_input_features=size_input,
                        num_output_features =size_output,
                        random_seed = random_seed,
                        hc_1 = hc_1,
                        hc_2 = hc_2,
                        hc_3 = hc_3,
                        printstat = printstat,                                      
                        )
    else:
        print (f"ERROR: model with {activation_fun=} not implemented")
    load_model = model
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
    ## SCHEDULER
    ##================================   
 
    scheduler = choose_scheduler(select_scheduler,
                                 optimizer,
                                 lr,
                                 Nepochs,
                                  )
    
    ## add information to the dict
    dict_text_output['scheduler']= scheduler.__class__.__name__
    
    ##================================
    ## RELOAD MODEL
    ##================================ 
    checkpoint = torch.load(reload_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    
    
    ##================================
    # Init Trainer Class
    ##================================

    
    trainer = Trainer(
                     # Model parameters
                     model = model,
                     device = device,
                     criterion = criterion,
                     optimizer = optimizer,
        
                     # control input
                     use_edges_attr = use_edges_attr,

                     # Data 
                     training_DataLoader = train_loader,
                     validation_DataLoader = val_loader,
                     lr_scheduler = scheduler,
                     select_scheduler = select_scheduler,

                     # trigger statements
                     printstat= printstat,
                     savestat = savestat,
                     plotstat = plotstat,

                     # store intermediate model as TEMP 
                     epochs = epoch+Nepochs,
                     epoch = epoch, # starting epoch 
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
        
                     # store dict at every TEMP step
                     dict_text_output = dict_text_output,
        
                     training_loss = training_losses,
                     validation_loss = validation_losses,
                     learning_rate = lr_load
                 
                     )
    
    ##================================
    # Train
    ##================================
    t_losses, v_losses, lr_r , load_paths, fig_paths = trainer.run_trainer()
    print(f"> Training completed")
    

    
    ##================================
    # STORE PARAMETERS
    ##================================
    if savestat:
        ## add information to the dict
        for k,v in load_paths.items():
            dict_text_output[k]= v
        
        ## add information to the dict
        dict_text_output['final_train_loss']= t_losses[-1]
        dict_text_output['final_val_loss']= v_losses[-1]
        #dict_text_output["figure_paths"] = fig_paths
        dict_text_output = update_dict_with_key("figure_paths",fig_paths, dict_text_output)
        

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
                
    ##================================
    # DELETE TEMP FILES
    ##================================
    print ("temporary files stored: ")
    tempfiles = glob.glob(MODEL_OUTPUT_PATH_TODAY+'/*TEMP_*')
    if run_unattended:
        deletestat = "y"
    else:
        print (*tempfiles, sep= "\n")
        print ("delete temporary files? y/n")
        deletestat = "n"#input(x)
    if deletestat in list_yes:
        for i, file_path in tqdm(enumerate(tempfiles)):
            try:
                os.remove(file_path)
            except OSError as e:
                print("Error: %s : %s" % (file_path, e.strerror))
        print ("temporary files deleted")
    else:
        print ("temporary files not deleted")
                
if __name__ == '__main__':
    
    ##--------------------------------
    # CHOSE WHICH MODEL YOU WANT TO CONTINUE TRAINING
    ##--------------------------------       
    this_date = "20210725-"
    ts = "13h51m08s"#22h36m37s"#"22h28m11s"#"21h30m51s"#"18h37m00s"#"20h33m28s"#"19h37m02s
    k = f"{this_date}{ts}"
    GCN_num = "014"
    
    this_GCN = f"GCN_{GCN_num}"
    MODEL_OUTPUT_PATH = os.path.join(OUTPUT_DIR,f"{this_GCN}/")
    MODEL_OUTPUT_PATH_TODAY = os.path.join(MODEL_OUTPUT_PATH,f"{k}/")
    
    # laod temporary saving if training had failed
    kep = 3625
    knep = 15000
    type_load = "TEMP"
    reload_model_path = os.path.join(MODEL_OUTPUT_PATH_TODAY,
                                     f"{k}EPOCH_{kep}of{knep}_{type_load}__full_GCN_014.pt")
    reload_train_losses_path = os.path.join(MODEL_OUTPUT_PATH_TODAY,
                                     f"{k}EPOCH_{kep}of{knep}_{type_load}__{type_load}__training_loss.pkl")
    reload_val_losses_path = os.path.join(MODEL_OUTPUT_PATH_TODAY,
                                     f"{k}EPOCH_{kep}of{knep}_{type_load}__{type_load}__validation_loss.pkl")
    reload_lr_path = os.path.join(MODEL_OUTPUT_PATH_TODAY,
                                     f"{k}EPOCH_{kep}of{knep}_{type_load}__{type_load}__learning_rate.pkl")
    
    # load losses
    t_losses_load = pkl.load(open(reload_train_losses_path, 'rb'))
    v_losses_load = pkl.load(open(reload_val_losses_path, 'rb'))
    lr_load = pkl.load(open(reload_lr_path, 'rb'))  
    
    # load losses from final savings
    #t_losses_load = pkl.load(open( reloaded_dict['train_losses_path'], 'rb'))
    #v_losses_load = pkl.load(open( reloaded_dict['val_losses_path'], 'rb'))
    #lr_load = pkl.load(open( reloaded_dict['lr_path'], 'rb'))
    
    
    ##--------------------------------
    # LOAD PARAMS
    ##-------------------------------- 
    dict_text_output_path = os.path.join(MODEL_OUTPUT_PATH_TODAY,
                                         f"{this_date}{ts}_training_parameters")
    reloaded_dict = pkl.load(open(f'{dict_text_output_path}.pkl',"rb"))
    
    hidden_layers_sizes = reloaded_dict["hidden_layers_sizes"]
    size_input = reloaded_dict['size_input']
    size_output = reloaded_dict['size_output']
    random_seed = reloaded_dict["random_seed"]
    

    
    combo = reloaded_dict["combo"]
    reduction = combo[0]
    batch_size = combo[1]
    select_optimizer = combo[2]
    select_criterion = combo[3]
    hidden_layers_sizes = combo[4]
    lr = combo[5]
    momentum = combo[6]
    weight_decay = combo[7]
    select_scheduler = combo [8]
    use_edges_attr = combo [9]
    activation_fun = combo [10]
    
    ##--------------------------------
    # UPDATE INFO PARAMS
    ##-------------------------------- 

    ## extract original info params
    origing_input_file_name = reloaded_dict["input_file_name"]
    train_size = reloaded_dict["train_size"]  
    orig_Nepochs = kep##reloaded_dict["Nepochs"]
    orig_batch_size = reloaded_dict["batch_size"] 
    
    updated_info_dict = {} 

    # num epochs
    print ("-"*80)
    print ("CHANGE N EPOCHS")
    print (f"model_{this_GCN}{k} was trained with {kep} epochs")
    print (">>>for how many more epochs do you want to train now?")
    Nepochs = int(input())
    save_every = int(Nepochs/20) 
    plot_every = int(Nepochs/20) 
    print (f"now training with {Nepochs=}")
    updated_info_dict["Nepochs"] = Nepochs

    # input file
    print ("-"*80)
    print ("CHANGE INPUT FILE")
    print (f"model_{this_GCN} {k} was trained with input: {origing_input_file_name}")
    print (">>>which input you want to use now?")
    # search for pre-existing datasets
    available_inputs_list = sorted(glob.glob(f"{GCN_INPUT_FOLDER}"+"/**/*"+f'*txt',
                                            recursive=True))
    short_available_inputs_list = [f"{os. path. splitext(os.path.split(e)[1])[0]}" for e in available_inputs_list]
    print (*short_available_inputs_list[-20:], sep = "\n")
    print ()
    input_file_name = input()
    assert input_file_name in short_available_inputs_list
    print (f"now training with {input_file_name=}")
    updated_info_dict["input_file_name"] = input_file_name
    #updated_info_dict["input_file_name_list"] = [origing_input_file_name,input_file_name]

    # batch size
    print ("-"*80)
    print ("CHANGE BATCH SIZE")
    print (f"model_{this_GCN} {k} was trained with batch_size: {orig_batch_size}")
    print (">>>which batch size you want to try now? ")
    batch_size = int(input())
    print (f"now training with {batch_size=}")
    updated_info_dict["batch_size"] = batch_size
    
    # lr
    print ("-"*80)
    print ("CHANGE LR")
    print (f"model_{this_GCN} {k} was trained with lr: {lr}")
    print (">>>which lr you want to try now? ")
    lr = float(input())
    print (f"now training with {lr=}")
    updated_info_dict["lr"] = lr
    
    # scheduler
    print ("-"*80)
    print ("CHANGE sscheduler")
    print (f"model_{this_GCN} {k} was trained with select_scheduler: {select_scheduler}")
    print (">>>which scheduler you want to try now? ")
    print (*["ReduceLROnPlateau","CosineAnnealingWarmRestarts","None","MultiStepLR"],sep ="\n")
    select_scheduler = str(input())
    print (f"now training with {select_scheduler=}")
    updated_info_dict["select_scheduler"] = select_scheduler
    
    print ("="*80)
    print (f"OVERVIEW NEW PARAMETERS")
    print (*[f"{k}:{v}" for k,v in updated_info_dict.items()], sep = "\n")
    print ("="*80)

    ##--------------------------------
    # HARDCODED PARAMETERS 
    ##--------------------------------     
    savestat = True
    transformstat = False
    plotstat = True
    run_unattended = True
    final_print = False
            
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
    # MAIN
    ##--------------------------------       

    main(
                input_file_name,
                random_seed,
                train_size, 
                batch_size,
                Nepochs,
                combo,

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
                select_scheduler,
                activation_fun, 

                run_unattended,
                final_print,
                use_edges_attr,
        
                reload_model_path,
                training_losses = t_losses_load,
                validation_losses = v_losses_load,
                lr_rates = lr_load,
        
                dict_text_output= reloaded_dict,   


            )
    