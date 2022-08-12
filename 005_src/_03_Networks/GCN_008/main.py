#--------------------------------
## IMPORTS
#--------------------------------
import sys
import os

# set the path to find the modules
sys.path.insert(0, "/storage/remote/atcremers50/ss21_multiagentcontrol/005_src/") 
os.chdir("/storage/remote/atcremers50/ss21_multiagentcontrol/005_src/")

from config import *
from _03_Networks.GCN_007.GCN_model_007 import *
check_import()
from _03_Networks.GCN_007.GCN_trainer_007 import *
this_GCN, ts = check_import()
this_date = get_date()
device = cudaOverview()

print(f"{this_date}-{ts}")

#--------------------------------
## FULL PRINTING
#--------------------------------
printstat = True


#--------------------------------
## MAIN
#--------------------------------
def main(input_file_name,
                 random_seed,
                 train_size, 
                 batch_size,

                 Nepochs,
                 savestat,
                 intentionstat,
                 
                 save_every,
                 plot_every ,
                 transformstat,
                 plotstat,
                 edges_attr,
                 exclude_yaw,
                 concatenatestat,
                 hl_1,

                 lr,
                 momentum,
                 weight_decay,
                 select_criterion,
                 select_optimizer,
         
                 paddingstat = None, 
        ):
    
    ##================================
    ## IMPORTS
    ##================================
    #if edges_attr:
        #from _03_Networks.GCN_008.GCN_model_008 import *
        #check_import()
        #from _03_Networks.GCN_008.GCN_trainer_008 import *
        #this_GCN, late_ts = check_import()
    
    ##================================
    ## INIT SAVING PATHS
    ##================================

    MODEL_OUTPUT_PATH = os.path.join(OUTPUT_DIR,f"{this_GCN}/")
    MODEL_OUTPUT_PATH_TODAY = os.path.join(MODEL_OUTPUT_PATH,f"{this_date}{ts}/")
    print (MODEL_OUTPUT_PATH_TODAY)
    if not os.path.exists(MODEL_OUTPUT_PATH_TODAY):
        Path(MODEL_OUTPUT_PATH_TODAY).mkdir(parents=True, exist_ok=True)

    dict_text_output = {}
    
    ##================================
    ## LOAD DATA
    ##================================
    
    dict_text_output["input_file_name"] = input_file_name
   
    print (f"\n> SELECTING INPUT ROWS")
    df_all,df_selected = select_rows_from_data(input_file_name,drop_col = True)
    df_input = df_selected
    printif(df_all.loc[15],printstat)
    printif(f"selected {len(df_selected)} coherent rows over {len(df_all)} total",printstat)
    
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
                        'intentionstat':intentionstat
                        })
    
    ##================================
    ## TRAIN AND VALIDATION 
    ##================================
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
    
    ## add information to the dict
    dict_text_output['num_rows_training'] = len(train_frames)
    dict_text_output['num_rows_validation'] = len(valid_frames)
    dict_text_output['num_rows_test'] = len(test_frames)

    ##================================
    ## MODEL VAR
    ##================================
    hc_1 =hl_1
    add_parameter = 0
    if intentionstat and concatenatestat:
        add_parameter = 1

    printif("",printstat)
    if concatenatestat:
        size_input = 4 + add_parameter
        printif(f"Input: concatenation(data_x,data_pos) [bacth_size,{size_input}]",printstat)
    else:
        size_input = 2 + add_parameter
        printif(f"Input: concatenation(data_x) [bacth_size,{size_input}]",printstat)
    if exclude_yaw:
        size_output = 2
        printif(f"Predicting: X, Y [bacth_size,{size_output}]",printstat)
    else:
        size_output = 3
        printif(f"Predicting: X, Y, Yaw [bacth_size,{size_output}]",printstat)
        
    ## add information to the dict
    dict_text_output['exclude_yaw'] = exclude_yaw
    dict_text_output['concatenatestat'] = concatenatestat
    dict_text_output['paddingstat'] = paddingstat
    dict_text_output['size_input'] = size_input
    dict_text_output['size_output'] = size_output

    ##================================
    ## CREATING THE DATASET
    ##================================
    print (f"> CREATING THE DATASET")

    # here I have two Dataset creators, 
    # Dataset_1 without intention, 
    # Dataset_GCN with intention

    if size_input% 2 == 0:
        printif("Training without intention",printstat)
        time.sleep(1)
        dataset_train = Dataset_1(
                             df_input,
                             train_frames,
                             #transform=transforms_training,
                             M = M,
                             printstat = False,
                             concatenatestat = concatenatestat,
                             paddingstat = paddingstat,
                             exclude_yaw = exclude_yaw
                             )
        dataset_val = Dataset_1(
                             df_input,
                             valid_frames,
                             M = M,
                             concatenatestat = concatenatestat,
                             paddingstat = paddingstat,
                             exclude_yaw = exclude_yaw
                             )
        dataset_test = Dataset_1(
                             df_input,
                             test_frames,
                             M = M,
                             concatenatestat = concatenatestat,
                             paddingstat = paddingstat,
                             exclude_yaw = exclude_yaw
                             )

    else:
        printif("Training with intention",printstat)
        time.sleep(1)
        dataset_train = Dataset_GCN(
                             df_input,
                             train_frames,
                             #transform=transforms_training,
                             M = M,
                             printstat = False,
                             concatenatestat = concatenatestat,
                             paddingstat = paddingstat,
                             exclude_yaw = exclude_yaw
                             )
        dataset_val = Dataset_GCN(
                             df_input,
                             valid_frames,
                             M = M,
                             concatenatestat = concatenatestat,
                             paddingstat = paddingstat,
                             exclude_yaw = exclude_yaw
                             )
        dataset_test = Dataset_GCN(
                             df_input,
                             test_frames,
                             M = M,
                             concatenatestat = concatenatestat,
                             paddingstat = paddingstat,
                             exclude_yaw = exclude_yaw
                             )
    printif(dataset_train,printstat, n = 10)
    printif(f"\nexample of data from train: \n{dataset_train[0]}",printstat)
    
    # save dataset_test
    dataset_test_path = os.path.join(MODEL_OUTPUT_PATH_TODAY,f'dataset_test'+'.pkl')
    with open(dataset_test_path, 'wb') as f:
        pkl.dump(dataset_test, f)
        
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

    model = GCN(num_input_features=size_input,
                num_output_features =size_output,
                random_seed = random_seed,
                hc_1 = hc_1,
                #hc_2 = 32
               )
    if select_criterion == "MSE":
        criterion = torch.nn.MSELoss()  # Define loss criterion for regression
    elif select_criterion == "L1":
        criterion = torch.nn.L1Loss()
    else:
        print (f" {select_criterion} not implemented")
    
    if select_optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif select_optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    else:
        print (f" {select_optimizer} not implemented")
    
    ## add information to the dict
    printif(f"\n{model=}",printstat)
    dict_text_output['model_architecture'] = model
    dict_text_output['criterion']= criterion
    dict_text_output['optimizer']= optimizer
    
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
                     shuttle_train = dataset_train[0],
                     shuttle_val = None,

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
    printif(f"> Dataset_test_path stored in: \n{dataset_test_path}",printstat)

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
            if edges_attr:
                print (model(data.x, data.edge_index, data.edge_attr)).detach().detach().numpy()
            else:
                print (model(data.x, data.edge_index).detach().detach().numpy())
                
if __name__ == '__main__':
    input_file_name = "20210710-20h38m27s_timesteps14930_ec3500_em7000" #15000
    input_file_name = "20210711-17h59m44s_timesteps30000_ec3500_em7000" #30000
    number_layers = [1]
    batch_sizes = [16,32,64,128,512]
    
    optimizers = ["SGD","Adam"]
    criterion = ["MSE", "L1"]
    
    hl_sizes = [16,64,128]
    lr_sizes = [0.001,0.01]
    momentum_sizes = [0.6]#,0.9]
    weight_decays = [0]#,1e-4]
    
    intentionstat_list = [True,False]
    edges_attr_list = [True]#,False]
    exclude_yaw_list = [True]#,False]
    concatenatestat_list = [True]#,False]
    

    list_parameters_space = [number_layers,
                             batch_sizes,
                             
                             optimizers,
                             criterion,
                             
                             hl_sizes,
                             lr_sizes,
                             momentum_sizes,
                             weight_decays,
                             
                             intentionstat_list,
                             edges_attr_list,
                             exclude_yaw_list,
                             concatenatestat_list
    ]

    all_combo = list(itertools.product(*list_parameters_space))
    # save the list
    all_combo_path = os.path.join(TRAIN_PARAMS_FOLDER,f'001_all_combos_params'+'.pkl')
    with open(all_combo_path, 'wb') as f:
        pkl.dump(all_combo, f)
    print(f" computing {len(all_combo)} combinations")
    
    
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
        
        random_seed = 457362
        train_size = 0.9  
        Nepochs =100        
        save_every = int(Nepochs/2) # temp pkl, pt, png, can delete after final is stored
        plot_every = int(Nepochs/10)
        savestat = True
        transformstat = False
        plotstat = True
        
        
        num_layer = combo[0]
        batch_size = combo[1]
        
        select_optimizer = combo[2]
        select_criterion = combo[3]
        
        hl_1 = combo[4]
        lr = combo[5]
        momentum = combo[6]
        weight_decay=combo[7]

        intentionstat = combo[8]
        edges_attr = combo[9]
        exclude_yaw = combo[10]
        concatenatestat = combo[11]
        
        
        main(input_file_name,
                 random_seed,
                 train_size, 
                 batch_size,

                 Nepochs,
                 savestat,
                 intentionstat,
                 save_every,
                 plot_every ,
                 transformstat,
                 plotstat,
                 edges_attr,
                 exclude_yaw,
                 concatenatestat,
                 hl_1,

                 lr,
                 momentum,
                 weight_decay,
                 select_criterion,
                 select_optimizer,
                )
        tried_combos.append(combo)
        with open(tried_combo_path, 'wb') as f:
            pkl.dump(tried_combos, f)
        
        #except Exception as e:
            #print (combo)
            #print ("")
            
            
    
    