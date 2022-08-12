#--------------------------------
## IMPORTS
#--------------------------------
import sys
import os

# set the path to find the modules
sys.path.insert(0, "/storage/remote/atcremers50/ss21_multiagentcontrol/005_src/") 
os.chdir("/storage/remote/atcremers50/ss21_multiagentcontrol/005_src/")

from config import *
#from _03_Networks.GCN_009.GCN_model_009 import *
from _03_Networks.GCN_010.GCN_model_010 import *
check_import()
#from _03_Networks.GCN_009.GCN_trainer_009 import *
from _03_Networks.GCN_010.GCN_trainer_010 import *
this_GCN, ts = check_import()
this_date = get_date()
device = cudaOverview()

print('funktioniert cuda')
print(device)

print(f"{this_date}-{ts}")

#--------------------------------
## FULL PRINTING
#--------------------------------
printstat = False


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
    
                 hl_1,
                 input_columns,
                 output_columns,

                 lr,
                 momentum,
                 weight_decay,
                 select_criterion,
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

    dict_text_output = {}
    
    ##================================
    ## LOAD DATA
    ##================================
    
    dict_text_output["input_file_name"] = input_file_name
   
    print (f"\n> SELECTING INPUT ROWS")
    df_all,df_selected = select_rows_from_data(input_file_name,drop_col = True)
    
    
    print('TEESSSSTTTT: df_all')
    print(df_all)
    #print(df_all.columns)
    #print('TEESSSSTTTT: df_selected')
    print(df_selected.size)
    #print(df_selected.columns)
    
    print (f"\n> CORRECTING COLUMNS DATA_POS, DATA_Y_POS, DATA_Y_YAW")
    df_selected = adjust_columns(df_selected)
    
    #print('TEESSSSTTTT: df_selected adjusted')
    #print(df_selected)
    #print(df_selected.columns)
    
    # for now we take only the coherent rows
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
                        'intentionstat':"obsoleted"
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
    
    ####################################################################################
    ## add information to the dict

    train_frames=train_frames[:200]
    valid_frames=valid_frames[:20]
   
    dict_text_output['num_rows_training'] = len(train_frames)
    dict_text_output['num_rows_validation'] = len(valid_frames)
    dict_text_output['num_rows_test'] = len(test_frames)

    ##================================
    ## MODEL VAR
    ##================================
    hc_1 =hl_1
    
    print('----------input check-----------------')
    print(df_input)
    print('----------input check-----------------')
    
    size_input = 0
    size_output = 0
    for select_column in input_columns:
        if select_column == "data_x":
            size_input += 3
        elif select_column == "data_pos_zc":
            size_input += 2
        elif select_column == 'Still_vehicle':
            size_input +=1
        else: 
            print (f"{select_column} not a foreseen input")
    
    #print('THAT ARE THE INPUT COLUMNS')
    #print(input_columns)
    
    for select_column in output_columns:
        if select_column == "data_y_zc":
            size_output += 2
        elif select_column == "data_y_yaw":
            size_output += 1
        else: 
            print (f"{select_column} not a foreseen output")
    
    #print('THAT ARE THE OUTPUT COLUMNS')
    #print(output_columns)
    
    printif(f"\nInput: {input_columns} \n [{batch_size=},{size_input=}]",printstat)
    printif(f"\nPredicting: {output_columns} \n [{batch_size=},{size_output=}]",printstat)
                    
    ## add information to the dict
    dict_text_output['exclude_yaw'] = "obsoleted"
    dict_text_output['concatenatestat'] = "obsoleted"
    dict_text_output['paddingstat'] = "obsoleted"
    #dict_text_output['size_input'] = size_input
    dict_text_output['size_input'] = 7
    dict_text_output['size_output'] = size_output

    ##================================
    ## CREATING THE DATASET
    ##================================
    print (f"> CREATING THE DATASET")

    # here I have two Dataset creators, 
    # Dataset_1 without intention, 
    # Dataset_GCN with intention
    
    #print('DAAAAASS IST EIN TEEEST')
    #print(df_input)
    #print('DAAAAASS IST EIN TEEEST')
    
    #dataset_train = Dataset_GCN_CORRECTED_lanes(
    dataset_train = Dataset_GCN_CORRECTED(
                             df_input,
                             train_frames,
                             #M = M,
                             printstat = False,
                             input_columns = input_columns,
                             output_columns = output_columns,                            
                             )
    
    #dataset_val = Dataset_GCN_CORRECTED_lanes(
    dataset_val = Dataset_GCN_CORRECTED(
                             df_input,
                             valid_frames,
                             #M = M,
                             input_columns = input_columns,
                             output_columns = output_columns,
                             )
    
    #dataset_test = Dataset_GCN_CORRECTED_lanes(
    dataset_test = Dataset_GCN_CORRECTED(
                             df_input,
                             test_frames,
                             #M = M,
                             input_columns = input_columns,
                             output_columns = output_columns,
                             )

    #print('-------Train Dataset--------')
    #print(dataset_train)
    #print('-------Train Dataset--------')
    printif(dataset_train,printstat, n = 10)
    #print('-------------------------------------------')
    printif(f"\nexample of data from train: \n{dataset_train[0]}",printstat)
    #print('-------------------------------------------')
    
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

    #model = GCN(num_input_features=size_input,
    model = GCN(num_input_features=6,
                num_output_features =size_output,
                random_seed = random_seed,
                hc_1 = hc_1,
                #hc_2 = 32
               )
    model.to(device)
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
                     #shuttle_train = dataset_train[5],
                     shuttle_train = dataset_train[25],
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
    print('UEBERSICHT LOSSES')
    print('--t loss-')
    print(t_losses)
    print('--v loss-')
    print(v_losses)
    #print('---')
    #print(lr_r)
    #print('---')
    #print(load_paths)
    #print('---')
    #print(fig_paths)
    print('UEBERSICHT LOSSES')
    
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
            #print ("--------------")
            #print ("\nINPUT DATA_POS,DATA_X")
            #print (np.around(data.x.detach().numpy(),2))
            #print ("\nTARGET DATA_Y")
            #print (data.y.detach().numpy())
            #print ("\nPREDICTIONS")
            if edges_attr:
                print (model(data.x, data.edge_index, data.edge_attr)).detach().detach().numpy()
            else:
                print (model(data.x, data.edge_index).detach().detach().numpy())
                
if __name__ == '__main__':
    #input_file_name = "20210710-20h38m27s_timesteps14930_ec3500_em7000" #15000
    #input_file_name = "20210711-17h59m44s_timesteps30000_ec3500_em7000" #30000
    #input_file_name = "20210721-16h25m22s_timesteps200_ec3500_em7000" #30000
    #input_file_name = "20210722-10h51m45s_timesteps200_ec3500_em7000" #30000
    
    #without lanes route 13
    input_file_name = "20210723-13h43m44s_timesteps5000_ec3500_em7000"
     
    #with lanes route 13
    #input_file_name = "20210723-08h46m45s_timesteps5000_ec3500_em7000"

    # the imports are possible only at module level
    # the new Data creator accepts whatever size as from the df ( so no need to load different ones)
    # need to update the ts tho otherwise it just overwrites..
    number_layers = [2]
    batch_sizes = [#256
                   #1,
                   #4 
                   #8
                   #16,
                   32#,
                   #64#,
                   #128#,
                   #256
                   #512
                  ]
    
    optimizers = [
                  "Adam"
                  #"SGD"
                  ]
    criterion = [
                #"MSE",
                 "L1"
                ]
    
    hl_sizes = [#512#
                256#,
                #128#,
                #64
                #16
                #32
                ]
    lr_sizes = [#0.001#,
                0.01#,
               #0.0001, 
                #0.1
               ]
    
    momentum_sizes = [#0.1,
                      0.9]
    weight_decays = [0]#,1e-4]
    
    

    list_parameters_space = [number_layers,
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
    #with open(all_combo_path, 'wb') as f:
    #    pkl.dump(all_combo, f)
    #print(f" computing {len(all_combo)} combinations")
    
    
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
        Nepochs =5000       
        save_every = int(Nepochs/2) # temp pkl, pt, png, can delete after final is stored
        #plot_every = int(Nepochs/10)
        plot_every = int(Nepochs/5)
        savestat = True
        transformstat = False
        plotstat = True
        
        input_columns = [
                         'data_pos_zc',  #x,y, size 2
                         'data_x',# speed, yaw, intention size 3
                         'Still_vehicle', # is the vehicle moving or not (similar to traffi light info), size 1
                         ]
        
        output_columns = [
                           #'data_y',
                           'data_y_zc',
                           #'data_y_yaw' 
                         ]
        
        
        num_layer = combo[0]
        batch_size = combo[1]
        
        select_optimizer = combo[2]
        select_criterion = combo[3]
        
        hl_1 = combo[4]
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

                 hl_1,
                 input_columns,
                 output_columns,

                 lr,
                 momentum,
                 weight_decay,
                 select_criterion,
                 select_optimizer,
                )
        tried_combos.append(combo)
        
        #######################wieder einkommentieren
        #'/storage/remote/atcremers50/ss21_multiagentcontrol/004_data/combo_training_parameters/001_tried_combos_params.pkl'
        #with open(tried_combo_path, 'wb') as f:
        #    pkl.dump(tried_combos, f)
        
        #except Exception as e:
            #print (combo)
            #print ("")
            
            
    
    