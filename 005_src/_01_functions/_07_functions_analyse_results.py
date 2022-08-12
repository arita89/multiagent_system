#--------------------------------
## IMPORTS
#--------------------------------
from _00_configuration._00_settings import *
from _00_configuration._01_variables import *
from _01_functions._00_helper_functions import *

def analyse_results(
                    this_date = "20210721-",
                    ts = "15h53m43s" ,
                    GCN_num = "013"
                    ):
    
    this_GCN = f"GCN_{GCN_num}"
    print (this_GCN)
    
    # load appropriate GCN
    
    if GCN_num == "011": 
        from _03_Networks.GCN_011.GCN_model_011 import GCN_HL01,GCN_HL02,GCN_HL03
        from _03_Networks.GCN_011.GCN_trainer_011 import Trainer, check_import
        
        this_GCN, ts = check_import()
        return this_GCN, ts
    
    elif GCN_num == "012": 
        from _03_Networks.GCN_012.GCN_model_012 import *
        from _03_Networks.GCN_012.GCN_trainer_012 import Trainer, check_import
        
        this_GCN, ts = check_import()
        return this_GCN, ts
    
    elif GCN_num == "013": 
        from _03_Networks.GCN_013.GCN_model_013 import *
        from _03_Networks.GCN_013.GCN_model_013 import GCN_HL01_relu,GCN_HL02_relu,GCN_HL03_relu
        from _03_Networks.GCN_013.GCN_trainer_013 import Trainer

    else:
        print (f"{GCN_num} not yet implemented")
    
    MODEL_OUTPUT_PATH = os.path.join(OUTPUT_DIR,f"{this_GCN}/")
    MODEL_OUTPUT_PATH_TODAY = os.path.join(MODEL_OUTPUT_PATH,f"{this_date}{ts}/")
    print (MODEL_OUTPUT_PATH_TODAY)

    dateset_test_path = f"/storage/remote/atcremers50/ss21_multiagentcontrol/006_model_output/{this_GCN}/{this_date}{ts}/dataset_test"
    dict_text_output_path = f"/storage/remote/atcremers50/ss21_multiagentcontrol/006_model_output/{this_GCN}/{this_date}{ts}/{this_date}-{ts}_training_parameters"

    # load dictionary and variables
    reloaded_dict = pkl.load(open(f'{dict_text_output_path}.pkl',"rb"))   
    input_file_name = reloaded_dict["input_file_name"]
    architecture = reloaded_dict["model_architecture"]
    hidden_layers_sizes = reloaded_dict["hidden_layers_sizes"]
    size_input = reloaded_dict['size_input']
    size_output = reloaded_dict['size_output']
    random_seed = reloaded_dict["random_seed"]
    activation_fun = reloaded_dict["random_seed"]
    print (f"{architecture=}")
    
    # load losses
    t_losses_load = pkl.load(open( reloaded_dict['train_losses_path'], 'rb'))
    v_losses_load = pkl.load(open( reloaded_dict['val_losses_path'], 'rb'))
    lr_load = pkl.load(open( reloaded_dict['lr_path'], 'rb'))
    
    
    # plot losses and learning rate
    fig = plot_training(t_losses_load,
                      v_losses_load,
                      learning_rate = lr_load,
                      gaussian=True,
                      sigma=2,
                      figsize=(20, 6),
                      mytitle = 'Training & validation loss'
                      )

    description = f"{this_date}-{ts}"
    new_png = f"{description}_train_val_loss_plot.png"
    new_png_path = os.path.join(MODEL_OUTPUT_PATH_TODAY,new_png)
    plt.savefig(new_png_path)
    print (f"Plot saved in {new_png_path}")
    
    ## all the models as from the main 
    num_hidden_layers = len(hidden_layers_sizes)
    if activation_fun == "tanh":
            if num_hidden_layers == 1:

                hc_1 = hidden_layers_sizes[0]
                hc_2 = None
                hc_3 = None

                model = GCN_HL01_tanh(
                                    num_input_features=size_input,
                                    num_output_features =size_output,
                                    random_seed = random_seed,
                                    hc_1 = hc_1,
                                   )
            elif num_hidden_layers == 2:

                hc_1 = hidden_layers_sizes[0]
                hc_2 = hidden_layers_sizes[1]
                hc_3 = None

                model = GCN_HL02_tanh(
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

                model = GCN_HL03_tanh(
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

                model = GCN_HL03_tanh(num_input_features=size_input,
                            num_output_features =size_output,
                            random_seed = random_seed,
                            hc_1 = hc_1,
                            hc_2 = hc_2,
                            hc_3 = hc_3,
                           )

    #### RELU       

    elif activation_fun == "relu": 
            if num_hidden_layers == 1:

                hc_1 = hidden_layers_sizes[0]
                hc_2 = None
                hc_3 = None

                model = GCN_HL01_relu(
                                    num_input_features=size_input,
                                    num_output_features =size_output,
                                    random_seed = random_seed,
                                    hc_1 = hc_1,
                                   )
            elif num_hidden_layers == 2:

                hc_1 = hidden_layers_sizes[0]
                hc_2 = hidden_layers_sizes[1]
                hc_3 = None

                model = GCN_HL02_relu(
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

                model = GCN_HL03_relu(
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

                model = GCN_HL03_relu(num_input_features=size_input,
                            num_output_features =size_output,
                            random_seed = random_seed,
                            hc_1 = hc_1,
                            hc_2 = hc_2,
                            hc_3 = hc_3,
                           )

    elif activation_fun == "leaky_relu": 
            if num_hidden_layers == 1:

                hc_1 = hidden_layers_sizes[0]
                hc_2 = None
                hc_3 = None

                model = GCN_HL01_leaky_relu(
                                    num_input_features=size_input,
                                    num_output_features =size_output,
                                    random_seed = random_seed,
                                    hc_1 = hc_1,
                                   )
            elif num_hidden_layers == 2:

                hc_1 = hidden_layers_sizes[0]
                hc_2 = hidden_layers_sizes[1]
                hc_3 = None

                model = GCN_HL02_leaky_relu(
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

                model = GCN_HL03_rrelu(
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

                model = GCN_HL03_leaky_relu(num_input_features=size_input,
                            num_output_features =size_output,
                            random_seed = random_seed,
                            hc_1 = hc_1,
                            hc_2 = hc_2,
                            hc_3 = hc_3,
                           )
    #load the model
    load_model = model              
    load_path = reloaded_dict['model_path']
    load_model.load_state_dict(torch.load(load_path))
    load_model.eval()
    
    # find the datasets
    MODEL_OUTPUT_PATH = os.path.join(OUTPUT_DIR,f"{this_GCN}/")
    MODEL_OUTPUT_PATH_DATASETS = os.path.join(MODEL_OUTPUT_PATH,f"DATASETS/")
    datasets_list = sorted(glob.glob(f"{MODEL_OUTPUT_PATH_DATASETS}"+"/**/*"+f'{input_file_name}_dataset*',
                                         recursive=True))
    if len( datasets_list) == 5:                               
            print ("\n> DATASETS FOUND")
            print (*datasets_list, sep = "\n")

            dataset_train = torch.load(datasets_list[0], map_location=torch.device('cpu') )
            # map_location=lambda storage, loc: storage.cuda(0))
            dataset_val = torch.load(datasets_list[1], map_location=torch.device('cpu') )
            dataset_test = torch.load(datasets_list[2], map_location=torch.device('cpu') )
            dataset_shuttle_train = torch.load(datasets_list[3], map_location=torch.device('cpu') )
            dataset_shuttle_val = torch.load(datasets_list[4], map_location=torch.device('cpu') )
            
            
    path_val_GIF = build_gif(folder = os.path.join(OUTPUT_DIR,f'/GCN_{GCN_num}/{this_date}{ts}/figures_validation_set/'),
                                      title = "Predictions over epochs",
                                      search = "", 
                                      fps=0.5,
                                      recursive = True,
                                      delete_tempFiles = False,
                                      max_n_images = 200
                                     )
                            
    print (f"{path_val_GIF=}")
                            
    path_train_GIF= build_gif(folder = os.path.join(OUTPUT_DIR,
    f'/GCN_{GCN_num}/{this_date}{ts}/figures_training_set/'),
                                      title = "Predictions over epochs",
                                      search = "", 
                                      fps=0.5,
                                      recursive = True,
                                      delete_tempFiles = False,
                                      max_n_images = 200
                                     )
                            
    print (f"{path_train_GIF=}")
                               
    deleted_folders = delete_empty_r(directory= MODEL_OUTPUT_PATH,printstat = True)
     
    return new_png_path, path_train_GIF, path_val_GIF
                    
                            
                            
    