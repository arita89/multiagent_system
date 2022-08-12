#--------------------------------
## IMPORTS
#--------------------------------
import sys
import os
# current notebook
this_notebook = os.path.split(os.path.realpath(__file__))[1]

# set the path to find the modules
sys.path.insert(0, '../005_src/') #use relative path
#os.chdir("../005_src")

from config_GCN_018 import *

##--------------------------------
# INPUT FILES
##--------------------------------
### old inputs with intention right, left, u turn
#input_file_name = "20210710-20h38m27s_timesteps14930_ec3500_em7000" #15000 70-76
#input_file_name = "20210711-17h59m44s_timesteps30000_ec3500_em7000" #30000
#input_file_name = "20210710-11h46m35s_timesteps200_ec3500_em7000" # frames 35 to 50
#parse_data_stat = True

### new inputs with intention N,S,E,W,C
#input_file_name = "20210724-19h49m31s_timesteps150000_ec3500_em7000"
#input_file_name = "20210725-16h24m21s_timesteps200_ec3500_em7000"
#parse_data_stat = True

# 5 vehicles at each frame
#input_file_name = "20210801-08h53m15s_timesteps50000_ec3500_em7000"
#parse_data_stat = True


## ONLY moving vehicles, set parse_data_stat to False, since this data is already parsed
#input_file_name = "df_only_moving_vehicles"
#input_file_name = "df_balanced_20320"
#input_file_name = "20210710-11h46m35s_timesteps200_ec3500_em7000_size11"
#parse_data_stat = False

## dataset with vehicle acceleration as fourth parameter in data_x
input_file_name = "20210812-15h45m39s_timesteps199_ec3500_em7000"
parse_data_stat = True


##--------------------------------
# PARAMETERS HARDCODED
##--------------------------------
random_seed = 4763498
train_size = 0.8  
Nepochs = 100000

save_every = int(Nepochs/20) # temp pkl, pt, png, can delete after final is stored

# plot every can be an integer or a list of frames
#plot_every = int(Nepochs/20) 
plot_every = list_flatten([list(range(0,101,10)),
                            list(range(100,1001,100)),
                            list(range(1000,10001,1000)),
                            list(range(10000,100001,10000))
                            ])
                
if Nepochs <=10:
    save_every = 1                
    plot_every = 1

savestat = True
transformstat = False
plotstat = True
run_unattended = True
final_print = False
early_stopping_stat = True

#if Nepochs <1000:
    #savestat = False
    #plotstat = False
            
input_columns = [
                        #'data_pos',
                        'data_pos_zc',  #x,y, size 2
                        #'data_x_speed',
                        #'data_x_yaw',
                        #'data_x',
                        'data_x_rad',#  yaw in rad, speed, intention size 3
                        #'data_x_acceleration',
                        'Still_vehicle', # is the vehicle moving or not (similar to traffi light info), size 1
    
                        ]
    
output_columns = [
                        #'data_y',                   
                        'data_y_zc',
                        #'data_y_delta',
                        'data_y_yaw',
                        ]
shuttle_train = 70# 36#None # if none is set manually an appropriate one will be randomly chosen
shuttle_val = 76# 50#None

# OPTIONS
reduction_options = [
                         #'mean', 
                         'sum', 
                        ]
batch_sizes = [
                   #1,
                   #4
                   #16,
                   #32,
                   64,
                   #128,
                   #256,
                   #512,
                   #1024
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
                 "MSE",
                 #"L1",
                 #"Custom_Loss_1"
                 #"Custom_Loss_2"
                 #"Custom_Loss_3",
                 #"Custom_Loss_6",
                 #"Custom_Loss_7",
                ]
    
hl_sizes = [
                [1024],
                #[64,128,256],
                #[256,512,256],
                #[128,256,512,256,128,256,512,256,128,64]
                ]
    
lr_sizes = [
                #0.1,            
                0.001,
                #0.01,
                #0.0001
                ]
    
momentum_sizes = [
                     # 0.1,
                      0.6,
                     # 0.9
                    ]
    
weight_decays = [
                     0,
                     #1e-4
                    ]
    
# https://pytorch.org/docs/stable/optim.html
lr_schedulers = [

                    "CosineAnnealingWarmRestarts",
                    #"ReduceLROnPlateau",                    
                    #"None",
                    #"MultiStepLR",
                    ]
    
use_edges_attributes = [
                        True,
                        #False,
                            ]
activation_functions = [
                            "relu",
                            #"tanh",
                            #"leakyrelu", #
                            #"rrelu",
                            ]

layer_type = "GraphConv"
normalization = "GraphNorm" 

print (f"imported parameters from: {this_notebook} at {get_timestamp()}")  