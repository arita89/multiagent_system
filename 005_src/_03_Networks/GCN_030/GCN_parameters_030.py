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

from config import *


##--------------------------------
# PARAMETERS HARDCODED
##--------------------------------
### old inputs with intention right, left, u turn
#input_file_name = "20210710-20h38m27s_timesteps14930_ec3500_em7000" #15000
#input_file_name = "20210711-17h59m44s_timesteps30000_ec3500_em7000" #30000
#input_file_name = "20210710-11h46m35s_timesteps200_ec3500_em7000"
#parse_data_stat = True

### new inputs with intention N,S,E,W,C
#input_file_name = "20210724-19h49m31s_timesteps150000_ec3500_em7000"
#input_file_name = "20210725-16h24m21s_timesteps200_ec3500_em7000"
#parse_data_stat = True

# 5 vehicles at each frame
input_file_name = "20210801-08h53m15s_timesteps50000_ec3500_em7000"
parse_data_stat = True


## ONLY moving vehicles, set parse_data_stat to False, since this data is already parsed
#input_file_name = "df_only_moving_vehicles"
#input_file_name = "df_balanced_20320"
#input_file_name = "20210710-11h46m35s_timesteps200_ec3500_em7000_size11"
#parse_data_stat = False



random_seed = 4767
train_size = 0.8  
Nepochs = 10000
save_every = int(Nepochs/20) # temp pkl, pt, png, can delete after final is stored
#save_every = 5
# plot every can be an integer or a list of frames
plot_every = int(Nepochs/20) 
#plot_every = flatten([list(range(0,101,10)),
                            #list(range(100,1001,100)),
                            #list(range(1000,5001,1000)),
                            #list(range(10000,100001,10000))
                            #])
#plot_every = 1

savestat = True
transformstat = False
plotstat = True
run_unattended = True
final_print = False

#if Nepochs <1000:
    #savestat = False
    #plotstat = False
            
input_columns = [
                        'data_pos_zc',  #x,y, size 2
                        'data_x_speed',
                        'data_x_yaw',
                        #'data_x_rad',#  yaw in rad, speed, intention size 3
                        'Still_vehicle', # is the vehicle moving or not (similar to traffi light info), size 1
    
                        ]
    
output_columns = [
                        'data_y_zc',
                        'data_y_yaw',
                        #'data_y_delta',
                        ]
shuttle_train = None # if none is set manually an appropriate one will be randomly chosen
shuttle_val = None

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
                   #64,
                   #128,
                   #256,
                   512,
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
                 #"MSE",
                 #"Custom_Loss_1"
                 #"Custom_Loss_2"
                ]
    
hl_sizes = [
                #[64],
                #[64,128],
                [64,128,256],
                #[256,512,256]
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
                        False,
                        True,
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