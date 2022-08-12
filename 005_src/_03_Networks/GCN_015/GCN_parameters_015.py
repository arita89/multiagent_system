#--------------------------------
## IMPORTS
#--------------------------------
import sys
import os

# current notebook
this_notebook = os.path.split(os.path.realpath(__file__))[1]

# set the path to find the modules
sys.path.insert(0, '../005_src/') #use relative path

from config import *


##--------------------------------
# PARAMETERS HARDCODED
##--------------------------------   
#input_file_name = "20210710-20h38m27s_timesteps14930_ec3500_em7000" #15000
#input_file_name = "20210711-17h59m44s_timesteps30000_ec3500_em7000" #30000
#input_file_name = "20210724-19h49m31s_timesteps150000_ec3500_em7000"
#input_file_name = "20210710-11h46m35s_timesteps200_ec3500_em7000"

#traffic light data
#input_file_name = "20210807-12h59m50s_timesteps200_ec3500_em7000"
#input_file_name = "20210807-14h52m34s_timesteps2000_ec3500_em7000"
input_file_name = "20210807-15h06m18s_timesteps20000_ec3500_em7000"

#balanced data
#input_file_name = "df_balanced_30480_normalized_standardized"
#input_file_name = "df_balanced_91_normalized_standard_100"
#input_file_name = "df_balanced_91_normalized_standard_100"
#input_file_name = "df_balanced_1971_normalized_standard"
#parse_data_stat = False

#with acceleration
#input_file_name = "20210812-18h12m53s_timesteps29785_ec3500_em7000"

parse_data_stat = True

random_seed = 42
train_size = 0.9  
Nepochs = 20
save_every = int(Nepochs/10) # temp pkl, pt, png, can delete after final is stored

# plot every can be an integer or a list of frames
plot_every = int(Nepochs/10) 
#plot_every = flatten([list(range(0,101,10)),
                            #list(range(100,1001,100)),
                            #list(range(1000,5001,1000)),
                            #list(range(10000,100001,10000))
                            #])
savestat = True
transformstat = False
plotstat = True
run_unattended = True
final_print = False

if Nepochs <10:
    savestat = False
    plotstat = False
            
input_columns = [
                        #'data_pos',  #x,y, size 2
                        'data_pos_zc',  #x,y, size 2
                        #'data_pos_zc_norm',  #x,y, size 2
                        #'data_pos_zc_norm_graph',
    
                        'data_x',#  yaw in rad, speed, intention, tl status, tl duration
                        #'data_x_rad_norm',#  yaw in rad, speed, intention size 3
                        #'data_x_rad_norm_graph',
    
                        #'Still_vehicle', # is the vehicle moving or not (similar to traffi light info), size 1
                        #'data_x_acceleration',
                        ]
    
output_columns = [
                        #'data_y',
                        'data_y_zc',
                        #'data_y_delta',
                        #'data_y_yaw'
                        #'data_y_zc_norm' 
                        ]
shuttle_train = None # if none is set manually an appropriate one will be randomly chosen
shuttle_val = None
#shuttle_train = 35 #if a specifical frame should be looked at
#shuttle_val = 70

#9264, 13590
# OPTIONS
reduction_options = [
                         #'mean', 
                         'sum', 
                         #'none' 
                        ]
batch_sizes = [
                   #1,
                   #2
                   #4
                   #16,
                   #32,
                   #64,
                   #128,
                   256,
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
                #"MSE",
                 "L1"
                ]
    
hl_sizes = [
                #[64],
                [64,128],
                #[64,128,256]
                #[128,256,64]
                ]
    
lr_sizes = [
                #0.001,
                0.01,
                #0.0001
                ]
    
momentum_sizes = [
                     # 0.1,
                      #0.6,
                      0.9
                    ]
    
weight_decays = [
                     0,
                     #1e-4
                    ]
    
# https://pytorch.org/docs/stable/optim.html
lr_schedulers = [
                    #"ReduceLROnPlateau",
                    "CosineAnnealingWarmRestarts",
                    #"None",
                    #"MultiStepLR",
                    ]
    
use_edges_attributes = [
                            True,
                            #False
                            ]
activation_functions = [
                            #"relu",
                            #"tanh",
                            "leaky_relu" 
                            #"rrelu",
                            ]


print (f"imported parameters from: {this_notebook} at {get_timestamp()}")  