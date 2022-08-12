# NOTE run from cd <pathtodir>/project/
#from _00_configuration._00_settings import *

##--------------------------------
# PARAMETERS HARDCODED
##--------------------------------   
#input_file_name = "20210710-20h38m27s_timesteps14930_ec3500_em7000" #15000
#input_file_name = "20210711-17h59m44s_timesteps30000_ec3500_em7000" #30000
input_file_name = "20210710-11h46m35s_timesteps200_ec3500_em7000"

random_seed = 42
train_size = 0.9  
Nepochs = 5000   
save_every = int(Nepochs/2) # temp pkl, pt, png, can delete after final is stored

# plot every can be an integer or a list of frames
plot_every = int(Nepochs/5) 
#plot_every = flatten([list(range(0,101,10)),
                            #list(range(100,1001,100)),
                            #list(range(1000,10001,1000)),
                            #list(range(10000,100001,10000))
                            #])
savestat = True
transformstat = False
plotstat = True
run_unattended = True
final_print = False

if Nepochs <1000:
    savestat = False
    plotstat = False
            
input_columns = [
                        'data_pos_zc',  #x,y, size 2
                        'data_x_rad',#  yaw in rad, speed, intention size 3
                        'Still_vehicle', # is the vehicle moving or not (similar to traffi light info), size 1
                        ]
    
output_columns = [
                        'data_y_zc',
                        #'data_y_yaw' 
                        ]
shuttle_train = None # if none is set manually an appropriate one will be randomly chosen
shuttle_val = None

# OPTIONS
reduction_options = [
                         #'mean', 
                         'sum', 
                         #'none' 
                        ]
batch_sizes = [
                   #1,
                   #16,
                   #32,
                   #64,
                   128,
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
                #"MSE",
                 "L1"
                ]
    
hl_sizes = [
                #[64],
                #[64,128],
                [64,128,256],
                #[128,256,64]
                ]
    
lr_sizes = [
                #0.001,
                0.01,
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
                    #"ReduceLROnPlateau",
                    "CosineAnnealingWarmRestarts",
                    "None",
                    "MultiStepLR",
                    ]
    
use_edges_attributes = [
                            True,
                            False
                            ]
activation_functions = [
                            "relu",
                            "tanh",
                            #"leaky_relu" ,
                            #"rrelu",
                            ]