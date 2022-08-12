# NOTE run from cd <pathtodir>/project/
from _00_configuration._00_settings import *

## imports and variables 
CONFIG_FOLDER = os.path.join(ROOT, "005_src/_00_configuration/")

## output dir
OUTPUT_DIR = os.path.join(ROOT,'006_model_output/')
if not os.path.exists(OUTPUT_DIR):
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
## input dir
DATA_FOLDER = os.path.join(ROOT,'004_data/')
if not os.path.exists(DATA_FOLDER):
    Path(DATA_FOLDER).mkdir(parents=True, exist_ok=True)
    
## parameters_options
TRAIN_PARAMS_FOLDER = os.path.join(DATA_FOLDER,'combo_training_parameters/')
if not os.path.exists(TRAIN_PARAMS_FOLDER):
    Path(TRAIN_PARAMS_FOLDER).mkdir(parents=True, exist_ok=True)

## GCN input
GCN_INPUT_FOLDER = os.path.join(DATA_FOLDER,'GCN_input/')
if not os.path.exists(GCN_INPUT_FOLDER):
    Path(GCN_INPUT_FOLDER).mkdir(parents=True, exist_ok=True)

## figures graph folder
GRAPH_FOLDER = os.path.join(DATA_FOLDER,'figures/graph/')
if not os.path.exists(GRAPH_FOLDER):
    Path(GRAPH_FOLDER).mkdir(parents=True, exist_ok=True) 
    
## csv folder
CSV_FOLDER = os.path.join(DATA_FOLDER,'CSV_FILES/')
if not os.path.exists(CSV_FOLDER):
    Path(CSV_FOLDER).mkdir(parents=True, exist_ok=True)   
    

## dictionaries to uniform the columns from xml and from script

columns_names_xml = {
                'time':'timestep',
                'id':'vehID', 
                'x':'X', 
                'y':'Y', 
                'angle':'yaw', 
                'type':'type', 
                'speed':'speed', 
                'pos':'pos',  
                'lane':'lane', 
                'slope':'slope', 
        }

columns_names_script = {
                         'id':'vehID', 
                          'x':'X', 
                          'y':'Y', 
                          'angle':'yaw', 
                          'type':'type', 
                          'speed':'speed', 
                          'pos':'pos', #? 
                          'lane':'lane', 
                          'slope':'slope', 
                         }

intention_dict_1 = {0: "U_turn",1:"Straight",2:"Turn_Left",3:"Turn_Right"}
intention_dict_2 = {0: "North",1:"East",2:"South",3:"West"}

TL_color_dict = {'r': 0,
                 'y': 1,
                 'G': 2
                }

list_yes = ["yes","y","yeah","yeps","true"]
list_no = ["no","n","nope","false"]

dash = "-"*100
DASH = "="*100

print ("Variables import successful")
