#--------------------------------
## IMPORTS
#--------------------------------
import sys
import os

# set the path to find the modules
sys.path.insert(0, '../005_src/') #use relative path
os.chdir("/storage/remote/atcremers50/ss21_multiagentcontrol/005_src")

#from config import *
from _50_config import *

#--------------------------------
## Step 1 - Get Input Data in Right Format
#--------------------------------

def step1_select_input_data(data_path, 
                            printstat, 
                            simulation_max_duration,
                            time_col = "timestep"):
    
    # read data format
    #data_name, data_ext = os.path.splitext(data_path)
    
    # if xml do smth   
    if data_path.lower().endswith('.xml'):
        # read file into a dataframe
        df_raw = xml_to_df(data_path) 
        df = format_df(df_raw, source = ".xml", printstat=printstat ) 
    
    # read df in zip file
    elif data_path.lower().endswith('.zip'):
        df = pd.read_csv(data_path) 
        df = format_df(df,source =".zip",printstat=printstat)
        
    # exceptions 
    else:
        print ("only .xml and .zip files are expected")
        
    ## note : first reduce the df then do stuff on it
    # decide to mask eventually for shorter simulations   
    all_timesteps = list(df[time_col].unique())
    if simulation_max_duration is None:
        limit_duration =  len(all_timesteps)
    else:
        limit_duration = int(min(simulation_max_duration,len(all_timesteps)))
    df = mask_timestep(df,all_timesteps[:limit_duration])
    printif(f"the simulation duration is {limit_duration} timesteps" , printstat) 
    
    # check all vehicles in the sim
    unique_vehicles = df.vehID.unique().tolist()
    printif(f"there are {len(unique_vehicles)} unique vehicles in the first {limit_duration} timesteps of the sim" , printstat)
    printif(df.head(5), printstat)  

    #add intention
    df = add_intention_absolute(df, unique_vehicles)
    #df = add_intention(df, unique_vehicles)
    printif(df.head(5), printstat)
    
    
    return df, limit_duration
