#--------------------------------
## IMPORTS
#--------------------------------
import sys
import os

# set the path to find the modules
sys.path.insert(0, '../src/') #use relative path
os.chdir("/storage/remote/atcremers50/ss21_multiagentcontrol/src")

from config import *

#--------------------------------
## Step 1 - Get Input Data in Right Format
#--------------------------------

def step1_select_input_data(xml_data, printstat, simulation_max_duration):
    # read file into a dataframe
    df_raw = xml_to_df(xml_data)  # columns: time, id, x, y, angle, type, speed, pos, lane, slope
    
    # custom formatting
    df = format_df(df_raw) # columns: time, vehID, X, Y, yaw, type, speed, pos, lane, slope   

    # check all vehicles in the sim
    unique_vehicles = df.vehID.unique().tolist()
    printif(f"there are {len(unique_vehicles)} unique vehicles" , printstat) 
    
    #add intention
    df = add_intention(df, unique_vehicles)

    # decide to mask eventually for shorter simulations   
    all_timesteps = list(df.time.unique())
    if simulation_max_duration is None:
        limit_duration =  len(all_timesteps)
    else:
        limit_duration = int(min(simulation_max_duration,len(all_timesteps)))
    df_test = mask_timestep(df,all_timesteps[:limit_duration])
    printif(f"the simulation duration is {limit_duration} timesteps" , printstat) 
    
    return df_test, limit_duration