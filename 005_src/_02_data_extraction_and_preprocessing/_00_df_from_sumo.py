## see https://www.pdc.kth.se/software/software/SUMO/centos7/1.0.0/index_using.html
## see https://www.eclipse.org/lists/sumo-user/msg03307.html
## https://sumo.dlr.de/pydoc/traci._vehicle.html
## https://sumo.dlr.de/pydoc/traci._trafficlight.html
#!/usr/bin/env python 
"""
Name: df_from_sumo.py
Time: 2021/06/22 16:00
Desc: aligning with output from xml file
"""
# ==================================================================================================
# -- env.    ---------------------------------------------------------------------------------------
# ==================================================================================================

"""
requires carla_sumo_env
"""

# ==================================================================================================
# -- imports ---------------------------------------------------------------------------------------
# ==================================================================================================
import os, glob, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import datetime
import pdb

sys.path.insert(0, '../005_src/') #use relative path
os.chdir("/storage/remote/atcremers50/ss21_multiagentcontrol/005_src/")

from _01_functions._02_functions_xml import *

# Import traci module
#import traci  # pylint: disable=wrong-import-position

# ==================================================================================================
# -- utility functions -----------------------------------------------------------------------------
# ==================================================================================================

def pad(string,p = "0", minl= 2):
    l = len(str(string))
    if l < minl: 
        return str(p*(minl-l))+str(string)
    else:
        return str(string)

def get_timestamp(sep= "-"):
    """
    returns current time as a string in format %hours%minutes%seconds
    used for saving files and directories 
    """
    now = datetime.datetime.now()
    time_obj = now.strftime("%Hh%Mm%Ss")
    return time_obj

def get_date(sep= "-"):
    """
    returns current date as a string in format %year%month%day
    used for saving files and directories 
    """
    date_obj= datetime.date.today()
    date_obj = date_obj.strftime("%Y%m%d")+sep
    return date_obj
    
# ==================================================================================================
# -- find carla module -----------------------------------------------------------------------------
# ==================================================================================================
print ()
try:
    sys.path.append(
        glob.glob('../Carla_0.9.11/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' %
                  (sys.version_info.major, sys.version_info.minor,
                   'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    print ("Carla installation found")
except IndexError:
    pass

# ==================================================================================================
# -- find traci module -----------------------------------------------------------------------------
# ==================================================================================================

if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
    sumo_home = os.environ['SUMO_HOME']
    print (f"SUMO_HOME found {sumo_home}")
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

# ==================================================================================================
# -- sumo integration imports ----------------------------------------------------------------------
# ==================================================================================================
sys.path.insert(0, '../Carla_0.9.11/Co-Simulation/Sumo/')


# ==================================================================================================
# -- main                     ----------------------------------------------------------------------
# ==================================================================================================

## variables
# env vars
date = get_date()
ts = get_timestamp()
running_usr = os.getenv('HOME')
running_env = os.getenv('CONDA_PREFIX')

max_num_veh = 5
group = 0
previous_IDlist = None
num_veh_per_timeframe = {}
num_col_per_timeframe = {}
veh_params_list = []

# saving vars
savestat = True
plotstat = True
printstat = False

## load configure file 
print ("insert number of configuration file to load eg 3")
conf_ID = int(input())
conf_ID = '%03d' %conf_ID
conf_file = f"../004_data/cross_{conf_ID}/cross.sumocfg"
route_file = f"../004_data/cross_{conf_ID}/cross.rou.xml"
flow_info_list = get_flow_details(route_file, spacing_factor = 5)
print (f"\nrunning configuration:{conf_ID}\n")

# simulation vars
step = 0
print ("for how long do you want the simulation to last?")
laststep = int(input())
#laststep = 100500 

## create command 
print (f"path exists: {os.path.exists(conf_file)}")
sumo_cmd=["sumo", "-c", conf_file,"--max-num-vehicles",f"{max_num_veh}"] # CHANGE

#Start the traci interface
traci.start(sumo_cmd, port=8813, label="sim1")
conn1 = traci.getConnection("sim1")
conn1.setOrder(1)

start = time.time()
# Run the simulations
while step <= laststep:

    traci.simulationStep( )
    
    # get vehicles IDS 
    IDList = traci.vehicle.getIDList()
    
    if previous_IDlist is not None:
        if IDList != previous_IDlist:
            group += 1
    
    num_collisions = traci.simulation.getCollidingVehiclesNumber()
    
    # get vehicles parameters
    # find more and more here: https://sumo.dlr.de/docs/TraCI/Vehicle_Value_Retrieval.html
    # https://sumo.dlr.de/daily/pydoc/traci._simulation.html#SimulationDomain-getCollidingVehiclesNumber
    # https://sumo.dlr.de/daily/pydoc/traci._vehicle.html
    for vehID in IDList:
        veh_params = {
                      ## as from xml
                      "timestep":step,
                      "vehID": vehID,
                      "position (X,Y)": traci.vehicle.getPosition(vehID) , # X,Y
                      "yaw": traci.vehicle.getAngle(vehID) ,            
                      "type": traci.vehicle.getTypeID(vehID) ,
                      "speed": traci.vehicle.getSpeed(vehID) ,
                      "pos": traci.vehicle.getLanePosition(vehID),
                      "lane": traci.vehicle.getLaneID(vehID) ,
                      "slope": traci.vehicle.getSlope(vehID) ,
            
                      ## additional 
                      "group": group, # timesteps belonging to the same group are consistent and can be used to train the model
                      "num_collisions": num_collisions, 
                     }
        veh_params_list.append(veh_params)
    
    
    num_veh_per_timeframe[step] = len(IDList)
    num_col_per_timeframe[step] = int(num_collisions)
    previous_IDlist = IDList
        
    step += 1

end = time.time()

print (f" simulation duration: {(end - start)} ")

# ==================================================================================================
# -- plot                     ----------------------------------------------------------------------
# ==================================================================================================
if plotstat: 

    #directories
    sumo_data_plot_path = f"../004_data/figures/plots_data/"
    Path(sumo_data_plot_path).mkdir(parents=True, exist_ok=True)
    all_dirs = os.listdir(sumo_data_plot_path)
    all_png = [f for f in all_dirs if f.endswith(".png")]
    # add one to the name
    l_im = '%06d' % (len(all_png)+1)
    new_png_path = os.path.join(sumo_data_plot_path,f'{l_im}_veh_per_timestep.png')
    if printstat:
        print (f"new plot in {new_png_path}")

    #plot
    last_veh = veh_params_list[-1]["vehID"]
    y1 = list(num_veh_per_timeframe.values())
    y2 = list(num_col_per_timeframe.values())
    x = list(num_veh_per_timeframe.keys())
    
    #print (data)

    plt.xlim([min(x)-1, max(x)+1])

    plt.plot(x,y1, c="blue", label= "num vehicles")
    plt.plot(x,y2, c="red", label= "num collisions")
    plt.legend(loc='upper right')
    plt.title(f'Simulation {l_im} overview,\nlast vehicle: {last_veh},\nconfiguration: cross_{conf_ID}')
    plt.xlabel('timesteps')
    plt.ylabel('count')
    #plt.show()
    
    
    
    plt.savefig(new_png_path)
    print (f"Plot saved in {new_png_path}")
    
else:
    l_im = "None"
    hist_path = "No plotting"
# ==================================================================================================
# -- save dataframe           ----------------------------------------------------------------------
# ==================================================================================================

# store as pandas dataframe
df = pd.DataFrame(veh_params_list)
if printstat: 
    print ()
    print (df.head())
    print ()
    print (df.tail())

if savestat:
    # create dataframe path
    dataframes_path = "../004_data/dataframes/"
    print (f"path exists: {os.path.exists(dataframes_path)}")
    # check num of existing files
    all_dirs = os.listdir(dataframes_path)
    all_zip_files = [f for f in all_dirs if f.endswith(".zip")]
    # add one to the name
    l = '%06d' % (len(all_zip_files)+1)
    
    print (f"figure:{l_im}, data ID:{l}")

    # save the dataframe
    df_name = f"df_sim_{l}"
    df_savepath = os.path.join(dataframes_path,f"{df_name}.zip")
    compression_opts = dict(method='zip',
                            archive_name=f"{df_name}.csv")  
    df.to_csv(df_savepath, index=False,
              compression=compression_opts) 
    print (f"SUMO DATAFRAME SAVED IN: {df_savepath}")
    
    
    ## create txt file that contains all the information about the simulation
    # create the path in the same folder, with a correspondent name, just different extension
    ## NOTE: dont leave spaces between words, only when you want to id a new element (key_text: v)
    DF_INFO = os.path.join(dataframes_path,f"{df_name}.txt")
    with open(DF_INFO, 'w') as filehandle:
        filehandle.write(f'max_num_veh: {max_num_veh}\n')
        filehandle.write(f'\n')
        filehandle.write(f'date: {date}\n')
        filehandle.write(f'time: {ts}\n')
        filehandle.write(f'usr: {running_usr}\n') 
        filehandle.write(f'env: {running_env}\n')       
        filehandle.write(f'\n')
        filehandle.write(f'sim_duration_timesteps: {laststep}\n')
        filehandle.write(f'sim_duration_actualseconds: {(end - start)}\n')
        filehandle.write(f'\n')
        filehandle.write(f'savestat: {savestat}\n')
        filehandle.write(f'plotstat: {plotstat}\n')
        filehandle.write(f'\n')
        filehandle.write(f'conf_ID: {conf_ID}\n')
        filehandle.write(f'conf_file: {conf_file}\n')
        filehandle.write(f'conf_file: {route_file}\n')
        filehandle.writelines([f"{row}\n" for row in flow_info_list])
        filehandle.write(f'\n')
        filehandle.write(f'path_figure: {new_png_path}\n')
        filehandle.write(f'path_df: {df_savepath}\n')
        filehandle.close()
    
    
## end of simulation 
traci.close()




