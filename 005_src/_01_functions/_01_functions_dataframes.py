from _00_configuration._00_settings import *
from _00_configuration._01_variables import *
from _01_functions._00_helper_functions import *

#--------------------------------
# MASKING
#--------------------------------

def mask_veh(df,veh):
    """
    function to mask a df by veh 
    """   
    if not isinstance(veh,list):
        mask = (df.vehID == str(veh))
        return df[mask]
    else:
        return df[df['vehID'].isin(veh)]

def mask_timestep(df,timestep,time_col = "timestep"):
    """
    function to mask a df by timestep 
    """
    if not isinstance(timestep,list):
        mask = (df[time_col] == timestep)
        return df[mask]
    else:
        return df[df[time_col].isin(timestep)]

#--------------------------------------------
# READ SIMULATION TXT FILES INTO DICT
#--------------------------------------------    
    
def read_txt_data(txt_data, split_char = ": "):
    """
    read files such as "../004_data/dataframes/df_sim_000005.txt"
    into a dictionary d['max_num_veh'] = '10'
    the split char is coming from the formatting of the txt files
    """
    d = {}
    with open(f"{txt_data}") as f:
        for line in f:
            if line != "\n":
                key, val,*other = line.split(split_char)
                d[key] = (val.strip())
    return d    

#--------------------------------
# OVERVIEW OF DATA AND PARAMETERS
#--------------------------------
def update_df_overview(txt_files= None,savestat = True):
    """
    loads updates and overwrites the table with the most recent entries
    """
    # read overview table
    file_name = f"OVERVIEW_TABLE"
    df_overview = pd.read_pickle(f"../004_data/{file_name}.pkl")
    
    if txt_files is None:
        # get files currently in the df
        existing_txt_files = set(list(df_overview.txt_file))
        # collect all current txt files from folder
        all_txt_files = glob.glob(os.path.join(GCN_INPUT_FOLDER, "*.txt"))
        # get the "new" txt files
        txt_files = [f for f in all_txt_files if f not in existing_txt_files]
    
    for txt_data in txt_files: 
    
        d1 = read_txt_data(txt_data)
        d = {}
        try: 
            d2 = read_txt_data(d1['info_input_df'])
            d["txt_file"] = txt_data
            d.update(d1)
            d.update(d2)
            df_overview = df_overview.append(d,ignore_index=True)
     
        except Exception as e:
            continue
    
    if savestat:
        # store pkl file with GCN input 
        file_to_write = open(f"../004_data/{file_name}.pkl", "wb")
        pkl.dump(df_overview, file_to_write)
    
    return df_overview

#--------------------------------
# CLEANING/FORMATTING DF
#--------------------------------

def change_columns_names(df,columns_names = columns_names_script):        
    df.rename(columns = columns_names, inplace = True)
    return df
    
def change_columns_format_to_numeric(df,columns_numeric = None):
    if columns_numeric is None:
        columns_numeric = [
                    #'timestep',
                    #'vehID',
                    'X','Y', 
                    'yaw','pos', 
                    #'lane', # this one has letters too
                    'speed',
                    'slope',
                    #'signals'
                    ]
    df[columns_numeric] = df[columns_numeric].apply(pd.to_numeric)
    return df

def change_vehicles_names(df, column_name = "vehID"):
    df[column_name]= df[column_name].map(lambda x: x[6:])
    return df

def format_df (df, 
               column_names = None, 
               columns_numeric = None, 
               source = ".xml", 
               shift_coordinates = (0,0),
               printstat = False,
              ):
    """
    custom df formatting 
    """
    if source == ".xml":
        printif(f"SOURCE {source}",printstat)
        printif(df.columns,printstat)
        printif(f"global variable {columns_names_xml}")
        df = change_columns_names(df,columns_names = columns_names_xml)
        df = change_vehicles_names(df, column_name = "vehID")
        df = change_columns_format_to_numeric(df,columns_numeric)
        
    elif source == ".zip":
        assert 'position (X,Y)' in df.columns
        X,Y = [],[]
        for e in list(map(eval,df['position (X,Y)'])): 
            X.append(e[0]+shift_coordinates[0])
            Y.append(e[1]+shift_coordinates[1])
        df.insert(2, 'Y', Y)
        df.insert(2, 'X', X)
        df.drop(columns=['position (X,Y)'] ,inplace = True)
        df = df.round(2)
        df = change_vehicles_names(df, column_name = "vehID")
        df = change_columns_format_to_numeric(df,columns_numeric)
        
    else:
        print ("accepting only xml,zip and csv files")
        
    return df

#--------------------------------
# ADD INTENTION COLUMN
#--------------------------------


def add_intention_absolute(df, 
                           unique_vehicles, 
                           time_col = "timestep", 

                          ):
    """
    this function returns the intention as one of 5 values: 
    # North = 0, East = 1, South = 2, West = 3, "else" or Center of Crossing = 4
    # 4 needed when simulation is interrupted before all cars have left 
    
    # upper and lower threshold depend on the geometry of the crossing
    # our crossing, un parsed, is centered in 100,100
    """
    #factor = 0.15
    upper_threshold_x = 175 # round(df.X.max())*(1-factor)
    lower_threshold_x = 25  # round(df.X.max())*factor
    upper_threshold_y = 175
    lower_threshold_y = 25
    
    
    for u in tqdm(unique_vehicles):
        
        # take only the rows where our vehicle is 
        mask_vehID = df['vehID'] == u
        df_vehicle = df[mask_vehID].sort_values(by=[time_col])
        
        ## initial position
        #first_row = df_vehicle.iloc[0]
        #X_i,Y_i = first_row.X.item(), first_row.y.item()
        
        ## final positions of the vehicle
        last_row = df_vehicle.iloc[-1]
        X_f,Y_f = last_row.X.item(), last_row.Y.item()
           
        
        if Y_f > upper_threshold_y: # moves north
            df.loc[mask_vehID, 'intention'] = 0
        elif X_f > upper_threshold_x: # moves east
            df.loc[mask_vehID, 'intention'] = 1
        elif Y_f < lower_threshold_y: # moves south
            df.loc[mask_vehID, 'intention'] = 2
        elif X_f < lower_threshold_x: # moves west
            df.loc[mask_vehID, 'intention'] = 3
        else: # moves center
            df.loc[mask_vehID, 'intention'] = 4 

    return df



def add_intention(df, unique_vehicles, time_col = "timestep"):
    for u in unique_vehicles:
        
        # take only the rows where our vehicle is 
        mask_vehID = df['vehID'] == u
        df_vehicle = df[mask_vehID]
        
        # the final destination is taken at the last timestep thevehicle appear
        # NOTE: I dont know if this is actually what we want
        max_timestemp = df_vehicle[time_col].max()
        min_timestemp = df_vehicle[time_col].min()
        
        # final position
        maxTime_X = df_vehicle.loc[df_vehicle[time_col] == max_timestemp, 'X'].item()
        maxTime_Y = df_vehicle.loc[df_vehicle[time_col] == max_timestemp, 'Y'].item()
        
        # initial position
        minTime_X = df_vehicle.loc[df_vehicle[time_col] == min_timestemp, 'X'].item()
        minTime_Y = df_vehicle.loc[df_vehicle[time_col] == min_timestemp, 'Y'].item()
        
        ## alternatively one could sort the df by the timestep
        # take the first row xy and the last row xy
        
        # zero center - can be probably deleted
        maxTime_X = maxTime_X - 100
        maxTime_Y = maxTime_Y - 100
        minTime_X = minTime_X - 100
        minTime_Y = minTime_Y - 100
        
        ## u-turn get the value 0, straight the value 1, left the value 2 and right the value 3
        # Case one (lower left corner)
        if minTime_X<0 and minTime_Y<0: #1
            if maxTime_X<0 and maxTime_Y>0: #U turn 0
                df.loc[df['vehID'] == u, 'intention'] = 0
                
            elif maxTime_X>0 and maxTime_Y<0: #Straight 1
                df.loc[df['vehID'] == u, 'intention'] = 1
                
            elif maxTime_X>0 and maxTime_Y>0: #Left 2
                df.loc[df['vehID'] == u, 'intention'] = 2
                
            elif maxTime_X<0 and maxTime_Y<0: #Right 3
                df.loc[df['vehID'] == u, 'intention'] = 3
        
        # Case two (lower right corner)
        elif minTime_X>0 and minTime_Y<0: #2
            if maxTime_X<0 and maxTime_Y<0: #U turn 0
                df.loc[df['vehID'] == u, 'intention'] = 0
                
            elif maxTime_X>0 and maxTime_Y>0: #Straight 1
                df.loc[df['vehID'] == u, 'intention'] = 1
                
            elif maxTime_X<0 and maxTime_Y>0: #Left 2
                df.loc[df['vehID'] == u, 'intention'] = 2
                
            elif maxTime_X>0 and maxTime_Y<0: #Right 3
                df.loc[df['vehID'] == u, 'intention'] = 3 
        
        # Case three (upper right corner)
        elif minTime_X>0 and minTime_Y>0: #3
            if maxTime_X>0 and maxTime_Y<0: #U turn 0
                df.loc[df['vehID'] == u, 'intention'] = 0
                
            elif maxTime_X<0 and maxTime_Y>0: #Straight 1
                df.loc[df['vehID'] == u, 'intention'] = 1
                
            elif maxTime_X<0 and maxTime_Y<0: #Left 2
                df.loc[df['vehID'] == u, 'intention'] = 2
                
            elif maxTime_X>0 and maxTime_Y>0: #Right 3
                df.loc[df['vehID'] == u, 'intention'] = 3            
        
        # Case four (upper left corner)
        elif minTime_X<0 and minTime_Y>0: #4
            if maxTime_X>0 and maxTime_Y>0: #U turn 0
                df.loc[df['vehID'] == u, 'intention'] = 0
                
            elif maxTime_X<0 and maxTime_Y<0: #Straight 1
                df.loc[df['vehID'] == u, 'intention'] = 1
                
            elif maxTime_X>0 and maxTime_Y<0: #Left 2
                df.loc[df['vehID'] == u, 'intention'] = 2
                
            elif maxTime_X<0 and maxTime_Y>0: #Right 3
                df.loc[df['vehID'] == u, 'intention'] = 3            

    return df    



print (f"Functions import successful")