from _00_configuration._00_settings import *
from _00_configuration._01_variables import *
from _01_functions._00_helper_functions import *

#--------------------------------
# MASKING
#--------------------------------

def mask_veh(df,veh):
    if not isinstance(veh,list):
        mask = (df.vehID == str(veh))
        return df[mask]
    else:
        #mask = (df.vehID in veh)
        return df[df['vehID'].isin(veh)]

def mask_timestep(df,timestep,columnname = "time"):
    if not isinstance(timestep,list):
        mask = (df[columnname] == str(timestep))
        return df[mask]
    else:
        #mask = (df.vehID in veh)
        return df[df[columnname].isin(timestep)]


#--------------------------------
# CLEANING/FORMATTING DF
#--------------------------------

def change_columns_names(df,column_names = None):
    if column_names is None:
        columns_names = {'id':'vehID', 
                'x':'X', 
                'y':'Y', 
                'angle':'yaw', 
                'type':'type', 
                'speed':'speed', 
                'pos':'pos', #? 
                'lane':'lane', 
                'slope':'slope',
                #'signals':'signals', # traffic lights? 
        }
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
    #df[column_name]= df[column_name].map(lambda x: x.lstrip('flow1.'))
    df[column_name]= df[column_name].map(lambda x: x[6:])
    return df

def format_df (df, column_names = None, columns_numeric = None, source = ".xml", shift_coordinates = (0,0)):
    """
    custom df formatting 
    """
    if source == ".xml":
        df = change_columns_names(df,column_names)
        df = change_vehicles_names(df, column_name = "vehID")
        df = change_columns_format_to_numeric(df,columns_numeric)
        
    elif source.isin([".zip",".csv"]):
        assert 'position (X,Y)' in df.columns
        X,Y = [],[]
        for e in list(map(eval,df['position (X,Y)'])): 
            X.append(e[0]+shift_coordinates[0])
            Y.append(e[1]+shift_coordinates[1])
        df.insert(2, 'Y', Y)
        df.insert(2, 'X', X)
        df.drop(columns=['position (X,Y)'] ,inplace = True)
        
    else:
        print ("accepting only xml,zip and csv files")
        
    return df


#--------------------------------
# PLOTTING 
#--------------------------------

def plot_2D(df,palette = "Paired" , 
            legend = "True", 
            grid = "True",
            figsize=(30,30),
            linewidth=7.0):
    """
    visualization of paths
    """
    unique_veh = df.vehID.unique().tolist()
    l = len(unique_veh)
    print (f"{l} unique vehicles found in the simulation")
    if l < 10: 
        print (*unique_veh, sep = "\n")
    cmap = list(sns.color_palette(palette,len(unique_veh)).as_hex())
    fig, ax = plt.subplots(figsize= figsize)
    ax.set_title("2D map");
    ax.set_xlabel("X");
    ax.set_ylabel("Y");
    for i,veh in enumerate(unique_veh):
        df_veh = mask_veh(df,veh)
        if "X" in df_veh.columns:
            X = df_veh.X.tolist()
            Y = df_veh.Y.tolist()
            #ax.plot(X,Y,cmap[i], label = str(veh) ,linewidth=linewidth)
        else: 
            pos_veh = list(map(eval, df_veh.Position3D))
            X,Y = list(), list()
            for row in pos_veh: 
                X.append(row[0])
                Y.append(row[1])

        ax.plot(X,Y,cmap[i], label = str(veh) ,linewidth=linewidth)
        if legend:
            ax.legend()
        if grid:
            ax.grid(grid)
            ax.xaxis.grid(True, which='minor')

        for i,x in enumerate(X):
            if i% 7 == 0:
                ax.annotate(str(i), (x, Y[i]), ha='center', va='center', size=14)

print (f"Functions import successful")


#--------------------------------
# ADD INTENTION COLUMN
#--------------------------------
def add_intention(df, unique_vehicles):
    for u in unique_vehicles:
        mask_vehID = df['vehID'] == u
        df_vehicle = df[mask_vehID]
        max_timestemp = df_vehicle['time'].max()
        min_timestemp = df_vehicle['time'].min()
        
        maxTime_X = df_vehicle.loc[df_vehicle['time'] == max_timestemp, 'X'].item()
        maxTime_Y = df_vehicle.loc[df_vehicle['time'] == max_timestemp, 'Y'].item()
        
        minTime_X = df_vehicle.loc[df_vehicle['time'] == min_timestemp, 'X'].item()
        minTime_Y = df_vehicle.loc[df_vehicle['time'] == min_timestemp, 'Y'].item()
        
        #zero center - can be probably deleted
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
