from _00_configuration._00_settings import *
from _00_configuration._01_variables import *
from _01_functions._00_helper_functions import *
from pandas.core.common import flatten

def is_veh_still(data_pos, data_y):
    """
    compares the initial location as in data_pos 
    with the future location as in data_y (first two values)
    if the location is the same, returns True, as the vehicle is still
    output is a newlist to be assigned as: df[newcolumn] = newlist
    """
    return [[veh[:2] == data_pos[i]] for i,veh in enumerate(data_y)]

def zero_center_coordinates(coord, delta = 100):
    """
    moves x and y of delta given
    not really flex but covers our use
    """
    return ([coord[0]-delta,coord[1]-delta])

def zero_center_column(column, delta = 100):
    """
    shifts of minus delta the first two numbers in each element of each column
    output is a newlist to be assigned as: df[newcolumn] = newlist
    """
    return [zero_center_coordinates(feat[:2],delta) for i,feat in enumerate(column)]

def get_data_x_radiants(data_x):
    """
    data_x = [yaw in degrees, speed,intention]
    data_x_output = [yaw in radians, speed,intention]
    """
    return [[math.radians(feat) if i == 0 else feat 
             for i, feat in enumerate(veh)] 
            for veh in data_x]

def get_yaw_separately(data_y):
    """
    yaw in a separate column
    output is a newlist to be assigned as: df[newcolumn] = newlist
    also converts to radians
    """
    return [[math.radians(feat[2])] for feat in data_y]

def get_yaw_from_data_x(data_x):
    """
    yaw in a separate column, x_speed
    if after get_data_x_radiants then it is already in radiants  
    """
    return [veh[0] for veh in data_x]


def get_speed_from_data_x(data_x):
    """
    speed in a separate column, x_speed
    """
    return [[veh[1]] for veh in data_x]

def get_intention_from_data_x(data_x):
    """
    intention in a separate column, x_speed
    """
    return [[veh[2]] for veh in data_x]

def get_accel_from_data_x(data_x):
    """
    acceleration in a separate column, x_speed
    """
    return [[veh[3]] for veh in data_x]


def check_all_veh_moving(Still_vehicle):
    """
    are all vehicles in the frame moving ?
    this is to choose the shuttle
    """
    for i, bool_val in enumerate(Still_vehicle):
        
        try:
            if bool_val[0]: # bool_val is class list 
                return False
            else:
                continue
        except Exception as e:
            print (Exception)
            return False
    return True


def delta_output_2(data_y_zc, data_pos_zc):
    """
    Get instead of the absolute position of the vehicles the delta position
    """
        
    target_pos = np.array(data_y_zc)  # target position x,y at t+deltat
    initial_pos = np.array(data_pos_zc) # initial position x,y at t 
    try:
        diff_list = (target_pos-initial_pos).tolist()
        return diff_list
    except Exception as e: 
        print (e)
        print (f"{initial_pos=}")
        print (f"{target_pos=}")
        print (" you have probably selected the whole df, not only the coherent rows, select only rows with equal number of cars in t and t+delta")
        pdb.set_trace()

def delta_output(data_y_zc, data_pos_zc):
    """
    Get instead of the absolute position of the vehicles the delta position
    """
    return (np.array(data_y_zc)-np.array(data_pos_zc)).tolist()


def min_max_normalization(input_column, min_max_list):
    """
    normalize the column via a min max normalization (x_i-min)/(max-min)
    """
    min_val=min_max_list[0]
    max_val=min_max_list[1]
    
    
    min_val = np.array(min_val)  
    max_val = np.array(max_val)
    input_column = np.array(input_column)
    
    first_divisor=np.subtract(input_column, min_val)
    
    second_divisor=np.subtract(max_val, min_val)
    normalized_data=np.divide(first_divisor, second_divisor)
    normalized_data_list = normalized_data.tolist()
    
    return normalized_data_list

def count_num_veh(row):
    """
    row functions for lambda fun. counting the number of vehicles moving and not moving 
    """
    list_vehicles  = list_flatten(row.Still_vehicle)
    return (len(list_vehicles))

def count_num_veh_still(row):
    """
    row functions for lambda fun. counting the number of vehicles not moving 
    """
    list_vehicles  = list_flatten(row.Still_vehicle)
    num_veh_still = sum(list_vehicles)
    return num_veh_still

def count_num_veh_moving (row):
    """
    row functions for lambda fun. counting the number of vehicles moving 
    """
    list_vehicles  = list_flatten(row.Still_vehicle)
    num_veh_still = sum(list_vehicles)
    num_veh_moving = len(list_vehicles)-num_veh_still
    return num_veh_moving

#--------------------------------
# stuff for classification 
#--------------------------------
def create_grid_range(min_value = 0,
                      max_value = 199, 
                      delta = 20):
    grid_range = [(e, e+20) for e in list(range(min_value,max_value,delta))]
    all_ranges_xy = {e:class_num for class_num,e in enumerate(list(product(grid_range,grid_range)))}
    
    return all_ranges_xy

def get_classes_for_point_list(point_list,printstat = False):
    
    all_ranges_xy = create_grid_range()
    classes_points = []
    printif (f" ", printstat)
    for p in point_list:
        printif (f"{p=}", printstat)
        for i,range_xy in enumerate(list(all_ranges_xy.keys())):
            range_x = range_xy[0]
            range_y = range_xy[1]
            #printif (f"{range_x=},{range_y=}", printstat)
            if (range_x[0]<= p[0] <= range_x[1]) and (range_y[0]<= p[1] <= range_y[1]):
                printif (f"{all_ranges_xy[range_xy]=}", printstat)
                classes_points.append([all_ranges_xy[range_xy]])
                break
    return classes_points

# function to create classes
def get_all_classes(data_pos,printstat = False):
    return get_classes_for_point_list(data_pos,printstat = printstat)

        
def adjust_columns_2(df_all, 
                     return_subselection_of_dataset = False,
                     return_selected_columns=[
                                            'data_x_speed',
                                            #'data_x_acceleration',
                                            'data_x_yaw',
                                            'data_x_intention',
                                            'data_x_rad',
                                            'data_pos_zc',
                                            'data_y_zc',
                                            'data_y_yaw',
                                            'Still_vehicle',
                                            'all_veh_moving',
                                            'data_y_delta',
                                            'data_classes',
                                            'data_y_classes',
                                           ],
                     printstat = True
                    ):
    """
    applies various adjustments to the columns of df_all
    """
    #--------------------------------
    ## create and add columns
    #--------------------------------
    traffic_info = df_all.apply(lambda x: is_veh_still(x['data_pos'],x['data_y']),axis=1)
    df_all.loc[:,"Still_vehicle"] = traffic_info
    
    data_pos_zc = df_all.apply(lambda x: zero_center_column(x['data_pos']),axis=1)
    df_all.loc[:,"data_pos_zc"] = data_pos_zc
    
    data_y_zc = df_all.apply(lambda x: zero_center_column(x['data_y']),axis=1)
    df_all.loc[:,"data_y_zc"] = data_y_zc
    
    data_y_yaw = df_all.apply(lambda x: get_yaw_separately(x['data_y']),axis=1)
    df_all.loc[:,"data_y_yaw"] = data_y_yaw
    
    data_x_rad = df_all.apply(lambda x: get_data_x_radiants(x['data_x']),axis=1)
    df_all.loc[:,"data_x_rad"] = data_x_rad
    
    data_x_speed = df_all.apply(lambda x: get_speed_from_data_x(x['data_x']),axis=1)
    df_all.loc[:,"data_x_speed"] = data_x_speed
    
    data_x_yaw = df_all.apply(lambda x: get_yaw_from_data_x(x['data_x']),axis=1)
    df_all.loc[:,"data_x_yaw"] = data_x_yaw
    
    data_x_intention = df_all.apply(lambda x: get_intention_from_data_x(x['data_x']),axis=1)   
    df_all.loc[:,"data_x_intention"] = data_x_intention
    
    # get delta column
    data_y_delta = df_all.apply(lambda x: delta_output_2(x['data_y_zc'],x['data_pos_zc']),axis=1)  
    df_all.loc[:,"data_y_delta"] = data_y_delta                                               
    
    # add info on number of vehicles moving per frame
    all_veh_moving = df_all.apply(lambda x: check_all_veh_moving(x['Still_vehicle']),axis=1)
    df_all.loc[:,"all_veh_moving"] = all_veh_moving
    
    df_all.loc[:,"num_veh_tot"] = df_all.apply (lambda row: count_num_veh(row), axis=1)
    df_all.loc[:,"num_veh_moving"] = df_all.apply (lambda row: count_num_veh_moving(row), axis=1)
    df_all.loc[:,"num_veh_still"] = df_all.apply (lambda row: count_num_veh_still(row), axis=1)

    #======================================================================================================
    # stuff for classification 
    df_all.loc[:,"data_classes"] = df_all.apply(lambda x: get_all_classes(x['data_pos']),axis=1)
    df_all.loc[:,"data_y_classes"] = df_all.apply(lambda x: get_all_classes(x['data_y']),axis=1)
    
    #======================================================================================================

    if len(df_all.data_x.iloc[0][0]) == 4: # means there is also the acceleration column
        df_all.loc[:,"data_x_acceleration"] = df_all.apply (lambda x: get_accel_from_data_x(x['data_x']), axis=1)
    
    if return_subselection_of_dataset:
        printif(return_selected_columns,printstat)
        #pdb.set_trace()
        return df_all[return_selected_columns]
    
    printif (df_all.Still_vehicle, printstat)
    return df_all


def adjust_columns(df_all, return_subselection_of_dataset = False):
    #--------------------------------
    ## create and add columns
    #--------------------------------
    traffic_info = df_all.apply(lambda x: is_veh_still(x['data_pos'],x['data_y']),axis=1)
    data_pos_zc = df_all.apply(lambda x: zero_center_column(x['data_pos']),axis=1)
    data_y_zc = df_all.apply(lambda x: zero_center_column(x['data_y']),axis=1)
    data_y_yaw = df_all.apply(lambda x: get_yaw_separately(x['data_y']),axis=1)
    data_x_rad = df_all.apply(lambda x: get_data_x_radiants(x['data_x']),axis=1)
    
    
    df_all.loc[:,"data_pos_zc"] = data_pos_zc
    df_all.loc[:,"Still_vehicle"] = traffic_info
    df_all.loc[:,"data_y_zc"] = data_y_zc
    df_all.loc[:,"data_y_yaw"] = data_y_yaw
    df_all.loc[:,"data_x_rad"] = data_x_rad
    
    # get delta column
    data_y_delta = df_all.apply(lambda x: delta_output(x['data_y_zc'],x['data_pos_zc']),axis=1)  
    df_all.loc[:,"data_y_delta"] = data_y_delta                                               
    
    # add info on number of vehicles moving per frame
    all_veh_moving = df_all.apply(lambda x: check_all_veh_moving(x['Still_vehicle']),axis=1)
    df_all.loc[:,"all_veh_moving"] = all_veh_moving
    
    df_all["num_veh_tot"] = df_all.apply (lambda row: count_num_veh(row), axis=1)
    df_all["num_veh_moving"] = df_all.apply (lambda row: count_num_veh_moving(row), axis=1)
    df_all["num_veh_still"] = df_all.apply (lambda row: count_num_veh_still(row), axis=1)

    
    if return_subselection_of_dataset:
        pdb.set_trace()
        return df_all[[
                        'data_x',
                        'data_x_rad',
                        'data_pos_zc',
                        'data_y_zc',
                        'data_y_yaw',
                        'Still_vehicle',
                        'all_veh_moving',
                        'data_y_delta',   
                        
       ]]
    
    return df_all


def adjust_columns_3(df_all, return_subselection_of_dataset = False):
    #--------------------------------
    ## create and add columns
    #--------------------------------
    traffic_info = df_all.apply(lambda x: is_veh_still(x['data_pos'],x['data_y']),axis=1)
    data_pos_zc = df_all.apply(lambda x: zero_center_column(x['data_pos']),axis=1)
    data_y_zc = df_all.apply(lambda x: zero_center_column(x['data_y']),axis=1)
    data_y_yaw = df_all.apply(lambda x: get_yaw_separately(x['data_y']),axis=1)
    data_x_rad = df_all.apply(lambda x: get_data_x_radiants(x['data_x']),axis=1)
    
    
    df_all.loc[:,"data_pos_zc"] = data_pos_zc
    df_all.loc[:,"Still_vehicle"] = traffic_info
    df_all.loc[:,"data_y_zc"] = data_y_zc
    df_all.loc[:,"data_y_yaw"] = data_y_yaw
    df_all.loc[:,"data_x_rad"] = data_x_rad
    
    # get delta column
    data_y_delta = df_all.apply(lambda x: delta_output(x['data_y_zc'],x['data_pos_zc']),axis=1)  
    df_all.loc[:,"data_y_delta"] = data_y_delta                                               
    
    # add info on number of vehicles moving per frame
    all_veh_moving = df_all.apply(lambda x: check_all_veh_moving(x['Still_vehicle']),axis=1)
    df_all.loc[:,"all_veh_moving"] = all_veh_moving
    
    df_all["num_veh_tot"] = df_all.apply (lambda row: count_num_veh(row), axis=1)
    df_all["num_veh_moving"] = df_all.apply (lambda row: count_num_veh_moving(row), axis=1)
    df_all["num_veh_still"] = df_all.apply (lambda row: count_num_veh_still(row), axis=1)

    
    if return_subselection_of_dataset:
        pdb.set_trace()
        return df_all[[
                        'data_x_rad',
                        'data_pos_zc',
                        'data_y_zc',
                        'data_y_yaw',
                        'Still_vehicle',
                        'all_veh_moving',
                        'data_y_delta',   
                        
       ]]
    
    
    return df_all


def get_min_max_of_column(input_column, value):
    """
    get the min and max value for each value in a column that consists of a nested list
    """
    
    input_column=input_column.to_list()
    input_column= list(flatten(input_column))
    
    #with 3 values
    if value==3:
        first_elem_list = input_column[0::3]
        second_elem_list = input_column[1::3]
        third_elem_list = input_column[2::3]

        max_first=max(first_elem_list)
        max_second=max(second_elem_list)
        max_third=max(third_elem_list)
        min_first=min(first_elem_list)
        min_second=min(second_elem_list)
        min_third=min(third_elem_list)
        
        output = [[min_first,min_second,min_third],[max_first,max_second,max_third]]
        print(output)
    
    #with 2 values
    elif value==2:
        first_elem_list = input_column[0::2]
        second_elem_list = input_column[1::2]

        max_first=max(first_elem_list)
        max_second=max(second_elem_list)
        min_first=min(first_elem_list)
        min_second=min(second_elem_list)
        
        output = [[min_first,min_second],[max_first,max_second]]
    
    #with 1 values
    elif value==1:
        first_elem_list = input_column[0::1]

        max_first=max(first_elem_list)
        min_first=min(first_elem_list)
        
        output = [[min_first],[max_first]]
    
    
    return output


def min_max_normalization(input_column, min_max_list):
    """
    normalize input column via a min max normalization, i.e. via (x_i-min)/(max-min)
    """
    min_val=min_max_list[0]
    max_val=min_max_list[1]
    
    
    min_val = np.array(min_val)  
    max_val = np.array(max_val)
    input_column = np.array(input_column)
    
    first_divisor=np.subtract(input_column, min_val)
    
    second_divisor=np.subtract(max_val, min_val)
    normalized_data=np.divide(first_divisor, second_divisor)
    normalized_data_list = normalized_data.tolist()
    
    return normalized_data_list

def min_max_normalize_columns(df_all):
    """
    select which columns to normalize via min max normalization
    """    
    
    #3 values
    min_max_x_rad_list=get_min_max_of_column(df_all['data_x_rad'], 3)
    data_x_rad_norm = df_all.apply(lambda x: min_max_normalization(x['data_x_rad'],min_max_x_rad_list),axis=1)
    df_all.loc[:,"data_x_rad_norm"] = data_x_rad_norm
    
    #2 values
    min_max_pos_zc_list=get_min_max_of_column(df_all['data_pos_zc'], 2)
    data_x_pos_norm = df_all.apply(lambda x: min_max_normalization(x['data_pos_zc'],min_max_pos_zc_list),axis=1)
    df_all.loc[:,"data_pos_zc_norm"] = data_x_pos_norm
    
    #2 values
    min_max_y_zc_list=get_min_max_of_column(df_all['data_y_zc'], 2)
    data_y_zc_norm = df_all.apply(lambda x: min_max_normalization(x['data_y_zc'],min_max_y_zc_list),axis=1)
    df_all.loc[:,"data_y_zc_norm"] = data_y_zc_norm
    
    #1 value
    min_max_y_yaw_list=get_min_max_of_column(df_all['data_y_yaw'], 1)
    data_y_yaw_norm = df_all.apply(lambda x: min_max_normalization(x['data_y_yaw'],min_max_y_yaw_list),axis=1)
    df_all.loc[:,"data_y_yaw_norm"] = data_y_yaw_norm
    
    return df_all


def get_mean_std_of_column(input_column, value):
    """
    get the mean and the standard deviation for each value in a column that consists of a nested list
    """
    
    input_column=input_column.to_list()
    input_column= list(flatten(input_column))
    
    #with 3 values
    if value==3:
        
        first_elem_list = input_column[0::3]
        second_elem_list = input_column[1::3]
        third_elem_list = input_column[2::3]
        
        f_elem_list=np.array(first_elem_list)
        s_elem_list=np.array(second_elem_list)
        t_elem_list=np.array(third_elem_list)
        
        std_first_elem=np.std(f_elem_list)
        std_second_elem=np.std(s_elem_list)
        std_third_elem=np.std(t_elem_list)
        
        mean_first_elem=np.mean(f_elem_list)
        mean_second_elem=np.mean(s_elem_list)
        mean_third_elem=np.mean(t_elem_list)
        
        output = [[mean_first_elem, mean_second_elem, mean_third_elem],[std_first_elem, std_second_elem, std_third_elem]]
    
    #with 2 values
    elif value==2:
        first_elem_list = input_column[0::2]
        second_elem_list = input_column[1::2]

        f_elem_list=np.array(first_elem_list)
        s_elem_list=np.array(second_elem_list)
        
        std_first_elem=np.std(f_elem_list)
        std_second_elem=np.std(s_elem_list)
        
        mean_first_elem=np.mean(f_elem_list)
        mean_second_elem=np.mean(s_elem_list)
        
        output = [[mean_first_elem, mean_second_elem],[std_first_elem, std_second_elem]]
    
    #with 1 values
    elif value==1:
        first_elem_list = input_column[0::1]
        f_elem_list=np.array(first_elem_list)
        
        std_first_elem=np.std(f_elem_list)
        
        mean_first_elem=np.mean(f_elem_list)
        
        output = [[mean_first_elem],[std_first_elem]]
    
    
    return output


def standard_normalization(input_column, standard_mean_list):
    """
    normalize the column via a standard normalization, i.e. via (x_i-mean)/standard deviation
    """
        
    mean_val=standard_mean_list[0]
    std_val=standard_mean_list[1]
    
    mean_val = np.array(mean_val)  
    std_val = np.array(std_val)
    
    input_column = np.array(input_column)
    
    first_divisor=np.subtract(input_column, mean_val)
    
    normalized_data=np.divide(first_divisor, std_val)
    normalized_data_list = normalized_data.tolist()
    
    return normalized_data_list


def standard_normalize_columns(df_all):
    """
    select which columns to normalize via standard normalization  
    """    
    
    #3 values
    standard_x_rad_list=get_mean_std_of_column(df_all['data_x_rad'], 3)
    data_x_rad_norm = df_all.apply(lambda x: standard_normalization(x['data_x_rad'],standard_x_rad_list),axis=1)
    df_all.loc[:,"data_x_rad_norm"] = data_x_rad_norm
    
    #2 values
    standard_pos_zc_list=get_mean_std_of_column(df_all['data_pos_zc'], 2)
    data_x_pos_norm = df_all.apply(lambda x: standard_normalization(x['data_pos_zc'],standard_pos_zc_list),axis=1)
    df_all.loc[:,"data_pos_zc_norm"] = data_x_pos_norm
    
    #2 values
    standard_y_zc_list=get_mean_std_of_column(df_all['data_y_zc'], 2)
    data_y_zc_norm = df_all.apply(lambda x: standard_normalization(x['data_y_zc'],standard_y_zc_list),axis=1)
    df_all.loc[:,"data_y_zc_norm"] = data_y_zc_norm
    
    #1 value
    standard_y_yaw_list=get_mean_std_of_column(df_all['data_y_yaw'], 1)
    data_y_yaw_norm = df_all.apply(lambda x: standard_normalization(x['data_y_yaw'],standard_y_yaw_list),axis=1)
    df_all.loc[:,"data_y_yaw_norm"] = data_y_yaw_norm
    
    return df_all


def single_row_standard_normalize(row, amount):
    """
    standardize per row
    """    
    arr = np.array(row)
    
    if amount==3:
        first_col=arr[:,0]
        sec_col=arr[:,1]
        third_col=arr[:,2]

        mean_first_elem=np.mean(first_col)
        mean_second_elem=np.mean(sec_col)
        mean_third_elem=np.mean(third_col)

        std_first_elem=np.std(first_col)
        std_second_elem=np.std(sec_col)
        std_third_elem=np.std(third_col)
        
        if std_first_elem == 0:
            std_first_elem=1
            mean_first_elem=0
        
        if std_second_elem == 0:
            std_second_elem=1
            mean_second_elem=0
        
        if std_third_elem == 0:
            std_third_elem=1
            mean_third_elem=0
            
    
        list_mean=[mean_first_elem, mean_second_elem, mean_third_elem]
        list_std=[std_first_elem, std_second_elem, std_third_elem]

        first_divisor=np.subtract(row, list_mean)

    if amount==2:
        first_col=arr[:,0]
        sec_col=arr[:,1]

        mean_first_elem=np.mean(first_col)
        mean_second_elem=np.mean(sec_col)

        std_first_elem=np.std(first_col)
        std_second_elem=np.std(sec_col)

        if std_first_elem == 0:
            std_first_elem=1
            mean_first_elem=0

        if std_second_elem == 0:
            std_second_elem=1
            mean_second_elem=0


        list_mean=[mean_first_elem, mean_second_elem]
        list_std=[std_first_elem, std_second_elem]
        
        
    if amount==1:
        first_col=arr[:,0]

        mean_first_elem=np.mean(first_col)

        std_first_elem=np.std(first_col)

        if std_first_elem == 0:
            std_first_elem=1
            mean_first_elem=0

        list_mean=[mean_first_elem]
        list_std=[std_first_elem]
        

    first_divisor=np.subtract(row, list_mean)

    normalized_data=np.divide(first_divisor, list_std)
    normalized_data_list = normalized_data.tolist()

    return normalized_data_list
   

def standard_normalize_graph(df_all):
    """
    select which columns to normalize via standard normalization per graph
    """    
    
    #3 values
    list_norm_x_rad_col=[]
    for ind in df_all.index:
        normalized_row=single_row_standard_normalize(df_all['data_x_rad'][ind], 3)
        list_norm_x_rad_col.append(normalized_row)
    df_all['data_x_rad_norm_graph']=list_norm_x_rad_col
    
    #2 values
    list_norm_pos_zc_col=[]
    for ind in df_all.index:
        normalized_row=single_row_standard_normalize(df_all['data_pos_zc'][ind], 2)
        list_norm_pos_zc_col.append(normalized_row)
    df_all['data_pos_zc_norm_graph']=list_norm_pos_zc_col
    
    #2 values
    list_y_zc_col=[]
    for ind in df_all.index:
        normalized_row=single_row_standard_normalize(df_all['data_y_zc'][ind], 2)
        list_y_zc_col.append(normalized_row)
    df_all['data_y_zc_norm_graph']=list_y_zc_col
    
    #1 value
    list_y_yaw_col=[]
    for ind in df_all.index:
        normalized_row=single_row_standard_normalize(df_all['data_y_yaw'][ind], 1)
        list_y_yaw_col.append(normalized_row)
    df_all['data_y_yaw_norm_graph']=list_y_yaw_col
    
    return df_all
 
# leave this as last line 
print (f"Functions Data Adjustments import successful")   
    