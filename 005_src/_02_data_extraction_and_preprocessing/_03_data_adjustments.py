#--------------------------------
## IMPORTS
#--------------------------------
import sys
import os

# set the path to find the modules
sys.path.insert(0, '../005_src/') #use relative path
os.chdir("/storage/remote/atcremers50/ss21_multiagentcontrol/005_src")

from config import *

#--------------------------------
## DEBUGGING OPTIONS
#--------------------------------
# mark True during dubugging to trigger the prints
printstat = False 

#--------------------------------
## INPUT 
#--------------------------------

print (f"which GCN_input file in pkl format do you want to correct? \neg '20210711-17h59m44s_timesteps30000_ec3500_em7000' ")
list_all_zip_files = sorted(glob.glob(os.path.join(GCN_INPUT_FOLDER ,f"*pkl")))
list_all_zip_files_short = [os.path.split(file)[1] for file in list_all_zip_files]
printif(list_all_zip_files, True, 10)
input_file_name = input()

file_to_read = open(f"../004_data/GCN_input/{input_file_name}.pkl", "rb")
data = pkl.load(file_to_read)
df_all = pd.DataFrame(data).T
#df_all,df_selected = select_rows_from_data(input_file_name,drop_col = True)

#--------------------------------
## create and add columns
#--------------------------------
traffic_info = df_all.apply(lambda x: is_veh_still(x['data_pos'],x['data_y']),axis=1)
data_pos_zc = df_all.apply(lambda x: zero_center_column(x['data_pos']),axis=1)
data_y_zc = df_all.apply(lambda x: zero_center_column(x['data_y']),axis=1)
data_y_yaw = df_all.apply(lambda x: get_yaw_separately(x['data_y']),axis=1)

df_all["data_pos_zc"] = data_pos_zc
df_all["Still_vehicle"] = traffic_info
df_all["data_y_zc"] = data_y_zc
df_all["data_y_yaw"] = data_y_yaw

# the corrected input: data_x concat data_pos_zc concat Stil_vehicle 
# to predict the corrected output: data_y_zc

print (df_all.head(5))

# store pkl file with GCN input 
gcn_input_file_name = f"{input_file_name}_CORRECTED"
new_df_path = f"../004_data/GCN_input/{gcn_input_file_name}.pkl"
file_to_write = open(new_df_path, "wb")
print (f"new df stored in: {new_df_path}")



pkl.dump(df_all, file_to_write)
