from configuration._00_settings import *
from configuration._01_variables import *
from configuration._02_helper_functions import *
from configuration._03_functions_dataframes import *
from math import hypot
from itertools import combinations

#--------------------------------
# ADD DISTANCE INFO TO DF 
#--------------------------------

def distance(p1,p2):
    """
    Euclidean distance between two points
    """
    x1,y1 = p1
    x2,y2 = p2
    return hypot(x2 - x1, y2 - y1)

def min_distance_to_border(p1):
    
    border_left = (-100, 0)
    border_right = (100, 0)
    border_top = (0, 100)
    border_buttom = (0, -100)
    dist_car_to_left_border = distance(p1, border_left)
    dist_car_to_right_border = distance(p1, border_right)
    dist_car_to_top_border = distance(p1, border_top)
    dist_car_to_buttom_border = distance(p1, border_buttom)
    dist = min(dist_car_to_left_border, dist_car_to_right_border, dist_car_to_top_border, dist_car_to_buttom_border)
    
    return dist

def weights_for_edges(p1, p2):
    dist1 = min_distance_to_border(p1)
    dist2 = min_distance_to_border(p2)
    
    weight= 1/(dist1+dist2)
    
    return weight


def veh_distance(df,timestep,i,
                edge_creation_radius = 40,   
                edge_maintenance_radius = 40 + 10,
                ):
    """
    computes the distance between each pair of vehicles from the df
    returns a dataframe with  veh_a,veh_b,(xa,ya),(xb,yb), distance
    """
    df_timestep = mask_timestep(df,timestep)
    # create all the unique combinations of vehicles at the timestep
    list_combinations = [combo for combo in combinations(df_timestep.vehID.unique().tolist(),2)]
    all_row_dict = []
    for combo in list_combinations:
        
        veh_a,veh_b = combo
        #print (veh_a,veh_b)
        p1 = (float(mask_veh(df_timestep,veh_a)['X']),
                float(mask_veh(df_timestep,veh_a)['Y']))
        p2 = (float(mask_veh(df_timestep,veh_b)['X']),
                float(mask_veh(df_timestep,veh_b)['Y']))
        d = distance(p1,p2)

        ## edges creation and maintenance rules
        if i == 0:
            edge_c = 0 # no edge at the first time step
            edge_m = 0
        else:
            if d < edge_creation_radius:
                edge_c = 1 # if the vehicles are close the edge is there
                edge_m = 1
            else:
                edge_c = 0 # no edge is created
                if d < edge_maintenance_radius:
                    edge_m = 1 # but in this case, if exists is maintained
                else:
                    edge_m = 0 # otherwise is destroyed

        # create the row of the dictionary
        row= {
            'timestep': timestep,
            'veh_a': veh_a,
            'veh_b': veh_b,
            '(xa,ya)': p1,
            '(xb,yb)': p2,
            'distance': d,
            f'edge_c': edge_c, # inside the radius for edge creation 
            f'edge_m': edge_m, # inside the radius for edge maintenance
        }
        all_row_dict.append(row)
    return pd.DataFrame(all_row_dict)
    
def get_df_per_timestep(df,
                        edge_creation_radius = 40,   
                        edge_maintenance_radius = 40 + 10
                ):
    """
    creates a list of dataframes
    one dataframe of distances per timestep
    """
    all_df_distances_per_timestep = [veh_distance(df,
                                                    timestep,
                                                    i,
                                                    edge_creation_radius,
                                                    edge_maintenance_radius
                                                  ) 
                                                for i,timestep in enumerate(df.time.unique())]
    return all_df_distances_per_timestep

#--------------------------------
# ADD EDGES INFORMATION PER DF 
#--------------------------------

def add_edge_info_per_veh(df):
    """
    looks at edge_c and edge_m to set
    if there is an edge (True) or not (False)
    """
    merged_list = df.edge_c *10 + df.edge_m
    l = len(merged_list)
    edges_list = [False]* l
    for i,e in enumerate(merged_list):
        if e == 0:
            edges_list[i:] = [False]*(l-i)
        elif e == 11:
            edges_list[i:] = [True]*(l-i)
        else:
            continue
    df["edge"] = edges_list
    return df

def add_edge_info_per_df(df):
    # all the unique combinations of vehicles veh_a, veh_b
    combinations = list(set(zip(df.veh_a, df.veh_b)))

    new_dfs = []
    for combo in combinations:
        veh_a,veh_b = combo
        df_veh = df[(df.veh_a == veh_a) & (df.veh_b == veh_b)]
        new_dfs.append(add_edge_info_per_veh(df_veh))
    
    return pd.concat(new_dfs)

#--------------------------------
# BUILD GRAPH AND EDGE LIST
#--------------------------------

def find_extremes_coord(df):
    """
    find the frame extreme coordinates
    this is just to have a stable plot 
    """
    assert "(xa,ya)" in df.columns 
    assert "(xb,yb)" in df.columns 
    xcoor, ycoor = [],[]
    for e in df["(xa,ya)"].tolist():
        xcoor.append(e[0]) 
        ycoor.append(e[1])
    for e in df["(xb,yb)"].tolist():
        xcoor.append(e[0]) 
        ycoor.append(e[1])

    xmin = min(xcoor)
    xmax = max(xcoor)
    ymin = min(ycoor)
    ymax = max(ycoor)
    return xmin,xmax,ymin,ymax

def build_graph(df,
                ec,
                em,
                SAVE_TEMP,
                date,
                ts,
                savestat = True, 
                delete_tempFiles= True,
                minl= 6):
    """
    build a graph per frame, adding vehicles as nodes and edge when existing
    SAVE_TEMP is the folder where to store the figures
    minl is just giving the zero padding to the timesteps if savestat = True
    """
    #find the frame extreme coordinates
    xmin,xmax,ymin,ymax = find_extremes_coord(df)

    # initialize a graph object
    G = nx.Graph()

    # create placeholder nodes
    G.add_node("A",pos = (xmin,ymin))
    G.add_node("B",pos = (xmin,ymax))
    G.add_node("C",pos = (xmax,ymin))
    G.add_node("D",pos = (xmax,ymax))
    

    # total number of frame, formatted to be in the title
    tmax = pad(len(df.timestep.unique())-1,minl= minl)

    # dict of list of edges per frame
    dict_per_frame = {}

    # draw the graph per time step
    for i,timestep in enumerate(sorted(df.timestep.unique())):
        #print (timestep)

        # get the df with only the vehicles in this timestep
        mask = df.timestep == timestep
        df_time = df[mask]

        # get the vehicles in this timestep
        unique_veh_a = np.unique(df_time[['veh_a']].values).tolist()
        unique_veh_b = np.unique(df_time[['veh_b']].values).tolist()
        unique_veh = list(set(unique_veh_a+unique_veh_b))

        # create dict of coordinates, key is vehicle
        positions = {}

        for veh in unique_veh_a:
            if not veh in positions:
                df_veh = df_time[df_time.veh_a == veh]
                positions[veh] = tuple(df_veh["(xa,ya)"].values)[0]

        for veh in unique_veh_b:
            if not veh in positions:
                df_veh = df_time[df_time.veh_b == veh]
                positions[veh] = tuple(df_veh["(xb,yb)"].values)[0]

        # get the list of edges 
        mask = (df_time.edge == True)
        
        # with weights
        for index, row in df_time[mask].iterrows():
            pos_a=getattr(row, '(xa,ya)')
            pos_b=getattr(row, '(xb,yb)')
            weight_one_edge = weights_for_edges(pos_a,pos_b)
            
            elist = [(row.veh_a,row.veh_b,weight_one_edge)]
        
        # without weights
        elist_simpler = [(row.veh_a,row.veh_b) 
                    for index, row in df_time[mask].iterrows()]

        # add the list of edges to the dict, key is timestep
        #dict_per_frame[timestep] = [elist_simpler]
        dict_per_frame[timestep] = [elist]
        
        if savestat: 

            # initialize a graph object
            H = G.copy()
            #color_map = ['gray','gray','gray','gray']
            # add the nodes with the position attribute
            for veh in unique_veh:
                H.add_node(veh,pos=positions[veh])
                #color_map.append("green")
            pos=nx.get_node_attributes(H,'pos')

            #add the edges
            H.add_weighted_edges_from(elist)

            # draw
            plt.figure(3,figsize=(12,12))
            nx.draw(H,pos, with_labels=True)#, node_color=color_map)

            # save as png image
            t = pad(i,minl= minl)
            plt.title(f"ec:{ec} em:{em} timestep: {t}/{tmax}",y=1.0, pad=-14, loc='right')
            plt.savefig(os.path.join(SAVE_TEMP,f"filename{t}.png"));

            # clearing the current plot
            plt.clf();

    if savestat:
        titleGif = build_gif(SAVE_TEMP,f"{date}{ts}_em{em}_ec{ec}_graph",search = "", 
                                fps=1,delete_tempFiles =delete_tempFiles)
    else: 
        titleGif = None
    return dict_per_frame,titleGif

def create_GNC_input():
    ## create the tensors that are going to be input to the GCN
    # https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html 
    # tensor of features data.x one per vehicle[[features_veh_0]...[features_veh_n]]
    # tensor of data.edges [[veh_1_from...node_from],[veh_2_to...node_to]]
    # tensor of targets data.y [[]]
    # tensor of positions data.pos [[X_veh_1,Y_veh_1]...[X_veh_n,Y_veh_nn]

    for index, row in df_time[mask].iterrows():

        data_x = 0
        data_pos = 0
        data_edges = 0
        data_y = 0
        #dict_per_frame[timestep] = [data_x,data_pos,data_edges,data_y]


def visualize_edges(df):
    for i,timestep in enumerate(df.timestep.unique()):
        G = nx.Graph()
        plt.figure(3,figsize=(12,12))

        # get the df with only the vehicles in this timestep
        mask = df.timestep == timestep
        df_time = df[mask]

        # get the vehicles in this timestep
        unique_veh_a = np.unique(df_time[['veh_a']].values).tolist()
        unique_veh_b = np.unique(df_time[['veh_b']].values).tolist()
        unique_veh = list(set(unique_veh_a+unique_veh_b))

        
        # get the coordinates
        positions = {}

        for veh in unique_veh_a:
            if not veh in positions:
                df_veh = df_time[df_time.veh_a == veh]
                positions[veh] = tuple(df_veh["(xa,ya)"].values)[0]

        for veh in unique_veh_b:
            if not veh in positions:
                df_veh = df_time[df_time.veh_b == veh]
                positions[veh] = tuple(df_veh["(xb,yb)"].values)[0]

        # add the nodes with the position attribute
        #G.add_nodes_from(unique_veh)
        for veh in unique_veh:
            G.add_node(veh,pos=positions[veh])
        pos=nx.get_node_attributes(G,'pos')
        #print (pos)

        # get the list of edges 
        mask = (df_time.edge == True)
        elist = [(row.veh_a,row.veh_b,distance) 
                    for index, row in df_time[mask].iterrows()]
        #add the edges
        G.add_weighted_edges_from(elist)

        # draw
        nx.draw(G,pos, with_labels=True)

        # save as png image
        t = pad(i,minl= 6)
        plt.title(f"timestep: {t}",y=1.0, pad=-14, loc='right')
        plt.savefig(os.path.join(SAVE_TEMP,f"filename{t}.png"));
        # clearing the current plot
        plt.clf();

print (f"Functions graph import successful")
