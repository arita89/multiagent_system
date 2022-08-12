import sys  
sys.path.insert(0, '../week3/') #use relative path

from _00_settings import *
from _01_functions_utilities import *
from _02_functions_readimages import *
from _03_pointNet import *
from _04_function_plot_results import plot_training

device = cudaOverview()
print(device)



# create the data as Data Class 
# cams : 0-5
# frames : 0-499
def load_data(allcams, allframes):
    
    dict_y = {}
    data_list = []
    
    for cam1 in allcams: 
        for frame1 in tqdm(allframes):
            
            frame1 = format(frame1, '05d')
            
            # find the images
            rgb_file1   = os.path.join(FOLDER, "CameraRGB{}/image_{}.png".format(cam1, frame1))
            depth_file1 = os.path.join(FOLDER, "CameraDepth{}/image_{}.png".format(cam1, frame1))
            seg_file1 = os.path.join(FOLDER, "CameraSemSeg{}/image_{}.png".format(cam1, frame1))

            # read the images
            rgb1 = read_rgb(rgb_file1, False)
            depth1 = read_depth(depth_file1, False)
            seg1 = read_rgb(seg_file1, False)

            #Semantic labels are contained in the R channel  # THIS IS THE TARGET
            R_channel = seg1[:,:,0]
            cityscapes_seg = labels_to_cityscapes_palette(R_channel)

            # create point clouds
            pc1, color1 = depth_to_local_point_cloud(depth1, 
                                                     color=rgb1, 
                                                     k = K,
                                                     max_depth=0.05)

            pc1, seg1 = depth_to_local_point_cloud(depth1, 
                                                   color=cityscapes_seg, 
                                                   k = K,
                                                   max_depth=0.05)
            
            #grouping 
            #pos = torch.from_numpy(np.asarray(pcdseg.points))
            pos = torch.from_numpy(pc1)
            edge_index = knn_graph(pos, k=6).long()
            #y = torch.from_numpy(np.asarray(pcdseg.colors))#.long()
            
            # update dictionary with new keys
            for i,raw in enumerate(np.unique(np.asarray(seg1[:,0]*1000))):
                # if there are new ones
                if raw not in dict_y.keys():
                    dict_y[raw] = i
            
            seg1_labels = [dict_y[raw] for raw in seg1[:,0]*1000]
            #print (len(seg1_labels))
            #y = torch.from_numpy(np.asarray(seg1[:,0]*1000)).long()
            y = torch.from_numpy(np.asarray(seg1_labels)).long()
            #print (len(y))
            #print (np.unique(y))
            #print (np.unique(torch.from_numpy(np.asarray(seg1[:,0])).long(), axis = 0))
            
            # create a data object to append to the list 
            # https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html
            data_list.append(Data
                                 (
                                 pos = pos, # X,Y,Z coordinates
                                 edge_index = edge_index, # equivalent of sparse matrix, just a non-sparse list of edges
                                 y = y # targets
                                )
                            )
            
    return dict_y,data_list