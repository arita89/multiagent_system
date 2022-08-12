from _00_configuration._00_settings import *
from _00_configuration._01_variables import *
from _01_functions._00_helper_functions import *

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
                
intention_dict_1 = {0: "U_turn",1:"Straight",2:"Turn_Left",3:"Turn_Right"}


def plot_results_model(
                   model,
                   dataset,
                   device = 'cpu',
                   start_from = 0,
                   end_at = 10,
                   edges_attr = True,
                   figsize = (10,10),
                   title = None,
                   padding_title = -15,
    
                   plot_input = True,
                   plot_target = True,
                   plot_prediction = True,
                   plot_intention = True, 
    
                   save_dir = None,
                   printstat = False,
                   plotstat = False,
    
                   xlim = (0,200),
                   ylim = (0,200),
    
                   loss_stat = None,
                   reduction = "mean",
    
                   predicting_delta = False,
                  ):
    """
    plotting without outputting the loss
    """
    
    if isinstance(dataset,list) == False:
        dataset = [dataset]
        start_from = 0
        end_at = 1
    
    figsize = figsize
    loss_values = []
    for i,data in enumerate(dataset[start_from:end_at]):
        IDX = i+start_from
        printif("",printstat)
        printif(IDX,printstat)
        model.to(device)
        
        x = data.x.to(device).float()
        edge_index = data.edge_index.to(device).long()
        target = data.y.to(device).float() 
        
        if edges_attr:
            edge_attr = data.edge_attr.to(device).float()
            pred_tensor = model(x,
                         edge_index,
                         edge_attr
                        )
            pred = pred_tensor.detach().numpy()
        else:
            edge_attr = None
            pred_tensor = model(x,
                         edge_index,
                        )
            pred = pred_tensor.detach().numpy()
        
        printif("\n INPUT",printstat)
        printif(f"{x=},\n{edge_index=},\n{edge_attr=}",printstat)
        printif("\n\n TARGET",printstat)
        printif(target,printstat)
        printif("\n PREDICTION",printstat)
        printif(pred_tensor,printstat)
        
        ## INITIALIZE FIGURE
        fig, ax = plt.subplots(figsize= figsize) 
        #cmap = list(sns.color_palette("Paired",len(pred)).as_hex())
        if title is None: 
             ax.set_title(f"2D map of data {IDX}");
        else:
            ax.set_title("%s"%title, pad=padding_title)
            
        ax.set_xlabel("X");
        ax.set_ylabel("Y"); 
            
        ## PLOT INPUT      
        initial_X,initial_Y = [], []
        INTENTION = []
        for e in data.x.detach().numpy():
            initial_X.append(e[0])
            initial_Y.append(e[1])
            INTENTION.append(e[-1])
        if plot_input:
            ax.scatter(initial_X,initial_Y, c ="orange",alpha = 0.8,s = 300, label= "initial positions at time t")
            for i,x in enumerate(initial_X):
                ax.annotate(str(i), (x, initial_Y[i]), ha='center', va='center', size=14)  
                if plot_intention:
                    try:
                        ax.annotate(intention_dict_1[INTENTION[i]], (x+5, initial_Y[i]+5), 
                                    ha='center', va='center', size=10 , c = "red")
                    except Exception as e:
                        print ("No intention information available")
            
        if predicting_delta:
            delta_list_X = initial_X
            delta_list_Y = initial_Y
        else:
            delta_list_X = [0]*len(initial_X)
            delta_list_Y = [0]*len(initial_Y)            
            
        ## PLOT MODEL PREDICTIONS
        if plot_prediction: 
            X,Y = [], []
            for i,e in enumerate(pred):
                X.append(e[0]+delta_list_X[i])
                Y.append(e[1]+delta_list_Y[i])
            ax.scatter(X,Y, c ="lightblue",alpha = 0.5, s = 300, label= "predicted positions at time t+2")

            for i,x in enumerate(X):
                ax.annotate(str(i), (x, Y[i]), ha='center', va='center', size=14)

        ## PLOT TARGETS
        if plot_target: 
            X,Y = [], []
            for i,e in enumerate(data.y.detach().numpy()):
                X.append(e[0]+delta_list_X[i])
                Y.append(e[1]+delta_list_Y[i])

            ax.scatter(X,Y, c ="lightgreen",alpha = 0.5,s = 500, label = "target positions at time t+2")
            for i,x in enumerate(X):
                ax.annotate(str(i), (x, Y[i]), ha='center', va='center', size=14)
        
        
        ax.set_xlim(xlim[0],xlim[1])
        ax.set_ylim(ylim[0],ylim[1])
        ax.legend()
        ax.grid()

        if save_dir is not None:
            new_png_path =os.path.join(save_dir,f"{title}.png")
            plt.savefig(new_png_path)
            printif(f"Plot saved in {new_png_path}", printstat)
        elif plotstat:
            plt.show()
        else:
            continue


def plot_training(training_losses,
                  validation_losses,
                  learning_rate = None,
                  gaussian=True,
                  sigma=2,
                  figsize=(8, 6),
                  mytitle = 'Training & validation loss'
                  ):
    """
    Returns a loss plot with training loss, validation loss and learning rate.
    """

    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from scipy.ndimage import gaussian_filter

    list_len = len(training_losses)
    x_range = list(range(1, list_len + 1))  # number of x values
    if learning_rate is not None:
        ncols = 2
    else:
        ncols = 1
        
    fig = plt.figure(figsize=figsize)
    grid = gridspec.GridSpec(ncols=ncols, nrows=1, figure=fig)

    subfig1 = fig.add_subplot(grid[0, 0])
    if learning_rate is not None:
        subfig2 = fig.add_subplot(grid[0, 1])

    subfigures = fig.get_axes()

    for i, subfig in enumerate(subfigures, start=1):
        subfig.spines['top'].set_visible(False)
        subfig.spines['right'].set_visible(False)

    if gaussian:
        training_losses_gauss = gaussian_filter(training_losses, sigma=sigma)
        validation_losses_gauss = gaussian_filter(validation_losses, sigma=sigma)

        linestyle_original = '.'
        color_original_train = 'lightcoral'
        color_original_valid = 'lightgreen'
        color_smooth_train = 'red'
        color_smooth_valid = 'green'
        alpha = 0.45
    else:
        linestyle_original = '-'
        color_original_train = 'red'
        color_original_valid = 'green'
        alpha = 1.0

    # Subfig 1
    subfig1.plot(x_range, training_losses, linestyle_original, color=color_original_train, label='Training',
                 alpha=alpha)
    subfig1.plot(x_range, validation_losses, linestyle_original, color=color_original_valid, label='Validation',
                 alpha=alpha)
    if gaussian:
        subfig1.plot(x_range, training_losses_gauss, '-', color=color_smooth_train, label='Training', alpha=0.75)
        subfig1.plot(x_range, validation_losses_gauss, '-', color=color_smooth_valid, label='Validation', alpha=0.75)
    subfig1.title.set_text(mytitle)
    subfig1.set_xlabel('Epoch')
    subfig1.set_ylabel('Loss')

    subfig1.legend(loc='upper right')

    if learning_rate is not None:
        # Subfig 2
        lr_range = list(range(0,len(learning_rate)))
        subfig2.plot(lr_range, learning_rate,".",alpha = 0.7, color='black')
        subfig2.title.set_text('Learning rate')
        subfig2.set_xlabel('Epoch')
        subfig2.set_ylabel('LR')

    return fig       
        
def compare_losses(
                   losses_dict,
                   title_1 = "losses_1",
                   columns = ["run_time"], # for the labels
                   
                   losses_dict_2 = None,
                   title_2 = "losses_2",
                   
                   columns_2 = ["run_time"],
                   gaussian = True,
                   sigma=2,
                   figsize=(8, 6),
                   ):
    """
    Returns a loss plot with training loss, validation loss and learning rate.
    """

    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from scipy.ndimage import gaussian_filter
    import pylab as pplot
    params = {'legend.fontsize': 20,
              'legend.handlelength': 2}
    pplot.rcParams.update(params)
    
    #check input format
    if not isinstance(columns, list):
        columns = [columns]
    
    if not isinstance(columns_2, list):
        columns_2 = [columns_2]

    models_1 = losses_dict.keys()
    if losses_dict_2 is not None: 
        models_2 = losses_dict_2.keys()
        assert models_1 == models_2
        
    num_models = len(models_1)
    print (f"Comparing {num_models} models")
    
    list_t_losses = [v[0] for k,v in losses_dict.items()]
    list_column_keys = list(losses_dict.values())[0][1].keys()
    print (f"possible keys to choose from:\n: {list(list_column_keys)}")
    list_lengths = [len(e) for e in list_t_losses]
    if losses_dict_2 is not None: 
        list_t_losses_2 = [v[0] for k,v in losses_dict_2.items()]
        list_lengths_2 = [len(e) for e in list_t_losses_2]
        assert list_lengths == list_lengths_2
    
    # check that all lists are the same length
    it = iter(list_t_losses)
    the_len = len(next(it))
    
    min_len = min(list_lengths)
    print (f"{min_len=}")
    
    if not all(len(l) == the_len for l in it):
        print ('not all lists of losses have same length!')
        print (f'they will be plotted according to the minimal length {min_len}')
        # reduce the losses all to the same legth
        list_t_losses = [losses[:min_len+1] for losses in list_t_losses]
        list_lengths = [len(e) for e in list_t_losses]
        
    # how many plot we have
    if losses_dict_2 is not None:
        if losses_dict.keys() == losses_dict_2.keys():
            ncols = 2
        else: 
            print (f"keys of the two dictionaries dont correspond, plottig only the first")
            ncols = 1
            losses_dict_2 = None
    else:
        ncols = 1
    
    # create the figure
    fig = plt.figure(figsize=figsize)
    grid = gridspec.GridSpec(ncols=ncols, nrows=1, figure=fig)
    
    # create the subfigure(s)
    subfig1 = fig.add_subplot(grid[0, 0])
    if losses_dict_2 is not None:
        subfig2 = fig.add_subplot(grid[0, 1])
    subfigures = fig.get_axes()

    for i, subfig in enumerate(subfigures, start=1):
        subfig.spines['top'].set_visible(False)
        subfig.spines['right'].set_visible(False)

    # set colors and alphas
    alpha = 0.55
    alpha_gauss = 0.75
    cmap = list(sns.color_palette("Paired",len(losses_dict.keys())).as_hex())
    cmap_dict = {k: cmap[i] for i,k in enumerate(list(losses_dict.keys()))}
    
    
    
    # length of x axis
    x_range = list(range(0, min_len))  # number of x values
    #print (f"{len(x_range)=}")
        
    # Subfig 1
    for k,v in losses_dict.items():
        losses = v[0][:min_len]
        #print (len(losses))
        model_details = v[1]
        label = ""
        for column in columns:
            label += f"{model_details[column]}_"
        label = label[:-1]
        color = cmap_dict[k]
        
        subfig1.plot(x_range, losses, '-', color=color, label=label,
                     alpha=alpha)
        if gaussian:
            losses_gauss = gaussian_filter(losses, sigma=sigma)
            subfig1.plot(x_range, losses_gauss, '-', color=color, alpha=alpha_gauss)    
    subfig1.title.set_text(title_1)
    subfig1.set_xlabel('Epoch')
    subfig1.set_ylabel('Loss')
    subfig1.legend(loc='upper right')

    # Subfig 2
    if losses_dict_2 is not None:
        for k,v in losses_dict_2.items():
            losses = v[0][:min_len]
            model_details = v[1]
            label = ""
            for column in columns_2:
                label += f"{model_details[column]}_"
            label = label[:-1]
            color = cmap_dict[k]
            subfig2.plot(x_range, losses, '-',color=color, label=label,
                     alpha=alpha)
            if gaussian:
                losses_gauss = gaussian_filter(losses, sigma=sigma)
                subfig2.plot(x_range, losses_gauss, '-', color=color, alpha=alpha_gauss)
        subfig2.title.set_text(title_2)
        subfig2.set_xlabel('Epoch')
        subfig2.set_ylabel('Loss')
        subfig2.legend(loc='upper left')

    return fig 
        
print (f"Functions Plotting import successful")