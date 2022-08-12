#--------------------------------
## IMPORTS
#--------------------------------
import sys
import os

# current GCN
this_GCN = os.path.split(os.path.split(os.path.realpath(__file__))[0])[1]


# set the path to find the modules
sys.path.insert(0, '../005_src/') #use relative path
os.chdir("../005_src")

from config import *


class Trainer:
    def __init__(self,
                 
                 # Model parameters
                 model: torch.nn.Module,
                 device: torch.device,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 
                 # Data 
                 training_DataLoader: torch.utils.data.Dataset,
                 validation_DataLoader: torch.utils.data.Dataset = None,
                 lr_scheduler: torch.optim.lr_scheduler = None,
                 
                 # trigger statements
                 printstat= False,
                 savestat = True,
                 plotstat = True,
                 
                 # store intermediate model as TEMP 
                 #epochs: int = 100,
                 epochs: int = 1,
                 epoch: int = 0,
                 save_every: int = 50,
                 #plot_every: int = 10, 
                 plot_every: int = 10, 
                 
                 # get intermediate results for each epoch
                 shuttle_train = None,
                 shuttle_val = None,
                 
                 # saving directory and description
                 save_dir = "",
                 model_name = "",
                 date = "",
                 ts = "",
                 notebook: bool = False,
                 ):

        self.model = model        
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer

        self.training_DataLoader = training_DataLoader
        self.validation_DataLoader = validation_DataLoader
        self.lr_scheduler = lr_scheduler
        
        self.printstat = printstat
        self.savestat = savestat       
        self.plotstat = plotstat
        
        self.epochs = epochs
        self.epoch = epoch
        self.save_every = save_every # save TEMP model and losses 
        self.plot_every = plot_every # plot figures
        
        self.shuttle_train = shuttle_train
        self.shuttle_val = shuttle_val
        
        self.save_dir = save_dir
        self.model_name = model_name
        self.date = date
        self.ts = ts
        self.notebook = notebook
        
        self.paths = {}
        self.training_loss = []
        self.validation_loss = []
        self.learning_rate = []
        self.figures_paths  = []

    def run_trainer(self):
        
        # create a progress bar 
        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange
        progressbar = trange(self.epochs-self.epoch, desc='Progress')
        
        for i in progressbar:
            """Epoch counter"""
            self.epoch += 1  # epoch counter

            """Training block"""
            self._train()

            """Validation block"""
            if self.validation_DataLoader is not None:
                self._validate()

            """Learning rate scheduler block"""
            if self.lr_scheduler is not None:
                if self.validation_DataLoader is not None and self.lr_scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                    self.lr_scheduler.batch(self.validation_loss[i])  # learning rate scheduler step with validation loss
                else:
                    self.lr_scheduler.batch()  # learning rate scheduler step
                    
        return self.training_loss, self.validation_loss, self.learning_rate,self.paths,self.figures_paths


    def _train(self):
        ts= self.ts
        date = self.date
        shuttle_train = self.shuttle_train
        shuttle_val = self.shuttle_val
        save_dir = self.save_dir
        model_name = self.model_name
        epoch = self.epoch
        epochs = self.epochs
        save_every = self.save_every
        plot_every = self.plot_every
        plotstat = self.plotstat
        fig_paths = self.figures_paths
        
        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange
        
        if plotstat:
            ## figures dir
            FIG_DIR = os.path.join(save_dir,'figures/')
            if not os.path.exists(FIG_DIR):
                Path(FIG_DIR).mkdir(parents=True, exist_ok=True)       
        
        self.model.train()  # train mode
        
        train_losses = []  # accumulate the losses here

        batch_iter = tqdm(enumerate(self.training_DataLoader), 'Training', total=len(self.training_DataLoader),
                          leave=False)
        
        for i, data in batch_iter:
            
            x = data.x.to(self.device).float()
            edge_index = data.edge_index.to(self.device).long() 
            edge_attr = data.edge_attr.to(self.device).float()
            target = data.y.to(self.device).float()
            
            # zerograd the parameters
            self.optimizer.zero_grad() 
            ## one forward pass
            #print(x.is_cuda)
            #print(edge_index.is_cuda)
            #print(edge_attr.is_cuda)
            
            out = self.model(x,edge_index,edge_attr)
            
            #print(out.is_cuda)
            
            
            #if i % 15 == 0 and printstat:
                
                #printif("\n input",printstat)
                #printif(f"{x=},\n{edge_index=},\n{edge_attr=}",printstat)
                #printif("\n\n TARGET",printstat)
                #printif(target,printstat)
                #printif("\n PREDICTION",printstat)
                #printif(out,printstat)
                #pdb.set_trace()
            #loss
            loss = self.criterion(out, target)
            loss_value = loss.item()
            train_losses.append(loss_value)
            loss.backward()  # one backward pass
            self.optimizer.step()  # update the parameters

            batch_iter.set_description(f'Training: (loss {loss_value:.4f})')  # update progressbar

        if plotstat and (shuttle_train is not None) and (epoch % plot_every) == 0:
            epoch_title = f"{date}{ts}Model_{model_name}_epoch_{pad(epoch, minl = 3)}of{epochs}"
            self.plot_results_model(
                   self.model,
                   shuttle_train, 
                   start_from = 0,
                   end_at = 1,
                   edges_attr = True,
                   figsize = (10,10),
                   title = epoch_title,
                   padding_title = -15,
    
                   plot_input = False,
                   plot_target = True,
                   plot_prediction = True,
                   plot_intention = False, 
    
                   save_dir = FIG_DIR,
                
                   xlim = (-100,100),
                   ylim = (-100,100),
                  )
            fig_paths.append(os.path.join(FIG_DIR,f"{epoch_title}.png"))
              
            
            
        self.training_loss.append(np.mean(train_losses))
        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])

        batch_iter.close()
    
    def _validate(self):

        ts= self.ts
        date = self.date
        shuttle_train = self.shuttle_train
        shuttle_val = self.shuttle_val
        save_dir = self.save_dir
        model_name = self.model_name
        epoch = self.epoch
        epochs = self.epochs
        save_every = self.save_every
        
        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.eval()  # evaluation mode
        valid_losses = []  # accumulate the losses here
        batch_iter = tqdm(enumerate(self.validation_DataLoader), 'Validation', total=len(self.validation_DataLoader),
                          leave=False)

        for i, data in batch_iter:
            
            x = data.x.to(self.device).float()
            edge_index = data.edge_index.to(self.device).long() 
            edge_attr = data.edge_attr.to(self.device).float()
            target = data.y.to(self.device).float()
            
            with torch.no_grad():
                out = self.model(x,edge_index,edge_attr)
                
                #loss
                loss = self.criterion(out, target)
                #print(loss)
                loss_value = loss.item()
                valid_losses.append(loss_value)

                batch_iter.set_description(f'Validation: (loss {loss_value:.4f})')
        
        #print (np.mean(valid_losses))
        self.validation_loss.append(np.mean(valid_losses))
        
        if epoch % save_every == 0:
            
            if epoch < epochs:
                type_save = "TEMP_" 
            else: 
                type_save = "FINAL_" 
            
            description = f"{date}{ts}EPOCH_{epoch}of{epochs}_{type_save}"

            # save model
            model_descr = f"{description}_{model_name}.pt"
            model_path = os.path.join(save_dir,model_descr)
            
            # save the complete model
            #torch.save(model.state_dict(), filepath)
            torch.save (self.model.state_dict(), model_path)
            print (f"Saved model in {model_path}")
            
            #store losses amd lr
            tloss_path = os.path.join(save_dir,f'{description}_{type_save}_training_loss'+'.pkl')
            with open(tloss_path, 'wb') as f:
                    pkl.dump(self.training_loss, f)
            
            vloss_path = os.path.join(save_dir,f'{description}_{type_save}_validation_loss'+'.pkl')
            with open(vloss_path, 'wb') as f:
                    pkl.dump(self.validation_loss, f)
            
            lr_path = os.path.join(save_dir,f'{description}_{type_save}_learning_rate'+'.pkl')
            with open(lr_path, 'wb') as f:
                    pkl.dump(self.learning_rate, f)
                    
            #return only the last paths
            self.paths = {"model_path":model_path,
                          "train_losses_path":tloss_path,
                          "val_losses_path":vloss_path,
                          "lr_path":lr_path}

        batch_iter.close()


    intention_dict_1 = {0: "U_turn",1:"Straight",2:"Turn_Left",3:"Turn_Right"}
    def plot_results_model(self,
                       model,
                       dataset, 
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

                       xlim = (0,200),
                       ylim = (0,200),
                      ):
        figsize = figsize
        #print (type(dataset))
        if isinstance(dataset,list) == False:
            dataset = [dataset]
            start_from = 0
            end_at = 1

        for i,data in enumerate(dataset[start_from:end_at]):
            #print ("--------------")
            #print (data.x)
            #print (data.y)
            IDX = i+start_from
            model.to('cpu')
            if edges_attr:
                pred = model(data.x,
                             #data.x.to(self.device).float(),
                             data.edge_index,
                             #data.edge_index.to(self.device).long(),
                             data.edge_attr
                             #data.edge_attr.to(self.device).float()
                            ).detach().numpy()
            else: 
                pred = model(data.x,
                             #data.x.to(self.device).float(),
                             data.edge_index,
                             #data.edge_index.to(self.device).long(),
                            ).detach().numpy()            

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
            if plot_input:
                X,Y = [], []
                INTENTION = []
                for e in data.x.detach().numpy():
                    X.append(e[0])
                    Y.append(e[1])
                    INTENTION.append(e[-1])
                ax.scatter(X,Y, c ="orange",alpha = 0.8,s = 300, label= "initial positions at time t")
                for i,x in enumerate(X):
                    ax.annotate(str(i), (x, Y[i]), ha='center', va='center', size=14)  
                    if plot_intention:
                        try:
                            ax.annotate(intention_dict_1[INTENTION[i]], (x+5, Y[i]+5), 
                                        ha='center', va='center', size=10 , c = "red")
                        except Exception as e:
                            print ("No intention information available")


            ## PLOT MODEL PREDICTIONS
            if plot_prediction: 
                X,Y = [], []
                for e in pred:
                    X.append(e[0])
                    Y.append(e[1])
                ax.scatter(X,Y, c ="red",alpha = 0.5, s = 300, label= "predicted positions at time t+2")

                for i,x in enumerate(X):
                    ax.annotate(str(i), (x, Y[i]), ha='center', va='center', size=14)

            ## PLOT TARGETS
            if plot_target: 
                X,Y = [], []
                for e in data.y.detach().numpy():
                    X.append(e[0])
                    Y.append(e[1])

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
            else:
                plt.show()
            model.to('cuda:0')

def check_import():
    print (f"imported trainer: {this_GCN} at {get_timestamp()}")
    return this_GCN,get_timestamp()

