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
                 
                 # control input
                 use_edges_attr: bool,
                 
                 # Data 
                 training_DataLoader: torch.utils.data.Dataset,
                 validation_DataLoader: torch.utils.data.Dataset = None,
                 lr_scheduler: torch.optim.lr_scheduler = None,
                 select_scheduler: str = "None",
                 
                 # trigger statements
                 printstat= False,
                 savestat = True,
                 plotstat = False,
                 
                 # store intermediate model as TEMP 
                 epochs: int = 100,
                 epoch: int = 0,
                 save_every: int = 50,
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
        
        self.use_edges_attr = use_edges_attr

        self.training_DataLoader = training_DataLoader
        self.validation_DataLoader = validation_DataLoader
        self.lr_scheduler = lr_scheduler
        self.select_scheduler = select_scheduler
        
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
            if self.select_scheduler != "None":
                if self.select_scheduler ==  "ReduceLROnPlateau":
                    self.lr_scheduler.step(self.validation_loss[-1])
                elif self.select_scheduler == "MultiStepLR":
                    self.lr_scheduler.step()
                elif self.select_scheduler == "CosineAnnealingWarmRestarts":
                    self.lr_scheduler.step(self.epoch / len(self.training_DataLoader))
                else:
                    print (f"{self.select_scheduler} not implemented, select_scheduler set to None")
                    self.select_scheduler == "None"
                
                
                    
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
        printstat = self.printstat
        
        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange
        
        if plotstat:
            ## figures dir
            FIG_DIR = os.path.join(save_dir,'figures_training_set/')
            if not os.path.exists(FIG_DIR):
                Path(FIG_DIR).mkdir(parents=True, exist_ok=True)
                
        self.model.to(self.device)
        self.model.train()  # train mode
        
        train_losses = []  # accumulate the losses here
        iters = len(self.training_DataLoader)
        batch_iter = tqdm(enumerate(self.training_DataLoader), 
                          'Training', 
                          total=iters,
                          leave=False)
        
        for i, data in batch_iter:
            
            x = data.x.to(self.device).float()
            edge_index = data.edge_index.to(self.device).long() 
            if self.use_edges_attr:
                edge_attr = data.edge_attr.to(self.device).float()
            else: 
                edge_attr = None
            target = data.y.to(self.device).float()
            
            # zerograd the parameters
            self.optimizer.zero_grad() 
            ## one forward pass
            out = self.model(x,edge_index,edge_attr)
            
            if i % 15 == 0 and printstat:
                
                printif("\n INPUT",printstat)
                printif(f"{x=},\n{edge_index=},\n{edge_attr=}",printstat)
                printif("\n\n TARGET",printstat)
                printif(target,printstat)
                printif("\n PREDICTION",printstat)
                printif(out,printstat)
                
            #loss
            loss = self.criterion(out, target)
            loss_value = loss.item()
            train_losses.append(loss_value)
            loss.backward()  # one backward pass
            self.optimizer.step()  # update the parameters
            batch_iter.set_description(f'Training: (loss {loss_value:.4f})')  # update progressbar

        # plot intermediate training shuttle
        if isinstance (plot_every, int):
            condition_plot = (epoch % plot_every) == 0
        elif isinstance (plot_every, list):
            condition_plot = epoch in plot_every
        else:
            print (f"Error in the condition plot: {plot_every} should be an int or a list instead is {type(plot_every)}")
            print ("not plotting")
            condition_plot = False  
            
        if plotstat and (shuttle_train is not None) and condition_plot:
            epoch_title = f"{date}{ts}Model_{model_name}_shuttle_train_epoch_{pad(epoch, minl = 6)}of{epochs}"
            plot_results_model(
                   self.model,
                   #device = self.device,
                   dataset = shuttle_train, 
                   start_from = 0,
                   end_at = 1,
                   edges_attr = True,
                   figsize = (10,10),
                   title = epoch_title,
                   padding_title = -15,
    
                   plot_input = True,
                   plot_target = True,
                   plot_prediction = True,
                   plot_intention = False, 
    
                   save_dir = FIG_DIR,
                
                   xlim = (-100,100),
                   ylim = (-100,100),
                  )
            fig_paths.append(os.path.join(FIG_DIR,f"{epoch_title}.png"))
              
            
        # append losses and lr to list   
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
        plot_every = self.plot_every
        plotstat = self.plotstat
        fig_paths = self.figures_paths
        printstat = self.printstat
        
        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange
         
        if plotstat:
            ## figures dir
            FIG_DIR = os.path.join(save_dir,'figures_validation_set/')
            if not os.path.exists(FIG_DIR):
                Path(FIG_DIR).mkdir(parents=True, exist_ok=True)
        
        self.model.to(self.device)
        self.model.eval()  # evaluation mode
        valid_losses = []  # accumulate the losses here
        iters = len(self.validation_DataLoader)
        batch_iter = tqdm(enumerate(self.validation_DataLoader), 
                          'Validation', 
                          total = iters,
                          leave = False)

        for i, data in batch_iter:
            
            x = data.x.to(self.device).float()
            edge_index = data.edge_index.to(self.device).long() 
            if self.use_edges_attr:
                edge_attr = data.edge_attr.to(self.device).float()
            else: 
                edge_attr = None
            target = data.y.to(self.device).float()
            
            with torch.no_grad():
                out = self.model(x,edge_index,edge_attr)
                
                #loss
                loss = self.criterion(out, target)
                #print(loss)
                loss_value = loss.item()
                valid_losses.append(loss_value)

                batch_iter.set_description(f'Validation: (loss {loss_value:.4f})')

        running_loss = np.mean(valid_losses)
        self.validation_loss.append(running_loss)
        

        # plot intermediate validation
        if isinstance (plot_every, int):
            condition_plot = (epoch % plot_every) == 0
        elif isinstance (plot_every, list):
            condition_plot = epoch in plot_every
        else:
            print (f"Error in the condition plot: {plot_every} should be an int or a list instead is {type(plot_every)}")
            print ("not plotting")
            condition_plot = False
            
        if plotstat and (shuttle_val is not None) and condition_plot:
            epoch_title = f"{date}{ts}Model_{model_name}_shuttle_val_epoch_{pad(epoch, minl = 6)}of{epochs}"
            plot_results_model(
                   self.model,
                   dataset= shuttle_val,
                   #device = self.device,
                   start_from = 0,
                   end_at = 1,
                   edges_attr = True,
                   figsize = (10,10),
                   title = epoch_title,
                   padding_title = -15,
    
                   plot_input = True,
                   plot_target = True,
                   plot_prediction = True,
                   plot_intention = False, 
    
                   save_dir = FIG_DIR,
                
                   xlim = (-100,100),
                   ylim = (-100,100),
                  )
            fig_paths.append(os.path.join(FIG_DIR,f"{epoch_title}.png"))
        
        
        # save intermediate model and losses
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



def check_import():
    print (f"imported trainer: {this_GCN} at {get_timestamp()}")
    return this_GCN,get_timestamp()

