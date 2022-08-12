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

from config_GCN_018 import *
from _03_Networks.GCN_021.Custom_losses import *
from _03_Networks.GCN_021.early_stop import EarlyStopping


class Trainer:
    def __init__(self,
                 
                 # Model parameters
                 model: torch.nn.Module,
                 device: torch.device,
                 select_criterion: str ,#="L1", #torch.nn.Module,
                 reduction:str,
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
                 
                 # store dict at every TEMP step
                 dict_text_output: dict = {},
                 
                 training_loss : list = [],
                 validation_loss : list = [],
                 learning_rate : list = [],
                 
                 early_stopping: bool = False,
                 predicting_delta: bool = False,
                 save_trucated_training: bool = False
                 ):

        self.model = model        
        self.device = device
        self.select_criterion = select_criterion
        self.reduction = reduction
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
        self.training_loss = training_loss
        self.validation_loss = validation_loss
        self.learning_rate = learning_rate
        self.figures_paths = []
        
        self.dict_text_output = dict_text_output
        self.early_stopping = early_stopping
        self.predicting_delta = predicting_delta
        self.save_trucated_training = save_trucated_training


    def run_trainer(self):
        
        best_loss = None
        counter = 0
        
        # create a progress bar 
        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange
        progressbar = trange(self.epochs-self.epoch, desc='Progress')
        
        # printing only at the end of a training
          
        if self.epoch == self.epochs-2:
            self.printstat = True
            
        for i in progressbar:
            """Epoch counter"""
            self.epoch += 1  # epoch counter
            printif(f"\n{'='*60}",self.printstat)
            printif(f"EPOCH: {self.epoch}", self.printstat)

            """Training block"""
            self._train()

            """Validation block"""
            if self.validation_DataLoader is not None:
                self._validate()
                
            """Plotting block"""
            if self.plotstat:
                self._plot_results()

            """Learning rate scheduler block"""
            if self.select_scheduler != "None":
                if self.select_scheduler ==  "ReduceLROnPlateau":
                    self.lr_scheduler.step(self.validation_loss[-1])
                elif self.select_scheduler == "MultiStepLR" and self.epochs >=10:
                    self.lr_scheduler.step()
                elif self.select_scheduler == "CosineAnnealingWarmRestarts":
                    self.lr_scheduler.step(self.epoch / len(self.training_DataLoader))
                else:
                    print (f"{self.select_scheduler} not implemented, select_scheduler set to None")
                    self.select_scheduler == "None"
            
            """Saving block"""
            if self.savestat:
                self._save_information()
                
            """Early Stopping"""
            if self.early_stopping and self.epoch > 1000:
                #patience: how many epochs to wait before stopping when loss is not improving
                #min_delta: minimum difference between new loss and old loss for new loss to be considered as an improvement
                #variable_min_delta: if True the min delta is set to a certain % of the best loss (eg.1%)
                early_stopping_check = EarlyStopping(patience=5, 
                                                     #min_delta = 20,
                                                     variable_min_delta = True,
                                                    )
                best_loss, counter, early_stop = early_stopping_check(self.validation_loss[-1],best_loss,counter)
                #print (f"{self.validation_loss=}")
                #print (f"{early_stop=}")

                if early_stop:
                    print ("Early stop condition met")
                    self.save_trucated_training = True
                    self._save_information()
                    break
               
                
            #pdb.set_trace()
            
        print (DASH) 
        print(f"Final Train Loss: {self.training_loss[-1]:.4f}")
        print(f'Final Val Loss: {self.validation_loss[-1]:.4f}')
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
        reduction = self.reduction
        
        printif ("\n\t> TRAINING", printstat)
        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange       
                
        self.model.to(self.device)
        self.model.train()  # train mode
        
        train_losses = []  # accumulate the losses here
        iters = len(self.training_DataLoader)
        batch_iter = tqdm(enumerate(self.training_DataLoader), 
                          'Training', 
                          total=iters,
                          leave=False)
        
        #print ("\nBATCHITER:")
        #print (self.training_DataLoader)
        #print (len(self.training_DataLoader.dataset))
        #print (len(batch_iter))
        
        # optimizer to gpu
        optimizer_to(self.optimizer,self.device)
        
        for i, data in batch_iter:
            #pdb.set_trace()
            printif(f"{'-'*40}",printstat)
            printif(f"BATCH ITER:  {pad(i)}",printstat)
            printif(f"{len(batch_iter)=}",printstat)
            
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
            
            printif("\n INPUT",printstat)
            printif(f"{x=},\n{edge_index=},\n{edge_attr=}",printstat)
            printif("\n\n TARGET",printstat)
            printif(target,printstat)
            printif("\n PREDICTION",printstat)
            printif(out,printstat)
                
            ## loss
            #loss = self.criterion(out, target)
            loss = calculateLoss(losstat = self.select_criterion,
                                 reduction = reduction,
                                 out = out,
                                 target= target, 
                                 x = x,                                 
                                 criterion = None, 
                                 phase = None, 
                                 device = self.device,
                                 printstat = printstat,
                                )
            
            loss_value = loss.item()
            train_losses.append(loss_value)
            printif(f"{len(train_losses)=}",printstat)
            printif(f"{loss_value=}",printstat)
            loss.backward()  # one backward pass
            self.optimizer.step()  # update the parameters
            batch_iter.set_description(f'Training: (loss {loss_value:.4f})')  # update progressbar
            
        # append losses and lr to list
        mean_training_loss = np.mean(train_losses)
        printif(f"\nTRAINING CONCLUSION {epoch=}",printstat)
        printif(f"input size = {len(train_losses)}",printstat)
        printif(list(train_losses),printstat)
        printif(f"{mean_training_loss=}",printstat)
        #pdb.set_trace()
        self.training_loss.append(mean_training_loss)
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
        reduction = self.reduction
        
        #printif ("\n\t> VALIDATION", printstat)
        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange
        
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
                
                loss = calculateLoss(losstat = self.select_criterion,
                                     reduction = reduction,
                                     out=out, 
                                     target= target,
                                     x = x, 
                                     criterion= None, 
                                     phase = None, 
                                     device = self.device)
                loss_value = loss.item()
                printif(f"{loss_value=}",printstat)
                valid_losses.append(loss_value)
                batch_iter.set_description(f'Validation: (loss {loss_value:.4f})')
        
          
        mean_validation_loss = np.mean(valid_losses)
        printif(f"\nVALIDATION CONCLUSION {epoch=}",printstat)
        printif(f"input size = {len(valid_losses)}",printstat)
        printif(f"{mean_validation_loss=}",printstat)
        self.validation_loss.append(mean_validation_loss)     
        batch_iter.close()
    
    def _plot_results(self):
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
        reduction = self.reduction
        predicting_delta = self.predicting_delta
           
        # check that its time to plot 
        if isinstance (plot_every, int):
            condition_plot = (epoch % plot_every) == 0
        elif isinstance (plot_every, list):
            condition_plot = epoch in plot_every
        else:
            print (f"Error in the condition plot: {plot_every} should be an int or a list instead is {type(plot_every)}")
            print ("not plotting")
            condition_plot = False  
            
        #-----------------------    
        ## create directories
        #-----------------------
        if condition_plot:
                
            #create figures dir
            FIG_DIR_T = os.path.join(save_dir,'figures_training_set/')
            if not os.path.exists(FIG_DIR_T):
                Path(FIG_DIR_T).mkdir(parents=True, exist_ok=True)  
            FIG_DIR_T_ZOOM = os.path.join(FIG_DIR_T,"ZOOM/")
            if not os.path.exists(FIG_DIR_T_ZOOM):
                Path(FIG_DIR_T_ZOOM).mkdir(parents=True, exist_ok=True)  
                
            #create figures dir
            FIG_DIR_V = os.path.join(save_dir,'figures_validation_set/')
            FIG_DIR_V_ZOOM = os.path.join(FIG_DIR_V,"ZOOM/")
            if not os.path.exists(FIG_DIR_V):
                Path(FIG_DIR_V).mkdir(parents=True, exist_ok=True)
            if not os.path.exists(FIG_DIR_V_ZOOM):
                Path(FIG_DIR_V_ZOOM).mkdir(parents=True, exist_ok=True)
       
        #------------------
        ## TRAINING
        #------------------
        if (shuttle_train is not None) and condition_plot:
            epoch_title = f"{date}{ts}Model_{model_name}_shuttle_train_epoch_{pad(epoch, minl = 6)}of{epochs}"
            loss = plot_results_model(
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

                                       save_dir = FIG_DIR_T,

                                       xlim = (-100,100),
                                       ylim = (-100,100),
                
                                       loss_stat = self.select_criterion,
                                       reduction = reduction,
                                       predicting_delta = predicting_delta,
                                      )
            fig_paths.append(os.path.join(FIG_DIR_T,f"{epoch_title}_{loss[0]}.png"))
            
            #ZOOM
            zoom = 25
            epoch_title_zoom = f"{epoch_title}_ZOOM{zoom}"
            loss = plot_results_model(
                                       self.model,
                                       dataset= shuttle_val,
                                       #device = self.device,
                                       start_from = 0,
                                       end_at = 1,
                                       edges_attr = True,
                                       figsize = (10,10),
                                       title = epoch_title_zoom,
                                       padding_title = -15,

                                       plot_input = True,
                                       plot_target = True,
                                       plot_prediction = True,
                                       plot_intention = False, 

                                       save_dir = FIG_DIR_T_ZOOM,

                                       xlim = (-zoom,zoom),
                                       ylim = (-zoom,zoom),

                                       loss_stat = self.select_criterion,
                                       reduction = reduction,
                                       predicting_delta = predicting_delta,                
                                      )
            fig_paths.append(os.path.join(FIG_DIR_T_ZOOM,f"{epoch_title_zoom}.png"))  
            
        #------------------
        # VALIDATION
        #------------------       
        
        if (shuttle_val is not None) and condition_plot:
            epoch_title = f"{date}{ts}Model_{model_name}_shuttle_val_epoch_{pad(epoch, minl = 6)}of{epochs}"
            
            loss_shuttle = plot_results_model(
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

                                       save_dir = FIG_DIR_V,

                                       xlim = (-100,100),
                                       ylim = (-100,100),

                                       loss_stat = self.select_criterion,
                                       reduction = reduction,
                                       predicting_delta = predicting_delta,                             
                                      )
            fig_paths.append(os.path.join(FIG_DIR_V,f"{epoch_title}.png"))
            
            #ZOOM
            zoom = 25
            epoch_title_zoom = f"{epoch_title}_ZOOM{zoom}"
            loss_shuttle = plot_results_model(
                                       self.model,
                                       dataset= shuttle_val,
                                       #device = self.device,
                                       start_from = 0,
                                       end_at = 1,
                                       edges_attr = True,
                                       figsize = (10,10),
                                       title = epoch_title_zoom,
                                       padding_title = -15,

                                       plot_input = True,
                                       plot_target = True,
                                       plot_prediction = True,
                                       plot_intention = False, 

                                       save_dir = FIG_DIR_V_ZOOM,

                                       xlim = (-zoom,zoom),
                                       ylim = (-zoom,zoom),

                                       loss_stat = self.select_criterion,
                                       reduction = reduction,
                                       predicting_delta = predicting_delta,                
                                      )
            fig_paths.append(os.path.join(FIG_DIR_V_ZOOM,f"{epoch_title_zoom}.png"))
    
    
    
    def _save_information(self):

        ts= self.ts
        date = self.date
        save_dir = self.save_dir
        model_name = self.model_name
        epoch = self.epoch
        epochs = self.epochs
        save_every = self.save_every
        plot_every = self.plot_every
        plotstat = self.plotstat
        fig_paths = self.figures_paths
        printstat = self.printstat
        dict_text_output = self.dict_text_output
        save_trucated_training = self.save_trucated_training
        
        # save intermediate model and losses
        if self.savestat and (epoch % save_every == 0 or save_trucated_training):
            
            if epoch < epochs:
                type_save = "TEMP" 
                if save_trucated_training:
                    print ("saving files of truncated training")
                    type_save = "TRUNC" 
            else: 
                type_save = "FINAL" 
            
            description = f"{date}{ts}EPOCH_{epoch}of{epochs}_{type_save}"

            # save model
            model_descr = f"{description}_{model_name}.pt"
            full_model_descr = f"{description}_full_{model_name}.pt"
            model_path = os.path.join(save_dir,model_descr)
            full_model_path = os.path.join(save_dir,full_model_descr)
            
            # save the complete model
            #torch.save(model.state_dict(), filepath)
            torch.save (self.model.state_dict(), model_path)
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': self.training_loss[-1],
                        }, full_model_path)
            printif(f"Saved model in {model_path}",printstat or type_save == "FINAL_")
            printif(f"Saved full model in {full_model_path}",printstat or type_save == "FINAL_")
            
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
            
            for k,v in self.paths.items():
                dict_text_output[k]= v
            
            ## add information to the dict
            dict_text_output[f'final_train_loss']= self.training_loss[-1]
            dict_text_output[f'final_val_loss']= self.validation_loss[-1]
            #dict_text_output["figure_paths"] = self.figures_paths

            description = f"{date}{ts}"
            dict_text_output_descr = f"{description}_training_parameters"
            dict_text_output_path = os.path.join(save_dir,dict_text_output_descr)
            printif(f"\n> Parameters stored in: \n{dict_text_output_path}.pkl", printstat or type_save == "FINAL_")

            # store parameters as txt file
            TXT_OUTPUT = f'{dict_text_output_path}.txt'
            with open(TXT_OUTPUT, 'w') as filehandle:
                for k,v in dict_text_output.items():
                    filehandle.write(f'{k}: {v}\n')
                filehandle.close()

            # store parameters as a pickle
            with open(f'{dict_text_output_path}.pkl', 'wb') as handle:
                pkl.dump(dict_text_output, handle, protocol=pkl.HIGHEST_PROTOCOL)





def check_import():
    print (f"imported trainer: {this_GCN} at {get_timestamp()}")
    return this_GCN,get_timestamp()

