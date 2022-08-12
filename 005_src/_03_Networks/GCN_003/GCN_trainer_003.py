#--------------------------------
## IMPORTS
#--------------------------------
import sys
import os

# set the path to find the modules
sys.path.insert(0, '../005_src/') #use relative path
os.chdir("../005_src")

from config import *


class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 #criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 training_DataLoader: torch.utils.data.Dataset,
                 validation_DataLoader: torch.utils.data.Dataset = None,
                 lr_scheduler: torch.optim.lr_scheduler = None,
                 
                 epochs: int = 10,
                 epoch: int = 0,
                 notebook: bool = False,
                 save_every: int =5,
                 
                 losstat = "criterion",
                 maskstat = True,
                 timestamp = None,
                 
                 models_directory = None, 
                 resizeto = None,
                 ):

        self.model = model
        #self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.training_DataLoader = training_DataLoader
        self.validation_DataLoader = validation_DataLoader
        self.device = device
        self.epochs = epochs
        self.epoch = epoch
        self.notebook = notebook
        self.save_every =save_every 
        self.losstat = losstat
        self.maskstat = maskstat
        self.timestamp = timestamp
        self.resizeto = resizeto
        
        self.models_directory = models_directory
        self.paths = []
        self.training_loss = []
        self.validation_loss = []
        self.learning_rate = []
        self.figures_paths  = []

    def run_trainer(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange
            
        device = cudaOverview()
        
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
        
        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.train()  # train mode
        train_losses = []  # accumulate the losses here

        batch_iter = tqdm(enumerate(self.training_DataLoader), 'Training', total=len(self.training_DataLoader),
                          leave=False)
    
        
        timest= self.timestamp
        for i, data in batch_iter:
            x, target = data.x,data.y
            x, target = x.to(self.device), target.to(self.device)
            # zerograd the parameters
            self.optimizer.zero_grad() 
            ## one forward pass
            out = self.model(x)
            
            #printif(f"{out.shape=},{target.shape=}", printstat)
            #pdb.set_trace()

            ## calculate loss
            loss = criterion(out, target)
             
            loss_value = loss.item()
            train_losses.append(loss_value)
            
            loss.backward()  # one backward pass
            self.optimizer.step()  # update the parameters

            batch_iter.set_description(f'Training: (loss {loss_value:.4f})')  # update progressbar
        
        self.training_loss.append(np.mean(train_losses))
        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])

        batch_iter.close()

    def _validate(self):
        #MODELS_TODAY = self.models_directory
        timest= self.timestamp
        
        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.eval()  # evaluation mode
        valid_losses = []  # accumulate the losses here
        batch_iter = tqdm(enumerate(self.validation_DataLoader), 'Validation', total=len(self.validation_DataLoader),
                          leave=False)

        for i, data in batch_iter:
            x, target = data.x,data.y
            x, target = x.to(self.device), target.to(self.device)  # send to device (GPU or CPU)
            
            
            with torch.no_grad():
                out = self.model(x)
                                
                loss = criterion(out, target)
                    
                #print(loss)
                loss_value = loss.item()
                valid_losses.append(loss_value)

                batch_iter.set_description(f'Validation: (loss {loss_value:.4f})')
        
        #print (np.mean(valid_losses))
        self.validation_loss.append(np.mean(valid_losses))
        
        if self.epoch % self.save_every == 0:
            
            if self.epoch < self.epochs:
                typesave = "TEMP_" 
            else: 
                typesave = "FINAL_" 
            
            details_descr = f"_{self.losstat}_masked_{self.maskstat}_EPOCH_{self.epoch}of{self.epochs}-images_{len(self.training_DataLoader)}_"
                      
            model_name = timest + details_descr +f'{typesave}Regression_Unet06'+".pt"
            model_path = os.path.join(MODELS_TODAY,model_name)
            printif(model_path, printstat)

            torch.save({
                'epoch': self.epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss,
                }, model_path)
            
            #store losses amd lr
            tloss_path = os.path.join(MODELS_TODAY,timest +details_descr+f'{typesave}training_loss'+'.pkl')
            with open(tloss_path, 'wb') as f:
                    pickle.dump(self.training_loss, f)
            
            vloss_path = os.path.join(MODELS_TODAY,timest +details_descr+f'{typesave}validation_loss'+'.pkl')
            with open(vloss_path, 'wb') as f:
                    pickle.dump(self.validation_loss, f)
            
            lr_path = os.path.join(MODELS_TODAY,timest +details_descr+f'{typesave}learning_rate'+'.pkl')
            with open(lr_path, 'wb') as f:
                    pickle.dump(self.learning_rate, f)
                    
            #return only the last paths
            self.paths = [tloss_path,vloss_path,lr_path]

        batch_iter.close()
        
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
        alpha = 0.25
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
        subfig2.plot(x_range, learning_rate, color='black')
        subfig2.title.set_text('Learning rate')
        subfig2.set_xlabel('Epoch')
        subfig2.set_ylabel('LR')

    return fig



def check_import():
    print (f"imported model: GCN_003 at {get_timestamp()}")