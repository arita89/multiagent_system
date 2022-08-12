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


## edited code from: https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/
class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=10, min_delta=0,variable_min_delta = True):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        #self.counter = 0
        #self.best_loss = None
        self.early_stop = False
        self.variable_min_delta =  variable_min_delta
        
    def __call__(self, 
                 val_loss,
                 best_loss = None,
                 counter = 0, 
                ):
        if best_loss == None:
            best_loss = val_loss
            
        else: 
            if self.variable_min_delta:
                self.min_delta = 0.01*best_loss

            if best_loss - val_loss > self.min_delta:
                best_loss = val_loss
                counter = 0
            elif best_loss - val_loss < self.min_delta:
                counter += 1
                print(f"INFO: Early stopping counter {counter} of {self.patience}")
                if counter >= self.patience:
                    print('INFO: Early stopping')
                    self.early_stop = True

        return best_loss, counter, self.early_stop
          