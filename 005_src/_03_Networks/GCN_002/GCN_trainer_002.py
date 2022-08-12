  
#--------------------------------
## IMPORTS
#--------------------------------
import sys
import os
# current GCN
this_GCN = os.path.split(os.path.realpath(__file__))[1]
# set the path to find the modules
sys.path.insert(0, '../005_src/') #use relative path
os.chdir("../005_src")

from config import *

def train(model, 
          optimizer,
          criterion,
          train_loader,
          training_losses, 
          printstat= False
         ):
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    
    total_loss = 0
    for data in train_loader:
        out = model(data.x, data.edge_index, data.edge_attr) 
        loss = criterion(out, data.y)
        training_losses.append(loss.item())

        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        
        #total_loss += loss.item() * data.num_graphs
    return training_losses
        
def validation(model, 
          optimizer,
          criterion,
          val_loader,
          printstat= False
         ):
    model.eval()
    
    for data in val_loader:
        out = model(data.x, data.edge_index)
        loss = criterion(out, data.y)  # Compute the loss solely based on the training nodes.
        printif(f"{epoch_validation_losses}",printstat)
        validation_losses.append(loss.item())
        
def test(model, 
          optimizer,
          criterion,
          data,
          printstat= False
         ):
    model.eval()
    out = model(data.x, data.edge_index)
    loss = criterion(out, data.y)  
    return out



def check_import():
    print (f"imported trainer: {this_GCN} at {get_timestamp()}")