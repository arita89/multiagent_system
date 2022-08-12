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

def train(model, 
          optimizer,
          criterion,
          train_loader,
          epoch_training_losses, 
          printstat= False,
          
          savestat = True,
          Nepochs = 100,
          save_every = 10,
          epoch_num = "dummy_epoch",
          save_dir = os.path.join(OUTPUT_DIR,
             f"dummy_GCN/{get_date()}{get_timestamp('')}") ,
          model_name = "dummy_GCN",
          date = get_date(),
          ts = get_timestamp(),
          
         ):
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    
    total_loss = 0
    for data in train_loader:
        out = model(data.x, data.edge_index, data.edge_attr) 
        loss = criterion(out, data.y)
        epoch_training_losses.append(loss.item())

        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        
    description = None
    model_path = None
    if savestat:
        
        if epoch_num == Nepochs:
            type_save = "FINAL"
            description = f"{date}{ts}_epoch{epoch_num}_{type_save}"
            
            model_descr = f"{description}_{model_name}.pt"
            model_path = os.path.join(save_dir,model_descr)
            torch.save (model.state_dict(), model_path)
            
            #torch.save({
                #'epoch': epoch,
                #'model_state_dict': model.state_dict(),
                #'optimizer_state_dict': optimizer.state_dict(),
                #'loss': loss,
                #}, model_path)
            
            print (f"Saved model in {model_path}")
            
            #tloss_descr = f"{description}_training_loss"
            #tloss_path = os.path.join(save_dir,tloss_descr)
            
            #with open(tloss_path, 'wb') as f:
                    #pickle.dump(mean(epoch_training_losses), f)
            
            
        elif epoch_num % save_every == 0 :
            type_save = "TEMP"
            description = f"{date}{ts}{epoch_num}_{type_save}"
            
            model_descr = f"{description}_{model_name}.pt"
            model_path = os.path.join(save_dir,model_descr)
            torch.save (model.state_dict(), model_path)
            print (f"Saved model in {model_path}")
              
    return epoch_training_losses,description,model_path
        
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
    return this_GCN,get_timestamp()