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

"""
here custom loss function
* L1
* handmade L1 with mean 
"""    
    
class Custom_Loss_1(torch.nn.Module):
    """
    inputs are tensors
    """

    def __init__(self):
        super(Custom_Loss_1,self).__init__()

    def forward(self,out, target,device = "cuda"):
        
        # cast the tensors to numpy
        out_np = out.cpu().detach().numpy()#[0]
        target_np = target.cpu().detach().numpy()#[0]
        criterion = torch.nn.L1Loss()
        
        handmade_mean_loss = sum([criterion(row.type(torch.FloatTensor), target_np[e].type(torch.FloatTensor)) 
                         for e, row in enumerate(out_np)])
        loss = criterion(out, target)
        
        print (f"{handmade_mean_loss=}")
        print (f"{loss=}")
        pdb.set_trace()
        return loss

# here all the possible losses
def calculateLoss(losstat, out, target, criterion= None, phase = None):
    if losstat == "L1":
        criterion = torch.nn.L1Loss()
        loss = criterion(out, target)
    elif losstat == "L1_mean_handmade":
        loss = Custom_Loss_1(out,target) 
    elif losstat == "MSE":
        criterion = torch.nn.MSELoss()
        loss = criterion(out, target)
    else:
        criterion = torch.nn.L1Loss()
        loss = criterion(out, target)
    return loss

print ("Custom loss imported")