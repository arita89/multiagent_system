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

from _01_functions._00_helper_functions import *
from _00_configuration._00_settings import *
from _00_configuration._01_variables import *
from torch.autograd import Variable
#from config import *

"""
here custom loss function
* L1
* handmade L1 with mean 
"""    
class Custom_Loss_2(torch.nn.Module):
    """
    handmade L1 loss
    """

    def __init__(self):
        super(Custom_Loss_2,self).__init__()

    def forward(self,
                out,target,x, 
                device = "cuda",
                printstat= False, 
                reduction = "mean"):
        
        # cast the tensors to numpy
        x_np = x.cpu().detach().numpy()
        out_np = out.cpu().detach().numpy()#[0]
        target_np = target.cpu().detach().numpy()#[0]
        criterion = torch.nn.L1Loss(reduction = reduction)
        
        assert len(out_np) == len(target_np) 
        assert len(out_np) == len(x_np)
        
        handmade_loss = 0
        for out_vect,target_vect in list(zip(out_np,target_np)):
            target_vect = list(target_vect)
            out_vect = list(out_vect)
            printif(f"{reduction=}",printstat)
            printif(f"{out_vect=}",printstat)
            printif(f"{target_vect=}",printstat)
            #printif(target_vect[0])
            #pdb.set_trace()
            loss_veh = 0
            for i,t in enumerate(target_vect):
                printif(f"abs({out_vect[i]}- {t}) = {abs(out_vect[i]-t)}",printstat)
                loss_veh +=  abs(out_vect[i]-t)      
                
            #loss_veh = sum([abs(o-target_vect[i]) for i,o in out_vect])

            printif(f"{loss_veh=}",printstat)
            handmade_loss += loss_veh


        # mean over the size of the input
        if reduction == "mean":
            handmade_loss = handmade_loss/len(out_np)
        printif(f"{handmade_loss=}",printstat)
        handmade_loss = Variable(torch.from_numpy(np.array(handmade_loss)), requires_grad = True)
        loss_default = criterion(out, target)
        
        printif(f"{handmade_loss=}",printstat)
        printif(f"{loss_default=}",printstat)
        #pdb.set_trace()
        return handmade_loss
    
class Custom_Loss_1(torch.nn.Module):
    """
    weighted L1 loss depending on the vehicle speed 
    """

    def __init__(self):
        super(Custom_Loss_1,self).__init__()

    def forward(self,
                out, target,x, 
                device = "cuda",
                printstat= False, 
                reduction = "mean"):
        
        # cast the tensors to numpy
        x_np = x.cpu().detach().numpy()
        out_np = out.cpu().detach().numpy()#[0]
        target_np = target.cpu().detach().numpy()#[0]
        criterion = torch.nn.L1Loss(reduction = reduction)
        
        handmade_loss = 0
        for o,t,feats in list(zip(out_np,target_np,x_np)):
            loss_veh = criterion(torch.FloatTensor(o), torch.FloatTensor(t))
            speed = feats[3]
            printif(f"{loss_veh=}",printstat)
            printif(f"{feats=}",printstat)
            printif(f"{speed=}",printstat)
            handmade_loss += loss_veh*(1+speed*0.1)

        # mean over the size of the input
        handmade_loss = handmade_loss/len(x_np)
        handmade_loss = Variable(handmade_loss, requires_grad = True)

        return handmade_loss

# here all the possible losses
def calculateLoss(losstat,
                  reduction,
                  out, 
                  target, 
                  x = None, 
                  criterion= None, 
                  phase = None, 
                  device = "cuda",
                  printstat = False,
                 ):
    printif("CALCULATING LOSS",printstat)
    if losstat == "L1":
        criterion = torch.nn.L1Loss(reduction = reduction)
        loss = criterion(out, target)
        
    elif losstat == "Custom_Loss_1":
        criterion = Custom_Loss_1()
        loss = criterion(out,target,x,device,printstat,reduction = reduction) 
        
    elif losstat == "Custom_Loss_2":
        criterion = Custom_Loss_2()
        loss = criterion(out,target,x,device,printstat,reduction = reduction) 
        
    elif losstat == "MSE":
        criterion = torch.nn.MSELoss()
        loss = criterion(out, target)
         
    else:
        criterion = torch.nn.L1Loss()
        loss = criterion(out, target)
        
    printif(f"\n{losstat}",printstat)
    printif(f"{out=}",printstat)
    printif(f"{target=}",printstat)
    printif(f"{loss=}",printstat)
    #pdb.set_trace()
    return loss

print ("Custom loss imported")