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


def in_circle(x1, y1, x2, y2,acceptance_radius):
    """
    check if point x2.y2 is inside a circle of radius and center x1,y1
    """
    dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return dist <= acceptance_radius

def get_loss_weight(x_vect, out_vect,acceptance_radius):
    """
    x_vect = initial coordinates
    out_vect = predicted coordinates
    """
    if in_circle( out_vect[0], out_vect[1], x_vect[0], x_vect[1],acceptance_radius ) :
        return 1
    else:
        return 10

class Custom_Loss_6(torch.nn.Module):
    """
    handmade loss, using MSE as basis, given the current position and the 
        - max speed of veh (which is 20m/s in the sim)
        - current speed of veh (doesnt consider the possibility of acceleration in 2s)
    there is a much more limited domain of plausible locations where the prediction can be.
    using the max available speed will give all the vehicles acceptables subdomains of the same size(pi*r^2)
    where r = v_max* delta_t = 20m/s*2s = = 40m 
    """

    def __init__(self):
        super(Custom_Loss_6,self).__init__()

    def forward(self,
                out,target,x, 
                device = "cuda",
                printstat= False, 
                reduction = "mean",
                acceptance_radius = 40, 
               ):

        printif(f"{reduction=}",printstat)
        mse = torch.nn.MSELoss(reduction = reduction) 

        printif(f"\t{target=}",printstat)
        printif(f"\t{out=}",printstat)
        printif(f"\t{x=}",printstat) #x,y,yaw,speed,intention,still_veh?

         
        tot_loss = [mse(tv,ov)*get_loss_weight(xv, ov,acceptance_radius) 
                        for (xv,ov,tv) in zip(x,out,target)]
    
        printif(tot_loss,printstat)
          
        #handmade_loss =  Variable(sum(tot_loss), requires_grad = True)
        handmade_loss =  sum(tot_loss)
        printif(f"{handmade_loss=}",printstat)
        #pdb.set_trace()
        return handmade_loss
    
class Custom_Loss_4(torch.nn.Module):
    """
    handmade loss, using L1 formula, given the current position and the 
        - max speed of veh (which is 20km/h in the sim)
        - current speed of veh (doesnt consider the possibility of acceleration in 2s)
    there is a much more limited domain of plausible locations where the prediction can be.
    using the max available speed will give all the vehicles acceptables subdomains of the same size(pi*r^2)
    where r = v_max* delta_t = 20m/s*2s = = 40m 
    """

    def __init__(self):
        super(Custom_Loss_4,self).__init__()

    def forward(self,
                out,target,x, 
                device = "cuda",
                printstat= False, 
                reduction = "mean",
                acceptance_radius = 12, 
                variable_acceptance_radius = False,
               ):
        
         
        #criterion = torch.nn.L1Loss(reduction = reduction)
        
        handmade_loss = 0
        printif(f"{reduction=}",printstat)
        
        count_veh = 0
        
        # per target/prediction
        for out_vect,target_vect,x_vect in list(zip(out,target,x)):
            
            printif(f"\nVEH_{pad(count_veh)}",printstat)
            target_vect = list(target_vect)
            out_vect = list(out_vect)
            x_vect = list(x_vect)
            
            printif(f"\t{target_vect=}",printstat)
            printif(f"\t{out_vect=}",printstat)
            printif(f"\t{x_vect=}",printstat) #x,y,yaw,speed,intention,still_veh?
            
            #per vehicle
            loss_veh = 0
            for i,t in enumerate(target_vect):

                # check that point is within the acceptable radius
                # (x_f-x_i)^2 + (y_f-y_i)^2 <= acceptance_radius
                
                # predicted position
                x_pred = out_vect[0]
                y_pred = out_vect[1]
                
                x_i = x_vect[0]
                y_i = x_vect[1]
                
                if variable_acceptance_radius:
                    acceptance_radius = 2*x_vect[3]   
                
                if in_circle(x_pred, y_pred, x_i, y_i,acceptance_radius ) :
                    loss_weight = 1
                else: 
                    loss_weight = 10
                this_loss = abs(out_vect[i]-t)*loss_weight
                loss_veh += this_loss     
                printif(f"\tfeature_{pad(i)}   abs({out_vect[i]}-{t})*{loss_weight}={this_loss}",printstat)
                
            printif(f"\t---------------------------------------------------> {loss_veh=}",printstat)
            handmade_loss += loss_veh
            count_veh +=1


        # mean over the size of the input
        if reduction == "mean":
            handmade_loss = handmade_loss/len(out)

        
        printif(f"{handmade_loss=}",printstat)
        return handmade_loss
    
class Custom_Loss_5(torch.nn.Module):
    """
    handmade loss, using MSE as basis, given the current position and the 
        - max speed of veh (which is 20m/s in the sim)
        - current speed of veh (doesnt consider the possibility of acceleration in 2s)
    there is a much more limited domain of plausible locations where the prediction can be.
    using the max available speed will give all the vehicles acceptables subdomains of the same size(pi*r^2)
    where r = v_max* delta_t = 20m/s*2s = = 40m 
    """

    def __init__(self):
        super(Custom_Loss_5,self).__init__()

    def forward(self,
                out,target,x, 
                device = "cuda",
                printstat= False, 
                reduction = "mean",
                acceptance_radius = 12, 
                variable_acceptance_radius = False,
               ):
        
         
        #criterion = torch.nn.L1Loss(reduction = reduction)
        
        handmade_loss = 0
        printif(f"{reduction=}",printstat)
        
        count_veh = 0
        # per target/prediction
        for out_vect,target_vect,x_vect in list(zip(out,target,x)):
            
            printif(f"\nVEH_{pad(count_veh)}",printstat)
            target_vect = list(target_vect)
            out_vect = list(out_vect)
            x_vect = list(x_vect)
            
            printif(f"\t{target_vect=}",printstat)
            printif(f"\t{out_vect=}",printstat)
            printif(f"\t{x_vect=}",printstat) #x,y,yaw,speed,intention,still_veh?
            
            #per vehicle
            loss_veh = 0
            for i,t in enumerate(target_vect):
                
                # check that point is within the acceptable radius
                # (x_f-x_i)^2 + (y_f-y_i)^2 <= acceptance_radius
                
                # predicted position
                x_pred = out_vect[0]
                y_pred = out_vect[1]
                
                x_i = x_vect[0]
                y_i = x_vect[1]
                
                if variable_acceptance_radius:
                    acceptance_radius = 2*x_vect[3]   
                
                if in_circle(x_pred, y_pred, x_i, y_i,acceptance_radius ) :
                    loss_weight = 1
                else: 
                    loss_weight = 10
                    
                #mse = torch.nn.MSELoss(reduction = reduction)
                #this_loss = mse(out_vect[i],t)*loss_weight 
                this_loss = ((abs(out_vect[i]-t))**2)*loss_weight 
                loss_veh +=   this_loss
                printif(f"\tfeature_{pad(i)}   (abs({out_vect[i]}-{t})**2)*{loss_weight}={this_loss}",printstat)
                
            #loss_veh = sum([abs(o-target_vect[i]) for i,o in out_vect])

            printif(f"\t---------------------------------------------------> {loss_veh=}",printstat)
            handmade_loss += loss_veh
            count_veh +=1

        # mean over the size of the input
        #if reduction == "mean":
            #handmade_loss = handmade_loss/len(out)

        
        printif(f"{handmade_loss=}",printstat)
        return handmade_loss
    
class Custom_Loss_4(torch.nn.Module):
    """
    handmade loss, using L1 formula, given the current position and the 
        - max speed of veh (which is 20km/h in the sim)
        - current speed of veh (doesnt consider the possibility of acceleration in 2s)
    there is a much more limited domain of plausible locations where the prediction can be.
    using the max available speed will give all the vehicles acceptables subdomains of the same size(pi*r^2)
    where r = v_max* delta_t = 20m/s*2s = = 40m 
    """

    def __init__(self):
        super(Custom_Loss_4,self).__init__()

    def forward(self,
                out,target,x, 
                device = "cuda",
                printstat= False, 
                reduction = "mean",
                acceptance_radius = 12, 
                variable_acceptance_radius = False,
               ):
        
         
        #criterion = torch.nn.L1Loss(reduction = reduction)
        
        handmade_loss = 0
        printif(f"{reduction=}",printstat)
        
        count_veh = 0
        
        # per target/prediction
        for out_vect,target_vect,x_vect in list(zip(out,target,x)):
            
            printif(f"\nVEH_{pad(count_veh)}",printstat)
            target_vect = list(target_vect)
            out_vect = list(out_vect)
            x_vect = list(x_vect)
            
            printif(f"\t{target_vect=}",printstat)
            printif(f"\t{out_vect=}",printstat)
            printif(f"\t{x_vect=}",printstat) #x,y,yaw,speed,intention,still_veh?
            
            #per vehicle
            loss_veh = 0
            for i,t in enumerate(target_vect):

                # check that point is within the acceptable radius
                # (x_f-x_i)^2 + (y_f-y_i)^2 <= acceptance_radius
                
                # predicted position
                x_pred = out_vect[0]
                y_pred = out_vect[1]
                
                x_i = x_vect[0]
                y_i = x_vect[1]
                
                if variable_acceptance_radius:
                    acceptance_radius = 2*x_vect[3]   
                
                if in_circle(x_pred, y_pred, x_i, y_i,acceptance_radius ) :
                    loss_weight = 1
                else: 
                    loss_weight = 10
                this_loss = abs(out_vect[i]-t)*loss_weight
                loss_veh += this_loss     
                printif(f"\tfeature_{pad(i)}   abs({out_vect[i]}-{t})*{loss_weight}={this_loss}",printstat)
                
            printif(f"\t---------------------------------------------------> {loss_veh=}",printstat)
            handmade_loss += loss_veh
            count_veh +=1


        # mean over the size of the input
        if reduction == "mean":
            handmade_loss = handmade_loss/len(out)

        
        printif(f"{handmade_loss=}",printstat)
        return handmade_loss

class Custom_Loss_3(torch.nn.Module):
    """
    handmade L1 loss, which actually backpropagates
    note, dont move to np, stay with torch tensors
    """

    def __init__(self):
        super(Custom_Loss_3,self).__init__()

    def forward(self,
                out,target,x, 
                device = "cuda",
                printstat= False, 
                reduction = "mean"):
        
        # cast the tensors to numpy
        #x_np = x.cpu().detach().numpy()
        #out_np = out.cpu().detach().numpy()#[0]
        #target_np = target.cpu().detach().numpy()#[0]
        criterion = torch.nn.L1Loss(reduction = reduction)
        
        handmade_loss = 0
        printif(f"{reduction=}",printstat)
        
        count_veh = 0
        for out_vect,target_vect in list(zip(out,target)):
            
            printif(f"\nVEH_{pad(count_veh)}",printstat)
            target_vect = list(target_vect)
            out_vect = list(out_vect)
            
            printif(f"\t{target_vect=}",printstat)
            printif(f"\t{out_vect=}",printstat)

            #printif(target_vect[0])
            #pdb.set_trace()
            loss_veh = 0
            for i,t in enumerate(target_vect):
                printif(f"\tfeature_{pad(i)}   abs({out_vect[i]}-{t})={abs(out_vect[i]-t)}",printstat)
                loss_veh +=  abs(out_vect[i]-t)      
                
            #loss_veh = sum([abs(o-target_vect[i]) for i,o in out_vect])

            printif(f"\t---------------------------------------------------> {loss_veh=}",printstat)
            handmade_loss += loss_veh
            count_veh +=1


        # mean over the size of the input
        if reduction == "mean":
            handmade_loss = handmade_loss/len(out)
        #printif(f"{handmade_loss=}",printstat)
        #handmade_loss = Variable(torch.from_numpy(np.array(handmade_loss)), requires_grad = True)
        loss_default = criterion(out, target)
        
        printif(f"{handmade_loss=}",printstat)
        printif(f"{loss_default=}",printstat)
        #pdb.set_trace()
        return handmade_loss




"""
here custom loss function
* L1
* handmade L1 with mean 
"""    
class Custom_Loss_2(torch.nn.Module):
    """
    handmade L1 loss 
    this is just to check that the values of L1 are correct
    it does not back prop
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
        printif(f"{reduction=}",printstat)
        
        count_veh = 0
        for out_vect,target_vect in list(zip(out_np,target_np)):
            
            printif(f"\nVEH_{pad(count_veh)}",printstat)
            target_vect = list(target_vect)
            out_vect = list(out_vect)
            
            printif(f"\t{target_vect=}",printstat)
            printif(f"\t{out_vect=}",printstat)

            #printif(target_vect[0])
            #pdb.set_trace()
            loss_veh = 0
            for i,t in enumerate(target_vect):
                printif(f"\tfeature_{pad(i)}   abs({out_vect[i]}-{t})={abs(out_vect[i]-t)}",printstat)
                loss_veh +=  abs(out_vect[i]-t)      
                
            #loss_veh = sum([abs(o-target_vect[i]) for i,o in out_vect])

            printif(f"\t---------------------------------------------------> {loss_veh=}",printstat)
            handmade_loss += loss_veh
            count_veh +=1


        # mean over the size of the input
        if reduction == "mean":
            handmade_loss = handmade_loss/len(out_np)
        #printif(f"{handmade_loss=}",printstat)
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
    printif("\nCALCULATING LOSS",printstat)
    # L1 and similar
    if losstat == "L1":
        criterion = torch.nn.L1Loss(reduction = reduction)
        loss = criterion(out, target)
        
    elif losstat == "Custom_Loss_1":
        criterion = Custom_Loss_1()
        loss = criterion(out,target,x,device,printstat,reduction = reduction) 
        
    elif losstat == "Custom_Loss_2":
        criterion = Custom_Loss_2()
        loss = criterion(out,target,x,device,printstat,reduction = reduction) 
        
    elif losstat == "Custom_Loss_3":
        criterion = Custom_Loss_3()
        loss = criterion(out,target,x,device,printstat,reduction = reduction)
    
    elif losstat == "Custom_Loss_4":
        criterion = Custom_Loss_4()
        loss = criterion(out,target,x,device,printstat,reduction = reduction) 
    
    #MSE and similar
    elif losstat == "MSE":
        criterion = torch.nn.MSELoss(reduction = reduction)
        loss = criterion(out, target)
        
    elif losstat == "Custom_Loss_5":
        criterion = Custom_Loss_5()
        loss = criterion(out,target,x,device,printstat,reduction = reduction)
        
    elif losstat == "Custom_Loss_6":
        criterion = Custom_Loss_6()
        loss = criterion(out,target,x,device,printstat,reduction = reduction) 
         
         
    else:
        
        printonceif(f"error: {losstat} not implemented, using L1 loss",True, 1)
        criterion = torch.nn.L1Loss()
        loss = criterion(out, target)
        
    printif(dash, printstat)
    printif(f"\n{losstat}",printstat)
    printif(f"{out=}",printstat)
    printif(f"{target=}",printstat)
    printif(f"{loss=}",printstat)
    #pdb.set_trace()
    return loss

print ("Custom loss imported")