from _00_settings import *
from _01_functions_utilities import *

num_classes = 12-1


class PointNetLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        # Message passing with "max" aggregation.
        super(PointNetLayer, self).__init__('max')
        
        # Initialization of the MLP:
        # Here, the number of input features correspond to the hidden node
        # dimensionality plus point dimensionality (=3).
        self.mlp = Sequential(Linear(in_channels + 3, out_channels),
                              ReLU(),
                              Linear(out_channels, out_channels))
        
    def forward(self, h, pos, edge_index):
        # Start propagating messages.
        return self.propagate(edge_index, h=h, pos=pos)
    
    def message(self, h_j, pos_j, pos_i):
        # h_j defines the features of neighboring nodes as shape [num_edges, in_channels]
        # pos_j defines the position of neighboring nodes as shape [num_edges, 3]
        # pos_i defines the position of central nodes as shape [num_edges, 3]

        input = pos_j - pos_i  # Compute spatial relation.

        if h_j is not None:
            # In the first layer, we may not have any hidden node features,
            # so we only combine them in case they are present.
            input = torch.cat([h_j, input], dim=-1)

        return self.mlp(input)  # Apply our final MLP.
    
    

class PointNet(torch.nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()

        torch.manual_seed(12345)
        self.conv1 = PointNetLayer(3, 32)
        self.conv2 = PointNetLayer(32, 32)
        self.classifier = Linear(32, num_classes)
        
    def forward(self, pos, batch,edge_index):
        # Compute the kNN graph:
        # Here, we need to pass the batch vector to the function call in order
        # to prevent creating edges between points of different examples.
        # We also add `loop=True` which will add self-loops to the graph in
        # order to preserve central point information.
        
        #edge_index = knn_graph(pos, k=16, batch=batch, loop=True).long()
        
        # 3. Start bipartite message passing.
        h = self.conv1(h=pos, pos=pos, edge_index=edge_index)
        h = h.relu()
        h = self.conv2(h=h, pos=pos, edge_index=edge_index)
        h = h.relu()
        #print (h.shape, type(h))
       
        # 4. Global Pooling.
        #h = global_max_pool(h, batch)  # [num_examples, hidden_channels]
        
        # 5. Classifier.
        return self.classifier(h)
    
def train(model, 
          optimizer,
          criterion,
          train_loader,
          device="cpu"):
    model.train()
    
    total_loss = 0
    for data in train_loader:
        y = data.y
        optimizer.zero_grad()  # Clear gradients.
        
        data, y = data.to(device), y.to(device)  # send to device (GPU or CPU)
        
        logits = model(data.pos.float(), data.batch.long(), data.edge_index) # Forward pass.
        #print (data.batch.long().shape,logits.shape,y.shape, y.min(), y.max(),torch.unique(y).long())
        #print (torch.squeeze(logits))
        #pdb.set_trace()
        
        pred = torch.squeeze(logits)
        #print (pred,y)
        loss = criterion(pred, y)
        
        # shapes https://github.com/pytorch/pytorch/issues/5554
        #loss = criterion(torch.squeeze(logits), y)  # Loss computation.
        loss.backward()  # Backward pass.
        optimizer.step()  # Update model parameters.
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def val(model, 
        val_loader,
        criterion,
        optimizer,
        epoch,
        all_epochs,
        SAVE_DIR,
        save_every = 10,
        num_images = "NA",
       ):
    model.eval()
    
    total_correct = 0
    total = 1
    total_loss = 0
    for data in val_loader:
        y = data.y
        data, y = data.to(device), y.to(device) 
        logits = model(data.pos.float(), data.batch.long(),data.edge_index)
        
        #pred = logits.argmax(dim=-1)
        ## shapes https://github.com/pytorch/pytorch/issues/5554
        pred = torch.squeeze(logits)
        #print (pred,y)
        loss = criterion(pred, y)
        pred = logits.argmax(dim=-1) #this flattens the prediction
        #print (pred,y)
        #pdb.set_trace()
        total_correct += int((pred == data.y).sum())
        total += len(pred)
        total_loss += loss.item() * data.num_graphs
        
        if epoch % save_every == 0:
            
            if epoch == all_epochs:
                typesave = "FINAL"
            else:
                typesave = "TEMP"
            
            details_descr = f"{mytimestamp()[:-1]}_EPOCHS_{epoch}of{all_epochs}_IMAGES_{num_images}"
            model_name =  details_descr +f'{typesave}_pointNet_segmentation.pt'
            model_path = os.path.join(SAVE_DIR,model_name)
            printif(model_path, True)

            # save entire model
            torch.save(model, model_path)

    return total_correct / total, total_loss/len(val_loader.dataset) 


@torch.no_grad()
def inference(
                model, 
                inference_loader,
                criterion,
                optimizer,
               ):
    model.eval()
    
    total_correct = 0
    total = 1
    total_loss = 0
    for data in inference_loader:
        y = data.y
        logits = model(data.pos.float(), data.batch.long(),data.edge_index)
        
        #pred = logits.argmax(dim=-1) # this returns the exat labels.. 
        ## shapes https://github.com/pytorch/pytorch/issues/5554
        pred = torch.squeeze(logits)
        #print (pred,y)
        loss = criterion(pred, y)
        pred = logits.argmax(dim=-1) #this flattens the prediction
        #print (pred,y)
        #pdb.set_trace()
        total_correct += int((pred == data.y).sum())
        total += len(pred)
        total_loss += loss.item() * data.num_graphs
        
    return total_correct / total, total_loss/len(inference_loader.dataset) 