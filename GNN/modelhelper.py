import numpy as np 
from torch_geometric.data import Data

import torch 
import torch.nn as nn 
from torch.optim import Adam 

from models import Model, GCN, EdgePredictor



class ModelHelper:
  def __init__(self, args): 
    self.args= args 


  def train_model(self, data:Data ): # data= graph 
    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_size= data.x.shape[1]
    hidden_size= self.args.hidden_size
    output_size= 3 
    layers= 1 

    completed_epochs=0
    train_loss_data= []
    test_loss_data= []

    past_args=[]



    # model = GCN(input_size, hidden_size, output_size)
    model= Model(input_size, hidden_size, output_size )
    optimizer = Adam(model.parameters(), lr= self.args.lr )
    loss= nn.MSELoss(reduction='none')

    # for if you want to continue training with saved model
    if self.args.load_model: 
      saved_state= torch.load(self.args.load_model)
      model.load_state_dict(saved_state['model'])
      optimizer.load_state_dict(saved_state['optimizer'])

      completed_epochs= saved_state['epoch'] + 1 
      train_loss_data= saved_state['train_loss_data']
      test_loss_data= saved_state['test_loss_data']

      past_args = saved_state['args']

      print('Loading from' ,  self.args.load_model )



    data= data.to(device)
    model= model.to(torch.double) # put model to torch.double or put data.x and data.y to torch.float 
    model= model.to(device)

    print(data)

    

  
    for epoch in range(self.args.epochs): 
      # no datatset iteration here: for row in dataloader..... trainloss.append(loss.item()) 
      
      # train model 
      model.train()
      pred= model(data)
      
      # a mask will be used for train test loss evaluation
      # this means that there will be no reduction in loss and will be calculated by code directly
      epochloss= loss(pred, data.y) 
      epochloss= ((epochloss * data.train_mask).sum())/ data.train_mask.sum()
      train_loss_data.append(epochloss.item())

      optimizer.zero_grad()
      epochloss.backward()
      optimizer.step()

      epochtestloss= ((epochloss * data.test_mask).sum())/ data.test_mask.sum()
      test_loss_data.append(epochtestloss.item())

      epochloss, epochtestloss= np.sqrt(epochloss.item()), np.sqrt(epochtestloss.item())

      print('Epoch', epoch, 'train loss:', epochloss.item(), 'test loss', epochtestloss.item())


    curr_state= { 
        'model': model.state_dict(), 
        'optimizer': optimizer.state_dict(), 
        'epoch': epoch+ completed_epochs, 
        'train_loss_data': train_loss_data, 
        'test_loss_data': test_loss_data, 
        'args': past_args
        }



    if self.args.save_model:
      '''
      Save the state_dict, rather than the entire model and optimizer. 
      Torch states that saving the entire model means you have to replicate the directory as they state, 
      or the code can break. Save the state_dict to avoid future errors. 
      torch save is better than pickle.dump( model, open(args.model_saveto, 'wb'))
      '''

      torch.save(curr_state, self.args.save_model)
      print( "model saved to ", self.args.save_model,)


    return curr_state

 






