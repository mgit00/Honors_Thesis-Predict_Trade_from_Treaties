from pandas.io import pickle
import torch 
from sklearn.model_selection import train_test_split 
from torch.utils.data import DataLoader, TensorDataset

# for make_model()
import torch.nn as nn 
from torch import optim 

from model import LSTM 

import numpy as np 
import pickle 

from earlystopping import EarlyStopping 


'''
ModelHelper sets up the data in tensor form and 
passes it to the defined model 
- model.py 
'''

class ModelHelper:
  def __init__(self, x, y , args ): 
    self.x= x 
    self.y= y 
    self.args =args 

    self.train_set, self.test_set= self.loading()

    self.curr_state= self.make_model()


  def loading(self): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Tensor on', device)
    x= torch.tensor(self.x, dtype=torch.float32 ).to(device)
    y= torch.tensor(self.y, dtype=torch.float32).to(device)

    xtrain,xtest, ytrain,ytest= train_test_split(x,y, test_size=.3, random_state=0 )

    train= TensorDataset(xtrain, ytrain)
    train_set= DataLoader(train, batch_size=self.args.batch_size, shuffle=True)

    test= TensorDataset(xtest, ytest)
    test_set= DataLoader(test, batch_size= 256, shuffle= True )

    return train_set, test_set

  def make_model(self) -> dict : 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Model on', device)
    args= self.args
    train_set= self.train_set
    test_set= self.test_set

    input_size= 20 # feature size 
    num_layers = self.args.num_layers
    hidden_size= self.args.hidden_size 
    output_size= 1 # output size 

    lr = self.args.lr  
    epochs= self.args.epochs


    completed_epochs=0 # for resuming training from saved model 
    train_loss_data= []
    test_loss_data=[]

    past_args=[]


    # MODEL checkpoint??

    model= LSTM(input_size, hidden_size, num_layers, output_size ).to(device)

    loss = nn.MSELoss()
    optimizer= optim.Adam(model.parameters(), lr = lr )

    earlystop = EarlyStopping(self.args)

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

    past_args.append(self.args.__dict__)

    for epoch in range(epochs):
      model.train( ) # set to train mode 
      train_loss= 0 # loss for epoch 

      for xbatch, ybatch in train_set: 
        pred= model(xbatch) # forward 
        
        batchloss= loss(pred, ybatch)

        optimizer.zero_grad() # zero gradient 
        batchloss.backward() # propogate loss backwards 
        optimizer.step() # make a step with the optimizer 

        train_loss += batchloss.item() # add curr batch's loss to epoch's loss 


      model.eval() # set to evalualtion mode after train 
      with torch.no_grad(): # no need to call backward() on loss fun, so use no grad to reduce computation as no grad is calculated 
        test_loss= 0 

        for xbatch, ybatch in test_set: 
          pred= model(xbatch) # will be 1 iter with full batch size, predict_on_batch from TF

          batchloss= loss(pred, ybatch )

          # no zerograd, backward, or step 

          test_loss += batchloss.item()

        



      train_loss, test_loss= np.sqrt(train_loss), np.sqrt(test_loss)
      train_loss_data.append(float(train_loss))
      test_loss_data.append(float(test_loss))
      print('Epoch', epoch + completed_epochs, train_loss, test_loss)


      if earlystop.early_stop(train_loss): 
        print('Early Stopping Here')
        break 


    curr_state= { 
        'model': model.state_dict(), 
        'optimizer': optimizer.state_dict(), 
        'epoch': epoch+ completed_epochs, 
        'train_loss_data': train_loss_data, 
        'test_loss_data': test_loss_data, 
        'args': past_args
        }



    if args.save_model:
      '''
      Save the state_dict, rather than the entire model and optimizer. 
      Torch states that saving the entire model means you have to replicate the directory as they state, 
      or the code can break. Save the state_dict to avoid future errors. 
      torch save is better than pickle.dump( model, open(args.model_saveto, 'wb'))
      '''

      torch.save(curr_state, args.save_model)
      print( "model saved to ", args.save_model,)


    return curr_state







