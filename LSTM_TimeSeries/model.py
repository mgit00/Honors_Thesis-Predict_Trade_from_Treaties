import torch.nn as nn 

'''
Used by modelhelper, it defines the model to be used 
'''

class LSTM(nn.Module): 
  def __init__(self, input_size, hidden_size, num_layers, output_size) : 
    super().__init__()

    self.hidden_size= hidden_size
    self.num_layers=num_layers

    self.lstm= nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    self.linear= nn.Linear(hidden_size, output_size)


  def forward(self, x): 
    out,_= self.lstm(x)

    out= self.linear(out[:,-1,:])

    return out 


