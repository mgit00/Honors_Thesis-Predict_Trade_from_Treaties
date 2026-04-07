from logging import raiseExceptions
import pandas as pd
pd.set_option('display.max_columns', None )

from pandas._config import options
import seaborn as sns

import numpy as np
np.set_printoptions(suppress= True )
np.set_printoptions(precision=2 )
np.set_printoptions(edgeitems=100)
np.set_printoptions(linewidth=500)

# %matplotlib inline
import matplotlib.pyplot as plt


'''
It is the driver script which will move through taking
the stacked data and creating the full model with it. 

Needs: 
- model.py subclassing from nn.Module 
- stacking.py 
- train.py which calls on model.py for train and eval epochs 

'''


import argparse 
import json 
import sys 
import numpy as np 

from dataset_stacking import DatasetStacking
from modelhelper import ModelHelper
from args_commands import given_commands
from plotting import Plotting 

class Driver: 

  def __init__(self): 
    self.args= self.get_args()
    
    self.x, self.y= self.stack_data()

    self.train_model()



  def get_args(self) -> argparse.Namespace: 
    '''
    Use either .txt file for args or read in through command line 
    If you want .txt file to be read, specify 'command_file' and continue 
      -> command_file 
    If you want command line to be read, specify 'command_line' and continue with correct commands 
      -> command_line --unstacked_datafile unstacked_data.csv --stacking_years 5 ... 
    '''

    parser= argparse.ArgumentParser()

    subparser= parser.add_subparsers(title= 'pick pathway' , dest ='pathway')
    # if and only if .txt file is given 
    command_file= subparser.add_parser('command_file')
    # if and only if command line args are given
    command_line = subparser.add_parser('command_line')

    # command line groups: data, modelparameters, model 
    data= command_line.add_argument_group('data')
    modelparameters= command_line.add_argument_group('model parameters')
    model= command_line.add_argument_group('model')

    data.add_argument('--unstacked_datafile', default= 'unstacked_data.csv' , type= str)
    data.add_argument('--stacking_years', type= int, default= 5, choices=[5,10])
    data.add_argument('--load_stacked_data', type= str)
    data.add_argument('--save_stacked_data', nargs='?', const= 'stacked.npz' ,type= str)


    modelparameters.add_argument("--epochs", default= 1, type= int) # default for not given vals 
    modelparameters.add_argument("--batch_size", default= 1, type= int )
    modelparameters.add_argument('--lr', default=.01, type= float)
    modelparameters.add_argument('--num_layers', default= 2, type= int )
    modelparameters.add_argument('--hidden_size', default= 32, type= int)


    model.add_argument('--early_stopping_patience', nargs='?', const= 10 , type= int)
    model.add_argument('--early_stopping_gap', nargs='?', const= .01, type= float )
    model.add_argument('--save_model', nargs='?', const= 'lstm_model.pt') # None or value given if not called, if called then you have to specify filename 
    model.add_argument('--load_model', type=str)

    model.add_argument('--plot_loss', nargs='?', const= 'lossplot')



    # parser.print_help()
    args_= parser.parse_args()

    if args_.pathway== 'command_file': 
      args= parser.parse_args(args= ['command_line'] + given_commands) 
    else: 
      args= args_


    print(args)

    return args 



  def stack_data(self)-> tuple[np.array]: 
    x, y= DatasetStacking(self.args).get_stack()

    print(x.shape, y.shape)
    return x,y


  def train_model(self): 
    model= ModelHelper(self.x, self.y, self.args)
    
    if self.args.plot_loss: 
      Plotting(self.args).make_plot(model.curr_state)

    pass



    



if __name__== '__main__': 
  # print(sys.argv)
  Driver()



