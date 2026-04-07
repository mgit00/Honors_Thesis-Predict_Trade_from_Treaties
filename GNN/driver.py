import sys 
import argparse

from command_file import commands

if __name__=='__main__': 
  print(sys.argv)

  
parser= argparse.ArgumentParser()
subparser= parser.add_subparsers(dest= 'command_source', title= 'command file commands')
# command_line = parser.add_subparsers(dest= 'command_line', title= 'command line commands')

command_file= subparser.add_parser('command_file')
command_line= subparser.add_parser('command_line') # add arguments 


#### del 
command_line.add_argument('--mini', action=argparse.BooleanOptionalAction)
####----

# command_line.add_argument('--process_data', action= argparse.BooleanOptionalAction, required=True)
# command_line.add_argument('--data_filename', required=False, type= str) 
command_line.add_argument('--unprocessed_data', const='/content/Honors-Thesis-Extension--Predicting-Trade-from-Treaties/Data/pre_binary_encoding.csv', nargs= '?') 
command_line.add_argument('--save_graph', nargs='?', const='graph.pt')
command_line.add_argument('--train_test_val_split', default = {'train': .7, 'val': .0, 'test': .3})

command_line.add_argument('--preload_graph', nargs= '?', const= 'graph.pt', help='passes over DatasetCreation.make_graph()')

# model 
command_line.add_argument('--epochs', default= 1, type= int)
command_line.add_argument('--hidden_size', type= int, default= 16 ) 
command_line.add_argument('--lr', default= .01, type= float )

command_line.add_argument('--save_model', nargs='?', const= 'gnn_model.pt') # None or value given if not called, if called then you have to specify filename 
command_line.add_argument('--load_model', type=str)



args = parser.parse_args()

if args.command_source=='command_file': 
  args= parser.parse_args(args= ['command_line'] + commands)
else: 
  pass #keep args 

print(args)


'''
Main driver code. 
'''
from dataset_creation import DatasetCreation


data= DatasetCreation(args)
data_graph= data.make_data()
# network= data.make_network(data_graph)


from modelhelper import ModelHelper

mh= ModelHelper(args)
curr_model= mh.train_model(data_graph)







