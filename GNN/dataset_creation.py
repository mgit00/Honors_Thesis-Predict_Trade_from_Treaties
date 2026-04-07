# import torch 
# from torch_geometric.data import Data 
# from torch_geometric.utils import to_undirected, to_networkx

# import category_encoders as ce 

# import sys 

# import pandas as pd
# pd.set_option('display.max_columns', None )
# from pandas._config import options

# import seaborn as sns

# import numpy as np
# np.set_printoptions(suppress= True )
# np.set_printoptions(precision=2 )
# np.set_printoptions(edgeitems=100)
# np.set_printoptions(linewidth=500)

import pandas as pd 
import numpy as np 


import category_encoders as ce 

from pandas.core import resample
import torch 
from torch_geometric.data import Data 
from torch_geometric.utils import to_undirected, to_networkx

'''
This will create the dataset from torch geometric. 
'''


class DatasetCreation: 
  def __init__(self, args): 
    '''
    If pyg.Data is given as graph_data, then use that. 
    Otherwise, process the args.unprocessed_data file specified and save 
    '''
    self.args= args 
    
  def make_data(self) -> Data: 
    if self.args.preload_graph: 
      try: 
        return torch.load(self.args.preload_graph) 
      except: 
        return torch.load(self.args.preload_graph, weights_only=False)
        
    data= pd.read_csv(self.args.unprocessed_data)
    if self.args.mini: #------ 
      self.args.save_graph= 'mini_'+self.args.save_graph
      # data= data.sample(n=10, ignore_index = True, replace= True)
      data= data.iloc[:1000, :]
      pass



    graph= self.make_graph(data)

    # make train, validation, and test masks
    graph= self.make_masks(graph)
    
    # save data if needed 
    if self.args.save_graph: 
      torch.save(graph, self.args.save_graph)
      print('graph saved to:', self.args.save_graph)


    return graph 



  def make_graph(self, data ) -> Data: 
    '''
    This makes an undirected graph.
    This makes data for the graph, which will be fed into the gnn 
    '''


    # make node (x) data 
    l= data[['ccode1', 'year', 'dcaGeneralV2', 'dcaSectorV2', 'dcaAnyV2',
       'smoothtotrade', 'scaled_year', 'country1']]
    r= data[['ccode2', 'year', 'dcaGeneralV2', 'dcaSectorV2', 'dcaAnyV2',
       'smoothtotrade', 'scaled_year', 'country2']]
    r.columns= l.columns

    x= pd.concat([l,r], axis= 0 ) # vertically stack split dyads 
    x= x[['ccode1', 'year', 'scaled_year', 'country1']] 
    x.drop_duplicates(ignore_index=True, inplace= True) # remove multiple entries of same country and year combo
    x.reset_index(drop= False, inplace=True)
    x.set_index(keys= ['ccode1', 'year'], drop= False, inplace= True ) # for process() below



    # make edge index, attributes, and target (y) data (attributes and y are kept together until instance doubling in to_undirected())
    edgeindex= []
    edgeattributes_y= []

    print('Data length is:', len(data))
    c=0 
    def process(row): 
      '''
      This looks at each dyad row and forms the edge connections, features, and target info
      '''
      edgeindex.append([x.loc[(row.ccode1, row.year), 'index'], x.loc[(row.ccode2, row.year), 'index']])
      edgeattributes_y.append(row.loc['dcaGeneralV2' : 'smoothtotrade'])
      
      nonlocal c
      c+= 1 
      if c%1000==0: 
        print('Current row processed:', c )
      return 

    data.apply(process, axis=1)
    



    # format data into tensors
    x.drop('index', axis=1, inplace=True)
    encoder= ce.BinaryEncoder(cols= ['ccode1'])
    x= encoder.fit_transform(x)
    x= torch.tensor(np.array(x))

    edgeindex= torch.tensor(edgeindex, dtype=torch.long).T 
    edgeattributes_y= torch.tensor(edgeattributes_y)

    
    # make graph undirected
    edgeindex, edgeattributes_y= to_undirected(edge_index= edgeindex, edge_attr= edgeattributes_y) # doubles in size <- make edge idx, weight/attr unto undirected 
    
    # split edgeattributes_y into edgeattributes and y 
    edgeattributes, y=  edgeattributes_y[:, :-1],   edgeattributes_y[:,-1]

    # make Data 
    graph= Data(x=x, edge_index= edgeindex, edge_attr= edgeattributes, y= y)
    print('Graph info:', graph, '\n is validated:', graph.validate(), '\nis directed:', graph.is_directed())

    
    return graph 





  def make_masks(self, graph) -> Data: 
    '''
    This makes the train/validation/test masks for the graph. 
    This is called in make_graph, but can also be seperately called. 
    It can also be saved if needed, just call save_data after changing mask property 
    '''
    length = len(graph.y)
    idx= np.ones(length)

    idx[:    int(np.ceil(self.args.train_test_val_split['val'] * length))   ]= 2 # from the start 
    idx[- int(np.ceil( self.args.train_test_val_split['test']* length))    :]= 3 # from the end 
    
    np.random.seed(1)
    np.random.shuffle(idx)

    graph.train_mask, graph.val_mask, graph.test_mask= idx==1 , idx==2, idx==3 

    graph.train_mask, graph.val_mask, graph.test_mask=  torch.tensor(graph.train_mask), torch.tensor(graph.val_mask), torch.tensor(graph.test_mask)

    return graph 






  def make_network(self, graph): 
    
    return to_networkx(graph)




