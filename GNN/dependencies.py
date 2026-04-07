'''
pip install -r requirements.txt
'''

import importlib



import pandas as pd
pd.set_option('display.max_columns', None )
from pandas._config import options
import seaborn as sns
import numpy as np
np.set_printoptions(suppress= True )
np.set_printoptions(precision=2 )
np.set_printoptions(edgeitems=100)
np.set_printoptions(linewidth=500)


import matplotlib.pyplot as plt 


from sklearn.utils import shuffle


import torch
from torch_geometric import loader
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, to_undirected
torch.set_printoptions(profile="full")
torch.set_printoptions(precision=3, sci_mode=False)
# print(x) # prints the whole tensor
# torch.set_printoptions(profile="default") # reset
# print(x) # prints the truncated tensor
# torch.set_printoptions(threshold=10_000)

import networkx as nx
