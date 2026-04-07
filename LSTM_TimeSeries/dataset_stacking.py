
import pandas as pd
pd.set_option('display.max_columns', None ) 
import numpy as np
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=1 )
np.set_printoptions(edgeitems=10)
np.set_printoptions(linewidth=500)

'''
It stacks the data and splits into appropriate x and y arrays. 
'''

class DatasetStacking:  
  def __init__(self, args): 

    self.args= args 
 

  def get_stack(self) -> tuple[np.array]: 
    '''
    This is the only function which has to be called to get
    x and y arrays. 
    '''

    print('Args are:', self.args)

    if self.args.load_stacked_data: # if the x,y are already given, load them in and return 
      data= np.load(self.args.load_stacked_data, allow_pickle=True)
      x,y= data[data.files[0]], data[data.files[1]] # call with x_data and y_data keys

      print('x,y loaded in from', self.args.load_stacked_data)
      return x,y 




    data= pd.read_csv(self.args.unstacked_datafile) 

    stacked= self.stack (data)
    x,y = self.process(stacked)

    if self.args.save_stacked_data: 
      np.savez(self.args.save_stacked_data, x_data=x, y_data= y ) # savez, not save, makes .npz with keys x_data and y_data
      print(self.args.save_stacked_data+ ' for x_data and y_data data are saved. ')

    return x,y 




  def stack(self, data) -> pd.DataFrame: 
    years= self.args.stacking_years

    # sort by year, ascending 
    data= data.sort_values(by=['year'])


    # group by ccodes, sort value preserved within group 
    data_groups= data.groupby(by=['country1', 'country2'])
    print('There are: ', len(data_groups.groups ), ' groups')

    stacked= []
    c=0 
    for group, idx in data_groups.groups.items(): 

      # get the stacks of 6 instances for 5 year stack 

      for i in range(len(idx)-years): 
        stacked.append(data.loc[idx[i:i+years+1]]) # use loc, not iloc 


      c+=1 

      if c%1000 ==0: 
        print(c , ' groups completed')
        # break #### 
    



    stacked= np.array(stacked) 
    print('shape', stacked.shape )

    return stacked 






  def process(self, data) -> tuple[np.array] :
    '''
    To process the stacked dataset, drop the unscaled year, country1 name, and country2 name.
    Then, split to seperate the trade data (target feature)
    Then, take only the 6th year trade info as this is what is going to be predicted 
    '''    

    data= np.delete(data, [ -2, -3,-4], axis= 2 ) # del og years, ccode1, ccode2 features 
    x,y = np.split(data, [-1], axis= 2) # split from trade column  
    x= x[:,:-1,:] # keep first years set of dca data (eg 5 years)
    y= y[:,-1,:] # keep last year of trade data (eg 6th year)

    print('split shapes:', x.shape, y.shape)

    return x , y 
    









