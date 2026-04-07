class EarlyStopping: 
  '''
  This checks if the model should continue to train or if early stopping should be done 
  - patience = how many iterations to wait until stopping 
  - gap = min improvement gap needed to classify en epoch as an improvement epoch 
  '''
  def __init__(self, args ): 

    self.patience, self.gap = args.early_stopping_patience, args.early_stopping_gap 
    self.no_improvement= 0 
    self.lowest_loss= float('inf')

  def early_stop(self, curr_loss) -> bool : # train loss 
  
    if self.lowest_loss - self.gap > curr_loss: # curr loss is best loss 
      self.lowest_loss= curr_loss
      self.no_improvement =0 

    else: # curr loss is less than best loss 
      self.no_improvement += 1 
      if self.patience <= self.no_improvement: 
        return True 

    return False 
    
