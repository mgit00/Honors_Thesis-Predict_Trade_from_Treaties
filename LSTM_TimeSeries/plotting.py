import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 

class Plotting: 
  '''
  This makes and saves a plot which displays the train and test loss over epochs
  '''
  def __init__(self, args): 
    self.args= args 

  def make_plot(self, state): 

    df= pd.DataFrame(data= {'epochs':range(1, state['epoch']+2) , 'train': state['train_loss_data'], 'test': state['test_loss_data']})
    fig, ax = plt.subplots()
    sns.lineplot(ax= ax, data= df, x= 'epochs', y='train', label= 'Train Loss')
    sns.lineplot(ax = ax,  data=df, x= 'epochs', y= 'test', label= 'Test Loss')

    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    fig.get_figure().savefig(self.args.plot_loss)
    print('Figure saved to:', self.args.plot_loss)

