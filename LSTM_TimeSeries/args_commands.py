given_commands= [
  '--unstacked_datafile=unstacked_data.csv',
  '--stacking_years=5', 
  '--load_stacked_data=stacked_5.npz', 
  # '--save_stacked_data=stacked_5.npz', 
  '--batch_size=256',
  # '--load_stacked_data=stacked.npz', 
  # '--save_stacked_data=stacked.npz', 

  

  '--load_model=saved_model3/lstm_modelC2B.pt',
  '--save_model=saved_model3/lstm_modelC2B2.pt', 


  '--lr=.01', 
  '--epochs=100', 
  '--hidden_size=400', 
  '--num_layers=1',

  '--early_stopping_patience= 5 ', 
  '--early_stopping_gap=.002 ', 

  '--plot_loss=saved_model3/lossplotC2B2.png'
  


  ]

