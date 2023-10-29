# Copyright (C) 2023, Jorge S. S. Junior and Jerome Mendes
# <jorge.silveira@isr.uc.pt>,<jermendes@gmail.com>
# -
# The "NFN-LSTM" comes with ABSOLUTELY NO WARRANTY;
# In case of publication of any application of this method, cite the work:
# -
# Jorge S. S. Junior, Jerome Mendes, Francisco Souza, and Cristiano Premebida
# Hybrid LSTM-Fuzzy System to Model a Sulfur Recovery Unit.
# In Proc. of the 20th International Conference on Informatics in Control,
# Automation and Robotics (ICINCO 2023), pages XX-XX, Rome, Italy,
# November 13th-15th, 2023. DOI: http://doi.org/XX.XXXX/XXXXX.XXXX.XXXXXX
# -
# Available at https://www.jeromemendes.com/

#%% Import libraries and manual functions
import torch
from Functions.nfn_lstm_aux import load_n_normalize,Antecedent
from Functions.nfn_lstm_learning import NFN_LSTM,training,comparison

torch.set_default_dtype(torch.float64) # Set all tensors to be float64 (double)

#%% Load Dataset and Normalize (Standard Scaler)
NormData = load_n_normalize()

#%% Parameters
# Fixed:
input_dim = NormData['dataTotal_x'].shape[1] # input dimension
bias_model = torch.mean(NormData['dataTotal_y'], dim=0).item() # y0

# Changeable:
mf_dim = 5 # number of membership functions per input variable
layer_dim = 1 # LSTM's layer dimension
steps = 1 # time steps for forecasting
hidden_dim = (input_dim*mf_dim) # LSTM's hidden dimension

#%% Obtain Antecedent parameters (before training)
FuzzyVar = Antecedent(NormData,mf_dim)

#%% Set model
model = NFN_LSTM(input_dim=input_dim,
					 mf_dim=mf_dim,
					 hidden_dim=hidden_dim,
					 layer_dim=layer_dim,
					 steps=steps,
					 bias_model=bias_model
					 )

#%% Train model and evaluate
model = training(model,NormData,FuzzyVar)
comparison(model,NormData,FuzzyVar)

#%% end of code