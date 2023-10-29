# Copyright (C) 2023, Jorge S. S. Junior and Jerome Mendes
# <jorge.silveira@isr.uc.pt>,<jermendes@gmail.com>
# -
# The "NFN-LSTM" comes with ABSOLUTELY NO WARRANTY;
# In case of publication of any application of this method, cite the work:
# -
# Jorge S. S. Junior, Jerome Mendes, Francisco Souza, and Cristiano Premebida.
# Hybrid LSTM-Fuzzy System to Model a Sulfur Recovery Unit.
# In Proc. of the 20th International Conference on Informatics in Control, 
# Automation and Robotics (ICINCO 2023), pages XX-XX, Rome, Italy, 
# November 13-15 2023. DOI: http://doi.org/XX.XXXX/XXXXX.XXXX.XXXXXX
# -
# Available at https://www.jeromemendes.com/

#%% Import libraries and manual functions
import time
import torch
import numpy as np
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as MAE
from .nfn_lstm_aux import nrmse
torch.set_default_dtype(torch.float64)

#%%
class NFN_LSTM(torch.nn.Module):
	'''
	Hybridization of a neo-fuzzy neuron (NFN) system with a long short-term
	memory (LSTM) network.
	
	Args:
		input_dim (int): number of inputs.
		mf_dim (int): number of membership functions per input.
		hidden_dim (int): number of hidden units in the LSTM layer.
		layer_dim (int): number of LSTM layers.
		steps (int): number of time steps.
		bias_model (float): bias value for the model.
	'''
	def __init__(self,input_dim,mf_dim,hidden_dim,layer_dim,steps,bias_model):
		super(NFN_LSTM,self).__init__()
		
		# Assign parameters
		self.input_dim = input_dim
		self.mf_dim = mf_dim
		self.hidden_dim = hidden_dim
		self.layer_dim = layer_dim
		self.steps = steps
		self.bias_model = bias_model
		
		# Initialize LSTM
		self.lstm = torch.nn.LSTM(
			input_size=(input_dim*mf_dim),
			hidden_size=self.hidden_dim,
			num_layers=self.layer_dim,
			batch_first=True
			)
		
		# Initialize Consequent Parameters
		self.theta_size = (input_dim*mf_dim)
		self.theta = torch.nn.Parameter(torch.zeros(1,self.theta_size))
		
		# Ensure that hidden_dim goes to theta_size (suitable for fuzzy rules)
		if hidden_dim!=(input_dim*mf_dim):
			self.theta_redux = torch.nn.Linear(
				in_features=hidden_dim,
				out_features=self.theta_size,
				bias=False
				)
		else:
			self.theta_redux = torch.nn.Identity()
	
	# Convert the inputs into a time-series format
	def timeseriesseq(self,inp):
		[K,n] = inp.shape
		new_inp = torch.zeros(K-(self.steps)+1,self.steps,n)
		for i in range(self.steps):
			new_inp[:,i,:] = inp[i:K-(self.steps)+i+1,:]
		return new_inp
	
	# Perform a pass through all time steps in LSTM
	def lstm_pass(self,inp):
		inp = self.timeseriesseq(inp) # change input to timeseries
		batch_dim = inp.shape[0]
		h0 = torch.zeros(self.layer_dim,batch_dim,self.hidden_dim)
		c0 = torch.zeros(self.layer_dim,batch_dim,self.hidden_dim)
		out,(hn,cn) = self.lstm(inp,(h0,c0))
		return out[:,-1,:]
	
	# Calculate the final output based on the individual rules (only consequent)
	def final_output(self,x):
		individual = torch.einsum('ki,ji->ki',x,self.theta)
		output = individual.sum(1) + self.bias_model
		output = output.unsqueeze(1)
		return output
	
	# Computation of the final output with antecedent and consequent parts
	def full_est(self,antecedent):
		lstm_out = self.lstm_pass(antecedent)
		lstm_out = self.theta_redux(lstm_out)
		output = self.final_output(lstm_out)
		return output
	
	# Add a new attribute to the model
	def new_attr(self,name,value):
		setattr(self,name,value)
	
	# ----------------------------
	# Forward pass during training
	def forward(self,ant_train):
		output = self.full_est(ant_train)
		return output
	
	# ----------------------------------------------------
	# Forward pass during evaluation (validation and test)
	def evaluation(self,ant_ev):
		output = self.full_est(ant_ev)
		return output

#%%
def training(model,NormData,FuzzyVar,opt=1):
	'''
	Train the NFN-LSTM model and plot the training and validation losses.
	
	Parameters
	----------
	model: {torch.nn.Module}
		The NFN-LSTM model to be trained.
	NormData : {dictionary}
		Stores input and output datasets for train, test and validation, and
		individual scalers 'sx' and 'sy' for input and output.
	FuzzyVar : {dictionary}
		Stores the antecedent parameters.
	opt : {int}
		Allow (1) or not (0) the plotting of the losses after training.
	
	Returns
	-------
	model: {torch.nn.Module}
		The trained NFN-LSTM model.
	'''
	learning_rate = 1e-1 # Learning rate
	epochs = 100 # Number of epochs
	lr_size = 5 # Period of learning rate decay
	lr_gamma = 0.95 # Multiplicative factor of learning rate decay.
	
	# Loss function and optimizer
	criterion = torch.nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
	
	# Decays the learning rate after every 'lr_size' epochs by 'lr_gamma'
	scheduler = StepLR(optimizer,step_size=lr_size,gamma=lr_gamma)
	
	# Initialize arrays to store training and validation losses
	loss_train = torch.zeros(epochs)
	loss_val = torch.zeros(epochs)
	
	# Limit outputs based on time steps
	out_train = NormData['data_ytrain'][model.steps-1:,:]
	out_val = NormData['data_yval'][model.steps-1:,:]
	
	start_time = time.time() # Start elapsed training time
	
	for ep in range(epochs):
		model.train() # Set model to training mode
		outputs = model(FuzzyVar['norm_mf_train']) # Forward pass
		optimizer.zero_grad() # Zero gradients
		loss = criterion(outputs,out_train) # Compute the loss
		loss_train[ep]=loss.item() # Store the training loss
		loss.backward() # Backpropagation
		optimizer.step() # Update weights
		
		model.eval() # Set model to evaluation mode
		with torch.no_grad():
			out_est_val = model.evaluation(FuzzyVar['norm_mf_val'])
			loss_val[ep] = criterion(out_est_val,out_val).item()
		
		scheduler.step() # Adjust the learning rate every 'lr_size' epochs
	
	end_time = time.time() # End elapsed training time
	
	# Set elapsed time of arrival into model
	model.new_attr('eta',end_time-start_time)
	
	# Plot losses (optional)
	if opt==1:
		plt.figure(figsize=(5,3),dpi=500)
		plt.plot(loss_train,color='blue',label='Training')
		plt.plot(loss_val,color='green',label='Validation')
		plt.title('Loss Function')
		plt.xlabel('Epoch')
		plt.ylabel('MSE')
		plt.legend()
		plt.tight_layout()
		plt.show()
	
	return model

#%%
def comparison(model,NormData,FuzzyVar):
	'''
	Compare the real and estimated outputs using the trained NFN-LSTM model.
	
	Parameters
	----------
	model: {torch.nn.Module}
		The trained NFN-LSTM model.
	NormData : {dictionary}
		Stores input and output datasets for train, test and validation, and
		individual scalers 'sx' and 'sy' for input and output.
	FuzzyVar : {dictionary}
		Stores the antecedent parameters.
	'''
	# Retrieve necessary variables to perform comparison of outputs
	sy = NormData['sy']
	Y_train = NormData['data_ytrain']
	Y_test = NormData['data_ytest']
	
	# Estimated outputs using train and test datasets with the trained model
	Y_est_train = model.evaluation(FuzzyVar['norm_mf_train'])
	Y_est_test = model.evaluation(FuzzyVar['norm_mf_test'])
	
	# Scale the estimated outputs to the original representation
	Y_est_train = torch.Tensor(sy.inverse_transform(Y_est_train.detach()))
	Y_est_test = torch.Tensor(sy.inverse_transform(Y_est_test.detach()))
	
	# Scale the real outputs to the original representation
	Y_train = torch.Tensor(sy.inverse_transform(Y_train[model.steps-1:,:]))
	Y_test = torch.Tensor(sy.inverse_transform(Y_test[model.steps-1:,:]))
	
	# Plot train results
	xmin = 0
	xmax = Y_train.shape[0]
	ymin = min(min(Y_train.squeeze(1)),min(Y_est_train.squeeze(1)))
	ymax = max(max(Y_train.squeeze(1)),max(Y_est_train.squeeze(1)))

	plt.figure(figsize=(6, 4), dpi=1000)
	plt.plot(Y_train,color='black',label='Real')
	plt.plot(Y_est_train,color='red',linestyle='dashed',label='Estimated')
	plt.title('Train')
	plt.xlabel('Sample')
	plt.ylabel('Output')
	plt.legend(loc='upper right')
	plt.tight_layout()
	plt.axis([xmin,xmax,ymin,ymax])
	plt.show()
	
	# Plot test results
	xmin = 0
	xmax = Y_test.shape[0]
	ymin = min(min(Y_test.squeeze(1)),min(Y_est_test.squeeze(1)))
	ymax = max(max(Y_test.squeeze(1)),max(Y_est_test.squeeze(1)))

	plt.figure(figsize=(6, 4), dpi=1000)
	plt.plot(Y_test,color='black',label='Real')
	plt.plot(Y_est_test,color='red',linestyle='dashed',label='Estimated')
	plt.title('Test')
	plt.xlabel('Sample')
	plt.ylabel('Output')
	plt.legend(loc='upper right')
	plt.tight_layout()
	plt.axis([xmin,xmax,ymin,ymax])
	plt.show()
	
	# Error metrics: NRMSE and MAE
	nrmse_train = nrmse(Y_train,Y_est_train).item()
	nrmse_test = nrmse(Y_test,Y_est_test).item()
	mae_train = MAE(Y_train,Y_est_train)
	mae_test = MAE(Y_test,Y_est_test)
	
	# Print elapsed training time and error metrics
	print('Elapsed training time is',np.round(model.eta,4),'seconds.')
	print('Train:')
	print('- NRMSE:',np.round(nrmse_train,4))
	print('- MAE:',np.round(mae_train,4))
	print('Test:')
	print('- NRMSE:',np.round(nrmse_test,4))
	print('- MAE:',np.round(mae_test,4))
	
	return

#%% end of code