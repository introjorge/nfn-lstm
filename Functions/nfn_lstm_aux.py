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
# Available at https://bit.ly/nfn-lstm, https://www.jeromemendes.com/

#%% Import libraries
import torch
import scipy.io as sp
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as mse

#%%
def load_n_normalize():
	'''
	Load the dataset for estimating hydrogen sulfide (H2S) of a sulfur recovery
	unit (SRU), and apply data standardization.
	
	Returns
	-------
	NormData : {dictionary}
		Stores input and output datasets for train, test and validation, and
		individual scalers 'sx' and 'sy' for input and output.
	'''
	LoadData = sp.loadmat('Datasets/bench_sru_h2s_nd.mat')
	
	# Assign full dataset for Input (X) and Output (Y)
	dataTotal_x = torch.from_numpy(LoadData['X'])
	dataTotal_y = torch.from_numpy(LoadData['Y'])
	# Assign train dataset for Input and Output
	data_xtrain = torch.from_numpy(LoadData['X_train'])
	data_ytrain = torch.from_numpy(LoadData['Y_train'])
	# Assign test dataset for Input and Output
	data_xtest = torch.from_numpy(LoadData['X_test'])
	data_ytest = torch.from_numpy(LoadData['Y_test'])
	# Assign validation dataset for Input and Output
	data_xval = torch.from_numpy(LoadData['X_val'])
	data_yval = torch.from_numpy(LoadData['Y_val'])
	
	# Initialize and Fit StandardScaler for X and Y (mean is 0, variance is 1)
	sx = StandardScaler(); sx.fit(dataTotal_x)
	sy = StandardScaler(); sy.fit(dataTotal_y)
	
	# Standardize all assigned datasets
	dataTotal_x=torch.Tensor(sx.transform(dataTotal_x))
	dataTotal_y=torch.Tensor(sy.transform(dataTotal_y))
	data_xtrain=torch.Tensor(sx.transform(data_xtrain))
	data_ytrain=torch.Tensor(sy.transform(data_ytrain))
	data_xtest=torch.Tensor(sx.transform(data_xtest))
	data_ytest=torch.Tensor(sy.transform(data_ytest))
	data_xval = torch.Tensor(sx.transform(data_xval))
	data_yval = torch.Tensor(sy.transform(data_yval))
	
	# Create a dictionary (structure) with standardized dataset and scalers
	NormData = {
		'dataTotal_x':dataTotal_x,
		'data_xtrain':data_xtrain,
		'data_xval':data_xval,
		'data_xtest':data_xtest,
		'dataTotal_y':dataTotal_y,
		'data_ytrain':data_ytrain,
		'data_yval':data_yval,
		'data_ytest':data_ytest,
		'sx':sx,
		'sy':sy
		}
	
	return NormData

#%% 
def set_tmf_param(NormData,per_extra,mf_dim):
	'''
	Obtain parameters 'a','b' and 'c' of triangular membership functions for
	each input.
	
	Parameters
	----------
	NormData : {dictionary}
		Stores input and output datasets for train, test and validation, and
		individual scalers 'sx' and 'sy' for input and output.
	per_extra: {float}
		Percentage used to increase the limits of the universe of discourse of
		the inputs
	mf_dim : {integer}
		Number of membership functions for each input.
	
	Returns
	-------
	Tparam : {tensor (mf_dim,input_dim,3)}
		Parameters 'a','b' and 'c' for each input.
	'''
	dataTotal_x = NormData['dataTotal_x'] # input, full dataset
	input_dim = dataTotal_x.shape[1] # number of inputs
	
	# Initialize triangular membership functions' parameters for each input
	Tparam = torch.zeros(mf_dim,input_dim,3)
	
	for j in range(input_dim): # For each input:
		# Set minimum and maximum limits of input variable
		lim_min = dataTotal_x[:,j].min().item()
		lim_max = dataTotal_x[:,j].max().item()
		var_lim = [
			lim_min - per_extra*abs(lim_min),
			lim_max + per_extra*abs(lim_max)
			]
		# Define parameters
		for i in range(mf_dim):
			# Define ranges for variables
			ranges = (var_lim[1]-var_lim[0])/(mf_dim-1)
			if i == 0: # First membership function
				Tparam[i,j,0] = var_lim[0] # parameter a_{j,i}
				Tparam[i,j,1] = var_lim[0] # parameter b_{j,i}
				Tparam[i,j,2] = var_lim[0] + ranges # parameter c_{j,i}
			elif i == mf_dim-1: # Last membership function
				Tparam[i,j,0] = var_lim[1] - ranges # a_{j,i}
				Tparam[i,j,1] = var_lim[1] # b_{j,i}
				Tparam[i,j,2] = var_lim[1] # c_{j,i}
			else: # Mid membership function
				Tparam[i,j,0] = Tparam[i-1,j,1] # a_{j,i}
				Tparam[i,j,1] = Tparam[i-1,j,1] + ranges # b_{j,i}
				Tparam[i,j,2] = Tparam[i,j,1] + ranges # c_{j,i}
	
	return Tparam

#%%
def TriangularMF(data,Tparam):
	'''
	Obtain the values of triangular membership functions (antecedent parameters)
	for each input, considering univariate fuzzy rules.
	
	Parameters
	----------
	data : {tensor (samples,input_dim)}
		Input data used to obtain the membership functions.
	Tparam : {tensor (mf_dim,input_dim,3)}
		Parameters 'a','b' and 'c' for each input.
	
	Returns
	-------
	mf : {tensor(mf_dim,input_dim,batch_dim)}
		Membership functions for each sample of each input.
	norm_mf : {tensor(mf_dim,input_dim,batch_dim)}
		Normalized membership functions for each sample of each input.
	
	'''
	batch_dim,input_dim = data.shape # 'batch_dim' samples, 'input_dim' inputs
	mf_dim = Tparam.shape[0] # number of membership functions for each input
	
	# Initialize membership functions
	mf = torch.zeros(mf_dim,input_dim,batch_dim)
	norm_mf = torch.zeros(mf_dim,input_dim,batch_dim) # Normalized version
	
	for j in range(input_dim): # For each input
		for i in range(mf_dim): # For each membership function
			ai,bi,ci = Tparam[i,j,:] # Parameters 'a', 'b', 'c'
			# Left Slope
			if ai != bi:
				idx = \
					torch.nonzero(torch.logical_and(ai<data[:,j],data[:,j]<bi))
				mf[i,j,idx] = (data[idx,j] - ai) / (bi-ai)
			# Right Slope
			if bi != ci:
				idx = \
					torch.nonzero(torch.logical_and(bi<data[:,j],data[:,j]<ci))
				mf[i,j,idx] = (ci - data[idx,j]) / (ci-bi)
			# Center
			idx = torch.nonzero(data[:,j] == bi)
			mf[i,j,idx] = 1
	
	# Activation - Normalize the membership functions
	for i in range(mf_dim):
		norm_mf[i,:,:] = torch.div(mf[i,:,:],(mf.sum(dim=0))+1e-20)
	
	return [mf,norm_mf]

#%%
def extend_as_rules(mf):
	'''
	Extend membership functions for fuzzy rule processing.
	
	Parameters
	----------
	mf : {tensor(mf_dim,input_dim,batch_dim)}
		Membership functions for each sample of each input.
	
	Returns
	-------
	mf_ext {tensor(mf_dim,input_dim*batch_dim)}
	'''
	mf_dim,input_dim,batch_dim = mf.shape # Dimensions of membership functions
	# Swap axes before reshaping
	mf_ext = torch.swapaxes(mf,0,2) # (batch_dim,input_dim,mf_dim)
	# Extend membership functions suitable for fuzzy rule processing
	mf_ext = torch.reshape(mf_ext,(batch_dim,input_dim*mf_dim))
	
	return mf_ext

#%%
def membership_fcn(data,Tparam):
	'''
	Consolidated computation for obtaining membership functions and their
	normalized versions.
	
	Parameters
	----------
	data : {tensor (samples,input_dim)}
		Input data used to obtain the membership functions.
	Tparam : {tensor (mf_dim,input_dim,3)}
		Parameters 'a','b' and 'c' for each input.
	
	Returns
	-------
	mf : {tensor(mf_dim,input_dim,batch_dim)}
		Membership functions for each sample of each input.
	norm_mf : {tensor(mf_dim,input_dim,batch_dim)}
		Normalized membership functions for each sample of each input.
	'''
	# Compute triangular membership functions and their normalized versions
	[mf,norm_mf] = TriangularMF(data,Tparam)
	# Extend membership functions for rule processing
	mf = extend_as_rules(mf)
	norm_mf = extend_as_rules(norm_mf)
	
	return [mf,norm_mf]


#%%
def Antecedent(NormData,mf_dim):
	'''
	Generate antecedent parameters using triangular membership functions.
	
	Parameters
	----------
	NormData : {dictionary}
		Stores input and output datasets for train, test and validation, and
		individual scalers 'sx' and 'sy' for input and output.
	mf_dim : {integer}
		Number of membership functions for each input.
	
	Returns
	-------
	FuzzyVar : {dictionary}
		Stores the antecedent parameters.
	'''
	per_extra = 0.05 # percentage for the limits of the universe of discourse
	
	# Generate triangular membership function parameters for the fuzzy rules
	Tparam = set_tmf_param(NormData,per_extra,mf_dim)
	
	# Compute membership functions for train, test and validation
	[_,norm_mf_train] = membership_fcn(NormData['data_xtrain'],Tparam)
	[_,norm_mf_test] = membership_fcn(NormData['data_xtest'],Tparam)
	[_,norm_mf_val] = membership_fcn(NormData['data_xval'],Tparam)
	
	# Dictionary to store the fuzzy antecedents
	FuzzyVar = {
		'Tparam':Tparam,
		'norm_mf_train':norm_mf_train,
		'norm_mf_test':norm_mf_test,
		'norm_mf_val':norm_mf_val,
		}
	
	return FuzzyVar

#%%
def nrmse(y_real,y_est):
	'''
	Compute the Normalized Root Mean Squared (NRMSE).between real and estimated
	outputs.
	
	Parameters
	----------
	y_real : {tensor (batch_dim,output_dim)}
		Values of real output.
	y_est : {tensor (batch_dim,output_dim)}
		Values of estimated output.
	
	Returns
	-------
	norm_rmse : {float}
		Value of the NRMSE between real and estimated outputs.
	
	'''
	# Compute the Root Mean Squared Error (RMSE)
	rmse = mse(y_real, y_est, squared=False)
	# Normalize RMSE
	norm_rmse = rmse/(max(y_real)-min(y_real))
	
	return norm_rmse

#%% end of code
