import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import TensorDataset
from imblearn.over_sampling import SMOTE

def evaluate_prediction(true_list, pred_list):
    '''
    This code evaluates sensitivity and specificity for class 1, a minor class.
    Recall = Sensitivity
    Specificity for class 1 is Sensitivity for class 0.
    '''
    sensitivity = np.sum((true_list*pred_list)==1)/np.sum(true_list==1)
    specificity = np.sum(((1-true_list)*(1-pred_list))==1)/np.sum(true_list==0)
    accuracy = np.mean(true_list == pred_list)
    average_accuracy = (sensitivity + specificity)/2.
    return sensitivity, specificity, accuracy, average_accuracy

def preprocessing_data(dt, tr_mean=0.1307, tr_std=0.3081):
	X = dt.data.cpu().numpy().astype(np.float)/255.
	X = (X-tr_mean)/tr_std
	Y = dt.targets.cpu().numpy().astype(np.float)
	Y = (Y==0)
	return X.astype(np.float32), Y.astype(np.int64)

def load_data_SMOTE(model, split, N_tr=60000, N_sample=200):
    sm = SMOTE(random_state=42, n_jobs=12)
    if split=='train':
        np.random.seed(1005)
        dt = datasets.MNIST('../data', train=True, download=True, transform=None)
        X_tr, Y_tr = preprocessing_data(dt)
        rnd_idx = np.random.choice(np.arange(N_tr), size=N_sample, replace=False)
        X_tr, Y_tr = X_tr[rnd_idx], Y_tr[rnd_idx]
        X_tr, Y_tr = sm.fit_resample(X_tr.reshape(-1,784), Y_tr)
        X_tr = X_tr.reshape(-1,1,28,28)
        return TensorDataset(torch.from_numpy(X_tr) , torch.from_numpy(Y_tr))
        
    elif split=='test':
        dt = datasets.MNIST('../data', train=False, transform=None)
        X_te, Y_te = preprocessing_data(dt)
        X_te = X_te[:,np.newaxis,:,:]
        return TensorDataset(torch.from_numpy(X_te) , torch.from_numpy(Y_te))
        
    else:
        assert False, 'Model should be one of [train,test].'

    return dataset  

def load_data(model, split, N_tr=60000, N_sample=200):
	if split=='train':
		dt = datasets.MNIST('../data', train=True, download=True, transform=None)
		X_tr, Y_tr = preprocessing_data(dt)
		np.random.seed(1005)
		rnd_idx = np.random.choice(np.arange(N_tr), size=N_sample, replace=False)
		X_tr, Y_tr = X_tr[rnd_idx], Y_tr[rnd_idx]
		X_tr = X_tr[:,np.newaxis,:,:]
		return TensorDataset(torch.from_numpy(X_tr) , torch.from_numpy(Y_tr))

	elif split=='test':
		dt = datasets.MNIST('../data', train=False, transform=None)
		X_te, Y_te = preprocessing_data(dt)
		X_te = X_te[:,np.newaxis,:,:]
		return TensorDataset(torch.from_numpy(X_te) , torch.from_numpy(Y_te))
		
	else:
		assert False, 'Model should be one of [train,test].'

	return dataset	



