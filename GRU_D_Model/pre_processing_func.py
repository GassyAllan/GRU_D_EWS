import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import pandas as pd

def train_val_test_split(data, labels, train_prop = 0.7, val_prop = 0.1, test_val = 0.2):
    
    assert train_prop + val_prop + test_val == 1, 'Proportions must = 1'

    sample_size = data.shape[0]
    index = np.arange(sample_size, dtype = int)
    np.random.seed(1024) # If crossval - This would need to change
    np.random.shuffle(index)
    
    data = data[index]
    
    labels = labels[index]
    
    # Get indices of when training/validation/test split    
    train_index = int(np.floor(sample_size * train_prop))
    valid_index = int(np.floor(sample_size * ( train_prop + val_prop)))
    
    train_data, train_label = data[:train_index], labels[:train_index] 
    valid_data, valid_label = data[train_index:valid_index], labels[train_index:valid_index]
    test_data, test_label = data[valid_index:], labels[valid_index:]

    return train_data, train_label, valid_data, valid_label, test_data, test_label , index[valid_index:]

def df_to_np_pipe (df,cols, time_steps_req):
    '''
    Takes df with vital signs and returns an np.array of data and array of episode order
    Input: DF (Spell, Time_step, Vitals signs....), number of time steps required
    Returns: np.array (Batch, Time_step, Features)
    Feature order: Time_lag, RR, SBP, DBP, HR, Temp, Spo2
    '''
    # Batch, seq_len, features
    episodes = df['episode_id'].unique()
    
    X = np.zeros((len(episodes), time_steps_req, len(cols)))

    id = 0
    for epi in episodes:
        data = (df[df['episode_id'] == epi]).reset_index() # Reset so you can directly index
        length = len(data)
        # If spell has less than required, add available data to beginning of array
        #Using Loc you have to directly index rather than reverse slice
        if length < time_steps_req:
            X[id,:length,:] = data.loc[0:length, cols].to_numpy()
            id += 1
        # If spell has more than required, last x time_steps of spell goes to data
        else:
            X[id,:,:] = data.loc[length-time_steps_req:length, cols].to_numpy()
            id += 1
    
    return X, episodes


def train_mask_delta_generator (train_data, train_label):
    ''' Normalises and generates masks requried for decay learning
    Learns parameters for normalisation and passes on to val_test generator
    Takes data array (Sample, Time_Step, Feature)
    returns 
    Model input(Sample, model input(4), time_step, feature)
    Length: Array seq_len for trimming (sample)
    X_mean: Means of values per time stamp (time_step, feature)
    train_means, train_std, max_delta: arrays required for normalisation
    '''
    
    print('Start to generate Mask, Delta, Last_observed_X ...')
    
    
    # Convert zero 'missing' timesteps to nan
    train_data[train_data == 0] = np.nan

    # Extract the feature space
    X = Xt[:,:,1:]
    # Where there is no observation - 0 == no Observation, 1 == observation present
    Mask = (X != -1)
    # Get training means to use for val_test generator
    train_means = np.nanmean(X, axis=(0, 1))
    train_std = np.nanstd(X, axis=(0, 1))
    # Normalise 0 mean / std
    X = (X - train_means) / train_std

    # Time_lags is S vector in paper -> Contains time lags of all examples (N x Time_steps)
    # Get lags from first index
    time_lags = Xt[:,:,0]
    # Find the lengths of each time series
    lengths =  (~np.isnan(time_lags)).sum(axis = 1)
    time_lags[:,0] = 0

    Delta = np.repeat(time_lags, X.shape[2], axis=1) # Like np.tile
    Delta = np.reshape(Delta, X.shape) # Reshape into data matrix shape
    
    X_last_obsv = np.copy(X)
    # Get the idx access
    missing_index = np.where(Mask == 0)
    # I: batch, j: time_step, k: feature
    for idx in range(missing_index[0].shape[0]):
        # Selects where there is a missing according to mask
        i = missing_index[0][idx] 
        j = missing_index[1][idx]
        k = missing_index[2][idx]
            # If previous not missing then delta = current time lag last observed + previous
        if j != 0 and j != (X.shape[1]-1): # This logic is to avoid first and last! need to alter j needs to be time step
            Delta[i,j+1,k] = Delta[i,j+1,k] + Delta[i,j,k]
        if j != 0:
            X_last_obsv[i,j,k] = X_last_obsv[i,j-1,k] # last observation
    # normalize Not sure this is required as S is not normalised - currently max/min scaled, keeps 0-1
    max_delta =  np.nanmax(Delta) 
    Delta = Delta / max_delta

    X = np.expand_dims(X, axis=1)
    X_last_obsv = np.expand_dims(X_last_obsv, axis=1)
    Mask = np.expand_dims(Mask, axis=1)
    Delta = np.expand_dims(Delta, axis=1)
    dataset_agger = np.concatenate((X, X_last_obsv, Mask, Delta), axis = 1)
    X_mean = np.nanmean(X, axis = 0)
    print('Finished')
    return dataset_agger, train_y, lengths, X_mean, train_means, train_std, max_delta

def val_test_mask_delta_generator (data, train_means, train_std, max_delta):
    ''' Normalises and generates masks required for decay learning
    Learns parameters for normalisation and passes on to val_test generator
    Takes: data array (Sample, Time_Step, Feature), 
    train_means, train_std, max_delta - parameters from training for normalisation 
    Returns 
    Model input(Sample, model input(4), time_step, feature)
    Length: Array seq_len for trimming (sample)
    '''
    print('Start to generate Mask, Delta, Last_observed_X ...')

    data[data == 0] = np.nan
    X = data[:,:,1:].round()

    # Where there is no observation 
    Mask = (X != -1)    
    # Normalise data to training ranges
    X = (X - train_means) / train_std
    time_lags = data[:,:,0]
    # Find the lengths of each time series
    lengths =  (~np.isnan(time_lags)).sum(axis = 1)
    time_lags[:,0] = 0

    # # Time_lags is S vector in paper -> Contains time lags of all examples (N x Time_steps)
    # time_lags = np.zeros((data.shape[0],data.shape[1]))
    # # Get lags from first index
    # time_lags[:,1:] = data[:,1:,0]

    Delta = np.repeat(time_lags, X.shape[2], axis=1) # Like np.tile
    Delta = np.reshape(Delta, X.shape) # Reshape into data matrix shape
    
    X_last_obsv = np.copy(X)
    # Get the idx access
    missing_index = np.where(Mask == 0)
    # I: batch, j: time_step, k: feature
    for idx in range(missing_index[0].shape[0]):
        # Selects where there is a missing according to mask
        i = missing_index[0][idx] 
        j = missing_index[1][idx]
        k = missing_index[2][idx]
            # If previous not missing then delta = current time lag last observed + previous
        if j != 0 and j != (X.shape[1]-1): # This logic is to avoid first and last! need to alter j needs to be time step
            Delta[i,j+1,k] = Delta[i,j+1,k] + Delta[i,j,k]
        if j != 0:
            X_last_obsv[i,j,k] = X_last_obsv[i,j-1,k] # last observation
    # normalize - currently max/min scaled keeps 0-1
    Delta = Delta / max_delta

    X = np.expand_dims(X, axis=1)
    X_last_obsv = np.expand_dims(X_last_obsv, axis=1)
    Mask = np.expand_dims(Mask, axis=1)
    Delta = np.expand_dims(Delta, axis=1)
    dataset_agger = np.concatenate((X, X_last_obsv, Mask, Delta), axis = 1)

    print('Finished')
    
    return dataset_agger , lengths


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Return a tuple (sample, label) based on the index
        sample = self.data[index]
        label = self.labels[index]
        return sample, label
    
def dataset_compiler (data, labels, lengths, length_required = 5):
    '''
    Removes instances where there are not enough observations
    Trims each instance based on the lengths from the mask_generator

    Takes: Data array, label array, length array, int: number of observations required
    Returns: Dataset
    '''
    # Removes short instances
    data = data[lengths > length_required]
    labels = labels[lengths > length_required]
    # Holding list to capture incoming tensors
    holding = []
    # Iteratrates through length array and trims data array in the time_step dim
    id = 0
    for i in lengths[lengths > length_required]:
        holding.append(torch.Tensor(data[id,:,:int(i),:]))
        id += 1

    # Combines list of tensors and labels
    dataset = CustomDataset(holding, labels)

    return dataset




def collate_fn(batch):
    '''Takes batch and pads to max seq_len within the batch
    '''
    inputs, labels = zip(*batch)
    max_length = max(matrix.size(1) for matrix in inputs)
    padded_inputs =  [torch.nn.functional.pad(matrix, (0, 0, 0, max_length - matrix.size(1))) for matrix in inputs]
    inputs = torch.stack(padded_inputs)
    inputs = torch.nan_to_num(inputs, nan = 0.0)
    stacked_labels = np.stack((labels))
    return inputs, torch.Tensor(stacked_labels)

# Sammpler that organises instances with the same length and organises into batches
# It then shuffles the batches for the dataloader so there are variable lengths delievered to the network


class SortByLengthSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.indices = list(range(len(dataset)))
        # Sort indices based on the maximum length of time steps within each matrix
        self.indices.sort(key=lambda x: self._get_max_length(self.dataset[x][0]))


    def __iter__(self):
        batches = [self.indices[i:i + self.batch_size] for i in range(0, len(self.indices), self.batch_size)]
        random_permutation = torch.randperm(len(batches)).tolist()
        shuffled_indices = [index for batch in random_permutation for index in batches[batch]]
        return iter(shuffled_indices)
    
    def __len__(self):
        return len(self.dataset)

    def _get_max_length(self, matrix):
        return matrix.size(1)

class WeightedSortByLengthSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, batch_size, class_weights):
        self.dataset = dataset
        self.batch_size = batch_size
        self.class_weights = class_weights
        self.indices = list(range(len(dataset)))
        # Sort indices based on the maximum length of time steps within each matrix
        self.indices.sort(key=lambda x: self._get_max_length(self.dataset[x][0]))

    def __iter__(self):
        batches = [self.indices[i:i + self.batch_size] for i in range(0, len(self.indices), self.batch_size)]
        # Calculate probabilities for each sample based on class weights
        probabilities = torch.tensor([self.class_weights[self.dataset[idx][1]] for idx in self.indices])
        probabilities /= probabilities.sum()
        # Sample indices based on weighted probabilities
        sampled_indices = torch.multinomial(probabilities, len(probabilities), replacement=True)
        sampled_indices = sampled_indices.tolist()
        # Shuffle the sampled indices
        random_permutation = torch.randperm(len(batches)).tolist()
        shuffled_indices = [index for batch in random_permutation for index in sampled_indices[batch]]
        return iter(shuffled_indices)
    
    def __len__(self):
        return len(self.dataset)

    def _get_max_length(self, matrix):
        return matrix.size(1)
    


def dataloader_compiler (dataset, batch_size = 20):
    sampler = SortByLengthSampler(dataset, batch_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                        collate_fn=collate_fn, 
                        sampler=sampler)
    
    return dataloader
