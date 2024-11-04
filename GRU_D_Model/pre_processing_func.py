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
    Input: DF (Spell, Time_step, Vitals signs....), cols: List of features,  number of time steps required
    Returns: X: (Batch, Time_step, Features), Y: (Batch, time_step), episodes (array of episode_id)
    Feature order: Anchor_time, Vitals (According list:[cols])
    '''
    # Batch, seq_len, features
    episodes = df['episode_id'].unique()
    
    X = np.zeros((len(episodes), time_steps_req, len(cols)))

    Y = -np.ones((len(episodes), time_steps_req))

    id = 0
    for epi in episodes:
        data = (df[df['episode_id'] == epi]).reset_index() # Reset so you can directly index
        length = len(data)
        # If spell has less than required, add available data to beginning of array
        #Using Loc you have to directly index rather than reverse slice
        if length < time_steps_req:
            X[id,:length,:] = data.loc[0:length, cols].to_numpy()
            Y[id,:length] = data.loc[0:length, 'outcome_in_24hrs'].to_numpy()
            id += 1
        # If spell has more than required, last x time_steps of spell goes to data
        else:
            X[id,:,:] = data.loc[length-time_steps_req:length, cols].to_numpy()
            Y[id,:] = data.loc[length-time_steps_req:length, 'outcome_in_24hrs'].to_numpy()
            id += 1
    
    return X, Y, episodes


def train_mask_delta_generator (data):
    
    ''' Normalises and generates masks requried for decay learning
    Learns parameters for normalisation and passes on to val_test generator
    Input: data: np.array (Sample, Time_Step, Feature)
    Output: dataset_agger: Model input(Sample, model input(4)[Vitals, last_obsv, Mask of presence, Delta of last seen], time_step, feature)
    Length: Array seq_len for trimming (sample)
    Timing: Anchor_times for later use if required
    X_mean: Means of values per time stamp (time_step, feature)
    train_means, train_std, max_delta: arrays required for normalisation
    '''

    print('Start to generate Training Mask, Delta, Last_observed_X ...')
    
    # Convert data flowrate data to small number -> If 0 assume no oxygen
    flow = data[:,:,7]
    data[:,:,7][flow == 0] = 0.1
    # Convert zero 'missing' timesteps to nan
    data[data == 0] = np.nan

    # Extract the feature space (i.e. not time)
    X = data[:,:,1:]
    # Where there is no observation - 0 == no Observation, 1 == observation present
    Mask = (X != -1)
    # Training means/std for val_test generator
    train_means = np.nanmean(X, axis=(0, 1))
    train_std = np.nanstd(X, axis=(0, 1))
    # Normalise 0 mean / std
    X = (X - train_means) / train_std

    # Get lags from first index
    # Alter the anchor_time to time_lags
    time_lags = data[:,:,0]
    lags = np.diff(time_lags)
    time_lags[:,1:] = lags
    # Use lags between each observations
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
    
    # Find the lengths of each time series
    lengths =  (~np.isnan(time_lags)).sum(axis = 1)

    # max/min scaled, keeps 0-1
    max_delta =  np.nanmax(Delta) 
    Delta = Delta / max_delta

    # Combine X, X_last_obs, Mask, Delta to combine
    X = np.expand_dims(X, axis=1)
    X_last_obsv = np.expand_dims(X_last_obsv, axis=1)
    Mask = np.expand_dims(Mask, axis=1)
    Delta = np.expand_dims(Delta, axis=1)
    dataset_agger = np.concatenate((X, X_last_obsv, Mask, Delta), axis = 1)

    timings = np.cumsum(time_lags, axis = 1)
    # Means for Decay Function
    X_mean = np.nanmean(X, axis = 0)

    print('Finished Training Set Pre-Processing')

    return dataset_agger, lengths, timings, X_mean, train_means, train_std, max_delta

def val_test_mask_delta_generator (data, train_means, train_std, max_delta):
    
    ''' Normalises and generates masks required for GRU-D Model

    Input: data: np.array (Sample, Time_Step, Feature)
    train_means, train_std, max_delta - parameters from training for normalisation 
    
    Output: dataset_agger: Model input(Sample, model input(4)[Vitals, last_obsv, Mask of presence, Delta of last seen], time_step, feature)
    Length: Array seq_len for trimming (sample)
    Timing: Anchor_times for later use if required

    Model input(Sample, model input(4), time_step, feature)
    Length: Array seq_len for trimming (sample)
    '''

    print('Start to generate Validation/Testing Mask, Delta, Last_observed_X ...')

    # Convert data flowrate data to small number -> If 0 assume no oxygen
    flow = data[:,:,7]
    data[:,:,7][flow == 0] = 0.1
    data[data == 0] = np.nan
    X = data[:,:,1:].round()

    # Where there is no observation 
    Mask = (X != -1)    
    # Normalise data to training ranges
    X = (X - train_means) / train_std
    
    # Get lags from first index
    lags = np.diff(data[:,:,0])
    # Alter the anchor_time to time_lags
    time_lags = data[:,:,0]
    time_lags[:,1:] = lags
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
    
    # Find the lengths of each time series
    lengths =  (~np.isnan(time_lags)).sum(axis = 1)    
    
    # normalize - currently max/min scaled keeps 0-1
    Delta = Delta / max_delta

    # Combine X, X_last_obs, Mask, Delta to combine
    X = np.expand_dims(X, axis=1)
    X_last_obsv = np.expand_dims(X_last_obsv, axis=1)
    Mask = np.expand_dims(Mask, axis=1)
    Delta = np.expand_dims(Delta, axis=1)
    dataset_agger = np.concatenate((X, X_last_obsv, Mask, Delta), axis = 1)

    timings = np.cumsum(time_lags, axis = 1)
    
    print('Finished Validation/Training Pre-Processing')
    
    return dataset_agger , lengths,  timings

def dataset_compiler_gru_shorten (data, labels, lengths, timings, hrs_inclusion_criteria = 24, hrs_trim = 48):
    '''
    Removes instances where there is LoS is < hrs_inclusion_criteria (default = 24hrs)
    Trims the last hrs_trim (default = 48hrs) hours within the episode

    Input: Data array, label array, length array, timings (anchor_time)
    Output: Dataset, seq_timings: list of time of each observation (for meta data in testing)
    '''
    # Exclude instances < 24hrs (1440 mins)
    inclusion_mask = np.nanmax(timings, axis = 1) >= (hrs_inclusion_criteria*60)
    data = data[inclusion_mask]
    labels = labels[inclusion_mask]
    lengths = lengths[inclusion_mask]
    timings = timings[inclusion_mask]

    # Find the time_steps within 48hrs of end of admission (2880 min)
    threshold_48 = np.nanmax(timings, axis = 1) - (hrs_trim*60)
    within_48 = (timings) >= threshold_48[:,np.newaxis]
    # First index within 48hrs
    indices_48 = np.where(within_48.any(axis=1), within_48.argmax(axis=1), -1)
    
    # Holding list to capture incoming tensors
    X = []
    y = []
    seq_timings = []

    # Iteratrates through length array and trims data,label and timings in the time_step dim
    for id in range(len(data)):
        start = int(indices_48 [id])
        end = int(lengths [id])
        X.append(torch.Tensor(data[id,:,start:end,:]))
        y.append(torch.Tensor(labels[id,start:end]))
        seq_timings.append(np.array(timings[id,start:end]))
    
    # Combines list of tensors and labels
    dataset = CustomDataset(X, y)

    return  dataset, seq_timings

def dataset_compiler_gru_full_time(data, labels, lengths, timings, hrs_inclusion_criteria = 24):
    '''
    Removes instances where there is LoS is < hrs_inclusion_criteria (default = 24hrs)
    Use this for the test set to give representive sample of the real world (i.e. throughout the admission)

    Inputs: Data array, label array, length array, timings (anchor_time)
    Returns: Dataset, Inclusion_mask: Indices of dataset that was included in test set (i.e. < 24hrs LoS)
    '''

    # Exclude instances < 24hrs (1440 mins)
    inclusion_mask = np.nanmax(timings, axis = 1) >= (hrs_inclusion_criteria *60)
    data = data[inclusion_mask]
    labels = labels[inclusion_mask]
    lengths = lengths[inclusion_mask]
    timings = timings[inclusion_mask]
    
    # Holding list to capture incoming tensors
    X = []
    y = []
    # Iteratrates through length array and trims data array in the time_step dim
    # id = 0
    for id in range(len(data)):
        end = int(lengths [id])
        X.append(torch.Tensor(data[id,:,:end,:]))
        y.append(torch.Tensor(labels[id,:end]))

    # Combines list of tensors and labels
    dataset = CustomDataset(X, y)

    return dataset, inclusion_mask

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
    
def collate_function(batch):
    '''
    Takes batch and pads to max seq_len within the batch for data and labels
    '''
    inputs, labels = zip(*batch)
    max_length = max(matrix.size(1) for matrix in inputs)
    padded_inputs =  [torch.nn.functional.pad(matrix, (0, 0, 0, max_length - matrix.size(1))) for matrix in inputs]
    inputs = torch.stack(padded_inputs)
    inputs = torch.nan_to_num(inputs, nan = 0.0)

    max_label_length = max(len(label) for label in labels)
    padded_labels = [np.pad(label, (0, max_label_length - len(label)), 'constant') for label in labels]
    stacked_labels = np.stack((padded_labels))
    
    return inputs, torch.Tensor(stacked_labels)

# Sampler that organises instances with the same length and organises into batches
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


def dataloader_compiler (dataset, batch_size = 20):
    sampler = SortByLengthSampler(dataset, batch_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                        collate_fn=collate_function, 
                        sampler=sampler)
    
    return dataloader
