import numpy as np
import torch
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
import torch.nn as nn
import time
from scipy.stats import mannwhitneyu


from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report

def Train_Model(model, train_dataloader, valid_dataloader, num_epochs = 300, patience = 10, min_delta = 0.00001, learning_rate = 0.00001):
    '''
    Training Loop for GRU-D
    Inputs: Model: Taken from GRUD.py
    train_dataloader, valid_dataloader: Data
    Training Parameters: num_epochs (Max epochs) patience (epochs for validation increasing), min_delta (best loss)
    Output: Trained model; Training Statistics (Training loss, Validation loss)
    '''
    print('Model Structure: ', model)
    print('Start Training ... ')
    
    if (type(model) == nn.modules.container.Sequential):
        output_last = model[-1].output_last
        print('Output type dermined by the last layer')
    else:
        output_last = model.output_last
        print('Output type dermined by the model')
        
    loss_BCE = torch.nn.BCELoss()    
    
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    use_gpu = torch.cuda.is_available()
    if use_gpu == False:
        device = 'cpu'

    losses_train = []
    losses_valid = []
    losses_epochs_train = []
    losses_epochs_valid = []
    
    cur_time = time.time()
    pre_time = time.time()
    
    # Variables for Early Stopping
    is_best_model = 0
    patient_epoch = 0
    for epoch in range(num_epochs):
        
        trained_number = 0
        valid_dataloader_iter = iter(valid_dataloader)   
        losses_epoch_train = []
        losses_epoch_valid = []
        
        for data in train_dataloader:
            inputs, labels = data    
            if use_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
            else: 
                inputs, labels = inputs.to(device), labels.to(device)
            
            model.zero_grad()
            outputs = model(inputs)            
            loss_train = loss_BCE(torch.squeeze(outputs), torch.squeeze(labels))          
            losses_train.append(loss_train.data)
            losses_epoch_train.append(loss_train.data)          
            optimizer.zero_grad()          
            loss_train.backward()           
            optimizer.step()
            
             # Validation Loop
            try: 
                inputs_val, labels_val = next(valid_dataloader_iter)
            except StopIteration:
                valid_dataloader_iter = iter(valid_dataloader)
                inputs_val, labels_val = next(valid_dataloader_iter)
            
            if use_gpu:
                inputs_val, labels_val = inputs_val.cuda(), labels_val.cuda()
            else: 
                inputs_val, labels_val = inputs_val.to(device), labels_val.to(device)
            
            model.zero_grad()           
            outputs_val = model(inputs_val)        
            loss_valid = loss_BCE(torch.squeeze(outputs_val), torch.squeeze(labels_val))
            losses_valid.append(loss_valid.data)
            losses_epoch_valid.append(loss_valid.data)
            trained_number += 1
            
        avg_losses_epoch_train = sum(losses_epoch_train).cpu().numpy() / float(len(losses_epoch_train))
        avg_losses_epoch_valid = sum(losses_epoch_valid).cpu().numpy() / float(len(losses_epoch_valid))
        losses_epochs_train.append(avg_losses_epoch_train)
        losses_epochs_valid.append(avg_losses_epoch_valid)
        
        # Early Stopping
        if epoch == 0:
            is_best_model = 1
            best_model = model
            min_loss_epoch_valid = 10000.0
            if avg_losses_epoch_valid < min_loss_epoch_valid:
                min_loss_epoch_valid = avg_losses_epoch_valid
        else:
            if min_loss_epoch_valid - avg_losses_epoch_valid > min_delta:
                is_best_model = 1
                best_model = model
                min_loss_epoch_valid = avg_losses_epoch_valid 
                patient_epoch = 0
            else:
                is_best_model = 0
                patient_epoch += 1
                if patient_epoch >= patience:
                    print('Early Stopped at Epoch:', epoch)
                    break
        
        # Print training parameters
        cur_time = time.time()
        print('Epoch: {}, train_loss: {}, valid_loss: {}, time: {}, best model: {}'.format( \
                    epoch, \
                    np.around(avg_losses_epoch_train, decimals=8),\
                    np.around(avg_losses_epoch_valid, decimals=8),\
                    np.around([cur_time - pre_time] , decimals=2),\
                    is_best_model) )
        pre_time = cur_time
                
    return best_model, [losses_train, losses_valid, losses_epochs_train, losses_epochs_valid]

def Test_Model(model, test_dataloader, title, classes = ['Survived', 'Died']):
    '''
    Testing Loop for GRU-D that plots CFM and AUROC,AUPRC using func "plot_confusion_matrix"
    Inputs: Trained Model
    test_dataloader: Data
    title: String For plots titles
    Output: prob_out: Sequence of probability output (as one)
    pred: Sequence of Predictions (Rounded of probability)
    ground_truth: Sequence of Labels
    lengths: Length of each sequence
    '''    

    if (type(model) == nn.modules.container.Sequential):
        output_last = model[-1].output_last
    else:
        output_last = model.output_last
    
    inputs, labels = next(iter(test_dataloader))
    batch_size= inputs.size(0)

    cur_time = time.time()
    pre_time = time.time()
    
    use_gpu = torch.cuda.is_available()
    if use_gpu == False:
        device = 'cpu'
    
    tested_batch = 0
    
    prob_out = []
    lengths = []
    ground_truth = []
    pred = []

    print("Testing Loop")
   
    model.eval()
    with torch.no_grad():    
        for data in test_dataloader:
            inputs, labels = data
            ground_truth.extend(labels)
            lengths.append(labels.size(1))
            if inputs.shape[0] != batch_size:
                continue
        
            if use_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
            else: 
                inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            prob_out.extend(outputs)

            if output_last:
                predictions = torch.round(outputs)
                pred.extend(predictions)
          
                
            else:
                predictions = torch.round(outputs)
                pred.extend(predictions)
            
            tested_batch += 1
        
            if tested_batch % 1000 == 0:
                cur_time = time.time()
                print('Tested #: {}, time: {}'.format( \
                    tested_batch * batch_size, \

                    np.around([cur_time - pre_time], decimals=8) ) )
                pre_time = cur_time
    
    print('Plotting Matrix')
    # Consolidate lists
    ground_truth = [item for sublist in ground_truth for item in sublist]
    prob_out = [item for sublist in prob_out for item in sublist]
    pred = [item for sublist in pred for item in sublist]
    
    ground_truth = torch.stack(ground_truth)
    prob_out = torch.stack(prob_out).squeeze()
    pred = torch.stack(pred).squeeze()
    
    # Calculate the confusion matrix
    conf_matrix = confusion_matrix((ground_truth), (pred))
    # Calculate the percentage of each class in the confusion matrix
    conf_matrix_percent = conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis]

    plt.figure()
    sns.heatmap(conf_matrix_percent, annot=conf_matrix, fmt='d', cmap='Blues', vmin=0, vmax=1)
    plt.xlabel('Predicted Label',fontsize=20)
    plt.ylabel('True Label',fontsize=20)    
    plt.yticks(np.arange(len(classes)) + 0.5, classes , rotation=45,fontsize=10)
    plt.xticks(np.arange(len(classes)) + 0.5, classes ,fontsize=10)
    plt.title(title ,fontsize=20)
    plt.show()

    acc = np.diag(conf_matrix).sum()/conf_matrix.sum()
    precision=conf_matrix[1,1]/(conf_matrix[1,1]+conf_matrix[0,1])
    spec = conf_matrix[0,0]/(conf_matrix[0,0]+conf_matrix[0,1])
    recall=conf_matrix[1,1]/(conf_matrix[1,1]+conf_matrix[1,0])
    bal_acc = (spec+recall)/2 

    print(f'Accuracy : {acc:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Specificity: {spec:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'Balanced Accuracy: {bal_acc:.4f}')

    print('Plotting ROC Curves')
    # Calcuations for AUC plots
    # ROC and PRC thresholds
    fpr, tpr, _ = roc_curve(ground_truth, prob_out)
    precis, rec, _ = precision_recall_curve(ground_truth,prob_out)
    
    # AUROC Plot
    plt.plot(fpr, tpr, marker='o',linestyle='-', label=title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUROC for '+ title)
    plt.legend()
    plt.show()
    print(f'AUROC: {auc(fpr, tpr):.4f}')

    #AUPRC Plot
    plt.plot(rec, precis, marker='o',linestyle='-', label=title)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('AUPRC for '+ title)
    plt.legend()
    plt.show()

    print(f'AUPRC: {auc(rec, precis):.4f}')

    return prob_out, pred, ground_truth, np.array(lengths)

def plot_confusion_matrix(cm, labels, normalize= False, title="Confusion Matrix"):
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title,fontsize=20)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def ews_time_point_results (df, test_episodes, EWS, long_or_short, hrs_to_trim = 24):
    '''
    Wrapper function that extracts EWS sequence with ground truth labels
    Uses Functions: EWS_seq_extraction_long/short to extract labels and EWS_metrics_time_point to plot
    Input: df: vital signs/EWS, test_episodes: Array of episode_id in test_set, EWS: string of EWS, long_or_short: string of eval type, hrs_to_trim: no. hours from end
    Output: CFM, AUROC, AUPRC plots and tuple of (Specificity, Recall, Balanced Accuracy, AUROC, AUPRC)
    '''
    assert long_or_short in {'long', 'short'}, 'Select Type of Eval: "long", "short"'
    assert EWS in {'mews','news2','ecart'}, 'Select valid EWS: "news2", "mews", "ecart"'

    if long_or_short == 'long':
        labels, ews_seq = EWS_seq_extraction_long(df, test_episodes, EWS)
    elif long_or_short == 'short':
        labels, ews_seq, _ , _ , _ = EWS_seq_extraction_short(df, test_episodes, EWS, hrs_to_trim)
    
    return EWS_metrics_time_point(labels, ews_seq, EWS , long_or_short)

def EWS_seq_extraction_long (df, test_episodes, EWS):
    '''
    Extracts all time points within test set
    Output: Tuple of 2x array: ground_truth , EWS Sequence
    '''
    # Selects episodes within Test set
    data = df[df['episode_id'].isin(test_episodes)]
    labels = data['outcome_in_24hrs']
    ews_seq = data[EWS]

    return labels, ews_seq


def EWS_metrics_time_point (labels, ews_seq, EWS , long_or_short = 'long', classes = ['Survived', 'Died']):
    '''
    Takes sequences of labels and EWS scores and applies threshold 
    Output: Plots of CFM, AUROC, AUPRC plots and tuple of (Specificity, Recall, Balanced Accuracy, AUROC, AUPRC)
    '''
    # Logic for Formatting
    if long_or_short == 'long':
        title = 'All Time Points '
    elif long_or_short == 'short':
        title = 'Last 24hrs '
    
    # Logic for thresholds of EWS
    if EWS == 'mews':
        threshold = 5
    elif EWS == 'news2':
        threshold = 7
    elif EWS == 'ecart':
        threshold = 20    

    # AUC/AUPRC Metrics
    # Iterate through the sequence of possible EWS scores for TPR/FPR/Precision
    TPR = []
    FPR = []
    prec_thres =[]
    max_score = int(ews_seq.max())
    for i in range(max_score):
        conf_matrix = confusion_matrix(labels , ews_seq >=i)
        recall=conf_matrix[1,1]/(conf_matrix[1,1]+conf_matrix[1,0])
        spec = conf_matrix[0,0]/(conf_matrix[0,0]+conf_matrix[0,1])
        precision=conf_matrix[1,1]/(conf_matrix[1,1]+conf_matrix[0,1])
        TPR.append(recall)
        prec_thres.append(precision)
        FPR.append(1-spec)
    
    auroc = auc(FPR, TPR)
    auprc = auc(TPR, prec_thres)

    # Metrics according to specificed threshold
    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(labels, ews_seq >= threshold)
    # Calculate the percentage of each class in the confusion matrix
    conf_matrix_percent = conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis]

    plt.figure()
    sns.heatmap(conf_matrix_percent, annot=conf_matrix, fmt='d', cmap='Blues', vmin=0, vmax=1)
    plt.xlabel('Predicted Label',fontsize=20)
    plt.ylabel('True Label',fontsize=20)    
    plt.yticks(np.arange(len(classes)) + 0.5, classes , rotation=45,fontsize=10)
    plt.xticks(np.arange(len(classes)) + 0.5, classes ,rotation=45,fontsize=10)
    plt.title(title + 'Confusion Matrix for ' + EWS ,fontsize=20)
    plt.show()

    acc = np.diag(conf_matrix).sum()/conf_matrix.sum()
    spec = conf_matrix[0,0]/(conf_matrix[0,0]+conf_matrix[0,1])
    recall=conf_matrix[1,1]/(conf_matrix[1,1]+conf_matrix[1,0])
    bal_acc = (spec+recall)/2 

    print(f'Accuracy : {acc:.4f}')
    print(f'Specificity: {spec:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'Balanced Accuracy: {bal_acc:.4f}')

    # AUROC Plot
    plt.plot(FPR, TPR, marker='o',linestyle='-', label=EWS)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title + 'AUROC for '+ EWS)
    plt.legend()
    plt.ylim(-0.01,1.01)
    plt.xlim(-0.01,1.01)
    plt.show()
    print(f'AUROC: {auroc:.4f}')

    #AUPRC Plot
    plt.plot(TPR, prec_thres, marker='o',linestyle='-', label=EWS)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title + 'AUPRC for '+ EWS)
    plt.ylim(-0.01,1.01)
    plt.xlim(-0.01,1.01)
    plt.legend()
    plt.show()
    print(f'AUPRC: {auprc:.4f}')

    return spec, recall, bal_acc, auroc, auprc


def gru_seq_extractor (last_24_length, ground_truth, prob_out, pred):
    short_ground_truth = []
    short_probilities = []
    short_predictions = []
    idx = 0
    for length in last_24_length:
        start = idx
        end = idx + length
        short_ground_truth.append(ground_truth[start:end])
        short_probilities.append(prob_out[start:end])
        short_predictions.append(pred[start:end])
        idx += length

    return short_ground_truth, short_probilities, short_predictions

def gru_seq_metric_deviation (ground_truth, prob_out):
    '''
    Evalutates metrics at a Sequence level
    Takes list of variable lengths arranaged in [Instance, Time_step]
    Outputs Time_Point Metrics in Tuple 
    Input: Ground_truth, Probability_output, Prediction (0/1)
    '''
    # Derive probability sequences in case you needed AUROC/AUPRC
    seq_prob = np.zeros(len(prob_out))
    for i in range(seq_prob.shape[0]):
        # If empty -> make it zero prediction
        if len(prob_out[i]) == 0:
            seq_prob[i]= 0
        # Take the max as this will be the breach of threshold
        else:
            seq_prob[i] = prob_out[i].max()

    # Derive the sequence level - Differs from the ground truth in shortened sequence as some may not have any time point
    seq_ground_truth = np.zeros(len(ground_truth))
    id= 0
    # each sequence take the max label which will be 0/1
    for seq in ground_truth:
        seq_ground_truth[id] = seq.max()
        id+= 1

    # Accuracy metrics -> Round threshold (0.5)
    conf_matrix = confusion_matrix(seq_ground_truth, np.round(seq_prob) )
    spec = conf_matrix[0,0]/(conf_matrix[0,0]+conf_matrix[0,1])
    recall=conf_matrix[1,1]/(conf_matrix[1,1]+conf_matrix[1,0])
    bal_acc = (spec+recall)/2 
    
  
    # Extract the sequences where there was a correct prediction
    indices = np.where((seq_ground_truth == np.round(seq_prob)) & (seq_ground_truth ==1))[0]

    print('Sequence Level Prediction for GRU-D')
    print(f'Specificity: {spec:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'Balanced Accuracy: {bal_acc:.4f}')

    return ( spec, recall, bal_acc), indices

def first_warning_timing_extraction (list_seq, list_timings, indices):
    first_warnings = np.zeros(indices.shape[0])
    id = 0
    for i in indices:
        timing_idx = int(find_first_occurrence(list_seq[int(i)], 1))
        first_warnings[id] = (list_timings[int(i)][timing_idx])
        id += 1

    return first_warnings

def find_first_occurrence(arr, value):
    '''
    Finds the index of event prediction
    Within the list/array, find the indices
    Return the index of the first occurance
    If no positive prediction -> returns NaN
    '''
    result = np.where(arr == value)[0]
    if result.size > 0:
        return int(result[0])
    else:
        return np.nan  # Return -1 if the no value is not found

def time_to_event_transform (timings):
    time_to_event = []
    for seq in timings:
        time_to_event.append(seq.max() - seq)

    return time_to_event


def gru_seq_metrics (prob_out, seq_lengths, pred, labels, short_timings, episodes, test_idx, indice_mask):
    '''
    Wrapper for gru
    '''

    short_ground_truth, short_probilities, short_predictions = gru_seq_extractor (seq_lengths, labels, prob_out, pred)
    seq_results, correct_idx = gru_seq_metric_deviation (short_ground_truth, short_probilities)
    
    short_timings = time_to_event_transform(short_timings)
    first_warnings = first_warning_timing_extraction(short_predictions, short_timings , correct_idx)

    correct_episodes = episodes[test_idx[indice_mask][correct_idx]]

    return seq_results, (first_warnings, correct_episodes)


def ews_24_extractor (seq_lengths, ground_truth_seq, ews_score_seq, time_seq):
    '''
    Converts as long sequences into the list of variable length sequences according to length
    Inputs: Seq_length (Array of lengths of each sequence), ground_truth_seq (raw sequence of labels), ews_score_seq (raw sequence of EWS Scores)
    outpus: Tuple (list of variable length sequences (labels, EWS Scores))
    '''
    label_sequences = []
    ews_sequences = []
    timing_sequences = []

    idx = 0
    for length in seq_lengths:
        start = idx
        end = idx + length
        label_sequences.append(ground_truth_seq[start:end])
        ews_sequences.append(ews_score_seq[start:end])
        # Need values for indexing later
        timing_sequences.append(time_seq[start:end].values)
        idx += length
    
    return label_sequences, ews_sequences, timing_sequences

def ews_seq_metric_deviation (label_sequences, ews_sequences, EWS):
    '''
    Evalutates metrics at a Sequence level
    Takes list of variable lengths arranaged in [Instance, Time_step]
    Outputs Sequence level metrics in Tuple 
    Input: Ground_truth, Probability_output
    '''

    # Logic for thresholds of EWS
    if EWS == 'mews':
        threshold = 5
    elif EWS == 'news2':
        threshold = 7
    elif EWS == 'ecart':
        threshold = 20    
    
    # Derive probability sequences in case you needed AUROC/AUPRC
    max_score = np.zeros(len(ews_sequences))
    for i in range(max_score.shape[0]):
        # If empty -> make it zero prediction
        if len(ews_sequences[i]) == 0:
            max_score[i]= 0
        # Take the max as this will be the breach of threshold
        else:
            max_score[i] = ews_sequences[i].max()

    # Derive the sequence level - Differs from the ground truth in shortened sequence as some may not have any time point
    seq_ground_truth = np.zeros(len(label_sequences))
    id= 0
    # each sequence take the max label which will be 0/1
    for seq in label_sequences:
        seq_ground_truth[id] = seq.max()
        id+= 1

    # Accuracy metrics -> Round threshold (0.5)
    conf_matrix = confusion_matrix(seq_ground_truth, (max_score>=threshold))
    spec = conf_matrix[0,0]/(conf_matrix[0,0]+conf_matrix[0,1])
    recall=conf_matrix[1,1]/(conf_matrix[1,1]+conf_matrix[1,0])
    bal_acc = (spec+recall)/2 
  
    # Extract the sequences where there was a correct prediction
    indices = np.where((seq_ground_truth == (max_score>=threshold) ) & (seq_ground_truth ==1))[0]
    

    print('Sequence Level Prediction for '+ EWS)
    print(f'Specificity: {spec:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'Balanced Accuracy: {bal_acc:.4f}')

    return ( spec, recall, bal_acc), indices, threshold

def first_warning_timing_extraction_ews (list_ews_scores, threshold, list_timings, indices):
    '''
    Function giving the time (mins) from event where the first warning is given.  Uses func find_first_occurance 
    Inputs: list_seq (list of variable length EWS scores), threshold (int of threshold for EWS), list_timings (list of time of each observation), indices (idx of correct sequences)
    Outpus: Array of time (mins) of first warning
    '''
    
    first_warnings = np.zeros(indices.shape[0])
    
    id = 0
    # Iterate through correctly identified seq in test_set
    for i in indices:
        # Pull each ews sequence and find where threshold breached
        seq_pred = list_ews_scores[i].values >= threshold
        # Find idx where threshold first breached
        timing_idx = find_first_occurrence(seq_pred, 1)

        # Add time to first warning array
        first_warnings[id] = (list_timings[int(i)][timing_idx])
        
        id += 1

    return first_warnings


def EWS_seq_extraction_short (df, test_episodes, EWS, hrs_to_trim = 24):
    '''
    Extracts time points within last Xhrs from each sequence test set
    Output: Tuple of 3x array: ground_truth , EWS Sequence, Sequence Length
    '''
    assert EWS in {'mews','news2','ecart'}, 'Select valid EWS: "news2", "mews", "ecart"'
    # Selects episodes within Test set
    data = df[df['episode_id'].isin(test_episodes)][['episode_id','anchor_time','outcome_in_24hrs', EWS]].reset_index()
    # For each episode find the maximum time and substract window hours (default 24hrs == 1440mins) and store in dict
    epi = (data.groupby('episode_id')['anchor_time'].max()-(hrs_to_trim*60)).index
    val = (data.groupby('episode_id')['anchor_time'].max()-(hrs_to_trim*60)).values
    epi_dict = dict(zip(epi, val))

    # Select all observations within last X hrs of sequence
    data['time_threshold'] = data['episode_id'].map(epi_dict)
    data = data[data['anchor_time'] >= data['time_threshold']].reset_index()

    labels = data['outcome_in_24hrs']
    ews_seq = data[EWS]
    lengths = np.array(data.groupby('episode_id').size())
    timings = data['anchor_time']

    return labels, ews_seq, lengths, np.array(epi), timings

def ews_short_sequence_metrics (df, test_episodes, EWS, hrs_to_trim = 24):
    '''
    Wrapper function to extract the last X hours of each sequence to evaluate the sequence level prediction
    Uses functions: EWS_seq_extraction_short (Extracts the last X hours), ews_24_extractor (compiles sequences into list of variable length sequences)
    ews_seq_metric_deviation (Provide metrics and episode_id of correctly identified sequences with adverse outcomes)
    Inputs: df (vital signs/EWS), test_episodes (array of episodes in test set), EWS: string of EWS type, hrs_to_trim: Xhrs to trim
    Outputs: 2x Tuples: seq_level_results: (Specificity, Recall, Balanced Accuracy), (first warning with episodes)
    '''
    assert EWS in {'mews','news2','ecart'}, 'Select valid EWS: "news2", "mews", "ecart"'
    
    ground_truth_seq, ews_score_seq, seq_lengths, episode_list, time_seq = EWS_seq_extraction_short(df, test_episodes, EWS, hrs_to_trim)

    label_sequences, ews_sequences, timing_sequences = ews_24_extractor (seq_lengths, ground_truth_seq, ews_score_seq, time_seq)
    seq_level_results, correct_idx, threshold = ews_seq_metric_deviation (label_sequences, ews_sequences, EWS)
    
    # Transform anchor_times to time to event
    timing_sequences = time_to_event_transform(timing_sequences)
    first_warnings = first_warning_timing_extraction_ews (ews_sequences, threshold, timing_sequences, correct_idx)
    correct_episodes = episode_list[correct_idx]

    return seq_level_results, (first_warnings, correct_episodes)

def mann_whitney_comparision (warnings, epi_warnings, EWS1, EWS2):
    assert EWS1 in {'MEWS','eCART','NEWS2', 'GRU-D'}, "Select valid EWS: 'MEWS','eCART','NEWS2', 'GRU-D'"
    assert EWS2 in {'MEWS','eCART','NEWS2', 'GRU-D'}, "Select valid EWS: 'MEWS','eCART','NEWS2', 'GRU-D'"

    if EWS1 == 'MEWS':
        first_idx = 0
    elif EWS1 == 'eCART':
        first_idx = 1
    elif EWS1 == 'NEWS2':
        first_idx = 2
    elif EWS1 == 'GRU-D':
        first_idx = 3
    
    if EWS2 == 'MEWS':
        sec_idx = 0
    elif EWS2 == 'eCART':
        sec_idx = 1
    elif EWS2 == 'NEWS2':
        sec_idx = 2
    elif EWS2 == 'GRU-D':
        sec_idx = 3

    first_group = warnings[first_idx][np.isin(epi_warnings[first_idx], epi_warnings[sec_idx])]
    sec_group = warnings[sec_idx][np.isin(epi_warnings[sec_idx], epi_warnings[first_idx])]

    _, p = mannwhitneyu((first_group), (sec_group), method="asymptotic")

    print('Pairwise Mann-Whitney-U Comparision between',EWS1, '&', EWS2, ':', p)

    return p

