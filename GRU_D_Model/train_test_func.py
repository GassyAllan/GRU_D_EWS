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

from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report

def Train_Binary_Model(model, train_dataloader, valid_dataloader, num_epochs = 300, patience = 10, min_delta = 0.00001, learning_rate = 0.00001):
    
    print('Model Structure: ', model)
    print('Start Training ... ')
    
    if (type(model) == nn.modules.container.Sequential):
        output_last = model[-1].output_last
        print('Output type dermined by the last layer')
    else:
        output_last = model.output_last
        print('Output type dermined by the model')
        
    loss_MSE = torch.nn.BCELoss()    
    
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
            
            loss_train = loss_MSE(torch.squeeze(outputs), torch.squeeze(labels))
        
            losses_train.append(loss_train.data)
            losses_epoch_train.append(loss_train.data)
            
            optimizer.zero_grad()
            
            loss_train.backward()
            
            optimizer.step()
            
             # validation 
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
            loss_valid = loss_MSE(torch.squeeze(outputs_val), torch.squeeze(labels_val))
            
            losses_valid.append(loss_valid.data)
            losses_epoch_valid.append(loss_valid.data)
            
            # output
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

def Test_Binary_Model(model, test_dataloader, title, classes = ['Survived', 'Died']):
    
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
            loss_BSE = torch.nn.BCELoss()

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

    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(ground_truth, pred)
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
    print(f'AUROC: {auc(fpr, tpr):.4f}')
    print(f'AUPRC: {auc(rec, precis):.4f}')

    ground_truth = torch.stack(ground_truth)
    prob_out = torch.stack(prob_out).squeeze()

    # AUROC Plot
    plt.plot(fpr, tpr, marker='o',linestyle='-', label=title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for '+ title)
    plt.legend()
    plt.show()

    #AUPRC Plot
    plt.plot(rec, precis, marker='o',linestyle='-', label=title)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PRC for '+ title)
    plt.legend()
    plt.show()

    return prob_out, np.array(lengths), pred, ground_truth

def plot_confusion_matrix(cm, labels, normalize= False, title="Confusion Matrix"):
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title,fontsize=20)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    #plt.xticks(tick_marks, labels, rotation=45)
    #plt.yticks(tick_marks, labels)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def CV_Test_Binary_Model(model, test_dataloader):
    
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
    ground_truth = []
    pred = []

    print("Testing Loop")
   
    model.eval()
    with torch.no_grad():    
        for data in test_dataloader:
            inputs, labels = data
            ground_truth.extend(labels)
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
    
    print('Calculating Matrix')
    # Consolidate lists
    ground_truth = [item for sublist in ground_truth for item in sublist]
    prob_out = [item for sublist in prob_out for item in sublist]
    pred = [item for sublist in pred for item in sublist]

    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(ground_truth, pred)
    # Calculate the percentage of each class in the confusion matrix

    acc = np.diag(conf_matrix).sum()/conf_matrix.sum()
    precision=conf_matrix[1,1]/(conf_matrix[1,1]+conf_matrix[0,1])
    spec = conf_matrix[0,0]/(conf_matrix[0,0]+conf_matrix[0,1])
    recall=conf_matrix[1,1]/(conf_matrix[1,1]+conf_matrix[1,0])
    bal_acc = (spec+recall)/2 

    print(f'Accuracy :: {acc:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Specificity: {spec:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'Balanced Accuracy: {bal_acc:.4f}')


    print('Plotting ROC Curves')
    # Calcuations for AUC plots
    # ROC and PRC thresholds
    fpr, tpr, _ = roc_curve(ground_truth, prob_out)
    precis, rec, _ = precision_recall_curve(ground_truth,prob_out)
    auroc = auc(fpr, tpr)
    auprc = auc(rec, precis)
    print(f'AUROC: {auroc:.4f}')
    print(f'AUPRC: {auprc:.4f}')


    return acc, precision, spec, recall, bal_acc, auroc, auprc



def Short_Testing(model, test_dataloader):
    ''' Function to extract the probability/prediction from only the first part of admission - Relies on inverse dataset (i.e. data from > 24hrs ending)
    Inputs: model: Trained Model, test_dataloader: Data from Dataset > 24hrs from end (assumes batch size = 1 to keep track)
    outputs: Prob out: Array probability outputs,
            lengths: Length of each instance
            Pred: rounded predictions for each time step (0/1)
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
    # ground_truth = []
    pred = []

    print("Short Testing Loop")
   
    model.eval()
    with torch.no_grad():    
        for data in test_dataloader:
            inputs, labels = data
            # ground_truth.extend(labels)
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


    prob_out = [item for sublist in prob_out for item in sublist]
    pred = [item for sublist in pred for item in sublist]

    return prob_out, np.array(lengths), pred

def seq_level_metrics (model, test_dataloader, labels, mask, tolerance = 5):
    '''
    Outputs a binary classification CFM based on your tolerance
    Inputs: Predictions - Flattened sequence of predictions from dataloader
            Test_lengths - Array of lengths
            Labels - Test Labels of all in the test set (batch, seq_len)
            Mask - outputed from dataset
            Tolerance - Number of warning required
    Outputs: Indices of labels which were predicted death
    '''
    _ , test_lengths , predictions = Short_Testing(model, test_dataloader)
    ground_truth = (labels[mask]).max(axis = 1)
    # Selects the overall labels based on indices
    predictions = np.array(predictions).squeeze()
    seq_pred = np.zeros(len(ground_truth))
    indices = np.cumsum(test_lengths)
    indices = np.pad(indices, (1, 0), mode='constant')  
    # Takes sequences and finds whether the sequence is death or not
    for id in range(len(seq_pred)):
        start = indices[id]
        end = indices[id + 1]
        if (predictions[start:end]).sum() >= tolerance:   # Number of Warnings that are needed to trigger a death warning
            seq_pred[id] = 1

    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(ground_truth, seq_pred)

    acc = np.diag(conf_matrix).sum()/conf_matrix.sum()
    precision=conf_matrix[1,1]/(conf_matrix[1,1]+conf_matrix[0,1])
    spec = conf_matrix[0,0]/(conf_matrix[0,0]+conf_matrix[0,1])
    recall=conf_matrix[1,1]/(conf_matrix[1,1]+conf_matrix[1,0])
    bal_acc = (spec+recall)/2 

    print(f'Seq Level Accuracy : {acc:.4f}')
    print(f'Seq Level Precision: {precision:.4f}')
    print(f'Seq Level Specificity: {spec:.4f}')
    print(f'Seq Level Recall: {recall:.4f}')
    print(f'Seq Level Balanced Accuracy: {bal_acc:.4f}')

    
    return acc, precision, spec, recall, bal_acc,
