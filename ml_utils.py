# -*- coding: utf-8 -*-

from keras import backend as K
from keras.models import load_model

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import itertools

########################################

# Source: https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d

def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss

##############################################################


##############################################################
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.4f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    
def plot_confusion_matrix_v2(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        cm = cm.astype('int')
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2%' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#################################################################
    
def quantity_results_per_class(cm, classes, decimals=2):
    
    result_dict = dict()
    round_result_dict = dict()
    for i in range(len(classes)):
        aux_dict = dict() #{'TP':0, 'FP':0, 'FN':0}
        aux_dict['TP'] = cm[i][i]
        aux_dict['FP'] = sum([cm[i_real][i] for i_real in range(len(classes)) if i_real != i])
        aux_dict['FN'] = sum([cm[i][i_pred] for i_pred in range(len(classes)) if i != i_pred])
        
        aux_dict['precision'] = (aux_dict['TP'])/(aux_dict['TP']+ aux_dict['FP'])
        aux_dict['recall'] = (aux_dict['TP'])/(aux_dict['TP'] + aux_dict['FN'])
        
        aux_dict['f1-score'] = 2*(aux_dict['precision']*aux_dict['recall'])/(aux_dict['precision']+aux_dict['recall'])
        
        result_dict[classes[i]] = dict(aux_dict)
    
    result_dict['mean'] = dict()
    result_dict['mean']['precision'] = sum([result_dict[classes[i]]['precision'] for i in range(len(classes)) ])/len(classes)
    result_dict['mean']['recall'] = sum([result_dict[classes[i]]['recall'] for i in range(len(classes)) ])/len(classes)
    result_dict['mean']['f1-score'] = sum([result_dict[classes[i]]['f1-score'] for i in range(len(classes)) ])/len(classes)
    
    return result_dict

def macro_and_micro_metrics_per_class(cm, classes, decimals=2):
    
    result_dict = dict()
    round_result_dict = dict()
    
    TP_list = list()
    FP_list = list()
    FN_list = list()
    
    for i in range(len(classes)):
        aux_dict = dict() #{'TP':0, 'FP':0, 'FN':0}
        
        TP = cm[i][i]
        FP = sum([cm[i_real][i] for i_real in range(len(cm)) if i_real != i])
        FN = sum([cm[i][i_pred] for i_pred in range(len(cm)) if i != i_pred])
        TP_list.append(TP)
        FP_list.append(FP)
        FN_list.append(FN)
        
        aux_dict['TP'] = TP
        aux_dict['FP'] = FP
        aux_dict['FN'] = FN
        
        print("TP: {}, FP: {} and FN: {}".format(TP, FP, FN))
        
        aux_dict['precision'] = TP/(TP + FP)
        aux_dict['recall'] = TP/(TP + FN)
        
        aux_dict['f1-score'] = 2*(aux_dict['precision']*aux_dict['recall'])/(aux_dict['precision']+aux_dict['recall'])
        
        print("precision: {}, recall: {} and f1-score: {}".format(aux_dict['precision'], aux_dict['recall'], aux_dict['f1-score']))
        
        result_dict[classes[i]] = dict(aux_dict)
    
    
    
    
    precision_micro = sum(TP_list)/sum(TP_list+FP_list)
    recall_micro = sum(TP_list)/sum(TP_list+FN_list)
    
    result_dict['micro'] = dict()
    result_dict['micro']['precision'] = precision_micro
    result_dict['micro']['recall'] = recall_micro
    result_dict['micro']['f1-score'] = 2*(precision_micro*recall_micro)/(precision_micro+recall_micro)
    
    print("precision_u: {}, recall_u: {} and f1-score_u: {}".format(precision_micro, recall_micro, result_dict['micro']['f1-score']))
    
    result_dict['macro'] = dict()
    result_dict['macro']['precision'] = sum([result_dict[classes[i]]['precision'] for i in range(len(classes)) ])/len(classes)
    result_dict['macro']['recall'] = sum([result_dict[classes[i]]['recall'] for i in range(len(classes)) ])/len(classes)
    result_dict['macro']['f1-score'] = sum([result_dict[classes[i]]['f1-score'] for i in range(len(classes)) ])/len(classes)
    
    return result_dict
                
def store_research_set_info(researchset_path):
    
    target_weights = [1,1,1] # only for load model correctly
    weighted_loss_function = weighted_categorical_crossentropy(target_weights)
    research_set_table = pd.read_csv(researchset_path + '/' + researchset_path + '_research_summary.csv')
    
    
    configuration_labels = list(research_set_table.columns.values)
    train_labels = ['n_epochs','train_acc','train_loss','test_acc','test_loss']
    complexity_labels = ['n_params']
    # results_statistics_labels is elaborated in for
    
    result_list = list()
    
    for model_i in range(len(research_set_table)):
        
        model_id = model_i + 1
        print('===================================')
        print('model #' + str(model_id))
        print('===================================')
        ex_model_info = np.load(researchset_path + '/' + researchset_path +  '_model_id_' + str(model_id) + '_info.npy')
        ex_model_info = ex_model_info[()]
        additional_data = ex_model_info['optional_data']
            
        train_history = ex_model_info['train_history'] 
        
        eval_results = ex_model_info['eval_results']
        
        # TODO: TESTING OF extract loss and accuracy from last epoch
        
        last_train_accuracy,last_train_loss,last_accuracy, last_loss, n_epochs = \
        train_history['acc'][-1], train_history['loss'][-1], train_history['val_acc'][-1], train_history['val_loss'][-1], len(train_history['loss'])
        train_values = [n_epochs,last_train_accuracy,last_train_loss,last_accuracy, last_loss]
        print(train_labels)
        print(train_values)
        
        
        # TODO: extract best loss and accuracy attached
        
        model_path = researchset_path + '/' + researchset_path + '_model_id_' + str(model_id)  + '_fullmodel.h5'
        ex_model = load_model(model_path, custom_objects={'loss': weighted_loss_function})
        complexity_values = [ex_model.count_params()]
        print(complexity_labels)
        print(complexity_values)

        configuration_values = research_set_table.iloc[[model_i]].values.tolist()[0]
        print(configuration_labels)
        print(configuration_values)
        
        classes = ['BKG', 'ALERT', 'FALL']
        eval_statistics =  quantity_results_per_class(eval_results, classes)

        results_statistics_labels = list()
        results_statistics_values = list()
        for cl in eval_statistics.keys():
            metric_labels = eval_statistics[cl].keys()
            for met in metric_labels:
                results_statistics_values.append(eval_statistics[cl][met])
                results_statistics_labels.append(cl + '_' + met)
        print(results_statistics_labels)
        print(results_statistics_values)
        
        model_results = configuration_values + train_values + complexity_values + results_statistics_values
        result_list.append(model_results) 
        
    results_dataframe = pd.DataFrame(result_list, columns=(configuration_labels + train_labels + complexity_labels + results_statistics_labels))
    resumed_results = results_dataframe[(configuration_labels + train_labels + complexity_labels + ['BKG_f1-score','ALERT_f1-score','FALL_f1-score','mean_precision','mean_recall','mean_f1-score'])]
    
    resumed_results.to_csv(researchset_path + '/' + researchset_path + '_resumed_table.csv', index=False)
    results_dataframe.to_csv(researchset_path + '/' + researchset_path + '_table.csv', index=False)
    
    return results_dataframe, resumed_results
    
        
        
        
    
    