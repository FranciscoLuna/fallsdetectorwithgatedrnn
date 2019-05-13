# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.layers import CuDNNLSTM, CuDNNGRU, Dropout
from keras.optimizers import Adam, SGD
import math
import random
import numpy as np
import json as js
import os

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, balanced_accuracy_score

from keras.models import load_model
from keras.utils import to_categorical

from keras.callbacks import Callback, ModelCheckpoint

# Import own methods

import data_generator_modified as dg
import ml_utils as mlu


""" Generation of personalized RNN model """

def RnnMmodel(w = 256, rnn_type = 'lstm', two_rnn_layers=False, drop_coeff_rnn=0.2, drop_coeff_dense=0.5, first_dense=True):
    
    if not (rnn_type == 'lstm' or rnn_type == 'gru'):
        print("rnn_type must be 'lstm' o  'gru'")
        return
    
    rnn_model = Sequential()
    if (first_dense):
        rnn_model.add(Dense(32, batch_input_shape = (None, w, 3)))
        rnn_model.add(BatchNormalization())
    else:
        rnn_model.add(BatchNormalization(batch_input_shape = (None, w, 3)))
    
    rnn_model.add(Dropout(drop_coeff_rnn))
    
    if two_rnn_layers:
        
        if rnn_type == 'lstm':
            rnn_model.add(CuDNNLSTM((32), return_sequences=True))
        elif rnn_type == 'gru':
            rnn_model.add(CuDNNGRU((32), return_sequences=True))
        
        if(drop_coeff_rnn != 0):
            rnn_model.add(Dropout(drop_coeff_rnn))
    
    if rnn_type == 'lstm':
        rnn_model.add(CuDNNLSTM((32)))
    elif rnn_type == 'gru':
        rnn_model.add(CuDNNGRU((32)))
    
    if(drop_coeff_dense !=0):
        rnn_model.add(Dropout(drop_coeff_dense))
    
    rnn_model.add(Dense(3,activation='softmax'))
    
    return rnn_model

def calc_weights(labels):
    
    unique, counts = np.unique(labels, return_counts=True)
    
    dict_counts = dict(zip(unique,counts))
    
    print('samples: {} '.format(dict_counts))
    
    N_bkg = dict_counts[0]
    N_alert = dict_counts[1]
    N_fall = dict_counts[2]
    
    w_bkg = 1
    w_alert = N_bkg / N_alert 
    w_fall = N_bkg / N_fall
    
    return [w_bkg,w_alert,w_fall]

def micro_f1_score(y_true, y_pred):
    
    """
    
    cm = confusion_matrix(y_true, y_pred)
    
    TP_list = list()
    FP_list = list()
    FN_list = list()
    for i in range(len(cm)):
        TP = cm[i][i]
        FP = sum([cm[i_real][i] for i_real in range(len(cm)) if i_real != i])
        FN = sum([cm[i][i_pred] for i_pred in range(len(cm)) if i != i_pred])
        
        TP_list.append(TP)
        FP_list.append(FP)
        FN_list.append(FN)
        
    precision_micro = sum(TP_list)/sum(TP_list+FP_list)
    recall_micro = sum(TP_list)/sum(TP_list+FN_list)
        
    f1_score_micro = 2*(precision_micro*recall_micro)/(precision_micro+recall_micro)
    
    """
    
    return f1_score(y_true, y_pred, average='micro')

def micro_recall(y_true, y_pred):

    return recall_score(y_true, y_pred, average='micro')

def micro_precision(y_true, y_pred):

    return precision_score(y_true, y_pred, average='micro')


def macro_f1_score(y_true, y_pred):
    
    return f1_score(y_true, y_pred, average='macro')

def macro_recall(y_true, y_pred):

    return recall_score(y_true, y_pred, average='macro')

def macro_precision(y_true, y_pred):

    return precision_score(y_true, y_pred, average='macro')

class Metrics(Callback):
        """ Extracted from https://medium.com/@thongonary/how-to-compute-f1-score-for-each-epoch-in-keras-a1acd17715a2"""
        def on_train_begin(self, logs={}):
            self.val_f1s_u = []
            self.val_recalls_u = []
            self.val_precisions_u = []
            self.val_f1s_m = []
            self.val_recalls_m = []
            self.val_precisions_m = []
             
        def on_epoch_end(self, epoch, logs={}):
            val_predict = self.model.predict(self.validation_data[0])
            val_targ = self.validation_data[1]
            # print(val_predict[:10])
            # print(val_targ[:10])
            print(confusion_matrix(np.argmax(val_targ, axis = 1), np.argmax(val_predict, axis = 1)))
            
            to_one_hot = lambda x: [1,0,0] if np.argmax(x) == 0 else ( [0,1,0] if np.argmax(x) == 1 else [0,0,1] )
            
            val_predict = np.apply_along_axis(to_one_hot, 1, np.asarray(val_predict))
            
            # print(val_predict[:10])
            
            
            _val_micro_f1 = micro_f1_score(val_targ, val_predict)
            _val_micro_recall = micro_recall(val_targ, val_predict)
            _val_micro_precision = micro_precision(val_targ, val_predict)
            self.val_f1s_u.append(_val_micro_f1)
            self.val_recalls_u.append(_val_micro_recall)
            self.val_precisions_u.append(_val_micro_precision)
            
            _val_macro_f1 = macro_f1_score(val_targ, val_predict)
            _val_macro_recall = macro_recall(val_targ, val_predict)
            _val_macro_precision = macro_precision(val_targ, val_predict)
            self.val_f1s_m.append(_val_macro_f1)
            self.val_recalls_m.append(_val_macro_recall)
            self.val_precisions_m.append(_val_macro_precision)
            
            logs['val_micro_f1'] = _val_micro_f1
            logs['val_macro_f1'] = _val_macro_f1
            logs['val_balanced_acc'] = _val_macro_recall
            
            print (" — val_micro_f1: %f — val_micro_precision: %f — val_micro_recall %f" %(_val_micro_f1, _val_micro_precision, _val_micro_recall))
            print (" — val_macro_f1: %f — val_macro_precision: %f — val_macro_recall %f" %(_val_macro_f1, _val_macro_precision, _val_macro_recall))
            return

def trainRNNModel(dataTrVal, dataTrLab, dataTestVal, dataTestLab, epochs = 100, lr=0.001, w = 256, stride = 128, batch_size = 32, rnn_type = 'lstm', two_rnn_layers=False, drop_coeff_rnn=0.2, drop_coeff_dense=0.5, first_dense=True, best_model=False, best_model_path="./best_models/"):
    
    import re
    
    own_callbacks = list()
    
    best_model_metrics = ['val_macro_f1','val_micro_f1','val_balanced_acc']
    
    best_model_name_format_f1_macro = "f1-macro_{val_macro_f1:03f}_epoch_{epoch:03d}.hdf5"
    best_model_name_format_f1_micro = "f1-micro_{val_micro_f1:03f}_epoch_{epoch:03d}.hdf5"
    best_model_name_format_balanced_acc = "balanced-acc_{val_balanced_acc:03f}_epoch_{epoch:03d}.hdf5"
    
    
    best_model_name_formats = [best_model_name_format_f1_macro,
                               best_model_name_format_f1_micro,
                               best_model_name_format_balanced_acc] 
    
    
    metrics = Metrics()
    own_callbacks.append(metrics)
    
    dataTrLabelOneHot = to_categorical(dataTrLab)
    dataTestLabelOneHot = to_categorical(dataTestLab)
    
    target_weights = calc_weights(dataTrLab)
    
    model = RnnMmodel(w, rnn_type, two_rnn_layers, drop_coeff_rnn, drop_coeff_dense, first_dense)
    
    model.summary()
    
    # model_json = model.to_json()
    
    if (best_model):
        for i in range(len(best_model_metrics)):
            best_model_route = os.path.join(best_model_path, best_model_name_formats[i])
            checkpoint = ModelCheckpoint(filepath=best_model_route, monitor=best_model_metrics[i], save_best_only=True, mode='max')
            own_callbacks.append(checkpoint)
            
    opt_adam = Adam(lr=lr)
    
    model.compile(optimizer=opt_adam, loss=mlu.weighted_categorical_crossentropy(target_weights), metrics = ['accuracy'])

    model_train_history = model.fit(dataTrVal, dataTrLabelOneHot, 
                                    batch_size=batch_size, epochs=epochs, 
                                    validation_data = (dataTestVal, dataTestLabelOneHot), 
                                    callbacks = own_callbacks)
    
    if(best_model):
        
        results = dict()
        
        list_files = os.listdir(best_model_path)
        best_model_file_name = ""
        
        regexp_metrics = ["f1-macro","f1-micro","balanced-acc"]
        regexp_common_segment = "_[0-9]+\.[0-9]+_epoch_[0-9]{3}\.hdf5"
        
        for metric in regexp_metrics:
            
            pattern = re.compile(metric + regexp_common_segment)
            
            list_files_filtered = list(filter(pattern.match, list_files))
            
            # list_of_epochs = list(map(lambda x : int(x.split(".")[0].split("_")[-1]), list_files_filtered))
            
            list_of_results = list(map(lambda x : float(x.split("_")[1]), list_files_filtered))
            
            best_model_file_name = list_files_filtered[np.argmax(list_of_results)]
            
            if len(best_model_file_name) == 0:
                print("Best model not found")
                return
            
            model_route = os.path.join(best_model_path, best_model_file_name)
            print("loading model with best metric: {}".format())
            model = load_model(model_route, custom_objects={'loss': mlu.weighted_categorical_crossentropy(target_weights)})
    
            model_prediction = model.predict(dataTestVal)
            # print(model_prediction[:10])
            model_confusion_matrix = confusion_matrix(np.argmax(dataTestLabelOneHot, axis = 1), np.argmax(model_prediction, axis = 1))
            # print(model_confusion_matrix)
            
            results[metric] = model, model_train_history, model_confusion_matrix
        
        return results
    
    else:
        model = load_model(model_route, custom_objects={'loss': mlu.weighted_categorical_crossentropy(target_weights)})
    
        model_prediction = model.predict(dataTestVal)
        # print(model_prediction[:10])
        model_confusion_matrix = confusion_matrix(np.argmax(dataTestLabelOneHot, axis = 1), np.argmax(model_prediction, axis = 1))
    
        return model, model_train_history, model_confusion_matrix


def generate_model_params(epochs = 100, rnn_types=['lstm', 'gru'], batch_sizes = [32,64], frequency_reductions=[0], 
                          batch_norm= [False, True], learning_rates = [0.001], first_dense = [False, True], second_rnn = [False, True], 
                          first_dropout=[0,0.2], second_dropout=[0, 0.2]):
    #TODO: implement change of stride and window_length
    params_list_results = list()
    
    for rnnt in rnn_types:
        for bs in batch_sizes:
            for fr in frequency_reductions:
                for bn in batch_norm:
                    for lr in learning_rates:
                        for fd in first_dense:
                            for srnn in second_rnn:
                                for fdr in first_dropout:
                                    for sdr in second_dropout:
                                        params_list_results.append({
                                                        'epochs' : epochs,
                                                        'learning_rate' : lr,
                                                        'frequency_reduction' : fr,
                                                        'window_length' : int(256/(2**(fr))),
                                                        'stride' : int(256/(2**(fr+1))),
                                                        'batch_size' : bs,
                                                        'rnn_type' : rnnt,
                                                        'second_rnn_layer' : srnn,
                                                        'first_dropout' : fdr,
                                                        'second_dropout' : sdr,
                                                        'first_dense_layer' : fd
                                                        })
    return params_list_results