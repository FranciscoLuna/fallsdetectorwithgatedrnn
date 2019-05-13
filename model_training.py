# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


#research_combination_set_name = 'researchset1'
#w_list = [256]
#stride_list = [128]
#rnntype_list = ['lstm', 'gru']
#lr_list = [0.001, 0.0001, 0.00001]
#batchsize_list = [32,64,96]
#
## TODO: create data_frame with | model_id | param_1 | ... | param_n | as columns
#
#info_list = list()
#model_id = 1
#for w in w_list:
#    for stride in stride_list:
#        for rnntype in rnntype_list:
#            for lr in lr_list:
#                for batchsize in batchsize_list:
#                    print([model_id, w, stride, rnntype, lr, batchsize, 0.2, 0, False, False])
#                    info_list.append([model_id, w, stride, rnntype, lr, batchsize, 0.2, 0, False, False])
#                    model_id += 1
#
#info_dataframe = pd.DataFrame(info_list, columns = ['model_id', 'w', 'stride', 'rnn_type', 'lr', 'batch_size', 'rnn_dropout', 'dense_dropout', 'two_rnn_layers', 'first_dense'])
#
#info_dataframe.to_csv('pepe.csv', index=False)
#info_dataframe.head()

research_set_folder = 'researchset_pre128_1'

research_set_table = pd.read_csv(research_set_folder + '/' + research_set_folder + '_research_summary.csv')

ex_model_info = np.load(research_set_folder + '/' + research_set_folder +  '_model_id_16_info.npy')

ex_model_info = ex_model_info[()]

eval_results = ex_model_info['eval_results']

eval_statistics =  quantity_results_per_class(eval_results, ['BKG', 'ALERT', 'FALL'])
