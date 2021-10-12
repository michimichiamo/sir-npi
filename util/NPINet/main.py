#!/usr/bin/env python
# coding: utf-8

# # A predictive model for epidemic scenarios (SARS-CoV-2) exploiting Artificial Neural Networks

## Imports

import os
#os.environ['PYTHONHASHSEED']=str(42)
import urllib
import datetime

#import random as python_random
#python_random.seed(42)

import pandas as pd
import numpy as np
#np.random.seed(42)
import tensorflow as tf
#tf.random.set_seed(42)

from matplotlib import pyplot as plt
import ipywidgets as widgets

from tqdm import tqdm

from util.NPINet.plotter import plotter, plot_loss_history, evaluate_model, SIR_evaluation
from util.NPINet.reader import download_data, load_data, apply_lbdays, split_data, convert_to_tensor
from util.NPINet.model import opts, customize_hyperparameters, NPINet
from sklearn.model_selection import train_test_split

# ## Data preprocessing

# Download
data = download_data()

start_date=datetime.date(2020, 8, 15) 
end_date=datetime.date(2021, 4, 22)
split_date=datetime.date(2021, 2, 23) # train-test split
encoding_method = 'normalize'


# Preprocessing
P, SIR, beta, T, M = load_data(data, start_date, end_date, split_date, encoding_method)


# Lookback days: sequences as input
lbdays=21
I = apply_lbdays(P, SIR, beta, T, M, start_date, lbdays)

# ## Training

# ### Split data

# Load training data
X_train_val, Y_train_val = split_data(P, I, beta, T, M, 'train', lbdays)
# Split into training and validation
X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.1, shuffle=False)
print(f'Training shapes: {X_train.shape}, {Y_train.shape}')
print(f'Validation shapes: {X_val.shape}, {Y_val.shape}')

# Convert to tensors
X_train_val, Y_train_val = convert_to_tensor(X_train_val, Y_train_val)
X_train, Y_train, X_val, Y_val = convert_to_tensor(X_train, Y_train, X_val, Y_val)


# Load test data
X_test, Y_test = split_data(P, I, beta, T, M, 'test', lbdays)
print(f'Test shapes: {X_test.shape}, {Y_test.shape}')

X_test, Y_test = convert_to_tensor(X_test, Y_test)


# Pack up data
X_true = tf.concat([X_train_val, X_test], axis=0)
Y_true = tf.concat([Y_train_val, Y_test], axis=0) 
print('Shapes:', X_true.shape, Y_true.shape)


# ### Set Hyperparameters

epochs = widgets.SelectionSlider(options=[500, 1000, 1500, 2000], value=500, description='Epochs:')
opt = widgets.Select(options=opts.keys(), value='adam', description='Optimizer:')
loss = widgets.Select(options=['mean_squared_error', 'mean_absolute_percentage_error'],
                      description='Loss:')
batch_size = widgets.SelectionSlider(options=[1, 32, X_train.shape[0]], value=X_train.shape[0], description='Batch size:')
lr_init = widgets.FloatLogSlider(base=10, min=-5, max=0, step=1, value=1e-3, description='LR initial value')
lr_decay = widgets.Checkbox(value=True, description='LR decay:')
lr_decay_rate = widgets.SelectionSlider(options=[0.1, 0.3, 0.5, 0.8, 0.9, 0.95], value=0.5, description='Rate:')
act = widgets.Select(options=['relu', 'softmax', 'sigmoid', 'tanh'], value='relu', description='Activation:')
es = widgets.Checkbox(value=True, description='Early stopping')
es_delta = widgets.FloatLogSlider(base=10, min=-7, max=-4, step=1, value=1e-7, description='Delta')
es_patience = widgets.SelectionSlider(options=[epochs.value, epochs.value/2, epochs.value/5, epochs.value/10, epochs.value/20, epochs.value/50, epochs.value/100], value=epochs.value/20, description='Patience:')
tb = widgets.Checkbox(value=False, description='TensorBoard')

box1 = widgets.VBox([epochs, loss, batch_size, act])
box2 = widgets.VBox([opt, lr_init, lr_decay, lr_decay_rate])
box3 = widgets.VBox([es, es_delta, es_patience, tb])
ui = widgets.HBox([box1, box2, box3])
display(ui)


hyperparameters = {
    'epochs' : epochs.value,
    'opt' : opt.value,
    'loss' : loss.value,
    'batch_size' : batch_size.value,
    'lr_init' : lr_init.value,
    'lr_decay' : lr_decay.value,
    'lr_decay_rate' : lr_decay_rate.value,
    'act' : act.value,
    'es' : es.value,
    'es_delta' : es_delta.value,
    'es_patience' : es_patience.value,
    'tb' : tb.value
    }

opt, loss, epochs, batch_size, cbks = customize_hyperparameters(X_train.shape[0], hyperparameters)


# ### Training

#from itertools import product
#dims = [8, 16, 32]
#h = [[d] for d in dims] +\
#         [list(i) for i in product(dims, dims)] +\
#         [list(i) for i in product(dims, dims, dims)]

#import keras_tuner as kt
#class MyHyperModel(kt.HyperModel):
#    def build(self, hp):
#        lrs = hp.Choice('hidden', range(len(h)))
#        model = NPINet(input_dim=X_train.shape[1], hidden=h[lrs])
#        model.compile(optimizer=opt, loss=loss)
#        return model

#tuner = kt.BayesianOptimization(
#    MyHyperModel(),
#    objective='val_loss',
#    max_trials=100)


#tuner.search(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val,Y_val), shuffle=False)

#tuner.results_summary(num_trials=10)
#best_hp = tuner.get_best_hyperparameters()[0]
#model = tuner.hypermodel.build(best_hp)


model = NPINet(input_dim=X_train.shape[1], hidden=[32, 32])
model.compile(optimizer=opt, loss=loss)

history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val,Y_val), callbacks=cbks, shuffle=False)


# ## Inspecting results

# ### Loss history

plot_loss_history(history)


#%load_ext tensorboard
#%tensorboard --logdir logs


# ### Predictions



evaluate_model(model, X_train_val, Y_train_val, mode='train', split={'val':X_train.shape[0]});


# ## Testing

# ### Predictions on $\beta$

evaluate_model(model, X_test, Y_test, mode='test');


# ### Predictions on $\beta$ (training+test sets)


evaluate_model(model, X_true, Y_true, mode='train_test', split={'val': X_train.shape[0],
                                                                'test': X_train_val.shape[0]});


# ### Prediction through SIR model


I_true, I_pred, I_fit = SIR_evaluation(model, X_true, Y_true, beta, start_date, end_date, method='RK',
                                       split={'val': X_train.shape[0],'test': X_train_val.shape[0]},
                                       lbdays=lbdays)

# ### Saving


#import json
#model_filename = 'data/models/model_'
#history_filename = 'data/models/history_'
#
#write=False
#i=1
#while write is False:
#    m_file = model_filename+str(i)+'.h5'
#    h_file = history_filename+str(i)+'.json'
#    try:
#        open(m_file)
#    except FileNotFoundError:
#        model.save(m_file)
#        with open(h_file, 'w') as hf:
#            json.dump(history.history, hf, indent=4)
#        write=True
#    else:
#        i=i+1
#
#with open('history.json', 'r') as hf:
#    data = json.load(hf)


# In[ ]:




