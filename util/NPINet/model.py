#!/usr/bin/env python
# coding: utf-8

# ## Imports

import tensorflow as tf
from tensorflow.keras import backend as K
from matplotlib import pyplot as plt

from sklearn.model_selection import KFold


# ## DEFINITION

class NPINet(tf.keras.Model):
    def __init__(self, input_dim, hidden={}):
        super(NPINet, self).__init__()
        # Build the model
        tf.random.set_seed(42)
        # Input layer
        self.lrs = [tf.keras.layers.Dense(input_dim=input_dim,
                               units=32,
                               kernel_initializer='random_normal',
                               bias_initializer='zeros',
                               kernel_regularizer='l2',
                               activation='relu',
                               dtype='float64',
                               name='input')]
        # Hidden layers
        self.lrs += [tf.keras.layers.Dense(h,
                                          kernel_initializer='random_normal',
                                          bias_initializer='zeros',
                                          #kernel_regularizer='l2',
                                          activation='relu',
                                          dtype='float64',
                                          name=f'h{i}') for i,h in enumerate(hidden)]
        #self.lrs.append(tf.keras.layers.Dropout(.2))
        # Output layer
        self.lrs.append(tf.keras.layers.Dense(1,
                                              kernel_initializer='random_normal',
                                              bias_initializer='zeros',
                                              activation='linear',
                                              dtype='float64',
                                              name='output'))

    #    # Loss trackers
    #    self.alpha = alpha
    #    self.ls_tracker = tf.keras.metrics.Mean(name='loss')
    #    self.mse_tracker = tf.keras.metrics.Mean(name='mse')
    #    self.reg_tracker = tf.keras.metrics.Mean(name='reg')
    #    self.val_ls_tracker = tf.keras.metrics.Mean(name='val_loss')

    def call(self, data):
        x = data
        for layer in self.lrs:
            x = layer(x)
        return x

    #def train_step(self, data):
    #    x, y_true = data
#
    #    with tf.GradientTape() as tape:
    #        # Obtain the predictions
    #        y_pred = self(x, training=True)
    #        # Compute the main loss
    #        mse = K.mean(K.square(y_pred-y_true))
    #        # Compute the regularization term
    #        reg = K.sum([K.sum(K.square(var)) for var in self.trainable_variables])
    #        loss = mse + self.alpha * reg
#
    #    # Compute gradients
    #    tr_vars = self.trainable_variables
    #    grads = tape.gradient(loss, tr_vars)
#
    #    # Update the network weights
    #    self.optimizer.apply_gradients(zip(grads, tr_vars))
#
    #    # Track the loss change
    #    self.ls_tracker.update_state(loss)
    #    self.mse_tracker.update_state(mse)
    #    self.reg_tracker.update_state(reg)
    #    return {'mse': self.mse_tracker.result(),
    #            'reg': self.reg_tracker.result()}
#
    #def test_step(self, data):
    #    x, y_true = data
#
    #    # Obtain the predictions
    #    y_pred = self(x, training=False)
    #    # Compute the main loss
    #    mse = K.mean(K.square(y_pred-y_true))
    #    # Compute the regularization term
    #    reg = K.sum([K.sum(K.square(var)) for var in self.trainable_variables])
    #    loss = mse + self.alpha * reg
#
    #    # Track the loss change
    #    self.val_ls_tracker.update_state(loss)
    #    return {'loss': self.val_ls_tracker.result()}
#
    #@property
    #def metrics(self):
    #    return [self.ls_tracker,
    #            self.mse_tracker,
    #            self.reg_tracker]


# ## TRAINING

# Optimizers
opts = {
    'adadelta': tf.keras.optimizers.Adadelta,
    'adagrad': tf.keras.optimizers.Adagrad,
    'adam': tf.keras.optimizers.Adam,
    'adamax': tf.keras.optimizers.Adamax,
    'nadam': tf.keras.optimizers.Nadam,
    'rmsprop': tf.keras.optimizers.RMSprop,
    'sgd': tf.keras.optimizers.SGD
}

def customize_hyperparameters(input_size, hyperparameters):
    # Hyperparameters
    epochs = 2000
    if 'epochs' in hyperparameters.keys():
        epochs = hyperparameters['epochs']
    opt = 'adam'
    if 'opt' in hyperparameters.keys():
         opt = hyperparameters['opt']
    loss = 'mean_squared_error'
    if 'loss' in hyperparameters.keys():
         loss = hyperparameters['loss']
    batch_size = input_size
    if 'batch_size' in hyperparameters.keys():
         batch_size = hyperparameters['batch_size']
    lr_init = 1e-3
    if 'lr_init' in hyperparameters.keys():
         lr_init = hyperparameters['lr_init']
    lr_decay = True
    if 'lr_decay' in hyperparameters.keys():
         lr_decay = hyperparameters['lr_decay']
    lr_decay_rate = 0.9
    if 'lr_decay_rate' in hyperparameters.keys():
         lr_decay_rate = hyperparameters['lr_decay_rate']
    act = 'relu'
    if 'act' in hyperparameters.keys():
         act = hyperparameters['act']
    es = True
    if 'es' in hyperparameters.keys():
         es = hyperparameters['es']
    if es is not False:
        es_delta = 1e-6
        if 'es_delta' in hyperparameters.keys():
            es_delta = hyperparameters['es_delta']
        es_patience = epochs/5
        if 'es_patience' in hyperparameters.keys():
             es_patience = hyperparameters['es_patience']
    tb = False
    if 'tb' in hyperparameters.keys():
         tb = hyperparameters['tb']
    # Optimizer
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr_init,
                                                                 decay_steps=epochs,
                                                                 decay_rate=lr_decay_rate)
    optimizer = opts[opt](learning_rate=lr_schedule) if lr_decay else opts[opt](learning_rate=lr_init)
    # Callbacks
    cbks, es_cbk, tb_cbk = [], None, None
    if es:
        es_cbk = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  min_delta=es_delta,
                                                  patience=es_patience,
                                                  mode='min',
                                                  restore_best_weights=True)
        cbks.append(es_cbk)
    if tb_cbk:
        tb_cbk = tf.keras.callbacks.TensorBoard(log_dir="./logs",
                                                histogram_freq=100)
        cbks.append(tb_cbk)

    return optimizer, loss, epochs, batch_size, cbks

## OLD

## Model definition and compilation
#def hidden_layer(units=32, name='hidden1', activation='relu'):
#    return tf.keras.layers.Dense(units=units,
#                                 kernel_initializer=tf.keras.initializers.RandomNormal(),
#                                 bias_initializer='zeros',
#                                 activation=activation,
#                                 dtype='float64',
#                                 name=name)
#
#
#def define_model(lbdays=0, encoding_method='normalize', act='relu'):
#    # Input size (policies)
#    input_size = 13 if encoding_method == 'normalize' else 37
#    # Temperature, mobility
#    input_size = input_size+2
#    # Lookback
#    input_size = input_size+lbdays
#        
#    tf.random.set_seed(42)
#    # Initializers
#    k_init = tf.keras.initializers.RandomNormal()
#    b_init = 'zeros'
#    #k_reg = None#tf.keras.regularizers.l2(l2=0.001)
#    #b_reg = None#tf.keras.regularizers.l2(l2=0.001)
#    
#    model = tf.keras.Sequential(
#        [tf.keras.layers.Dense(input_shape=(input_size,),
#                               units=32,
#                               kernel_initializer=k_init,
#                               bias_initializer=b_init,
#                               kernel_regularizer='l2',
#                               dtype='float64',
#                               name='input'),
#         #tf.keras.layers.BatchNormalization(),
#         #tf.keras.layers.Activation(act),
#         
#         hidden_layer(units=64, name='hidden1'),
#         hidden_layer(units=32, name='hidden2'),
#         #hidden_layer(units=256, name='hidden3'),
#         #hidden_layer(units=128, name='hidden4'),
#         #hidden_layer(units=64, name='hidden5'),
#         
#         
#         tf.keras.layers.Dense(units=1,
#                               kernel_initializer=k_init,
#                               bias_initializer=b_init,
#                               dtype='float64',
#                               activation='linear',
#                               name='output')],
#        name='Model')
#    return model
#
#def build_model(model, loss='mean_squared_error', opt='Adam'):
#    model.compile(loss=loss, optimizer=opt)

#def train_model(X_train, Y_train, lbdays=0, val_data=None, **hyperparameters):
##, batch_size=32, epochs=500, act='relu', loss='mean_squared_error', opt='Adam', lbdays=lbdays, cbks=None, val_data=None):
#    # Hyperparameters
#    epochs = 2000
#    if 'epochs' in hyperparameters.keys():
#        epochs = hyperparameters['epochs']
#    opt = 'adam'
#    if 'opt' in hyperparameters.keys():
#         opt = hyperparameters['opt']
#    loss = 'mean_squared_error'
#    if 'loss' in hyperparameters.keys():
#         loss = hyperparameters['loss']
#    batch_size = X_train.shape[0]
#    if 'batch_size' in hyperparameters.keys():
#         batch_size = hyperparameters['batch_size']
#    lr_init = 1e-3
#    if 'lr_init' in hyperparameters.keys():
#         lr_init = hyperparameters['lr_init']
#    lr_decay = 0.9
#    if 'lr_decay' in hyperparameters.keys():
#         lr_decay = hyperparameters['lr_decay']
#    act = 'relu'
#    if 'act' in hyperparameters.keys():
#         act = hyperparameters['act']
#    es = True
#    if 'es' in hyperparameters.keys():
#         es = hyperparameters['es']
#    if es is not False:
#        es_delta = 1e-6
#        if 'es_delta' in hyperparameters.keys():
#            es_delta = hyperparameters['es_delta']
#        es_patience = epochs/5
#        if 'es_patience' in hyperparameters.keys():
#             es_patience = hyperparameters['es_patience']
#    tb = False
#    if 'tb' in hyperparameters.keys():
#         tb = hyperparameters['tb']
#    # Optimizer
#    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr_init,
#                                                                 decay_steps=epochs,
#                                                                 decay_rate=lr_decay)
#    optimizer = opts[opt](learning_rate=lr_schedule)
#    # Callbacks
#    cbks, es_cbk, tb_cbk = [], None, None
#    if es:
#        es_cbk = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
#                                                  min_delta=es_delta,
#                                                  patience=es_patience,
#                                                  mode='min',
#                                                  restore_best_weights=True)
#        cbks.append(es_cbk)
#    if tb_cbk:
#        tb_cbk = tf.keras.callbacks.TensorBoard(log_dir="./logs",
#                                                histogram_freq=100)
#        cbks.append(tb_cbk)
#    # Define model
#    model = define_model(lbdays=lbdays, act=act)
#    # Build model
#    build_model(model, loss=loss, opt=optimizer)
#    # Fit model
#    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, callbacks=cbks, validation_data=val_data, shuffle=False)
#    return model, history


# ## CROSS-VALIDATION

def cross_validate(X_train_val, Y_train_val, batch_size=32, epochs=500, lbdays=0, n=5):
    kf = KFold(n_splits=n)
    
    # Save models and losses
    models = []
    histories = []
    # Loss, optimizer, callbacks
    loss='mean_squared_error'
    opt='adam'
    cbks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                     min_delta=1e-06,
                                                     patience=epochs/10,
                                                     mode='min',
                                                     restore_best_weights=True)]
        
    # Cross-validation
    for i, (train_index, val_index) in enumerate(kf.split(X_train_val)):
        print(f"Fold {i+1} of {n}")
        X_train, X_val = X_train_val[train_index], X_train_val[val_index]
        Y_train, Y_val = Y_train_val[train_index], Y_train_val[val_index]
        
        # Train
        m, h = train_model(X_train, Y_train, batch_size=batch_size, epochs=epochs, act='relu', loss=loss, opt=opt, lbdays=lbdays, cbks=cbks, val_data=(X_val,Y_val))
        # Save
        models.append(m)
        histories.append(h)
        
    return models, histories


