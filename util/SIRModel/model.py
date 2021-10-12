#!/usr/bin/env python
# coding: utf-8

# ## Imports

# System
import os

# Data manipulation
import numpy as np

# Optimization
import tensorflow as tf
from tensorflow.keras import backend as K

from tqdm import tqdm # nice progress bars

# ## Model definition
# ### The SIRModel class

class SIRModel():

    def __init__(self, S, I, R, N, gamma=1./17, eps_R=1e-3, lambda_beta=0, mu_beta=0):
        # Observed values
        self.S = tf.Variable(S, dtype='float64')
        self.I = tf.Variable(I, dtype='float64')
        self.R = tf.Variable(R, dtype='float64')
        self.N = tf.Variable(N, dtype='float64')
        self.T = len(S)
        # Approximated values
        self.S_ = tf.Variable([self.S[0]], dtype='float64')
        self.I_ = tf.Variable([self.I[0]], dtype='float64')
        self.R_ = tf.Variable([self.R[0]], dtype='float64')
        # Parameters
        self.initializer_beta = tf.keras.initializers.RandomUniform(minval=0, seed=42)
        self.betas = tf.Variable(self.initializer_beta(shape=(self.T,), dtype='float64'), trainable=True)

        # Hyperparameters
        self.gammas = tf.constant(value=gamma, dtype='float64', shape=(self.T,))
              
        # Lagrangian dual parameters
        self.mu_beta = mu_beta
        self.lambda_beta = tf.Variable(lambda_beta, dtype='float64')
        self.beta_cst = tf.Variable(0, dtype='float64')

        # Weight
        self.eps_R = tf.constant(eps_R, shape=(), dtype='float64')
        
        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam()
        
        # Loss history
        self.loss_history = tf.Variable([], dtype='float64')
        self.mse_history = tf.Variable([], dtype='float64')
        self.cst_history = tf.Variable([], dtype='float64')

    # Approximation step
    def __step(self, t):
        S_to_I = (self.betas[t] * self.I_[t] * self.S_[t]) / self.N
        I_to_R = self.gammas[t] * self.I_[t]
        self.S_ = tf.concat([self.S_, [self.S_[t] - S_to_I]], 0)
        self.I_ = tf.concat([self.I_, [self.I_[t] + S_to_I - I_to_R]], 0)
        self.R_ = tf.concat([self.R_, [self.R_[t] + I_to_R]], 0)
        
    # Approximation
    def run_simulation(self):
        self.S_ = tf.Variable([self.S[0]], dtype='float64')
        self.I_ = tf.Variable([self.I[0]], dtype='float64')
        self.R_ = tf.Variable([self.R[0]], dtype='float64')
        for t in range(self.T-1):
            self.__step(t)
            
    # Loss function
    def loss_fn(self):
        self.run_simulation()
        
        # Error on I
        err_I = K.mean(K.square(self.I_ - self.I))
        # Error on R
        err_R = self.eps_R * K.mean(K.square(self.R_ - self.R))

        mse = err_I + err_R

        # Regularization on beta
        self.beta_cst = K.mean(K.square(self.betas[:-1] - self.betas[1:]))
        cst = self.lambda_beta * self.beta_cst
    
        loss = (mse + cst)# / self.N # Scaled

        # Update loss history
        self.loss_history = tf.concat([self.loss_history, [loss]], 0)
        self.mse_history = tf.concat([self.mse_history, [mse]], 0)
        self.cst_history = tf.concat([self.cst_history, [cst]], 0)
        
        return loss

    # Update parameters
    def optimize_lagrangian_parameters(self):
        self.lambda_beta = self.lambda_beta + self.mu_beta * self.beta_cst

    # Train: minimize loss, update parameters
    def train(self, epochs, ldf=False):
        for _ in tqdm(range(epochs)):
            self.optimizer.minimize(self.loss_fn, var_list=[self.betas])#, self.gammas])
            if ldf:
                self.optimize_lagrangian_parameters()

    # Return S, I, R
    def get_SIR(self):
        self.run_simulation()
        return self.S_, self.I_, self.R_
    
    # Return beta, gamma
    def get_pars(self):
        return self.betas.numpy(), self.gammas.numpy()
    
    # Return loss history
    def get_loss_history(self):
        return [self.loss_history.numpy(),
                self.mse_history.numpy(),
                self.cst_history.numpy()]
    
    # (Roughly) estimate beta for next 'days' time steps
    def predict(self, days):
        self.run_simulation()
        S = [self.S_[-1].numpy()]
        I = [self.I_[-1].numpy()]
        R = [self.R_[-1].numpy()]
        N = self.N.numpy()
        
        # Compute beta: exponentially smoothed average of last 5 days
        beta = self.betas[-5:]
        beta_mean = tf.reduce_mean(beta)
        soft = tf.nn.softmax(tf.linspace(0., len(beta)-1, len(beta))).numpy()
        beta_avg = tf.reduce_mean((beta-beta_mean)*soft +beta_mean)
        
        gamma = self.gammas[0].numpy()
        for day in range(days):
            S_to_I = (beta_avg * I[-1] * S[-1]) / N
            I_to_R = I[-1] * gamma
            S.append(S[-1] - S_to_I)
            I.append(I[-1] + S_to_I - I_to_R)
            R.append(R[-1] + I_to_R)

        return S, I, R, beta_avg.numpy()



# ## Save parameters


def save_parameters(model, directory='./data/fit'):

    # Check whether directory exists
    if not os.path.exists(directory):
        os.mkdir(directory)

    # Save parameter
    beta, _ = model.get_pars()
    np.save(os.path.join(directory,'beta_it.npy'), beta)

    # Save fitted SIR
    S_, I_, R_ = model.get_SIR()
    np.save(os.path.join(directory,'SIR_it.npy'), np.vstack([S_,I_,R_]))

