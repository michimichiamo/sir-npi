#!/usr/bin/env python
# coding: utf-8

# ## Imports
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

from util.SIRModel.reader import read_SIR

figsize=(9,4)

def plotter(args, labels=None):
    plt.close('all')
    plt.figure(figsize=figsize)
    if labels:
        for arg, lbl in zip(args ,labels):
            plt.plot(arg, label=lbl)
        plt.legend()
    else:
        for arg in args:
            plt.plot(arg)
    plt.show()


# Plot loss and delta_loss over iterations
def plot_loss_history(model, reg=False, figsize=figsize):
    # New figure
    plt.close('all')
    # Extract loss history from model
    loss, mse, cst = model.get_loss_history()
    
    # Create two plots
    _, axs = plt.subplots(2, gridspec_kw={'hspace': 0.5}, figsize=figsize)
    
    # All iterations
    axs[0].plot(loss, label='loss', color='blue')
    if reg:
        axs[0].plot(mse, label='mse', color='orange')
        axs[0].plot(cst, label='cst', color='green')
    axs[0].legend(loc=0)
    axs[0].set_title('Loss history')
    axs[0].set_ylabel('Loss')
    
    # Last 200 iterations
    ll = axs[1].plot(loss[-200:], label='loss', color='blue')
    if reg:
        ll = axs[1].plot(mse[-200:], label='mse', color='orange')
        ll = axs[1].plot(cst[-200:], label='cst', color='green')

    axs[1].legend(loc=0)
    axs[1].set_title('Last 200 iterations')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Iterations')

    plt.show()

# OLD

## Plot loss and delta_loss over iterations
#def plot_loss_history(model, figsize=figsize):
#    # New figure
#    plt.close('all')
#    # Extract loss history from model
#    losses = model.get_loss_history()
#    # Loss variation
#    delta_losses = losses[:-1]-losses[1:]
#    
#    # Create two plots
#    _, axs = plt.subplots(2, gridspec_kw={'hspace': 0.5}, figsize=figsize)
#    
#    # All iterations
#    l = axs[0].plot(range(len(losses)), losses, label='loss', color='blue')
#    axs_0 = axs[0].twinx()
#    dl = axs_0.plot(range(len(delta_losses)), delta_losses, label='delta_loss', color='orange')
#    ls = l+dl
#    lbls = [l.get_label() for l in ls]
#    axs[0].legend(ls, lbls, loc=0)
#    axs[0].set_title('Loss and delta_loss history')
#    axs[0].set_ylabel('Loss')
#    
#    # Last 200 iterations
#    ll = axs[1].plot(range(len(losses[-200:])), losses[-200:], label='loss', color='blue')
#    axs_1 = axs[1].twinx()
#    dll = axs_1.plot(range(len(delta_losses[-200:])), delta_losses[-200:], label='delta_loss', color='orange')
#    lls = ll+dll
#    lblls = [l.get_label() for l in lls]
#    axs[1].legend(lls, lblls, loc=0)
#    axs[1].set_ylabel('Loss')
#    axs[1].set_xlabel('Iterations')
#
#    plt.show()
    
# Plot model parameters
def plot_pars(model, start_date, figsize=figsize):
    # New figure
    plt.close('all')
    plt.figure(figsize=figsize)
    # Extract parameters history from model
    beta, gamma = model.get_pars()

    # Extract date range
    days = pd.date_range(start=start_date, periods=len(beta), freq='D').date
    
    # Plot
    plt.title('Fitted beta parameters')
    plt.plot(days, beta, label='beta')
    #plt.plot(range(len(gamma)), gamma, label='gamma')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Value')
    
    plt.tight_layout()

# ## Runge-Kutta

# Differential equations for SIR model
def dsdt(model, S, I, R, par):
    return -par[0]*I*S/model.N

def didt(model, S, I, R, par):
    return par[0]*I*S/model.N -par[1]*I

def drdt(model, S, I, R, par):
    return par[1]*I

dfs = [dsdt, didt, drdt]

# Compute coefficients for Runge-Kutta algorithm
def RK_coeffs(model, t, S_, I_, R_):
    par = [model.betas[t], model.gammas[t]]
    k1 = [df(model, S_[t],I_[t],R_[t],par) for df in dfs]
    k2 = [df(model, S_[t] + 0.5*k1[0], I_[t] + 0.5*k1[1], R_[t] + 0.5*k1[2], par) for df in dfs]
    k3 = [df(model, S_[t] + 0.5*k2[0], I_[t] + 0.5*k2[1], R_[t] + 0.5*k2[2], par) for df in dfs]
    k4 = [df(model, S_[t] + k3[0], I_[t] + k3[1], R_[t] + k3[2], par) for df in dfs]
    return k1,k2,k3,k4

# Apply RungeKutta algorithm to get an approximation of S,I,R
def RungeKutta(model):
    # Compute T next values for S,I,R

    # Initialize variables
    S_ = tf.Variable([model.S[0]], dtype='float64')
    I_ = tf.Variable([model.I[0]], dtype='float64')
    R_ = tf.Variable([model.R[0]], dtype='float64')

    # Iterate for number of steps
    for t in range(model.T-1):
        #Apply RK Formulas to get S(t+1),I(t+1),R(t+1)
        kS,kI,kR = zip(*RK_coeffs(model, t, S_, I_, R_))

        # Update next value of S,I,R
        S_ = tf.concat([S_, [S_[t] + (1.0/6.0)*(kS[0] + 2*kS[1] + 2*kS[2] + kS[3])]], 0)
        I_ = tf.concat([I_, [I_[t] + (1.0/6.0)*(kI[0] + 2*kI[1] + 2*kI[2] + kI[3])]], 0)
        R_ = tf.concat([R_, [R_[t] + (1.0/6.0)*(kR[0] + 2*kR[1] + 2*kR[2] + kR[3])]], 0)
        
    return S_, I_, R_


# ## Plot SIR predictions
    
def plot_SIR(model, S, I, R, method='E', start_date=None, figsize=figsize):
    # New figure
    plt.close('all')
    # Extract S,I,R and parameters from model
    S_, I_, R_ = model.get_SIR()
    beta, gamma = model.get_pars()
    # Get Runge-Kutta predictions
    S_RK, I_RK, R_RK = RungeKutta(model)

    days = range(len(S))
    if start_date:
        # Extract date range
        days = pd.date_range(start=start_date, periods=len(beta), freq='D').date
    
    # Create two plots
    _, ax0 = plt.subplots(figsize=figsize)
    
    # Plot S
    ax0.plot(days, S, color='blue', label='S_true')
    if 'E' in method:
        ax0.plot(days, S_, color='purple', label='S_approx_E')
    if 'RK' in method:
        ax0.plot(days, S_RK, color='violet', label='S_approx_RK')
    # Plot I
    ax0.plot(days, I, color='black', label='I_true')
    if 'E' in method:
        ax0.plot(days, I_, color='brown', label='I_approx_E')
    if 'RK' in method:
        ax0.plot(days, I_RK, color='grey', label='I_approx_RK')
    # Plot R
    ax0.plot(days, R, color='red', label='R_true')
    if 'E' in method:
        ax0.plot(days, R_, color='orange', label='R_approx_E')
    if 'RK' in method:
        ax0.plot(days, R_RK, color='yellow', label='R_approx_RK')
    plt.legend(loc='upper right', bbox_to_anchor=(0.85, 1))
    ax0.set_title('SIR evolution')
    ax0.set_xlabel('Time')# (days) after '+start_date.strftime('\'%y %b %d'))
    ax0.set_ylabel('# people')
    
    # Plot parameters
    ax1 = ax0.twinx()
    ax1.plot(days, beta, '--', alpha=0.5, color='red', label='beta')
    #ax1.plot(range(len(gamma)), gamma, '--', alpha=0.5, color='green', label='gamma')
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
    ax1.set_ylabel('Parameters value')
    plt.tight_layout()


    # Compute scores
    mse, r2 = {}, {}
    
    mse['E'] = mean_squared_error(I/I.max(), I_/I.max())
    r2['E'] = r2_score(I/I.max(), I_/I.max())
    
    mse['RK'] = mean_squared_error(I/I.max(), I_RK/I.max())
    r2['RK'] = r2_score(I/I.max(), I_RK/I.max())
    
    print(f'R2 scores: ', end='')
    print([f'{v:.2f} ({k})' for k,v in r2.items()])
    print(f'mse:', end='')
    print([f'{v:.2e} ({k})' for k,v in mse.items()])
    
    
#def plot_prediction(model, start_date, end_date, dataframe, pred_days=14, figsize=figsize):
#    # New figure
#    plt.close('all')
#    # Read more data
#    S_t, I_t, R_t = read_SIR(dataframe,
#                         N=4463805,
#                         start_date=start_date,
#                         end_date=end_date,
#                         padding=True,
#                         pad_days=20)
#    
#    # Extract S,I,R from model
#    S_, I_, R_ = model.get_SIR()
#    
#    # Extract new I from data (not used for learning)
#    I_true = I_t[len(I_):len(I_)+pred_days]
#    
#    # Predict new S,I,R and beta from model
#    S_p, I_p, R_p, beta = model.predict(pred_days)
#    
#    # Create two plots
#    fig, ax1 = plt.subplots(figsize=figsize)
#    
#    # Plot I: true and predicted
#    ax1.plot(range(len(I_t)), I_t, color='blue', label='I_true')
#    ax1.plot(range(len(I_) -1, len(I_) + len(I_p)-1), I_p, color='yellow', label='I_predicted')
#    ax1.set_xlabel('Time (days) after '+start_date.strftime('\'%y %b %d'))
#    ax1.set_ylabel('# ppl')
#    plt.legend(loc='upper left')
#    
#    # Plot error on prediction
#    ax2 = ax1.twinx()
#    ax2.plot(range(len(I_), len(I_)+pred_days), (I_true-I_p[1:])/I_true, '--', alpha=0.5, color='red', label='error')
#    ax2.set_ylabel('Relative error')
#    plt.legend(loc='upper left', bbox_to_anchor=(0, 0.85))
#    
#    plt.show()



## OLD

## Plot Runge-Kutta approximation
#def plot_RK(model, I, figsize=figsize):
#    # New figure
#    plt.close('all')
#    # Get Runge-Kutta predictions
#    S_RK, I_RK, R_RK = RungeKutta(model)
#    
#    # Extract parameters from model
#    beta, gamma = model.get_pars()
#    
#    # Create two plots
#    fig, ax1 = plt.subplots(figsize=figsize)
#    
#    # Plot I
#    ax1.plot(range(len(I)), I, color='blue', label='I_true')
#    ax1.plot(range(len(I_RK)), I_RK, color='orange', label='I_approx')
#    ax1.set_xlabel('time (days)')
#    ax1.set_ylabel('# ppl')
#    plt.legend(loc='upper right', bbox_to_anchor=(0.85, 1))
#    
#    # Plot parameters
#    ax2 = ax1.twinx()
#    ax2.plot(range(len(beta)), beta, '--', alpha=0.5, color='red', label='beta')
#    ax2.plot(range(len(gamma)), gamma, '--', alpha=0.5, color='green', label='gamma')
#    ax2.set_ylabel('beta/gamma value')
#    plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
#    
#    plt.show()
#
#