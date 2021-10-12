#!/usr/bin/env python
# coding: utf-8

# ## Imports

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

figsize=(9,3)
split_colors = {'val': 'red', 'test': 'purple'}

# Generalized plotter

def plotter(*args):
    plt.close('all')
    plt.figure(figsize=figsize)
    for arg in args:
        plt.plot(arg)
        
    plt.show()

# Plot loss history

def plot_loss_history(history, figsize=figsize):
    # New figure
    plt.close('all')
    plt.figure(figsize=figsize)
    for metric in history.history.keys():
        plt.plot(history.history[metric], label=metric)
    if len(history.history.keys()) > 0:
        plt.legend()
    plt.title('Loss history')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.tight_layout()

# ## EVALUATION

# ### BETA

def evaluate_model(model, X_test, Y_test, start_date, mode='train', split={}):
    # New figure
    plt.close('all')
    plt.figure(figsize=figsize)
    
    # Extract predictions
    Y_pred = model.predict(X_test)

    # Extract date range
    days = pd.date_range(start=start_date, periods=len(Y_pred), freq='D').date

    # Plot predictions
    plt.plot(days, Y_test, label='true', color='blue')
    plt.plot(days, Y_pred, label='pred', color='orange')

    for s in split:
        plt.vlines(x=days[split[s]], ymin=np.min([np.min(Y_test), np.min(Y_test)]),
            ymax=np.max([np.max(Y_test), np.max(Y_test)]),
            label=f'{s} split', color=split_colors[s])
    plt.title('Beta parameters')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()



    # Compute scores
    mse, r2 = {}, {}
    if 'train' in mode:
        train_end = split['val']
        val_end = split.get('test',len(Y_pred))
        mse['train'] = mean_squared_error(Y_test[:train_end], Y_pred[:train_end])
        r2['train'] = r2_score(Y_test[:train_end], Y_pred[:train_end])

        mse['val'] = mean_squared_error(Y_test[train_end:val_end], Y_pred[train_end:val_end])
        r2['val'] = r2_score(Y_test[train_end:val_end], Y_pred[train_end:val_end])

    if 'test' in mode:
        test_start = split.get('test', 0)
        mse['test'] = mean_squared_error(Y_test[test_start:], Y_pred[test_start:])
        r2['test'] = r2_score(Y_test[test_start:], Y_pred[test_start:])


    print(f'R2 scores: ', end='')
    print([f'{v:.2f} ({k})' for k,v in r2.items()])
    print(f'mse:')
    print([f'{v:.5f} ({k})' for k,v in mse.items()])
    return mse, r2


# ### SIR

# #### Runge-Kutta

# Differential equations for SIR model
def dsdt(N, S, I, R, par):
    return -par[0]*I*S/N

def didt(N, S, I, R, par):
    return par[0]*I*S/N -par[1]*I

def drdt(N, S, I, R, par):
    return par[1]*I

dfs = [dsdt, didt, drdt]

# Compute coefficients for Runge-Kutta algorithm
def RK_coeffs(N, S_, I_, R_, beta, gamma, t):
    par = [beta[t], gamma]
    k1 = [df(N, S_[t],I_[t],R_[t],par) for df in dfs]
    k2 = [df(N, S_[t] + 0.5*k1[0], I_[t] + 0.5*k1[1], R_[t] + 0.5*k1[2], par) for df in dfs]
    k3 = [df(N, S_[t] + 0.5*k2[0], I_[t] + 0.5*k2[1], R_[t] + 0.5*k2[2], par) for df in dfs]
    k4 = [df(N, S_[t] + k3[0], I_[t] + k3[1], R_[t] + k3[2], par) for df in dfs]
    return k1,k2,k3,k4


# #### Simulation

# Approximation
def run_simulation(beta,
                   SIR,
                   method='E',
                   N=59384222,
                   gamma=1./17):

    S, I, R = SIR[0], SIR[1], SIR[2]
    
    S_ = np.array([S[0]])
    I_ = np.array([I[0]])
    R_ = np.array([R[0]])
    
    # Euler's method
    if method == 'E':
    # Iterate for number of steps
        for t in range(beta.shape[0]-1):
            S_to_I = (beta[t] * I_[t] * S_[t]) / N
            I_to_R = gamma * I_[t]
            S_ = np.hstack([S_, S_[t] - S_to_I])
            I_ = np.hstack([I_, I_[t] + S_to_I - I_to_R])
            R_ = np.hstack([R_, R_[t] + I_to_R])
            
            
    # Runge-Kutta 4 method
    elif method == 'RK':
    # Iterate for number of steps
        for t in range(beta.shape[0]-1):
            #Apply RK Formulas to get S(t+1),I(t+1),R(t+1)
            kS,kI,kR = zip(*RK_coeffs(N, S_, I_, R_, beta, gamma, t))
    
            # Update next value of S,I,R
            S_ = np.hstack([S_, S_[t] + (1.0/6.0)*(kS[0] + 2*kS[1] + 2*kS[2] + kS[3])])
            I_ = np.hstack([I_, I_[t] + (1.0/6.0)*(kI[0] + 2*kI[1] + 2*kI[2] + kI[3])])
            R_ = np.hstack([R_, R_[t] + (1.0/6.0)*(kR[0] + 2*kR[1] + 2*kR[2] + kR[3])])
        
    else:
        print('Method not recognized.')
    
    
    return S,I,R, S_,I_,R_

import datetime

def SIR_evaluation(model, X_true, Y_true, beta, start_date, end_date, method='E', split={}, lbdays=0):
    # New figure
    plt.close('all')
    plt.figure(figsize=figsize)

    # Beta true and predicted
    beta = beta.loc[start_date+datetime.timedelta(lbdays):end_date, 'beta'].values
    Y_pred = model.predict(X_true).reshape(X_true.shape[0])
    
    # SIR true and fitted
    start = (start_date-datetime.date(2020,8,15)+datetime.timedelta(lbdays)).days
    end = (datetime.date(2021,5,1)-end_date).days
    SIR_true = np.load('./data/SIR_it.npy')[:, start:-end]
    SIR_fit = np.load('./data/fit/SIR_it.npy')[:, start:-end]

    # Extract SIR predictions
    _, I_true, _,  _, I_pred, _ = run_simulation(Y_pred, SIR_true, method=method)
    #_, _, _,  _, I_fit, _ = run_simulation(beta, SIR_fit, method=method)
    I_fit = SIR_fit[1]

    # Extract date range
    days = pd.date_range(start=start_date, periods=len(Y_pred), freq='D').date


    # Plot SIR predictions    
    plt.plot(days, I_true, label='I_true', color='blue')
    plt.plot(days, I_pred, label='I_pred', color='orange')
    plt.plot(days, I_fit, label='I_fit', color='green')

    for s in split:
        plt.vlines(x=days[split[s]], ymin=np.min([np.min(I_true), np.min(I_pred), np.min(I_fit)]),
            ymax=np.max([np.max(I_true), np.max(I_pred), np.max(I_fit)]),
            label=f'{s} split', color=split_colors[s])
    plt.legend()
    plt.title('SIR evolution')
    plt.xlabel('Time')
    plt.ylabel('# people')
    plt.tight_layout()


    # Compute scores
    mse, r2 = {}, {}
    train_end = split['val']
    val_end = split['test']
    # Normalize
    norm = I_true.max()
    I_tr, I_pr= I_true/norm, I_pred/norm

    
    mse['train'] = mean_squared_error(I_tr[:train_end], I_pr[:train_end])
    r2['train'] = r2_score(I_tr[:train_end], I_pr[:train_end])
    
    mse['val'] = mean_squared_error(I_tr[train_end:val_end], I_pr[train_end:val_end])
    r2['val'] = r2_score(I_tr[train_end:val_end], I_pr[train_end:val_end])
    
    mse['test'] = mean_squared_error(I_tr[val_end:], I_pr[val_end:])
    r2['test'] = r2_score(I_tr[val_end:], I_pr[val_end:])
    
    print(f'R2 scores: ', end='')
    print([f'{v:.2f} ({k})' for k,v in r2.items()])
    print(f'mse:', end='')
    print([f'{v:.2e} ({k})' for k,v in mse.items()])

    return I_true, I_pred, I_fit




### PROGRESSIVE EVALUATION (not working)

# Idea:
# repeat:
#   - take last lbdays training data
#   - predict beta for next day
#   - launch SIR to predict I
#   - append I value to training data, remove oldest I



## COMPARE WITH: data[1][['totale_positivi']].set_index(pd.date_range(start=datetime.date(2020,2,24), end=datetime.date(2021,9,28), freq='D')).loc[mystart:myend+datetime.timedelta(test_days)]

# where:
#   data[1] is SIR_DATA
#   mystart, myend should determine the range of last training lbdays data
#   test_days is the number of days to predict



# Substitute (at most) last n elements of src into tgt
def substitute_values(src, tgt, n):
    tgt = np.array(tgt).reshape(1, -1)
    src = np.array(src).reshape(1,-1)
    m = len(src)
    if m < n:
        return np.hstack([tgt[:-m-2], src, tgt[-2:]])
    else:
        return np.hstack([tgt[:-n-2], src[-n:], tgt[-2:]])



def progressive_evaluation(X_val, X_test, beta, split_date, lbdays, method='E', test_days=20):
    # Select dates
    start_date = split_date - datetime.timedelta(lbdays+1)
    end_date = split_date - datetime.timedelta(1)
    
    # Select inputs and outputs
    myX = tf.concat([X_val[-1:], X_test[:50]], axis=0)
    X = myX[0].numpy().reshape(1,-1)
    Y_pred = beta.loc[start_date:end_date, 'beta'].values

    # SIR dataframe
    SIR_idx = pd.date_range(start=datetime.date(2020,8,15), end=datetime.date(2021,9,22), freq='D')
    SIR_data = np.load('./data/new/SIR_it.npy').T # Data from (2020,8,15) to (2021,9,22))
    SIR_columns = ['S','I','R']
    SIR_df = pd.DataFrame(index=SIR_idx, data=SIR_data, columns=SIR_columns)
    # Scaling factors
    I_max = SIR_df['I'].max()
    I_min = SIR_df['I'].min()
    I_scale = I_max - I_min
    # S, I, R
    S = SIR_df.loc[start_date:end_date, 'S'].values
    I = SIR_df.loc[start_date:end_date, 'I'].values
    R = SIR_df.loc[start_date:end_date, 'R'].values
    SIR_true = np.vstack([S,I,R])
    
    SIR_pred = SIR_true
    
    # To store infected and new inputs
    I_p, X_p = [], []
    
    for i in range(test_days):
        # Stack new beta predictions
        Y_pred = np.hstack([Y_pred, model.predict(X).reshape(1,)])
        
        # Extract SIR predictions with new beta
        _, I_true, _,  S_pred, I_pred, R_pred = run_simulation(Y_pred, SIR_pred, method=method)
        
        # Store values for Infected results
        I_p.append(I_pred[-1])
        # Store scaled values for new input
        X_p.append((I_pred[-1]-I_min)/I_scale)
        
        # Prepare new input
        # Substitute 
        #X = np.hstack([myX[i+1].numpy()[:-i-3], X_p, myX[i+1].numpy()[-2:]]).reshape(1,-1)
        X = substitute_values(X, myX[i+1], lbdays)
        S = np.hstack([S,S_pred[-1]])
        I = np.hstack([I,I_pred[-1]])
        R = np.hstack([R,R_pred[-1]])
        SIR_pred = np.vstack([S,I,R])
        print(i)
        
    return I_p
        