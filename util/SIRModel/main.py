#!/usr/bin/env python
# coding: utf-8

# # A descriptive optimization model for epidemic scenarios (SARS-CoV-2) exploiting the SIR compartmental model and Lagrangian Duality
# ## Imports

# Data management and manipulation
import datetime # dates
from util.SIRModel.reader import download_data, read_SIR

# Model
from util.SIRModel.model import SIRModel, save_parameters

# Plotting
from util.SIRModel.plotter import plotter, plot_loss_history, plot_pars, plot_SIR#, plot_prediction, plot_RK

from tqdm import tqdm # nice progress bars

%matplotlib widget
%load_ext autoreload
%autoreload 2


repository = 'https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/'
url = 'dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv'

pop_aug = 59392408 # Italy population at 1.08.2020
pop_sep = 59376037 # Italy population at 1.09.2020
N = round((pop_aug+pop_sep)/2) # Italy population estimate at 15.08.2020

# Read values
dataframe = download_data(repository+url)
start_date=datetime.date(2020, 8, 15)
end_date=datetime.date(2021, 5, 1)

# Read values
S, I, R = read_SIR(dataframe, start_date, end_date, N)

# ## Set hyperparameters

epochs = 300
eps_R=1e-3
mu_beta=1e16
gamma = 1./17

# ## Training
# ### Create model

lgd = SIRModel(S, I, R, N, gamma=gamma, eps_R=eps_R, lambda_beta=0, mu_beta=mu_beta)


# ## Training
# ### Train model

lgd.train(epochs, ldf=True)

# ## Plotting

# Loss history
plot_loss_history(lgd, reg=True)

# Parameters history
plot_pars(lgd, start_date)

# Approximated I
plot_SIR(lgd, I, start_date)

## Show some more data to compare prediction
#p_end_date=datetime.date(2021, 3, 30)
#
## Plot prediction (vs data)
#plot_prediction(model, start_date, p_end_date, dataframe, 14)
#
#
## Plot Runge-Kutta approximation
##plot_RK(model, I)
#


## SAVING
#save_parameters(model)


