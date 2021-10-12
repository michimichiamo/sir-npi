#!/usr/bin/env python
# coding: utf-8

# ## Imports

from urllib.error import HTTPError # check connection

# Data management and manipulation
import pandas as pd
import numpy as np
import datetime # dates

# ## Data preprocessing
# ### Download and clean

## Extract data information from string
#def dateparse (timestamp):
#    return datetime.datetime.strptime(timestamp[:10], '%Y-%m-%d').date()

# Create DataFrame from csv
def download_data(url, region=None):
    try:
        # Load data
        dataframe = pd.read_csv(url)
    except HTTPError:
        print("Error while loading data.")
    else:
        if region is not None:
            # Select region
            dataframe = dataframe[dataframe['denominazione_regione'] == region]
        # Select interesting columns
        dataframe = dataframe.filter(['data', 'totale_positivi', 'dimessi_guariti', 'deceduti'])
        # Parse dates
        dataframe['data'] = pd.to_datetime(dataframe['data'], format='%Y-%m-%d').dt.date
        # Set DatetimeIndex
        dataframe = dataframe.set_index(pd.DatetimeIndex(dataframe['data'])).drop('data', axis=1)
        #dataframe['data'] = dataframe['data'].apply(dateparse)
        return dataframe


# ## Data preprocessing
# ### Retrieve input data, apply padding

# Retrieve S,I,R from Dataframe and eventually add padding
def read_SIR(dataframe,
             start_date=datetime.date(2020, 2, 24),
             end_date=datetime.date(2020, 12, 31),
             N=59384222, # Italy pop 15.08.2020
             padding=False,
             pad_days=None):
    
    # Select dates
    dataframe = dataframe.loc[start_date:end_date]
    
    # Retrieve S, I, R from DataFrame
    I = np.array(dataframe['totale_positivi'], dtype='float64')
    recovered = np.array(dataframe['dimessi_guariti'], dtype='float64')
    deceased = np.array(dataframe['deceduti'], dtype='float64')
    R = recovered + deceased
    S = N - I - R
    
    # Padding
    if padding is not False:
        pad_value = 10
        try:
            S_pad = np.full(shape=(pad_days,), fill_value=N-pad_value, dtype='float64')
            I_pad = np.full(shape=(pad_days,), fill_value=pad_value, dtype='float64')
            R_pad = np.zeros(shape=(pad_days,), dtype='float64')
        except TypeError as e:
            print("If padding is set to true, pad_days must be specified")
            raise e
        else:
            S = np.insert(S, 0, S_pad)
            I = np.insert(I, 0, I_pad)
            R = np.insert(R, 0, R_pad)
    
    return S, I, R
