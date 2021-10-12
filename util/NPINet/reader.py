#!/usr/bin/env python
# coding: utf-8

# ## Imports

import os
import urllib
import datetime
import pandas as pd
import numpy as np
import tensorflow as tf

# ## READ DATA

# Data directory
directory = 'data'

# OxCGRT Policies dataset
P_url = 'https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv'
P_DATA = 'OxCGRT_latest.csv' # Local file

# Protezione Civile COVID-19 cases
SIR_url = 'https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv'

# Temperature dataset folder
T_DATA = 'ECA_blended_custom'

# Apple's Mobility dataset
M_DATA = 'applemobilitytrends-2021-05-09.csv'


## Extract data information from string
#def dateparse (timestamp, format):
#    if format == '%Y-%m-%d':
#        return datetime.datetime.strptime(timestamp[:10], '%Y-%m-%d').date()
#    elif format == '%Y%m%d':
#        return datetime.datetime.strptime(timestamp, '%Y%m%d').date()

# Create DataFrame from csv
def download_data(directory=directory, P_DATA=P_DATA, P_url=P_url, SIR_url=SIR_url, T_DATA=T_DATA, M_DATA=M_DATA):
    
    # Filepaths
    P_DATA = os.path.join(directory, P_DATA)
    T_DATA = os.path.join(directory, T_DATA)
    M_DATA = os.path.join(directory, M_DATA)
    
    ## Policies
    # Check file
    if not os.path.exists(directory):
        os.mkdir(directory)
    if not os.path.exists(P_DATA):
        urllib.request.urlretrieve(P_url, P_DATA)
    # Read file
    P_data = pd.read_csv(P_DATA, low_memory=False)
        
    ## Beta
    # Beta from (2020, 2, 24) to (2021, 4, 28) (Emilia-Romagna)
    #beta = np.array([2.00062831e-01, 1.97940256e-01, 1.96695305e-01, 1.95787060e-01, 1.94322605e-01, 1.91757735e-01, 1.88261581e-01, 1.84558733e-01, 1.81416989e-01, 1.79232315e-01, 1.78006728e-01, 1.77448671e-01, 1.77159233e-01, 1.76744276e-01, 1.75881416e-01, 1.74344350e-01, 1.72080848e-01, 1.69297753e-01, 1.66494932e-01, 1.64273025e-01, 1.63202395e-01, 1.63787354e-01, 1.66275814e-01, 1.70295972e-01, 1.74803694e-01, 1.78520203e-01, 1.80325475e-01, 1.79442988e-01, 1.75536229e-01, 1.68875186e-01, 1.60352517e-01, 1.51441123e-01, 1.43861836e-01, 1.38834015e-01, 1.36479209e-01, 1.36020452e-01, 1.36249810e-01, 1.35957267e-01, 1.34365055e-01, 1.31056140e-01, 1.25930528e-01, 1.19125402e-01, 1.11037007e-01, 1.02299951e-01, 9.36489461e-02, 8.59483081e-02, 7.97593208e-02, 7.49889680e-02, 7.09756319e-02, 6.69681379e-02, 6.26577863e-02, 5.80717529e-02, 5.32674043e-02, 4.84102810e-02, 4.34054177e-02, 3.80239489e-02, 3.21667553e-02, 2.62886869e-02, 2.11252252e-02, 1.74324758e-02, 1.54144846e-02, 1.45187198e-02, 1.39034861e-02, 1.29746094e-02, 1.16390895e-02, 1.04923411e-02, 1.09484585e-02, 1.29055543e-02, 1.61254746e-02, 1.99136158e-02, 2.33373300e-02, 2.58186558e-02, 2.70864718e-02, 2.71025454e-02, 2.59398059e-02, 2.38678323e-02, 2.13279620e-02, 1.88262873e-02, 1.66981869e-02, 1.49293044e-02, 1.32406236e-02, 1.13378936e-02, 9.05742215e-03, 6.53255352e-03, 4.10985262e-03, 2.13791821e-03, 7.13531498e-04,  -2.80749673e-04, -8.46571184e-04,  -8.03835961e-04,  -1.00306907e-04, 8.91181088e-04, 1.59492057e-03, 1.64327510e-03, 1.10543790e-03, 3.96357719e-04, 9.27110425e-05, 6.37157816e-04, 2.08922231e-03, 4.06448318e-03, 6.01790762e-03, 7.68835733e-03, 9.22036956e-03, 1.08939045e-02, 1.27925132e-02, 1.47413991e-02, 1.64360498e-02, 1.75870863e-02, 1.81054768e-02, 1.82334705e-02, 1.83984707e-02, 1.89380765e-02, 2.00033584e-02, 2.16610153e-02, 2.38518604e-02, 2.62354610e-02, 2.83102340e-02, 2.97987256e-02, 3.08605175e-02, 3.19098957e-02, 3.32974563e-02, 3.51752979e-02, 3.75488954e-02, 4.02774838e-02, 4.30090171e-02, 4.52416924e-02, 4.65227510e-02, 4.66341580e-02, 4.57084526e-02, 4.42684771e-02, 4.31015488e-02, 4.29006516e-02, 4.38591654e-02, 4.55595140e-02, 4.72852466e-02, 4.84876521e-02, 4.90544899e-02, 4.92500429e-02, 4.94516791e-02, 4.98908560e-02, 5.05289340e-02, 5.11353414e-02, 5.15389769e-02, 5.18206496e-02, 5.22991496e-02, 5.32726130e-02, 5.47945782e-02, 5.66821577e-02, 5.86695316e-02, 6.04813947e-02, 6.18656763e-02, 6.26382119e-02, 6.27272627e-02, 6.22690993e-02, 6.16720683e-02, 6.15354164e-02, 6.23443732e-02, 6.41823160e-02, 6.67083893e-02, 6.93909930e-02, 7.17644229e-02, 7.35708212e-02, 7.47662838e-02, 7.54634387e-02, 7.58511057e-02, 7.61290196e-02, 7.64485185e-02, 7.68954986e-02, 7.74946841e-02, 7.81606170e-02, 7.87962608e-02, 7.95237273e-02, 8.06206217e-02, 8.21267099e-02, 8.36493471e-02, 8.47163739e-02, 8.52519398e-02, 8.55944819e-02, 8.61085242e-02, 8.69290569e-02, 8.79520406e-02, 8.89835717e-02, 8.99174108e-02, 9.08033995e-02, 9.16621030e-02, 9.23699134e-02, 9.27884779e-02, 9.29588817e-02, 9.30846796e-02, 9.34314302e-02, 9.41179765e-02, 9.50155178e-02, 9.59236361e-02, 9.66932258e-02, 9.73856673e-02, 9.82302816e-02, 9.94060122e-02, 1.00841409e-01, 1.02275503e-01, 1.03337529e-01, 1.03737496e-01, 1.03415523e-01, 1.02624531e-01, 1.01828332e-01, 1.01413682e-01, 1.01440348e-01, 1.01682438e-01, 1.01843948e-01, 1.01633115e-01, 1.00901236e-01, 9.98358573e-02, 9.88955330e-02, 9.84609290e-02, 9.86564627e-02, 9.93905467e-02, 1.00702112e-01, 1.02688741e-01, 1.04884850e-01, 1.06596212e-01, 1.07749986e-01, 1.08790050e-01, 1.09836551e-01, 1.10769706e-01, 1.11095663e-01, 1.10692963e-01, 1.10029619e-01, 1.09764827e-01, 1.11173215e-01, 1.14002423e-01, 1.16306811e-01, 1.17333081e-01, 1.17730599e-01, 1.18119195e-01, 1.18648924e-01, 1.19329854e-01, 1.20041645e-01, 1.21313276e-01, 1.23452370e-01, 1.25954037e-01, 1.28096271e-01, 1.31057358e-01, 1.34618596e-01, 1.37038287e-01, 1.38832348e-01, 1.41025846e-01, 1.43204590e-01, 1.44059482e-01, 1.43777316e-01, 1.43445494e-01, 1.42814402e-01, 1.40610214e-01, 1.37623491e-01, 1.34926587e-01, 1.32676737e-01, 1.31120683e-01, 1.29375548e-01, 1.26981139e-01, 1.24484884e-01, 1.22113373e-01, 1.20465612e-01, 1.18674716e-01, 1.16105299e-01, 1.12769762e-01, 1.09196050e-01, 1.06428325e-01, 1.04218154e-01, 1.01647336e-01, 9.93824248e-02, 9.66282251e-02, 9.53408887e-02, 9.51555308e-02, 9.37883676e-02, 8.99839263e-02, 8.22537415e-02, 7.15941407e-02, 6.12666842e-02, 5.72587622e-02, 5.94239102e-02, 6.36415897e-02, 6.56259224e-02, 6.29067058e-02, 5.88762482e-02, 5.54673429e-02, 5.38350884e-02, 5.18956121e-02, 5.03310306e-02, 5.00782019e-02, 4.59944171e-02, 4.67022736e-02, 4.88450582e-02, 5.36796759e-02, 5.97589987e-02, 6.27610502e-02, 6.03919910e-02, 5.56752449e-02, 5.20189009e-02, 5.21601180e-02, 5.19831498e-02, 5.17347917e-02, 5.32860290e-02, 5.29860438e-02, 4.93884267e-02, 4.96775368e-02, 5.35542492e-02, 5.77104625e-02, 5.74087796e-02, 5.88317079e-02, 5.88839463e-02, 5.84889128e-02, 6.02972329e-02, 6.27335682e-02, 6.30579383e-02, 6.39283372e-02, 6.27590042e-02, 6.50714457e-02, 6.80265647e-02, 6.91497904e-02, 6.81292818e-02, 7.02774829e-02, 7.36732149e-02, 7.32606923e-02, 6.70565973e-02, 5.58970960e-02, 4.77413273e-02, 4.54822105e-02, 4.79470658e-02, 5.00103927e-02, 4.94966679e-02, 4.91173761e-02, 4.97318604e-02, 5.03943690e-02, 5.15559131e-02, 5.29529987e-02, 5.39297644e-02, 5.38652745e-02, 5.20839362e-02, 4.98257088e-02, 4.84729509e-02, 4.87837492e-02, 5.02299926e-02, 5.13710286e-02, 5.28465055e-02, 5.30268129e-02, 5.17753751e-02, 5.13651915e-02, 5.23152540e-02, 5.32712807e-02, 5.36978733e-02, 5.35821826e-02, 5.17903011e-02, 4.95086084e-02, 4.61541405e-02, 4.35162365e-02, 4.24526015e-02, 4.21834889e-02, 4.25400491e-02, 4.37848079e-02, 4.59423997e-02, 4.93015341e-02, 5.47432521e-02, 6.02493939e-02, 6.57743298e-02, 7.16078889e-02, 7.70361599e-02, 8.19202327e-02, 8.71184429e-02, 9.28994593e-02, 9.83072085e-02, 1.02981481e-01, 1.06351721e-01, 1.07094025e-01, 1.07034493e-01, 1.07569943e-01, 1.09268547e-01, 1.10529982e-01, 1.09907515e-01, 1.08134398e-01, 1.05150116e-01, 1.01516538e-01, 9.85065768e-02, 9.73257063e-02, 9.75990422e-02, 9.72711786e-02, 9.56841296e-02, 9.23470220e-02, 8.81729167e-02, 8.56910586e-02, 8.55114809e-02, 8.56748665e-02, 8.17879602e-02, 7.61350894e-02, 6.81200965e-02, 6.00547918e-02, 5.67773683e-02, 6.01793765e-02, 6.55273779e-02, 7.05496664e-02, 7.35340983e-02, 7.28147188e-02, 6.86498801e-02, 6.66445345e-02, 6.71775772e-02, 6.74661651e-02, 6.48261584e-02, 6.23162045e-02, 5.89063838e-02, 5.42142736e-02, 5.22922109e-02, 5.43826896e-02, 5.57661238e-02, 5.64032583e-02, 5.71453593e-02, 5.83455564e-02, 5.95022952e-02, 6.24385951e-02, 6.36015386e-02, 6.25536863e-02, 6.13990571e-02, 5.78213077e-02, 4.95690306e-02, 3.90496228e-02, 3.12753415e-02, 2.73290396e-02, 2.58048123e-02, 2.57760874e-02, 2.69955404e-02, 2.87174790e-02, 3.01570181e-02, 2.97644498e-02, 2.97653077e-02])
    
    # Beta from (2020, 2, 24) to (2021, 5, 1) (Italy) (+initial padding of 20 days)
    #beta_data = np.load('data/fit/betas/betas_.npy')

    # Beta from (2020, 8, 15) to (2021, 5, 1) (Italy) without padding
    fit_directory = 'fit'
    beta_data = np.load(os.path.join(directory, fit_directory, 'beta_it.npy'))

    ## Temperature
    T_data = pd.DataFrame()
    for filename in os.listdir(T_DATA):
        if filename.startswith('TG'):
            with open(os.path.join(T_DATA, filename)) as file:
                for line in file:
                    if line == ' SOUID,    DATE,   TG, Q_TG\n':
                        columns=line
                        break
                df = pd.read_table(file, sep=',', names=list(map(lambda x: x.strip(), columns.split(','))))
                T_data = pd.concat([T_data, df])
    
    ## Mobility
    M_data = pd.read_csv(M_DATA, low_memory=False)        
    
    ## Infected
    SIR_data = None
    try:
        SIR_data = pd.read_csv(SIR_url)
    except HTTPError as e:
        print("Error while loading data.")
        raise e
    else:
        return P_data, SIR_data, beta_data, T_data, M_data


# ## ENCODE DATA

# Perform "one-hot"/"normalize" encoding of DataFrame labels
def encode_dataframe(dataframe, method='normalize'):
    for column in dataframe.columns:
        if method == 'one-hot':
            values = np.sort(dataframe[column].unique())
            values_dict = {k:v for v,k in zip(range(len(values)), values)}
            one_hot = np.identity(len(values))
            dataframe[column] = dataframe[column].apply(lambda x: one_hot[values_dict[x]])
        elif method == 'normalize':
            dataframe[column] = dataframe[column]/dataframe[column].max()
        else:
            print('Method not recognized.')

# Load P(olicies), I(nfected) and beta DataFrames (selecting on dates)
# plus T(emperature) and M(obility)
def load_data(data, start_date=datetime.date(2020, 2, 24), end_date=datetime.date(2021, 4, 22), split_date=datetime.date(2021, 2, 23), encoding_method='normalize', N=59583924, beta_padding=0):

    # Extract dataframes
    P_data, SIR_data, beta_data, T_data, M_data = data
    
    ## Policies
    # Preprocessing
    P_dataframe = P_data[P_data.CountryName=='Italy'].drop(['CountryName','CountryCode','RegionName', 'RegionCode','Jurisdiction'],
                                                           axis=1)
    P_dataframe = P_dataframe.drop(['E1_Income support', 'E1_Flag', 'E2_Debt/contract relief', 'E3_Fiscal measures', 'E4_International support', 'M1_Wildcard', 'ConfirmedCases', 'ConfirmedDeaths', 'StringencyIndex', 'StringencyIndexForDisplay', 'StringencyLegacyIndex', 'StringencyLegacyIndexForDisplay', 'GovernmentResponseIndex', 'GovernmentResponseIndexForDisplay', 'ContainmentHealthIndex', 'ContainmentHealthIndexForDisplay', 'EconomicSupportIndex','EconomicSupportIndexForDisplay'],
                               axis=1)
    # I don't wanna take Flags into account
    P_dataframe = P_dataframe.drop(['C1_Flag', 'C2_Flag', 'C3_Flag', 'C4_Flag', 'C5_Flag', 'C6_Flag', 'C7_Flag', 'H1_Flag',  'H6_Flag', 'H7_Flag', 'H8_Flag'],
                                   axis=1)
    #P_dataframe.loc[:, 'Date'] = P_dataframe.loc[:, 'Date'].apply(lambda x: dateparse(str(x), format='%Y%m%d')) # parse dates
    P_dataframe = P_dataframe.set_index(pd.DatetimeIndex(pd.to_datetime(P_dataframe['Date'], format='%Y%m%d').dt.date, name='date')).drop('Date', axis=1)
    # Select dates
    P_dataframe = P_dataframe.loc[start_date:end_date]
    # Encoding (normalize/one-hot)
    encode_dataframe(P_dataframe, encoding_method)
    #Drop useless columns (with just one value)
    for column in P_dataframe.columns:
        if (encoding_method == 'normalize' and len(P_dataframe[column].unique())==1) or (encoding_method == 'one-hot' and len(P_dataframe[column][0])==1):
            P_dataframe = P_dataframe.drop(column, axis=1)
    # Train-test split
    P_dataframe.loc[start_date:split_date, 'split'] = 'train'
    P_dataframe.loc[split_date+datetime.timedelta(1):, 'split'] = 'test'
    

    ## SIR
    # Keep only interesting columns
    SIR_dataframe = SIR_data.filter(['data', 'totale_positivi', 'dimessi_guariti', 'deceduti'])
    # Create S,I,R columns
    SIR_dataframe.rename(mapper={'totale_positivi':'I'}, axis=1, inplace=True)
    SIR_dataframe.loc[:, 'R'] = SIR_dataframe['dimessi_guariti'] + SIR_dataframe['deceduti']
    SIR_dataframe.loc[:, 'S'] = N - SIR_dataframe['I'] - SIR_dataframe['R']
    SIR_dataframe = SIR_dataframe.drop(['dimessi_guariti', 'deceduti'], axis=1)
    # Parse dates, set index
    SIR_dataframe = SIR_dataframe.set_index(pd.DatetimeIndex(pd.to_datetime(SIR_dataframe['data']).dt.date, name='date')).drop('data', axis=1)
    #SIR_dataframe = SIR_dataframe.set_index(SIR_dataframe.apply(lambda x: dateparse(x['data'], format='%Y-%m-%d'), axis=1)).drop('data', axis=1)
    # Normalize
    SIR_dataframe = (SIR_dataframe-SIR_dataframe.min())/(SIR_dataframe.max()-SIR_dataframe.min())
    ## Set 'lookback' column
    #I_dataframe['lookback'] = I_dataframe.iloc[lbdays:].apply(lambda x: I_dataframe.loc[x.name-datetime.timedelta(lbdays):x.name-datetime.timedelta(1), 'totale_positivi'].to_numpy(), axis=1)
    # Select dates
    SIR_dataframe = SIR_dataframe.loc[start_date:end_date]
    # Train-test split
    SIR_dataframe.loc[start_date:split_date, 'split'] = 'train'
    SIR_dataframe.loc[split_date+datetime.timedelta(1):, 'split'] = 'test'
    
    
    ## Beta
    # Preprocessing
    beta_dataframe = pd.DataFrame()
    beta_dataframe.loc[:, 'beta'] = beta_data[beta_padding:]
    beta_dataframe = beta_dataframe.set_index(pd.DatetimeIndex(pd.date_range(start=start_date, periods=len(beta_dataframe['beta']), freq='D'), name='date'))
    #beta_dataframe['data'] = [start_date+datetime.timedelta(i) for i in range(len(beta_data)-beta_padding)]
    #beta_dataframe = beta_dataframe.set_index('data')
    # Normalize
    #beta_dataframe['beta'] = (beta_dataframe['beta']-beta_dataframe['beta'].min())/(beta_dataframe['beta'].max()-beta_dataframe['beta'].min())
    # Select dates
    beta_dataframe = beta_dataframe.loc[start_date:end_date]
    # Train-test split
    beta_dataframe.loc[start_date:split_date, 'split'] = 'train'
    beta_dataframe.loc[split_date+datetime.timedelta(1):, 'split'] = 'test'
    
    
    ## Temperature
    # Preprocessing
    T_df = T_data[T_data['Q_TG'] == 0].drop('Q_TG', axis=1) # Keep only valid data
    T_df['DATE'] = pd.to_datetime(T_df['DATE'], format='%Y%m%d') # parse dates
    T_df = T_df.set_index('DATE').sort_index() # sort dates
    T_df = T_df.loc[datetime.date(2006,1,1):] # keep only recent data
    T_df['DAY'] = T_df.index.day
    T_df['MONTH'] = T_df.index.month
    T_df['YEAR'] = T_df.index.year
    T_df = T_df.set_index(['YEAR', 'MONTH', 'DAY']) # to average over years
    # Create new dataframe
    T_dataframe = pd.DataFrame()
    T_dataframe['date'] = pd.date_range(start='1/1/2020', end='31/12/2021', freq='D') # index
    T_dataframe['temperature'] = T_df.groupby(['MONTH', 'DAY'])['TG'].mean().reset_index().drop(['MONTH', 'DAY'], axis=1)/10 # average over years, divide by 10
    T_dataframe = T_dataframe.set_index('date')
    T_dataframe.loc[datetime.date(2021,1,1):, 'temperature'] = T_dataframe.loc[datetime.date(2021,1,1):].apply(lambda x: T_dataframe.loc[[datetime.date(2020, x.name.month, x.name.day)],'temperature'].item(), axis=1) # copy data in both 2020 and 2021
    T_dataframe['mov-avg'] = T_dataframe.loc[:, 'temperature'].rolling(10).mean().fillna(method='ffill').fillna(method='bfill') # smooth by computing moving average
    # Normalize
    T_dataframe = (T_dataframe-T_dataframe.min())/(T_dataframe.max()-T_dataframe.min())
    # Select dates
    T_dataframe = T_dataframe.loc[start_date:end_date].filter(['mov-avg'])
    # Train-test split
    T_dataframe.loc[start_date:split_date, 'split'] = 'train'
    T_dataframe.loc[split_date+datetime.timedelta(1):, 'split'] = 'test'
    
    
    ## Mobility
    # Preprocessing
    M_dataframe = M_data[M_data['region']=='Italy'].drop(['geo_type',
                                                          'region',
                                                          'alternative_name',
                                                          'sub-region',
                                                          'country'],
                                                         axis=1)

    M_dataframe = M_dataframe.set_index('transportation_type').transpose() # to compute mean
    M_dataframe = M_dataframe.interpolate() # fill NaN
    M_dataframe['mov-avg'] = pd.Series(M_dataframe.mean(axis=1).rolling(5).mean().fillna(method='bfill')) # smooth by computing moving average
    M_dataframe = M_dataframe.set_index(pd.to_datetime(M_dataframe.index).rename('date'))
    M_dataframe.columns.name=None # Remove 'transporation_type'
    #M_dataframe = M_dataframe.set_index(M_dataframe.apply(lambda x: dateparse(x.name, format='%Y-%m-%d'), axis=1)) # parse dates
    #Normalize
    M_dataframe = (M_dataframe-M_dataframe.min())/(M_dataframe.max()-M_dataframe.min())
    # Select dates
    M_dataframe = M_dataframe.loc[start_date:end_date].filter(['mov-avg'])
    # Train-test split
    M_dataframe.loc[start_date:split_date, 'split'] = 'train'
    M_dataframe.loc[split_date+datetime.timedelta(1):, 'split'] = 'test'
    
    return P_dataframe, SIR_dataframe, beta_dataframe, T_dataframe, M_dataframe


# Courtesy of prof. Lombardi
def sliding_window_1D(data, wlen):
    m = len(data)
    lc = [data.filter(['I']).iloc[i:m-wlen+i+1] for i in range(0, wlen)]
    wdata = np.hstack(lc)
    wdata = pd.DataFrame(index=data.index[wlen-1:], data=wdata, columns=range(wlen))
    wdata['split']=data['split']
    return wdata

def apply_lbdays(P, SIR, beta, T, M, start_date, lbdays=0):
    I = SIR.filter(['I', 'split'])
    # Substitute 'nan' for first lbdays values (not large enough time window)
    if lbdays:
        for df in [P, beta, T, M]:
            df.loc[:start_date+datetime.timedelta(lbdays-1), 'split'] = np.nan

        I = sliding_window_1D(I, wlen=lbdays+1)

    # Restore values whenever 'apply_lbdays' had already been executed before
    else:
        for df in [P, beta, T, M]:
            df.fillna(value='train', inplace=True)
    return I



## TO CHECK LBDAYS ?
#for i in range(len(I)):
#    for j in range(22):
#        print(j,i)
#        assert np.equal(I[I['split']=='train'].iloc[j, i:22], I[I['split']=='train'].iloc[i:22, j].values).all()


## OLD

## Load P(olicies), I(nfected) and beta DataFrames (selecting on dates)
## plus T(emperature) and M(obility)
#def load_data(data, start_date=datetime.date(2020, 2, 24), end_date=datetime.date(2021, 4, 22), N=59583924, lbdays=0, split_date=datetime.date(2021, 2, 23), encoding_method = 'normalize', beta_padding=0):
#
#    # Extract dataframes
#    P_data, SIR_data, beta_data, T_data, M_data = data
#    
#    ## Policies
#    # Preprocessing
#    P_dataframe = P_data[P_data.CountryName=='Italy'].drop(['CountryName','CountryCode','RegionName', 'RegionCode','Jurisdiction'],
#                                                           axis=1)
#    P_dataframe = P_dataframe.drop(['E1_Income support', 'E1_Flag', 'E2_Debt/contract relief', 'E3_Fiscal measures', 'E4_International support', 'M1_Wildcard', 'ConfirmedCases', 'ConfirmedDeaths', 'StringencyIndex', 'StringencyIndexForDisplay', 'StringencyLegacyIndex', 'StringencyLegacyIndexForDisplay', 'GovernmentResponseIndex', 'GovernmentResponseIndexForDisplay', 'ContainmentHealthIndex', 'ContainmentHealthIndexForDisplay', 'EconomicSupportIndex','EconomicSupportIndexForDisplay'],
#                               axis=1)
#    # I don't wanna take Flags into account
#    P_dataframe = P_dataframe.drop(['C1_Flag', 'C2_Flag', 'C3_Flag', 'C4_Flag', 'C5_Flag', 'C6_Flag', 'C7_Flag', 'H1_Flag',  'H6_Flag', 'H7_Flag', 'H8_Flag'],
#                                   axis=1)
#    P_dataframe.loc[:, 'Date'] = P_dataframe.loc[:, 'Date'].apply(lambda x: dateparse(str(x), format='%Y%m%d')) # parse dates
#    P_dataframe = P_dataframe.set_index('Date')   
#    # Select dates
#    P_dataframe = P_dataframe.loc[start_date:end_date]
#    # Encoding (normalize/one-hot)
#    encode_dataframe(P_dataframe, encoding_method)
#    #Drop useless columns (with just one value)
#    for column in P_dataframe.columns:
#        if (encoding_method == 'normalize' and len(P_dataframe[column].unique())==1) or (encoding_method == 'one-hot' and len(P_dataframe[column][0])==1):
#            P_dataframe = P_dataframe.drop(column, axis=1)
#    # Train-test split
#    P_dataframe.loc[start_date+datetime.timedelta(lbdays):split_date, 'split'] = 'train'
#    P_dataframe.loc[split_date+datetime.timedelta(1):, 'split'] = 'test'
#    
#    
#    I_dataframe=None
#    ## Infected
#    if lbdays:
#        # Preprocessing
#        I_dataframe = SIR_data[['data', 'totale_positivi']]#, 'dimessi_guariti', 'deceduti']]
#        # Parse dates
#        I_dataframe = I_dataframe.assign(data=I_dataframe.apply(lambda x: dateparse(x['data'], format='%Y-%m-%d'), axis=1))
#        #I_dataframe.loc[:, 'data'] = I_dataframe.apply(lambda x: dateparse(x['data'], format='%Y-%m-%d'), axis=1)
#        I_dataframe = I_dataframe.set_index('data')
#        # Normalize
#        I_dataframe.loc[:, 'totale_positivi'] = I_dataframe.loc[:, 'totale_positivi']/I_dataframe.loc[:, 'totale_positivi'].max()
#        # Set 'lookback' column
#        I_dataframe['lookback'] = I_dataframe.iloc[lbdays:].apply(lambda x: I_dataframe.loc[x.name-datetime.timedelta(lbdays):x.name-datetime.timedelta(1), 'totale_positivi'].to_numpy(), axis=1)
#        # Select dates
#        I_dataframe = I_dataframe.loc[start_date:end_date]
#        # Train-test split
#        I_dataframe.loc[start_date+datetime.timedelta(lbdays):split_date, 'split'] = 'train'
#        I_dataframe.loc[split_date+datetime.timedelta(1):, 'split'] = 'test'
#    
#    
#    ## Beta
#    # Preprocessing
#    beta_dataframe = pd.DataFrame()
#    beta_dataframe['data'] = [start_date+datetime.timedelta(i) for i in range(len(beta_data)-beta_padding)]
#    beta_dataframe['beta'] = beta_data[beta_padding:]
#    # Normalize
#    #beta_dataframe['beta'] = (beta_dataframe['beta']-beta_dataframe['beta'].min())/(beta_dataframe['beta'].max()-beta_dataframe['beta'].min())
#    beta_dataframe = beta_dataframe.set_index('data')
#    # Select dates
#    beta_dataframe = beta_dataframe.loc[start_date:end_date]
#    # Train-test split
#    beta_dataframe.loc[start_date+datetime.timedelta(lbdays):split_date, 'split'] = 'train'
#    beta_dataframe.loc[split_date+datetime.timedelta(1):, 'split'] = 'test'
#    
#    
#    ## Temperature
#    # Preprocessing
#    T_df = T_data[T_data['Q_TG'] == 0].drop('Q_TG', axis=1) # Keep only valid data
#    T_df['DATE'] = pd.to_datetime(T_df['DATE'], format='%Y%m%d') # parse dates
#    T_df = T_df.set_index('DATE').sort_index() # sort dates
#    T_df = T_df.loc[datetime.date(2006,1,1):] # keep only recent data
#    T_df['DAY'] = T_df.index.day
#    T_df['MONTH'] = T_df.index.month
#    T_df['YEAR'] = T_df.index.year
#    T_df = T_df.set_index(['YEAR', 'MONTH', 'DAY']) # to average over years
#    # Create new dataframe
#    T_dataframe = pd.DataFrame()
#    T_dataframe['date'] = pd.date_range(start='1/1/2020', end='31/12/2021', freq='D') # index
#    T_dataframe['temperature'] = T_df.groupby(['MONTH', 'DAY'])['TG'].mean().reset_index().drop(['MONTH', 'DAY'], axis=1)/10 # average over years, divide by 10
#    T_dataframe = T_dataframe.set_index('date')
#    T_dataframe.loc[datetime.date(2021,1,1):, 'temperature'] = T_dataframe.loc[datetime.date(2021,1,1):].apply(lambda x: T_dataframe.loc[[datetime.date(2020, x.name.month, x.name.day)],'temperature'].item(), axis=1) # copy data in both 2020 and 2021
#    T_dataframe['mov-avg'] = T_dataframe.loc[:, 'temperature'].rolling(10).mean().fillna(method='ffill').fillna(method='bfill') # smooth by computing moving average
#    # Normalize
#    T_dataframe = (T_dataframe-T_dataframe.min())/(T_dataframe.max()-T_dataframe.min())
#    # Select dates
#    T_dataframe = T_dataframe.loc[start_date:end_date][['mov-avg']]
#    # Train-test split
#    T_dataframe.loc[start_date+datetime.timedelta(lbdays):split_date, 'split'] = 'train'
#    T_dataframe.loc[split_date+datetime.timedelta(1):, 'split'] = 'test'
#    
#    
#    ## Mobility
#    # Preprocessing
#    M_dataframe = M_data[M_data['region']=='Italy'].drop(['geo_type',
#                                                          'region',
#                                                          'alternative_name',
#                                                          'sub-region',
#                                                          'country'],
#                                                         axis=1)
#
#    M_dataframe = M_dataframe.set_index('transportation_type').transpose() # to compute mean
#    M_dataframe = M_dataframe.interpolate() # fill NaN
#    M_dataframe['mov-avg'] = pd.Series(M_dataframe.mean(axis=1).rolling(5).mean().fillna(method='ffill')).fillna(method='bfill') # smooth by computing moving average
#    M_dataframe = M_dataframe.set_index(M_dataframe.apply(lambda x: dateparse(x.name, format='%Y-%m-%d'), axis=1)) # parse dates
#    #Normalize
#    M_dataframe = (M_dataframe-M_dataframe.min())/(M_dataframe.max()-M_dataframe.min())
#    # Select dates
#    M_dataframe = M_dataframe.loc[start_date:end_date][['mov-avg']]
#    # Train-test split
#    M_dataframe.loc[start_date+datetime.timedelta(lbdays):split_date, 'split'] = 'train'
#    M_dataframe.loc[split_date+datetime.timedelta(1):, 'split'] = 'test'
#    
#    return P_dataframe, I_dataframe, beta_dataframe, T_dataframe, M_dataframe


# ## SPLIT DATA

# Load train and test splits
def split_data(P, I, beta, T, M, split, lbdays=0):
    
    # Policies
    P_split = P.loc[P['split']==split].drop('split', axis=1)
    # One vector for each day
    #P_split = P_split.apply(lambda x: np.hstack(x), axis=1)
    P_split = np.vstack(P_split.values)

    # Infected
    I_split = I.loc[I['split']==split].drop('split', axis=1)
    I_split = np.vstack(I_split.values)
    
    # Beta
    Y_split = beta.loc[beta['split']==split].drop('split', axis=1)
    Y_split = np.hstack(Y_split.values)

    # Temperature
    T_split = T.loc[T['split']==split].drop('split', axis=1)
    T_split = np.hstack(T_split.values).reshape(-1, 1)
    
    # Mobility
    M_split = M.loc[M['split']==split].drop('split', axis=1)
    M_split = np.hstack(M_split.values).reshape(-1, 1)
    

    
    # Check shapes
    print('Shapes')
    print('Policies:', P_split.shape)
    print('Infected:', I_split.shape)
    print('Beta:', Y_split.shape)
    print('Temperature:', T_split.shape)
    print('Mobility:', M_split.shape)
    assert(P_split.shape[0]==I_split.shape[0])
    assert(P_split.shape[0]==Y_split.shape[0])
    assert(P_split.shape[0]==T_split.shape[0])
    assert(P_split.shape[0]==M_split.shape[0])


    #X_split = np.hstack([P_split, T_split, M_split, I_split])


    # OLD
    ## Lookback
    #if lbdays:
    #    I_split = I.loc[I['split']==split].drop('split', axis=1)
    #    I_split = np.vstack(I_split['lookback'])
    #    print('Lookback days:', I_split.shape)
    #    assert(P_split.shape[0]==I_split.shape[0])
    
    # Infected
    X_split = np.hstack([P_split, I_split])# if lbdays else P_split
    
    # Temperature, mobility
    X_split = np.hstack([X_split, T_split, M_split])

    return X_split, Y_split


def convert_to_tensor(*args):
    return [tf.convert_to_tensor(arg, dtype='float64') for arg in args]



