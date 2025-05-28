# -*- coding: utf-8 -*-
"""
Created on Mon May 12 09:45:02 2025

@author: Tanja Liesch
"""

#%% paths and packages

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
from numpy.random import seed
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import keras as ks
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten
from keras.models import Model


pth_dt_dyn = "./data/dynamic"
pth_out = "./results_single/"   


#%% functions

def split_data(data, seq_length, start, val_start, test_start, end):
    Train = data[(data.index >= start) & (data.index <= val_start)]
    
    Val = data[(data.index >= val_start) & (data.index <= test_start)]
    Val_ext = pd.concat([Val.iloc[-seq_length:], Val], axis=0) 
    
    Test = data[(data.index >= test_start) & (data.index <= end)]
    Test_ext = pd.concat([Val.iloc[-seq_length:], Test], axis=0) 
    return  Train, Val, Val_ext, Test, Test_ext

def scale_data(Train, Val, Val_ext,Test, Test_ext): 
    scaler = StandardScaler().fit(Train)
    
    Train_n = pd.DataFrame(scaler.transform(Train), index=Train.index, columns=Train.columns)   
    Val_n = pd.DataFrame(scaler.transform(Val), index=Val.index, columns=Val.columns)
    Val_ext_n = pd.DataFrame(scaler.transform(Val_ext), index=Val_ext.index, columns=Val_ext.columns)
    Test_n = pd.DataFrame(scaler.transform(Test), index=Test.index, columns=Test.columns) 
    Test_ext_n = pd.DataFrame(scaler.transform(Test_ext), index=Test_ext.index, columns=Test_ext.columns)

    scaler_gwl = StandardScaler().fit(pd.DataFrame(Train.iloc[:,15]))         
       

    return Train_n, Val_n, Val_ext_n, Test_n, Test_ext_n, scaler_gwl

def sequence(data, seq_length, n_input):
    X, Y = list(), list()
    # step over the entire history one time step at a time
    for i in range(len(data)):
        # find the end of this pattern
        end_idx = i + seq_length
        # check if we are beyond the dataset
        if end_idx >= len(data):
            break
        # gather input and output parts of the pattern
        # seq_x, seq_y = data[i:end_idx, 1:], data[end_idx, 0]
        # X.append(seq_x)
        # Y.append(seq_y)
        seq_x = data[i:end_idx, :n_input]
        seq_y = data[end_idx, n_input:]
        X.append(seq_x)
        Y.append(seq_y)
        
    return np.array(X), np.array(Y)

def build_model(ini, seq_length, SETTINGS, inp):   
    #seed
    seed(42+ini)
    tf.random.set_seed(42+ini)
    
#################### Layer Structure    
    inp = Input(shape=(seq_length, inp.shape[2]), name='input')               
    CNN = Conv1D(filters=SETTINGS["filters"], kernel_size=SETTINGS["kernel_size"],
                 padding='same', activation='relu', name='CNN_1')(inp)       
    Pool = MaxPooling1D(pool_size=2, padding='same', name='max_pool')(CNN)
    Flat = Flatten(name='flatten')(Pool)   
    dense = Dense(SETTINGS["dense_size_cnn"], activation='relu', name='dense_gwl')(Flat)    
    out = Dense(1, activation='linear')(dense)
    
#################### Define Model 
    model = Model(inputs=inp, outputs=out)

#################### Optimizer     
    #optimizer = ks.optimizers.Adam(learning_rate=SETTINGS["learning_rate"], epsilon=10E-3, clipnorm=SETTINGS["clip_norm"], clipvalue=SETTINGS["clip_value"])    
    optimizer = ks.optimizers.Adam(learning_rate=SETTINGS["learning_rate"], epsilon=10E-3)    

#################### Compile    
    model.compile(loss='mse', optimizer=optimizer, metrics=[ks.metrics.MeanAbsoluteError()])            
    return model


        
        
#%% load time series data

#-----------------------------------

# list dynamic time series data files
dt_list_files = os.listdir(pth_dt_dyn)
temp = [i for i,sublist in enumerate(dt_list_files) if '.csv' in sublist]
dt_list_files = [dt_list_files[i] for i in temp]
del temp

# load dynamic time series
dt_list_dyn = list()
for i in range(len(dt_list_files)):
    temp = pd.read_csv(pth_dt_dyn + "/" + dt_list_files[i], 
                       parse_dates=[0], index_col=0, dayfirst = True, decimal = '.', sep=',')
    dt_list_dyn.append(temp)
del temp

# get ID names
dt_list_names = [item[:-4] for item in dt_list_files]

#-----------------------------------


# save IDs of remaining stations
pd.DataFrame({"ID": dt_list_names}).to_csv("./data/IDremainig.csv", sep = ";")


#-----------------------------------



#%% Model

all_scores_list = []


for ID in dt_list_names:
    
    result_file = os.path.join(pth_out, f"{ID}_obs_sim.csv")  # Updated filename format
    if os.path.exists(result_file):
        print(f"Skipping {ID}, already in results.")
        continue  # Skip this ID if it already exists

    seq_length = 52
    
    start = pd.to_datetime('07011991', format='%d%m%Y')
    val_start = pd.to_datetime('01012008', format='%d%m%Y') 
    test_start = pd.to_datetime('01012013', format='%d%m%Y')
    end = pd.to_datetime('31122022', format='%d%m%Y')
    
    
    setup = {
        'batch_size': 16,            
        'epochs': 30, 
        'learning_rate': 1e-3,                  
        'filters': 256, 
        'kernel_size': 3,           
        'dense_size_cnn': 32,
        }
    
    
    data = pd.read_csv(pth_dt_dyn + "/" + ID + '.csv', 
                       parse_dates=[0], index_col=0, dayfirst = True, decimal = '.', sep=',')
    
    data.reset_index(inplace=True)
    data.rename(columns={"index": "Date"}, inplace=True)
    data["Date"] = pd.to_datetime(data["Date"])
    data.set_index('Date', inplace=True)
    
    data.drop(['GWL_flag'], axis=1, inplace=True)
    data = data[[col for col in data.columns if col != 'GWL'] + ['GWL']]
        

    print(f"Processing ID: {ID}")
        
    Train, Val, Val_ext, Test, Test_ext = split_data(data, seq_length, start, val_start, test_start, end)
    Train_n, Val_n, Val_ext_n, Test_n, Test_ext_n, scaler_gwl = scale_data(Train, Val, Val_ext,Test, Test_ext)
    
    x_train, y_train = sequence(np.asarray(Train_n), seq_length, n_input = 15)
    x_val, y_val = sequence(np.asarray(Val_ext_n), seq_length, n_input = 15)
    x_test, y_test = sequence(np.asarray(Test_ext_n), seq_length, n_input = 15)    

    
    seeds = [1,52,123,2457,5321,16284,29752,39982,47665,59123]
    
 
    members = np.zeros((len(y_test), len(seeds)))
    for s in range(len(seeds)):
        ini = seeds[s]
        model = build_model(ini, seq_length, setup, x_train)
        
        es = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                min_delta=0,
                patience=5,
                verbose=1,
                mode='auto',
                baseline=None,
                restore_best_weights=True,
            )

        history = model.fit(
              x_train,y_train,
              validation_data=(x_val,y_val),
              epochs=setup["epochs"],
              verbose=2,
              batch_size=setup["batch_size"], 
              callbacks=[es])
        
        test_sim_n = model.predict(x_test)    
        test_sim = scaler_gwl.inverse_transform(test_sim_n)
        members[:, s] = test_sim[:, 0]

    
    
    ensemble_sim = pd.DataFrame(members, index=Test.index)    
    sim_obs = pd.concat([Test['GWL'], ensemble_sim], axis=1)
    sim_obs.to_csv(f'./results_single/{ID}_obs_sim.csv', sep=';', decimal='.')

