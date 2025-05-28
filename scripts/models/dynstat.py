# -*- coding: utf-8 -*-
"""
@author: Tanja Liesch
"""

#%% paths and packages

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import tensorflow as tf
#from keras import backend as K
from keras.callbacks import EarlyStopping
from scipy import stats
from datetime import datetime
#import matplotlib.pyplot as plt # auskommentiert für Cluster
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import copy
import tensorflow.keras.backend as K
import gc


pth_dt_dyn = "./data/dynamic"
pth_out = "./results_dynstat/"   


#%% functions

# ----- sequentialize  --------------------

def make_sequences(data, n_steps_in, n_input):
    #make the data sequential
    #modified after Jason Brownlee and machinelearningmastery.com
    X, Y = list(), list()
    # step over the entire history one time step at a time
    for i in range(len(data)):
        # find the end of this pattern
        end_idx = i + n_steps_in
        # check if we are beyond the dataset
        if end_idx >= len(data):
            break
        # gather input and output parts of the pattern
        seq_x = data[i:end_idx, :n_input]
        seq_y = data[end_idx, n_input:]
        X.append(seq_x)
        Y.append(seq_y)
    return np.array(X), np.array(Y)
    
# ----- learning rate scheduling  --------------------

class CustomLearningRateScheduler(tf.keras.callbacks.Callback):
    """Learning rate scheduler implementing linear warmup and cosine decay."""

    def __init__(self, warmup_steps, total_steps, target_lr=0.001, start_lr=0.0, hold=0):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.target_lr = target_lr
        self.start_lr = start_lr
        self.hold = hold
        self.global_step = 0  # Initialize step counter

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "learning_rate"):
            raise ValueError('Optimizer must have a "learning_rate" attribute.')
        
        # Compute new learning rate
        new_lr = self.lr_warmup_cosine_decay(self.global_step)
        self.model.optimizer.learning_rate.assign(new_lr)  # Assign new LR
        
        print(f"\nEpoch {epoch}: Learning rate is {float(new_lr)}")
        self.global_step += 1  # Increment step counter

    def lr_warmup_cosine_decay(self, global_step):
        """Computes learning rate with linear warmup and cosine decay."""
        if global_step >= self.total_steps:
            return self.target_lr  # Ensure no overshooting beyond total steps

        # Cosine decay
        learning_rate = 0.5 * self.target_lr * (1 + np.cos(np.pi * (global_step - self.warmup_steps - self.hold) / float(self.total_steps - self.warmup_steps - self.hold)))

        # Linear warmup
        warmup_lr = self.target_lr * (global_step / self.warmup_steps)

        # Apply warmup, hold, and decay logic
        if self.hold > 0:
            learning_rate = np.where(global_step > self.warmup_steps + self.hold, learning_rate, self.target_lr)

        learning_rate = np.where(global_step < self.warmup_steps, warmup_lr, learning_rate)
        return learning_rate
    


# ----- clear memory  --------------------

class ClearMemory(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        K.clear_session()
        gc.collect()
        
        
# ----- data generator  --------------------
        
from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    def __init__(self, X_dyn, X_stat, y, batch_size, shuffle=True):
        self.X_dyn = X_dyn
        self.X_stat = X_stat
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(y))
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.y) / self.batch_size))
    
    def __getitem__(self, index):
        # Generate batch indices
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Generate data
        X_dyn_batch = self.X_dyn[batch_indices]
        X_stat_batch = self.X_stat[batch_indices]
        y_batch = self.y[batch_indices]

        return (X_dyn_batch, X_stat_batch), y_batch

    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

        
# ----- BatchProgressCallback  --------------------

from keras.callbacks import Callback

class BatchProgressCallback(Callback):
    def on_train_batch_end(self, batch, logs=None):
        print(f"Batch {batch + 1} processed. Loss: {logs.get('loss'):.4f}")

    def on_epoch_begin(self, epoch, logs=None):
        print(f"\nStarting Epoch {epoch + 1}...\n")
        


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
#%% load static features
    
# static features
dt_static_features = pd.read_csv('./data/static/metadata_static.csv', sep=',', index_col=[0])
dt_static_features.isna().sum()


dt_static_features.drop(['Proj_ID','Operator', 'Depth', 'UpFilter', 'LoFilter',
                         'ScrLength', 'Pumping', 'PreState',
                          'Easting (EPSG:3035)', 'Northing (EPSG:3035)'], axis=1, inplace=True)


dt_static_features.set_index('MW_ID', inplace = True)
# Define LabelEncoder for categorical data
labelenc = LabelEncoder()

# Encode "AquiferMed" categories
x1 = pd.DataFrame(labelenc.fit_transform(dt_static_features[["AquiferMed"]]))
x1.columns = ["AquiferMed_" + str(item) for item in np.arange(0,len(x1.columns))]
x1.index = dt_static_features.index

# Encode "HUEK250_HU" categories
x2 = pd.DataFrame(labelenc.fit_transform(dt_static_features[["HUEK250_HU"]]))
x2.columns = ["HUEK250_HU_" + str(item) for item in np.arange(0,len(x2.columns))]
x2.index = dt_static_features.index

# Encode "HUEK250_RT" categories
x3 = pd.DataFrame(labelenc.fit_transform(dt_static_features[["HUEK250_RT"]]))
x3.columns = ["HUEK250_RT_" + str(item) for item in np.arange(0,len(x3.columns))]
x3.index = dt_static_features.index

# Encode "HUEK250_CT" categories
x4 = pd.DataFrame(labelenc.fit_transform(dt_static_features[["HUEK250_CT"]]))
x4.columns = ["HUEK250_CT_" + str(item) for item in np.arange(0,len(x4.columns))]
x4.index = dt_static_features.index

# Encode "HUEK250_DC" categories
x5 = pd.DataFrame(labelenc.fit_transform(dt_static_features[["HUEK250_DC"]]))
x5.columns = ["HUEK250_DC_" + str(item) for item in np.arange(0,len(x5.columns))]
x5.index = dt_static_features.index

# Encode "HUEK250_GC" categories
x6 = pd.DataFrame(labelenc.fit_transform(dt_static_features[["HUEK250_GC"]]))
x6.columns = ["huek250_GC_" + str(item) for item in np.arange(0,len(x6.columns))]
x6.index = dt_static_features.index

# Encode "HUMUS1000_OC" categories
x7 = pd.DataFrame(labelenc.fit_transform(dt_static_features[["HUMUS1000_OC"]]))
x7.columns = ["HUMUS1000_OC_" + str(item) for item in np.arange(0,len(x7.columns))]
x7.index = dt_static_features.index

# Drop unencoded categorical data
dt_static_features_dropped = dt_static_features.drop(["AquiferMed","HUEK250_HU",
                                                      "HUEK250_RT", "HUEK250_CT",
                                                      "HUEK250_DC","HUEK250_GC","HUMUS1000_OC"
                                                      ], axis = 1)

# Add encoded categorical data
dt_static_features_dropped = np.concatenate([dt_static_features_dropped,x1], axis = 1)
dt_static_features_dropped = np.concatenate([dt_static_features_dropped,x2], axis = 1)
dt_static_features_dropped = np.concatenate([dt_static_features_dropped,x3], axis = 1)
dt_static_features_dropped = np.concatenate([dt_static_features_dropped,x4], axis = 1)
dt_static_features_dropped = np.concatenate([dt_static_features_dropped,x5], axis = 1)
dt_static_features_dropped = np.concatenate([dt_static_features_dropped,x6], axis = 1)
dt_static_features_dropped = np.concatenate([dt_static_features_dropped,x7], axis = 1)



# Scaling
#scaler_static = StandardScaler()
scaler_static = MinMaxScaler(feature_range = (-1, 1))

scaler_static.fit(dt_static_features_dropped)
dt_static_features_n = scaler_static.transform(dt_static_features_dropped)
    

#-----------------------------------

#%% Model - Prepare Inputs and Hyperparameters

HP_seeds = [999, 206, 380, 471, 570, 624, 643, 778, 808, 973]

# Hyperparameters
n_steps_in = 52
HPi_lstm_size = 128
HPi_static_size =  128
HPi_comb_size =  256
HPi_dropout =  0.3
HPi_targetlr =  0.001
HPi_epochs = 20
HPi_batchsize = 512

# ii=0
for ii in range(len(HP_seeds)):
    
    HPi_seed = HP_seeds[ii]
    
    # create folder for outputs
    pth_out_i = pth_out+'/run'+str(HPi_seed)
    os.mkdir(pth_out_i)
    
    
    #% Data preprocessing
    
    #Initialize containers
    IDlen = list()
    datafull = list()
    scalers = list()
    scalers_y = list()
    X_train,Y_train = list(), list()
    X_stop,Y_stop = list(), list()
    X_test,Y_test = list(), list()
    X_train_stat, X_stop_stat, X_test_stat = list(),list(),list()
    
    # Define split dates
    date_start_stop = pd.to_datetime("2008-01-01", format = "%Y-%m-%d")
    date_start_test = pd.to_datetime("2013-01-01", format = "%Y-%m-%d")
    
    for i in range(len(dt_list_files)):
        
        # merge all input data
        tempdata = copy.deepcopy(dt_list_dyn[i])
        
        tempdata["ID"] = i
        tempdata.reset_index(inplace=True)
        tempdata.rename(columns={"index": "Date"}, inplace=True)
        tempdata["Date"] = pd.to_datetime(tempdata["Date"])
        tempdata.set_index('Date', inplace=True)
        
        tempdata.drop(['GWL_flag'], axis=1, inplace=True)
        tempdata = tempdata[[col for col in tempdata.columns if col != 'GWL'] + ['GWL']]
        
        # save original data for score calculations later on 
        datafull.append(tempdata)
        
        # fit scalers (on train+stop data) and transform (on full) data
        scalers.append(StandardScaler().fit(tempdata[(tempdata.index < date_start_test)]))
        scalers_y.append(StandardScaler().fit(pd.DataFrame(tempdata[(tempdata.index < date_start_test)]["GWL"])))
        tempdata_n = scalers[i].transform(tempdata)
        
        # Split data
        tempdata_n_train = tempdata_n[(tempdata.index < date_start_stop)]
        tempdata_n_stop = tempdata_n[(tempdata.index >= date_start_stop) & (tempdata.index < date_start_test)]
        tempdata_n_test = tempdata_n[(tempdata.index >= date_start_test)] 
        
        # ID tracker: Save length of test set to identify individual Mst after modelfit
        IDlen.append(np.repeat(i,len(tempdata_n_test)))
        
        # extend stop + Testdata to be able to fill sequence later
        tempdata_n_stop_ext = np.concatenate([tempdata_n_train[-n_steps_in:], tempdata_n_stop], axis=0)
        tempdata_n_test_ext = np.concatenate([tempdata_n_stop[-n_steps_in:], tempdata_n_test], axis=0)
        
        # sequentialize data and add static features
        temp_x, temp_y = make_sequences(np.asarray(tempdata_n_train), n_steps_in, n_input = 15)
        X_train.append(temp_x); Y_train.append(temp_y[:,1])
        X_train_stat.append(np.repeat(dt_static_features_n[dt_static_features.index == dt_list_names[i]], 
                                      len(temp_x), axis = 0))
        
        temp_x, temp_y = make_sequences(np.asarray(tempdata_n_stop_ext), n_steps_in, n_input = 15)
        X_stop.append(temp_x); Y_stop.append(temp_y[:,1])
        X_stop_stat.append(np.repeat(dt_static_features_n[dt_static_features.index == dt_list_names[i]], 
                                     len(temp_x), axis = 0))
        
        temp_x, temp_y = make_sequences(np.asarray(tempdata_n_test_ext), n_steps_in, n_input = 15)
        X_test.append(temp_x); Y_test.append(temp_y[:,1])
        X_test_stat.append(np.repeat(dt_static_features_n[dt_static_features.index == dt_list_names[i]], 
                                     len(temp_x), axis = 0))

    
    # Final merge
    X_train = np.concatenate(X_train)
    X_train_stat = np.concatenate(X_train_stat)
    Y_train = np.concatenate(Y_train)
    X_stop = np.concatenate(X_stop)
    X_stop_stat = np.concatenate(X_stop_stat)
    Y_stop = np.concatenate(Y_stop)
    X_test = np.concatenate(X_test)
    X_test_stat = np.concatenate(X_test_stat)
    Y_test = np.concatenate(Y_test)
    IDlen = np.concatenate(IDlen)
    datafull = pd.concat(datafull)
    datafullwide = pd.pivot(datafull.drop(['HYRAS_pr','HYRAS_tas','HYRAS_tasmax',
                                           'HYRAS_tasmin','HYRAS_hurs', 'DWD_evapo_p',
                                           'DWD_evapo_r','DWD_evapo_fao','DWD_soil_moist',
                                           'DWD_soil_temp5cm','ERA5_sro','ERA5_ssro',
                                           'ERA5_sdwe', 'ERA5_sm','ERA5_sf'
                                           ], axis = 1), columns = "ID")
    
    datafullwide.to_csv(pth_out+'/datafullwide.csv', float_format='%.4f', sep = ";")
    
    #% Modelling
    
    #-----------------------------    
    # Model
    #-----------------------------
    
    #take time
    now1 = datetime.now()
    
    # set seed
    np.random.seed(HPi_seed)
    tf.random.set_seed(HPi_seed)
    
    # Callbacks
    model_es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5, restore_best_weights=True)
    model_mc = tf.keras.callbacks.ModelCheckpoint(filepath=pth_out_i+'/model.keras', 
                                                  save_best_only=True)
    
    warmup_steps = 3  # Number of warmup steps
    total_steps = 20  # Total training steps
    target_lr = 0.001  # Peak learning rate  
    lr_callback = CustomLearningRateScheduler(warmup_steps, total_steps, target_lr=target_lr)
          

    # Input layers for dynamic and static model strands
    model_dyn_in = tf.keras.Input(shape=(n_steps_in, X_train.shape[2]))
    model_stat_in = tf.keras.Input(shape=(X_train_stat.shape[1],))
    
    # Dynamic model strand
    model_dyn = tf.keras.layers.LSTM(HPi_lstm_size)(model_dyn_in) #, activation='relu'
    model_dyn = tf.keras.layers.Dropout(HPi_dropout)(model_dyn)

    # Static model strand
    model_stat = tf.keras.layers.Dense(HPi_static_size, activation='relu')(model_stat_in)
    model_stat = tf.keras.layers.Dropout(HPi_dropout)(model_stat)
    
    # Combine dynamic and static strands
    model_comb = tf.keras.layers.concatenate([model_dyn, model_stat])
    model_comb = tf.keras.layers.Dense(HPi_comb_size, activation='relu')(model_comb)
    model_comb = tf.keras.layers.Dropout(HPi_dropout)(model_comb)
    
    # Define output layer for predictions
    model_output = tf.keras.layers.Dense(units=1, activation='linear', dtype=tf.float32)(model_comb)
    
    # Define model with both dynamic and static inputs
    model = tf.keras.Model(inputs=[model_dyn_in, model_stat_in], outputs=model_output)
    
    # Compile model with appropriate loss function and optimizer
    optimizer =  tf.keras.optimizers.Adam(epsilon = 0.0001)
    model.compile(loss='mse', optimizer=optimizer)


    # Instantiate data generators
    train_generator = DataGenerator(X_train, X_train_stat, Y_train, HPi_batchsize, shuffle=True)
    val_generator = DataGenerator(X_stop, X_stop_stat, Y_stop, HPi_batchsize, shuffle=False)
    
    # Use fit with generators
    model_history = model.fit(train_generator,
                              validation_data=val_generator,
                              epochs=HPi_epochs, verbose=1, 
                              callbacks=[model_es, model_mc, lr_callback,
                              BatchProgressCallback(),
                              ClearMemory()])
        

        
    # take time
    now2 = datetime.now()
    timetaken = round((now2-now1).total_seconds())/60
    print('\n timetaken = '+str(timetaken)+'\n')
    

    # predict - with saved model checkpoint
    loaded_model = tf.keras.models.load_model(pth_out_i+'/model.keras')
    sim_n = loaded_model.predict([X_test, X_test_stat])
    
    # inverse scaling
    sim = []
    for i in range(len(dt_list_names)):
        temp = sim_n[IDlen == i,0]
        temp = temp.reshape(-1,1)
        temp = scalers_y[i].inverse_transform(temp)
        temp = temp.reshape(-1,)
        temp = pd.DataFrame({"ID": np.repeat(i,len(temp)), "sim": temp})
        sim.append(temp)
    del temp
    sim = pd.concat(sim)
    
    
    
    #% Evaluate Model
    
    # Minimum val MSE
    MSEvalmin = np.min(model_history.history['val_loss'])
    
    results = []
    for i in range(len(dt_list_names)):
        temp = sim[sim.ID == i]
        temp.columns = (dt_list_names[i] + "_" + temp.columns).tolist()
        Y_test = Y_test.reshape(-1,1)
        temp[dt_list_names[i]+"_obs"] = scalers_y[i].inverse_transform(Y_test[IDlen == i])
        temp = temp.reset_index(drop = True)
        results.append(temp)
    del temp
    results = pd.concat(results, axis = 1)
    
    sim_test = results.filter(like = "sim")
    sim_test.columns = np.arange(0,len(dt_list_names))
    obs_test = results.filter(like = "obs")
    obs_test.columns = np.arange(0,len(dt_list_names))
    
    err_test = sim_test-obs_test
    err_nash = obs_test - np.mean(datafullwide[datafullwide.index < date_start_test], 
                                    axis = 0).values.reshape(-1,len(sim_test.columns))
    
    MSE_test =  np.mean((err_test) ** 2, axis = 0)
    RMSE_test = np.sqrt(np.mean((sim_test-obs_test) ** 2, axis = 0))
    
    if((sum(sim_test.isnull().any()) > 0)):
        rr_test = np.nan
    if((sum(sim_test.isnull().any()) == 0)):
        rr_test = np.zeros(len(sim_test.columns))
        for i in range(len(sim_test.columns)):
            rr_test[i] = stats.pearsonr(sim_test.iloc[:,i], obs_test.iloc[:,i])[0]
        
    NSE_test = 1 - ((np.sum(err_test ** 2, axis = 0)) / (np.sum((err_nash) ** 2, axis = 0)))
    
    Bias_test = np.mean(err_test, axis = 0)
    
    alpha_test = np.std(sim_test, axis = 0)/np.std(obs_test, axis = 0)
    beta_test = np.mean(sim_test, axis = 0)/np.mean(obs_test, axis = 0)
    KGE_test = 1-np.sqrt((rr_test-1)**2+(alpha_test-1)**2+(beta_test-1)**2)
    
    # concat scores
    scores = pd.DataFrame([NSE_test, KGE_test, rr_test**2, Bias_test, MSE_test, RMSE_test]).transpose()
    scores.index = dt_list_names
    scores.columns = ['NSE','KGE','R2','Bias','MSE','RMSE']
    
    
    #% Export results
     
    # train history
    pd.DataFrame(model_history.history).to_csv(pth_out_i+'/losshistory.csv', float_format='%.4f', sep = ";")
    
    # scores
    scores.to_csv(pth_out_i+'/scores.csv', float_format='%.4f', sep = ";")
    
    # export obs+sim data of test period
    results.to_csv(pth_out_i+'/results.csv', float_format='%.4f', sep = ";")
    
    
