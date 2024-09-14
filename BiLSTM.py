# BiLSTM CODE

import pandas as pd
import numpy as np

import random
import statistics
import time
import warnings

from itertools import product
from scipy.integrate import simps
from scipy.interpolate import interp1d
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Nadam

# Ignore warnings
warnings.filterwarnings("ignore")

# Common route
tfm = "WRITE DIRECTORY HERE"

# Seed
np.random.seed(200)

###############################################################################

# Functions for distances
def na_clean(x):
     k = len(x) - sum(x.isna())
     return list(x[0:k])
 
def step_inter(x_new, x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    x = np.insert(x, 0, 0)
    x = np.insert(x, len(x), 1)
    y = np.insert(y, 0, min(y))
    y = np.insert(y, len(y), max(y))
    f = interp1d(x, y, kind='previous')
    y_new = f(x_new)
    return y_new

def Linf(x1, y1, x2, y2):
    x_new = np.arange(0, 1, 0.0001)
    y1_new = step_inter(x_new, x1, y1)
    y2_new = step_inter(x_new, x2, y2)
    return np.max(np.abs(np.array(y1_new) - np.array(y2_new)))

def L2(x1, y1, x2, y2):
    x_new = np.arange(0, 1, 0.0001)
    y1_new = np.array(step_inter(x_new, x1, y1))
    y2_new = np.array(step_inter(x_new, x2, y2))
    return np.sqrt(simps((y1_new-y2_new)**2, x_new))

mercados = ["diario", "int1", "int2", "int3", "int4", "int5", "int6"]
var_den = ["den_diario", "den_int1", "den_int2", "den_int3", 
             "den_int4", "den_int5", "den_int6"]

for m, v in zip(mercados, var_den):
    ruta = tfm + f"Distribution Data/alt_den_{m}.csv"
    alt_den = pd.read_csv(ruta).values.flatten()
    globals()[v] = alt_den
    
dens = [den_diario, den_int1, den_int2, den_int3, den_int4, den_int5, den_int6]
    
var_grid = ["grid_diario", "grid_int1", "grid_int2", "grid_int3", 
             "grid_int4", "grid_int5", "grid_int6"]

for m, v in zip(mercados, var_grid):
    ruta = tfm + f"Distribution Data/grid_den_{m}.csv"
    grid = pd.read_csv(ruta).values.flatten()
    globals()[v] = grid
    
    
def L2_pond(x1, y1, x2, y2, den):
    x_new = np.arange(0, 1, 0.0001)
    y1_new = np.array(step_inter(x_new, x1, y1))
    y2_new = np.array(step_inter(x_new, x2, y2))
    return np.sqrt(simps((y1_new-y2_new)**2 * den, x_new))

def tiempo(secs):
    horas = secs // 3600
    mins = (secs % 3600) // 60
    secs_res = secs % 60
    return horas, mins

# Custom losses
def custom_loss_int1(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_true - y_pred) * grid_int1, axis=-1))

def custom_loss_int2(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_true - y_pred) * grid_int2, axis=-1))

def custom_loss_int3(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_true - y_pred) * grid_int3, axis=-1))

def custom_loss_int4(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_true - y_pred) * grid_int4, axis=-1))

def custom_loss_int5(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_true - y_pred) * grid_int5, axis=-1))

def custom_loss_int6(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_true - y_pred) * grid_int6, axis=-1))

custom_loss = [custom_loss_int1, custom_loss_int2, custom_loss_int3,
               custom_loss_int4, custom_loss_int5, custom_loss_int6]

###############################################################################

# Hyperparamer combinations
n_LSTM = [1, 2]
n_dense = [1, 2, 3, 4]
units = [10, 20, 50, 100]
neurons = [10, 40, 50, 100]
dropouts = [0, 0.1, 0.2]

iters = 300
comb = list(product(n_LSTM, n_dense, units, neurons, dropouts))

# Markets and respective hours of horizon
mercados = ["int1", "int2", "int3", "int4", "int5", "int6"]
horas = [24, 24, 28, 24, 20, 17, 12]

cont = -1
networks = 0

start = time.time()
for t in mercados:
    cont += 1
    
    # Import grid
    ruta_grid = tfm + f"Distribution Data/grid_{t}.csv"
    grid = pd.read_csv(ruta_grid).values.flatten()
    
    # Random hyperparameter combinations selection
    selected = random.sample(range(0, len(comb)), iters)
    
    # Import data
    train_input_path = tfm + f"New Train Train Data/Tensor Data/new_train_train_{t}_input_tensor.npy"
    train_output_path = tfm + f"New Train Train Data/Tensor Data/new_train_train_{t}_output_tensor.npy"
    val_input_path = tfm + f"New Validation Data/Tensor Data/new_val_{t}_input_tensor.npy"
    
    train_input = np.load(train_input_path)
    train_output = np.load(train_output_path)
    val_input = np.load(val_input_path)
    
    real_path_energia = tfm + f"New Train Data/Read Data/Energia/new_train_{t}_energia.csv"
    real_path_precio = tfm + f"New Train Data/Read Data/Precio/new_train_{t}_precio.csv"
    real_energia = pd.read_csv(real_path_energia)
    real_precio = pd.read_csv(real_path_precio)
    
    real_energia = real_energia.iloc[304*horas[cont+1]:,:]
    real_energia.reset_index()
    real_precio = real_precio.iloc[304*horas[cont+1]:,:]
    real_precio.reset_index()
    
    # Benchmark initial mean
    best_mean = 10000000000
    
    # Validation with each hyperparameter combination
    for c in range(iters):
        # Take hyperparameter combination
        sel = selected[c]
        
        # Neural network construction
        if comb[sel][0] == 1:
            lstm_model = models.Sequential([
                layers.Bidirectional(layers.LSTM(units=comb[sel][2], activation = "relu", 
                            input_shape = (horas[cont]+horas[cont+1], 150), 
                            return_sequences = False, dropout = comb[sel][4]))
            ])
            
            for k in range(0, comb[sel][1]):
                lstm_model.add(layers.Dense(comb[sel][3], activation = "relu"))
                
            lstm_model.add(layers.Dense(horas[cont+1]*150)),
            lstm_model.add(layers.Reshape((horas[cont+1],150)))
        
        else:
            lstm_model = models.Sequential([
                layers.Bidirectional(layers.LSTM(units=comb[sel][2], activation = "relu", 
                            input_shape = (horas[cont]+horas[cont+1], 150), 
                            return_sequences = True, dropout = comb[sel][4]))
            ])
            
            lstm_model.add(layers.Bidirectional(layers.LSTM(units=comb[sel][2], 
                                       return_sequences=False, 
                                       dropout = comb[sel][4])))
            
            for k in range(0, comb[sel][1]):
                lstm_model.add(layers.Dense(comb[sel][3], activation = "relu"))
                
            lstm_model.add(layers.Dense(horas[cont+1]*150)),
            lstm_model.add(layers.Reshape((horas[cont+1],150)))
            
        # Compile
        optimizer = Nadam(learning_rate=0.001)
        lstm_model.compile(optimizer=optimizer, loss=custom_loss[cont])
        
        try:
            # Fit
            lstm_model.fit(train_input, train_output, epochs = 50, verbose = 0)
            
            # Predict
            predictions = lstm_model.predict(val_input, verbose = 0)
            
            # Predictions from 3D tensor to 2D dataframe
            predictions2D = predictions.reshape(-1, predictions.shape[2])
            
            # Correction of predictions
            for l in range(61*horas[cont+1]):
                for i in range(150):
                    if predictions2D[l,i] < 0:
                        predictions2D[l,i] = 0
                
                for i in range(1,150):
                      if predictions2D[l,i] < predictions2D[l,i-1]:
                        predictions2D[l,i] = predictions2D[l,i-1]   
            
            # Calculation of distances
            L2ponds = []
            for q in range(61*horas[cont+1]):
                L2ponds.append(L2_pond(na_clean(real_precio.iloc[q,2:]), 
                              na_clean(real_energia.iloc[q,2:]), 
                              grid, predictions2D[q,:], dens[cont+1]))
             
            # Check if these are the best hyperparamerts so far and save them if so
            mean = statistics.mean(L2ponds)
            if mean < best_mean:
                best_mean = mean
                best_n_LSTM = comb[sel][0]
                best_n_dense = comb[sel][1]
                best_unit = comb[sel][2]
                best_neuron = comb[sel][3]
                best_dropout = comb[sel][4]
        
        except:
            pass
        
        # Estimated time
        networks += 1
        print(networks/((6)*iters) * 100,"%")
        print('ET:', tiempo((time.time()-start)/networks * ((6)*iters-networks))[0],
               "h", tiempo((time.time()-start)/networks * ((6)*iters-networks))[1],
               "min")
        
    # Save best hyperparameters        
    bests = [best_mean, best_n_LSTM, best_n_dense, 
                   best_unit, best_neuron, best_dropout]
    
    # Final fit using best hyperparameters

    # Import data
    train_input_path = tfm + f"New Train Data/Tensor Data/new_train_{t}_input_tensor.npy"
    train_output_path = tfm + f"New Train Data/Tensor Data/new_train_{t}_output_tensor.npy"
    test_input_path = tfm + f"Test Data/Tensor Data/New/test_{t}_input_tensor.npy"
    
    train_input = np.load(train_input_path)
    train_output = np.load(train_output_path)
    test_input = np.load(test_input_path)
    
    real_path_energia = tfm + f"Test Data/Read Data/Energia/test_{t}_energia.csv"
    real_path_precio = tfm + f"Test Data/Read Data/Precio/test_{t}_precio.csv"
    real_energia = pd.read_csv(real_path_energia)
    real_precio = pd.read_csv(real_path_precio)
    
    # Neural network construction
    if best_n_LSTM == 1:
        lstm_model = models.Sequential([
            layers.Bidirectional(layers.LSTM(units=best_unit, activation = "relu", 
                        input_shape = (horas[cont]+horas[cont+1], 150), 
                        return_sequences = False, dropout = best_dropout))
        ])
        
        for k in range(0, best_n_dense):
            lstm_model.add(layers.Dense(best_neuron, activation = "relu"))
            
        lstm_model.add(layers.Dense(horas[cont+1]*150)),
        lstm_model.add(layers.Reshape((horas[cont+1],150)))
    
    else:
        lstm_model = models.Sequential([
            layers.Bidirectional(layers.LSTM(units=best_unit, activation = "relu", 
                        input_shape = (horas[cont]+horas[cont+1], 150), 
                        return_sequences = True, dropout = best_dropout))
        ])
        
        lstm_model.add(layers.Bidirectional(layers.LSTM(units=best_unit, 
                                   return_sequences=False, 
                                   dropout = best_dropout)))
        
        for k in range(0, best_n_dense):
            lstm_model.add(layers.Dense(best_neuron, activation = "relu"))
            
        lstm_model.add(layers.Dense(horas[cont+1]*150)),
        lstm_model.add(layers.Reshape((horas[cont+1],150)))
        
    # Optimize
    optimizer = Nadam(learning_rate=0.001)
    lstm_model.compile(optimizer=optimizer, loss=custom_loss[cont])
    
    try:
        # Fit
        lstm_model.fit(train_input, train_output, epochs = 50, verbose = 0)
        
        # Predict
        predictions = lstm_model.predict(test_input, verbose = 0)
        
        # Predictions from 3D tensor to 2D dataframe
        predictions2D = predictions.reshape(-1, predictions.shape[2])
        
        # Correction of predictions
        for l in range(365*horas[cont+1]):
            for i in range(150):
                if predictions2D[l,i] < 0:
                    predictions2D[l,i] = 0
            
            for i in range(1,150):
                  if predictions2D[l,i] < predictions2D[l,i-1]:
                    predictions2D[l,i] = predictions2D[l,i-1]
                    
        # Calculation of distances          
        L2s = []
        for q in range(365*horas[cont+1]):
            L2s.append(L2_pond(na_clean(real_precio.iloc[q,2:]), 
                          na_clean(real_energia.iloc[q,2:]), 
                          grid, predictions2D[q,:], dens[cont+1]))
            
        Linfs = []
        for q in range(365*horas[cont+1]):
            Linfs.append(Linf(na_clean(real_precio.iloc[q,2:]), 
                              na_clean(real_energia.iloc[q,2:]), 
                              grid, predictions2D[q,:]))
        
    except:
        pass