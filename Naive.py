# CODE FOR NAIVE MODEL

import pandas as pd
import numpy as np

from scipy.integrate import simps
from scipy.interpolate import interp1d

import warnings

###############################################################################

# Ignore warnings
warnings.filterwarnings("ignore")

# Common route
tfm = "FILL"

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
    ruta = f"C:/Users/usuario/Documents/Universidad/2023-2024 Máster/TFM/Distribution Data/alt_den_{m}.csv"
    alt_den = pd.read_csv(ruta).values.flatten()
    globals()[v] = alt_den
    
dens = [den_diario, den_int1, den_int2, den_int3, den_int4, den_int5, den_int6]
    
var_grid = ["grid_diario", "grid_int1", "grid_int2", "grid_int3", 
             "grid_int4", "grid_int5", "grid_int6"]

for m, v in zip(mercados, var_grid):
    ruta = f"C:/Users/usuario/Documents/Universidad/2023-2024 Máster/TFM/Distribution Data/grid_den_{m}.csv"
    grid = pd.read_csv(ruta).values.flatten()
    globals()[v] = grid
    
    
def L2_pond(x1, y1, x2, y2, den):
    x_new = np.arange(0, 1, 0.0001)
    y1_new = np.array(step_inter(x_new, x1, y1))
    y2_new = np.array(step_inter(x_new, x2, y2))
    return np.sqrt(simps((y1_new-y2_new)**2 * den, x_new))

###############################################################################

# Markets and horizon hours
mercado = ["int1", "int2", "int3", "int4", "int5", "int6"]
horas = [24, 28, 24, 20, 17, 12]

# Naive model
for h, m in zip(horas, mercado):
    ruta_energia_train = "FILL"
    ruta_energia_test = "FILL"
    
    energia_train = pd.read_csv(ruta_energia_train)
    energia_train = energia_train.iloc[:,2:]
    
    energia_test = pd.read_csv(ruta_energia_test)
    fechas = energia_test.iloc[:,0:2]
    energia_test = energia_test.iloc[:,2:]
    
    energia = energia_test.iloc[:-24,:]
    energia = pd.concat([pd.DataFrame(energia_train.tail(24)), energia], axis=0, ignore_index=True)
    energia = pd.concat([fechas, energia], axis=1, ignore_index=True)