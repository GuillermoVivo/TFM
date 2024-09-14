# ERROR METRICS CODE

import statistics
import os
import pickle
import numpy as np
import pandas as pd

###############################################################################

# Function for error metrics
def stat(x):
    df = {"Mean": statistics.mean(x), "Median": statistics.median(x),
          "SD": statistics.stdev(x), "MAD": statistics.median(np.abs(x-statistics.median(x)))}
    return pd.DataFrame(df, index=[0])

# Creation of dataframes
big_summaryL2 = pd.DataFrame()
big_summaryLinf = pd.DataFrame()

# Models
modelos = ["1", "2", "3", "4", "5", "6", "7"]

# Apply function to each market and model pairs
for m in modelos:
    names = {0: "Intradiario 1 " + "Mod " + m, 1: "Intradiario 2 " + "Mod " + m, 
             2: "Intradiario 3 " + "Mod " + m, 3: "Intradiario 4 " + "Mod " + m, 4: "Intradiario 5 " + "Mod " + m,
             5: "Intradiario 6 " + "Mod " + m}
    
    path = f"FILL/Mod {m}/L2"
    
    summaryL2 = []
    
    for archivo in os.listdir(path):
        with open(os.path.join(path, archivo), 'rb') as f:
            data = pickle.load(f)
            summaryL2.append(stat(data))
    
    summaryL2 = pd.concat(summaryL2, axis=0, ignore_index=True)
    summaryL2 = summaryL2.rename(index=names)
    big_summaryL2 = pd.concat([big_summaryL2, summaryL2], axis=0, ignore_index=False)
    
    path = f"FILL/Mod {m}/Linf"
    
    summaryLinf = []
    
    for archivo in os.listdir(path):
        with open(os.path.join(path, archivo), 'rb') as f:
            data = pickle.load(f)
            summaryLinf.append(stat(data))
    
    summaryLinf = pd.concat(summaryLinf, axis=0, ignore_index=True)
    summaryLinf = summaryLinf.rename(index=names)
    big_summaryLinf = pd.concat([big_summaryLinf, summaryLinf], axis=0, ignore_index=False)