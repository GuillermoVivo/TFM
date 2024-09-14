# CODE FOR APPLYING DIEBOLD-MARIANO TEST

import pickle
import numpy as np
import pandas as pd

from dieboldmariano import dm_test

###############################################################################

mercados = ["int1", "int2", "int3", "int4", "int5", "int6"]

total = pd.DataFrame()

# Diebold-Mariano Test between Model A and Model B
for m in mercados:
    # Read files
    path = f"FILL"
    with open(path, 'rb') as f:
        modA = pickle.load(f)
        
    path = f"FILL"
    with open(path, 'rb') as f:
        modB = pickle.load(f)
    
    # Test
    test = pd.DataFrame(dm_test(np.zeros(len(modA)), modA, modB, one_sided=False)).T
    
    # Contains all markets
    total = pd.concat([total, test], axis = 0, ignore_index = True)
