# CODE FOR BUILDING TENSOR INPUTS FOR THE MODELS
# Change for train, test, train-train and train-validation

import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta

###############################################################################

# INTRADAY MARKETS

# Date range
start_date = datetime(2022, 1, 1)
end_date = datetime(2022, 12, 31)
fechas = []

current_date = start_date
while current_date <= end_date:
    fechas.append(current_date.strftime('%d/%m/%Y'))
    current_date += timedelta(days=1)

# MM = Same market, MA = predecessor market 
MMs = ["int1", "int2", "int3", "int4", "int5", "int6"]
MAs = ["diario", "int1", "int2", "int3", "int4", "int5"]
horas = [24, 24, 28, 24, 20, 17, 12]

cont = -1
for t in MMs:
    cont += 1
    
    ruta_MM = "FILL"
    ruta_MA = "FILL"

    MM = pd.read_csv(ruta_MM)
    MA = pd.read_csv(ruta_MA)
    
    input_MM = MM.iloc[0:364*horas[cont+1],:]
    input_MA = MA.iloc[horas[cont]:365*horas[cont],:]
    output = MM.iloc[horas[cont+1]:365*horas[cont+1],:]
    
    imput = []
    for i in tqdm(range(len(fechas)-1)):
        hoy = fechas[i+1]
        ayer = fechas[i]
        imput.append(pd.concat([input_MM[input_MM.iloc[:,1]==ayer].iloc[:,2:], 
                                input_MA[input_MA.iloc[:,1]==hoy].iloc[:,2:]], 
                               axis=0, ignore_index=True))
        
    autput = []
    for i in tqdm(range(len(fechas)-1)):
        hoy = fechas[i+1]
        autput.append(output[output.iloc[:,1]==hoy].iloc[:,2:])
        
    input_tensor = np.stack([df.values for df in imput])
    output_tensor = np.stack([df.values for df in autput])
    
    print("Input", input_tensor.shape)
    print("Output", output_tensor.shape)
    
###############################################################################

# DAILY MARKET

start_date = datetime(2022, 1, 1)
end_date = datetime(2022, 12, 30)
fechas = []

current_date = start_date
while current_date <= end_date:
    fechas.append(current_date.strftime('%d/%m/%Y'))
    current_date += timedelta(days=1)

ruta_MM = "FILL"
MM = pd.read_csv(ruta_MM)

input_MM = MM.iloc[0:364*24,:]
output = MM.iloc[24:365*24,:]

imput = []
for fecha in tqdm(fechas):
    imput.append(input_MM[input_MM.iloc[:,1]==fecha].iloc[:,2:])

start_date = datetime(2022, 1, 2)
end_date = datetime(2022, 12, 31)
fechas = []

current_date = start_date
while current_date <= end_date:
    fechas.append(current_date.strftime('%d/%m/%Y'))
    current_date += timedelta(days=1)
    
autput = []
for fecha in tqdm(fechas):
    autput.append(output[output.iloc[:,1]==fecha].iloc[:,2:])

input_tensor = np.stack([df.values for df in imput])
output_tensor = np.stack([df.values for df in autput])
