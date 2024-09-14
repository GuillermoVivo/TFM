# CODE FOR READING RAW FILES
# Change minor issues for intraday markets

import pandas as pd

import os
import warnings
import zipfile

###############################################################################

# Ignore warnings
warnings.filterwarnings("ignore")

# Dataframes
train_precio = []
train_energia = []

# File loop (one for each day)
for carpeta_zip in os.listdir("train_diario_prueba"):
    with zipfile.ZipFile(os.path.join("train_diario_prueba", carpeta_zip), 'r') as zip_file:
        for nombre_archivo in zip_file.namelist():
            with zip_file.open(nombre_archivo) as archivo_csv:
                
                # Read file
                dia = pd.read_csv(archivo_csv, delimiter=";", encoding='ISO-8859-1', header=1)
                
                # Remove last column and row (which are empty)
                dia = dia.iloc[:, :-1]
                dia = dia.iloc[:-1, :]
                
                # Filter for non-matched sell offers
                dia = dia[dia['Tipo Oferta'] == 'V']
                dia = dia[dia['Ofertada (O)/Casada (C)'] == 'O']
                
                # Remove redundant or unimportant columns
                dia = dia.drop('Tipo Oferta', axis=1)
                dia = dia.drop('Ofertada (O)/Casada (C)', axis=1)
                dia = dia.drop('Unidad', axis=1)
                dia = dia.drop('Pais', axis=1)
                
                # Reset indices
                dia = dia.reset_index(drop=True)
                
                # Fix number format
                dia['Energía Compra/Venta'] = dia['Energía Compra/Venta'].apply(lambda x: x.replace('.', ''))
                dia['Precio Compra/Venta'] = dia['Precio Compra/Venta'].apply(lambda x: x.replace('.', ''))
                dia = dia.replace(',', '.', regex=True)
                dia['Energía Compra/Venta'] = dia['Energía Compra/Venta'].astype(float)
                dia['Precio Compra/Venta'] = dia['Precio Compra/Venta'].astype(float)
                
                # Remove Spanish accents
                dia = dia.rename(columns={'Energía Compra/Venta': 'Energia'})
                dia = dia.rename(columns={'Precio Compra/Venta': 'Precio'})
                
                # Fix days with 23 or 25 hours
                if len(dia['Hora'].unique()) == 23:
                    hora_extra = dia[dia['Hora'] == 23]
                    hora_extra.loc[:, 'Hora'] = 24
                    dia = pd.concat([dia, hora_extra], axis=0, ignore_index=True)  
                elif len(dia['Hora'].unique()) == 25:
                    dia = dia[dia['Hora'] != 25]
                
                # Create offer curves
                dia = dia.sort_values(by=['Fecha','Hora','Precio'])
                dia['Energia'] = dia.groupby(['Hora','Fecha'])['Energia'].cumsum()
                index = dia.groupby(['Hora','Fecha','Precio'])['Energia'].idxmax()
                dia = dia.loc[index]
                dia = dia.sort_values(by=['Fecha','Hora'])
                dia = dia.reset_index(drop=True)
                
                # Add info to dataframe
                for h in range(1,25):
                    data_precio = {
                        'Hora': [h],
                        'Fecha': [dia['Fecha'][0]],
                    }
                    data_precio = pd.DataFrame(data_precio)
                    data_energia = {
                        'Hora': [h],
                        'Fecha': [dia['Fecha'][0]],
                    }
                    data_energia = pd.DataFrame(data_energia)
                    hora = dia[dia['Hora'] == h]
                    for i in range(0, len(hora)):
                        data_energia[i+2] = hora.iloc[i,2]
                        data_precio[i+2] = hora.iloc[i,3]
            
                    train_precio.append(data_precio)
                    train_energia.append(data_energia)
                
                print(dia['Fecha'][0])
 
# Add to big dataframe
train_energia = pd.concat(train_energia, axis=0, ignore_index=True)
train_precio = pd.concat(train_precio, axis=0, ignore_index=True)