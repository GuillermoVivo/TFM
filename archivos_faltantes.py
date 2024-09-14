# CODE TO LOOK FOR MISSING FILES
# Change for different markes

import os
import zipfile

from datetime import datetime, timedelta

###############################################################################

# First and last dates
fecha_inicio = datetime(2022, 1, 1)
fecha_fin = datetime(2023, 12, 31)

# Create all the names of the files that are supposed to exist
fechas_en_archivos = []
fecha_actual = fecha_inicio
while fecha_actual <= fecha_fin:
    nombre_archivo = "curva_pbc_uof_" + fecha_actual.strftime("%Y%m%d") + ".1"
    fechas_en_archivos.append(nombre_archivo)
    fecha_actual += timedelta(days=1)

# Names of the existing files
archivos_en_directorio = []
for carpeta_zip in os.listdir("FILL"):
  with zipfile.ZipFile(os.path.join("FILL", carpeta_zip), 'r') as zip_file:
      for nombre_archivo in zip_file.namelist():
          archivos_en_directorio.append(nombre_archivo)
    
# Check which files do exist 
archivos_faltantes = []
for nombre_archivo in fechas_en_archivos:
    if nombre_archivo not in archivos_en_directorio:
        archivos_faltantes.append(nombre_archivo)

if len(archivos_faltantes) == 0:
    print("All files exists.")
else:
    print("The following files do not exist:")
    print(archivos_faltantes)