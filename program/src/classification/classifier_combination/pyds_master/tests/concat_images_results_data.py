import pandas as pd

# Lista de nombres de los archivos CSV que quieres leer
archivos_csv = ["complet_table_old.csv", "th.csv"]

# Lista para almacenar los DataFrames de los archivos CSV
dataframes = []

# Leer los archivos CSV y agregarlos a la lista de DataFrames
for archivo in archivos_csv:
    df = pd.read_csv("images_result_data/combined/" + archivo)
    dataframes.append(df)

# Concatenar los DataFrames en uno solo
tabla_concatenada = pd.concat(dataframes, ignore_index=True)

# Ordenar la tabla resultante de mayor a menor respecto a la última columna
tabla_ordenada = tabla_concatenada.sort_values(by=tabla_concatenada.columns[-1], ascending=False)

# Nombre del archivo CSV de salida
archivo_salida = "images_result_data/combined/complet_table.csv"

# Guardar la tabla resultante en un archivo CSV
tabla_ordenada.to_csv(archivo_salida, index=False)

