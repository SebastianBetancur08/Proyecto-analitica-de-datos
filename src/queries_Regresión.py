# Este script se encarga de ejecutar las consultas SQL para el proyecto de Regresión y guardar los resultados en archivos CSV.


import pandas as pd
import sqlite3

#Conectar a la base de datos
conn= sqlite3.connect('database/regresion.db')

#Consultas SQL
query1= ""
query2= ""
query3= ""
query4= ""
query5= ""

#Cargar resultado en DataFrame
df1=pd.read_sql_query(query1, conn)
df2=pd.read_sql_query(query2, conn)
df3=pd.read_sql_query(query3, conn)
df4=pd.read_sql_query(query4, conn)
df5=pd.read_sql_query(query5, conn)

#Guardar resultado en CSV
df1.to_csv("data/processed/Clasificacion_query1.csv", index=False)
df2.to_csv("data/processed/Clasificacion_query2.csv", index=False)
df3.to_csv("data/processed/Clasificacion_query3.csv", index=False)
df4.to_csv("data/processed/Clasificacion_query4.csv", index=False)
df5.to_csv("data/processed/Clasificacion_query5.csv", index=False)

#Cerrar la conexión
conn.close()