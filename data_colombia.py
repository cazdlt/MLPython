# -*- coding: utf-8 -*-
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import regresionLineal as rl

sns.set_style("whitegrid")

#cargar y formatear dataset
data_all=pd.read_csv("Data/API_COL_DS2_es_csv_v2_449302/API_COL_DS2_es_csv_v2_449302.csv",skiprows=range(4),index_col="Indicator Name")
data_all=data_all.transpose().iloc[3:]

#filtrar dataset para encontrar buen data
data_valid=data_all.loc[map(str,range(1960,2020)),data_all.isna().sum()<3].infer_objects()
data_valid.index=data_valid.index.astype(int)
indicadores=list(data_valid.columns)

#extraer data a analizar y agregar data faltante, graficar
#data_tasacambio=data_valid.loc[:,"Tasa de cambio oficial (UMN por US$, promedio para un período)"]
data_tasacambio=data_valid.loc[:,"Crecimiento de la población (% anual)"]
print(data_tasacambio.head())
#data_tasacambio[2019]=3500
data_tasacambio=data_tasacambio.dropna()
sns.relplot(data=data_tasacambio)

#crear arreglos np
tasacambio=data_tasacambio.to_numpy()
año=np.array(data_tasacambio.index)

#pruebas ML
alfa=0.0000001
maxit=10000
umbral=0.1

theta,costos=rl.regresionLineal(año,tasacambio,alfa,maxit,umbral)
hipotesis=rl.h(año,theta)
rl.linePlot(año,hipotesis)

print(costos)
plt.show()

