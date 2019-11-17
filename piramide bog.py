# -*- coding: utf-8 -*-
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")


data_all=pd.read_csv("osb_demografia_piramide_poblacional_v2.csv")
data_bog2019=data_all[(data_all["Área"]=="Bogotá") & (data_all["Año"]==2019)]
data_bog2019=data_bog2019.astype({"Total":"int64"})

print(data_bog2019.describe())
sns.catplot(data=data_bog2019,x="Grupos de edad",y="Total",hue="Sexo",kind="point")
#plt.show()
