import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
matplotlib.rcParams.update({'font.size': 10})


version = input("Ingresa la versió dels plots generats: ")
df = pd.read_excel('datos_isotermas.xlsx')


# Estructura iso_T = [p,V_sist,pV,1/V]
iso_10 = [df.iloc[35:57,11],df.iloc[35:57,12],df.iloc[35:57,26], df.iloc[35:57,27]]
iso_15 = [df.iloc[35:55,3],df.iloc[35:55,4],df.iloc[35:55,20],df.iloc[35:55,21]]
iso_20 = [df.iloc[3:24,11],df.iloc[3:24,12],df.iloc[3:24,26],df.iloc[3:24,27]]
iso_25 = [df.iloc[3:26,3],df.iloc[3:26,4],df.iloc[3:26,20],df.iloc[3:26,21]]
iso_30 = [df.iloc[35:57,15],df.iloc[35:57,16],df.iloc[35:57,29],df.iloc[35:57,30]]
iso_35 = [df.iloc[35:55,7],df.iloc[35:55,8],df.iloc[35:55,23],df.iloc[35:55,24]]
iso_40 = [df.iloc[3:23,15],df.iloc[3:23,16],df.iloc[3:23,29],df.iloc[3:23,30]]
iso_45 = [df.iloc[3:26,7],df.iloc[3:26,8],df.iloc[3:26,23],df.iloc[3:26,24]]

# Estructura punts_sat = [p_sat,V_sat]
punts_sat = [df.iloc[62:78,3],df.iloc[62:78,4]]

plt.figure(figsize=(8,6))
plt.scatter(iso_10[0],iso_10[2],s=10,label='T = 10ºC')
plt.scatter(iso_15[0],iso_15[2],s=10,label='T = 15ºC')

plt.scatter(iso_20[0],iso_20[2],s=10,label='T = 20ºC')
plt.scatter(iso_25[0],iso_25[2],s=10,label='T = 25ºC')
plt.scatter(iso_30[0],iso_30[2],s=10,label='T = 30ºC')
plt.scatter(iso_35[0],iso_35[2],s=10,label='T = 35ºC')
plt.scatter(iso_40[0],iso_40[2],s=10,label='T = 40ºC')
plt.scatter(iso_45[0],iso_45[2],s=10,label='T = 45ºC')

plt.xlabel('$V_{sist}$ [ml]')
plt.ylabel('$pV$ [bar$\cdot$ml]')
plt.legend()
plt.savefig(f"../practica_IIIb/plots/amagat/Amagat_v{version}.png")