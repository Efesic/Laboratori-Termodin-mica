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

#--------------DIAGRAMA DE CLAPEYRON-----------------------
plt.figure(figsize=(8,6))

plt.xlabel('$V_{sist}$ [ml]')
plt.ylabel('$p$ [bar]')

plt.plot(iso_10[1],iso_10[0], label='T = 10ºC', marker = 'o', markersize = 3)
plt.plot(iso_15[1],iso_15[0], label='T = 15ºC',marker = 'o', markersize = 3)
plt.plot(iso_20[1],iso_20[0], label='T = 20ºC',marker = 'o', markersize = 3)
plt.plot(iso_25[1],iso_25[0], label='T = 25ºC',marker = 'o', markersize = 3)
plt.plot(iso_30[1],iso_30[0], label='T = 30ºC',marker = 'o', markersize = 3)
plt.plot(iso_35[1],iso_35[0], label='T = 35ºC',marker = 'o', markersize = 3)
plt.plot(iso_40[1],iso_40[0], label='T = 40ºC',marker = 'o', markersize = 3)
plt.plot(iso_45[1],iso_45[0], label='T = 45ºC',marker = 'o', markersize = 3)
plt.scatter(punts_sat[1],punts_sat[0], label = 'Punts de saturació', s =25, color = 'k',marker ='D')
plt.legend()

plt.savefig(f"../practica_IIIb/plots/clapeyron/Clapeyron_v{version}.png")
