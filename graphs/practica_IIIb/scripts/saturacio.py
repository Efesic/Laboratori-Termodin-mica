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

coef = np.polyfit(punts_sat[1].tolist(), punts_sat[0].tolist(), deg=3)
xp = np.linspace(min(punts_sat[1]), max(punts_sat[1]), 100)
yp = np.polyval(punts_sat[1],punts_sat[0])
yp = np.polyval(coef, xp)
plt.figure(figsize=(8,6))
plt.scatter(punts_sat[1],punts_sat[0], label = 'Punts de saturació', s =20, color = 'k',marker ='D')
plt.plot(xp,yp)
plt.savefig(f"../practica_IIIb/plots/saturacio/Corba_Saturació_v{version}.png")

V_sat = punts_sat[0].tolist()
p_sat = punts_sat[1].tolist()

V_crit = max(V_sat)
p_crit = p_sat[V_sat.index(V_crit)]
V_critt = max(yp)
ypp = yp.tolist()
p_critt = xp[ypp.index(V_critt)]
print()
print('Els coeficients del polinomi de tercer grau són:',coef)
print()
print('El punt (V_crit,p_crit) màxim de les dades mesurades és: ', [V_crit,p_crit])
print()
print('El punt (V_crit,p_crit) màxim de la corba ajustada és: ', [V_critt,p_critt])