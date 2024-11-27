import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

matplotlib.rcParams.update({'font.size': 12})
df = pd.read_excel('graphs\practica_Ia\dades_estacionari.xlsx')

T_amb = sum(df.iloc[0,6:9].tolist())/len(df.iloc[0,6:9].tolist())
T_Al = df.iloc[8,1:13].tolist()
T_Al = [a - T_amb for a in T_Al]
T_Llau = df.iloc[9,1:13].tolist()
T_LLau = [a - T_amb for a in T_Llau]
T_Fe = df.iloc[10,1:13].tolist()
T_Fe = [a - T_amb for a in T_Fe]
dist = df.iloc[11,1:13].tolist()

#Corba a la que ajustar
def exponencial(x,a,b):
    return a*np.exp(b*x)

p0 = (90,-0.1)  # Suponiendo un decaimiento exponencial
coef_Al, cov_Al = curve_fit(exponencial, dist, T_Al, p0=p0)
coef_Llau, cov_Llau = curve_fit(exponencial,dist,T_Llau, p0=p0)
coef_Fe, cov_Fe = curve_fit(exponencial,dist,T_Fe, p0=(120,-1))

print(coef_Al)
print(coef_Llau)
print(coef_Fe)

x1 = np.linspace(0,110,200)
y1 = coef_Al[0]*np.exp(coef_Al[1]*x1)

x2 = np.linspace(0,110,200)
y2 = coef_Llau[0]*np.exp(coef_Llau[1]*x2)

x3 = np.linspace(0,110,200)
y3 = coef_Fe[0]*np.exp(coef_Fe[1]*x3)
plt.figure(figsize=(8,6))

plt.scatter(dist,T_Al)

plt.scatter(dist,T_Llau)
plt.scatter(dist,T_Fe)

plt.plot(x1,y1)
plt.plot(x2,y2)
plt.plot(x3,y3)

plt.savefig('graphs/practica_Ia/plots/ln_theta.png')



