import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.rcParams.update({'font.size': 20})

df = pd.read_excel('graphs/practica_Ia/dades_permanent.xlsx',engine='openpyxl',usecols=[0,8,9,10,11,12,13],header=None)
temps = pd.read_excel('graphs/practica_Ia/dades_permanent.xlsx',usecols=[0])
temps_n = df[0].iloc[2:179]
t_g0 = pd.read_excel('graphs/practica_Ia/dades_permanent.xlsx',usecols=[1])
t_g10 = pd.read_excel('graphs/practica_Ia/dades_permanent.xlsx',usecols=[2])
t_g15 = pd.read_excel('graphs/practica_Ia/dades_permanent.xlsx',usecols=[3])

t_p0 = pd.read_excel('graphs/practica_Ia/dades_permanent.xlsx',usecols=[4])
t_p10 = pd.read_excel('graphs/practica_Ia/dades_permanent.xlsx',usecols=[5])
t_p20 = pd.read_excel('graphs/practica_Ia/dades_permanent.xlsx',usecols=[6])

t_g0_n = df[8].iloc[2:179]
t_g10_n = df[9].iloc[2:179]
t_g15_n = df[10].iloc[2:179]

t_p0_n = df[11].iloc[2:179]
t_p10_n = df[12].iloc[2:179]
t_p20_n = df[13].iloc[2:179]

plt.figure(figsize=(8,6))
plt.scatter(temps,t_g0, s=3, label= '$x$ = 0 cm',color="firebrick")
plt.scatter(temps,t_g10, s=3, label= '$x$ = 10 cm',color="limegreen")
plt.scatter(temps,t_g15, s=3, label= '$x$ = 15 cm',color="royalblue")
plt.xlabel('$t$ [s]')
plt.ylabel('$T$ [K]')
plt.minorticks_on()
plt.tick_params(which= 'major', direction='in',top = True,right =True,size = 10)
plt.tick_params(which= 'minor', direction='in',top = True,right =True,size = 5)
plt.grid(linestyle='--')

plt.text(4100, 78.3,"$x$ = 0 cm",fontsize=12,color="black",ha="center",va="center",bbox=dict(facecolor="firebrick",edgecolor="black",boxstyle="round,pad=0.3",linewidth=2,alpha=0.2))
plt.text(4100, 75,"$x$ = 10 cm",fontsize=12,color="black",ha="center",va="center",bbox=dict(facecolor="limegreen",edgecolor="black",boxstyle="round,pad=0.3",linewidth=2,alpha=0.2))
plt.text(4100, 70,"$x$ = 15 cm",fontsize=12,color="black",ha="center",va="center",bbox=dict(facecolor="royalblue",edgecolor="black",boxstyle="round,pad=0.3",linewidth=2,alpha=0.3))

plt.tight_layout()
plt.savefig('graphs/practica_Ia/plots/gran.png',dpi=300)

plt.figure(figsize=(8,6))
plt.scatter(temps,t_p0, s=3, label= '$x$ = 0 cm',color="firebrick")
plt.scatter(temps,t_p10, s=3, label= '$x$ = 10 cm',color="limegreen")
plt.scatter(temps,t_p20, s=3, label= '$x$ = 15 cm',color="royalblue")
plt.xlabel('$t$ [s]')
plt.ylabel('$T$ [K]')
plt.minorticks_on()
plt.tick_params(which= 'major', direction='in',top = True,right =True,size = 10)
plt.tick_params(which= 'minor', direction='in',top = True,right =True,size = 5)
plt.grid(linestyle='--')

plt.text(4100, 91,"$x$ = 0 cm",fontsize=12,color="black",ha="center",va="center",bbox=dict(facecolor="firebrick",edgecolor="black",boxstyle="round,pad=0.3",linewidth=2,alpha=0.2))
plt.text(4100, 85.5,"$x$ = 10 cm",fontsize=12,color="black",ha="center",va="center",bbox=dict(facecolor="limegreen",edgecolor="black",boxstyle="round,pad=0.3",linewidth=2,alpha=0.2))
plt.text(4100, 70,"$x$ = 20 cm",fontsize=12,color="black",ha="center",va="center",bbox=dict(facecolor="royalblue",edgecolor="black",boxstyle="round,pad=0.3",linewidth=2,alpha=0.3))
plt.tight_layout()
plt.savefig('graphs/practica_Ia/plots/petita.png',dpi=300)

plt.figure(figsize=(8,6))
plt.scatter(temps_n,t_g0_n, s=10, label= '$x$ = 0 cm',color="firebrick")
plt.scatter(temps_n,t_g10_n, s=10, label= '$x$ = 10 cm',color="limegreen")
plt.scatter(temps_n,t_g15_n, s=10, label= '$x$ = 15 cm',color="royalblue")
plt.xlabel('$t$ [s]')
plt.ylabel('$T$ [K]')
plt.minorticks_on()
plt.tick_params(which= 'major', direction='in',top = True,right =True,size = 10)
plt.tick_params(which= 'minor', direction='in',top = True,right =True,size = 5)
plt.grid(linestyle='--')

plt.legend(title='Posició',loc='upper right',fontsize=16,markerscale=2.5)
plt.tight_layout()
plt.savefig('graphs/practica_Ia/plots/gran_norm.png',dpi=300)

plt.figure(figsize=(8,6))
plt.scatter(temps_n,t_p0_n, s=10, label= '$x$ = 0 cm',color="firebrick")
plt.scatter(temps_n,t_p10_n, s=10, label= '$x$ = 10 cm',color="limegreen")
plt.scatter(temps_n,t_p20_n, s=10, label= '$x$ = 15 cm',color="royalblue")
plt.xlabel('$t$ [s]')
plt.ylabel('$T$ [K]')
plt.minorticks_on()
plt.tick_params(which= 'major', direction='in',top = True,right =True,size = 10)
plt.tick_params(which= 'minor', direction='in',top = True,right =True,size = 5)
plt.grid(linestyle='--')

plt.legend(title='Posició',loc='upper right',fontsize=16,markerscale=2.5)
plt.tight_layout()
plt.savefig('graphs/practica_Ia/plots/petit_norm.png',dpi=300)
