import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

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
plt.ylabel('$T$ [$^{\circ}$C]')
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
plt.ylabel('$T$ [$^{\circ}$C]')
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
plt.ylabel('$T$ [$^{\circ}$C]')
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
plt.ylabel('$T$ [$^{\circ}$C]')
plt.minorticks_on()
plt.tick_params(which= 'major', direction='in',top = True,right =True,size = 10)
plt.tick_params(which= 'minor', direction='in',top = True,right =True,size = 5)
plt.grid(linestyle='--')

plt.legend(title='Posició',loc='upper right',fontsize=16,markerscale=2.5)
plt.tight_layout()
plt.savefig('graphs/practica_Ia/plots/petit_norm.png',dpi=300)

#Regressió mitjanes temporals de la temperatura
est = pd.read_excel('graphs/practica_Ia/dades_permanent.xlsx',usecols=[15,16,17,18,19,20], header=None)

xg_i = est.iloc[1,0:3].tolist()
pg_i = est.iloc[2,0:3].tolist()
xg_lr = np.linspace(min(xg_i),max(xg_i),100)

xp_i = est.iloc[1,3:6].tolist()
pp_i = est.iloc[2,3:6].tolist()
xp_lr = np.linspace(min(xp_i),max(xp_i),100)

def lr(x,m,b):
    return m*x+b

print(xg_i)

coefg_lr, covg_lr = curve_fit(lr,xg_i,np.log(pg_i))

u_x_i = [0.5,0.5,0.5]
coefp_lr, covp_lr = curve_fit(lr,xp_i,np.log(pp_i))
plt.figure(figsize=(8,6))
plt.errorbar(xg_i,np.log(pg_i),xerr=u_x_i,color='k',linestyle='',capsize=5,elinewidth=0.7,markersize=4,marker='D',label='Punts experimentals')
plt.plot(xg_lr,coefg_lr[0]*xg_lr+coefg_lr[1],color='darkolivegreen',linestyle='--',label='Barra gran')

plt.errorbar(xp_i,np.log(pp_i),xerr=u_x_i,color='k',linestyle='',capsize=5,elinewidth=0.7,markersize=4,marker='D')
plt.plot(xp_lr,coefp_lr[0]*xp_lr+coefp_lr[1],color='firebrick',linestyle='--',label='Barra petita')

plt.xlabel('$x_i$ [cm]')
plt.ylabel('$\ln{(\\bar{\\theta})}$')
plt.minorticks_on()
plt.tick_params(which= 'major', direction='in',top = True,right =True,size = 10)
plt.tick_params(which= 'minor', direction='in',top = True,right =True,size = 5)
plt.legend(fontsize='12')
plt.grid(linestyle='--')

plt.savefig('graphs/practica_Ia/plots/linear_reg.png',dpi=300)

#Càlcul de r^2:
y_data = np.log(pg_i)
y_model = [coefg_lr[0]*q + coefg_lr[1] for q in xg_i]
res = y_data - y_model
ss_res = np.sum(res**2)
ss_tot = np.sum((y_data - np.mean(y_data))**2)
r2= 1 -(ss_res/ss_tot)
print('Coeficient r2 per la regressió de la barra gran: ',r2)

y_data = np.log(pp_i)
y_model = [coefp_lr[0]*q + coefp_lr[1] for q in xp_i]
res = y_data - y_model
ss_res = np.sum(res**2)
ss_tot = np.sum((y_data - np.mean(y_data))**2)
r2= 1 -(ss_res/ss_tot)
print('Coeficient r2 per la regressió de la barra petita: ',r2)



#Màxims i minims locals:

def amplitud(maxims,minims,temp):
    return [0.5*(temp[a]-temp[b]) for a in maxims for b in minims if maxims.tolist().index(a) == minims.tolist().index(b)]

#Barra gran
maxims_t_g0, properties = find_peaks(t_g0.iloc[:, 0].tolist(),distance=50)
minims_t_g0, properties = find_peaks((-t_g0).iloc[:, 0].tolist(),distance=50)
amplituds_t_g0 = amplitud(maxims_t_g0,minims_t_g0,t_g0.iloc[:, 0].tolist())
a_g0 = np.mean(amplituds_t_g0)
print('Llista d\'amplituds per a la barra gran a x=0: ',amplituds_t_g0)
print('Amplitud de T per a la barra gran a x=0: ',a_g0)
print()

maxims_t_g10, properties = find_peaks(t_g10.iloc[:, 0].tolist(),distance=50)
minims_t_g10, properties = find_peaks((-t_g10).iloc[:, 0].tolist(),distance=50)
amplituds_t_g10 = amplitud(maxims_t_g10,minims_t_g10,t_g10.iloc[:, 0].tolist())
a_g10 = np.mean(amplituds_t_g10)
print('Llista d\'amplituds per a la barra gran a x=10: ',amplituds_t_g10)
print('Amplitud de T per a la barra gran a x=10: ',a_g10)
print()

maxims_t_g15, properties = find_peaks(t_g15.iloc[:, 0].tolist(),distance=50)
minims_t_g15, properties = find_peaks((-t_g15).iloc[:, 0].tolist(),distance=50)
amplituds_t_g15 = amplitud(maxims_t_g15,minims_t_g15,t_g15.iloc[:, 0].tolist())
a_g15 = np.mean(amplituds_t_g15)
print('Llista d\'amplituds per a la barra gran a x=15: ',amplituds_t_g15)
print('Amplitud de T per a la barra gran a x=15: ',a_g15)
print()

#Barra petita
maxims_t_p0, properties = find_peaks(t_p0.iloc[:, 0].tolist(),distance=50)
minims_t_p0, properties = find_peaks((-t_p0).iloc[:, 0].tolist(),distance=50)
amplituds_t_p0 = amplitud(maxims_t_p0,minims_t_p0,t_p0.iloc[:, 0].tolist())
a_p0 = np.mean(amplituds_t_p0)
print('Llista d\'amplituds per a la barra petita a x=0: ',amplituds_t_p0)
print('Amplitud de T per a la barra petita a x=0: ',a_p0)
print()

maxims_t_p10, properties = find_peaks(t_p10.iloc[:, 0].tolist(),distance=50)
minims_t_p10, properties = find_peaks((-t_p10).iloc[:, 0].tolist(),distance=50)
amplituds_t_p10 = amplitud(maxims_t_p10,minims_t_p10,t_p10.iloc[:, 0].tolist())
a_p10 = np.mean(amplituds_t_p10)
print('Llista d\'amplituds per a la barra petita a x=10: ',amplituds_t_p10)
print('Amplitud de T per a la barra petita a x=10: ',a_p10)
print()

maxims_t_p20, properties = find_peaks(t_p20.iloc[:, 0].tolist(),distance=50)
minims_t_p20, properties = find_peaks((-t_p20).iloc[:, 0].tolist(),distance=50)
amplituds_t_p20 = amplitud(maxims_t_p20,minims_t_p20,t_p20.iloc[:, 0].tolist())
a_p20 = np.mean(amplituds_t_p20)
print('Llista d\'amplituds per a la barra petita a x=20: ',amplituds_t_p20)
print('Amplitud de T per a la barra petita a x=10: ',a_p20)
print()

#Regressió de les amplituds
amplituds_gran = [a_g0,a_g10,a_g15]
amplituds_petit= [a_p0,a_p10,a_p20]

plt.figure(figsize=(8,6))
plt.plot(xg_i,np.log(amplituds_gran))
plt.plot(xp_i,np.log(amplituds_petit))
plt.savefig('graphs/practica_Ia/plots/reg_ampli.png',dpi=300)