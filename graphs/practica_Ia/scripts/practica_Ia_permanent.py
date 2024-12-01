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

print('Temperatura mitjana')
xg_i = est.iloc[1,0:3].tolist()
pg_i = est.iloc[2,0:3].tolist()
xg_lr = np.linspace(min(xg_i),max(xg_i),100)
print()
print('Temperatures mitjanes de la barra gran ', pg_i)
print('La incertesa és la mateixa que per a la temperatura: ', 0.000001, 'ºC')
xp_i = est.iloc[1,3:6].tolist()
pp_i = est.iloc[2,3:6].tolist()
xp_lr = np.linspace(min(xp_i),max(xp_i),100)
print()
print('Temperatures mitjanes de la barra petita ', pp_i)
print('La incertesa és la mateixa que per a la temperatura: ', 0.000001, 'ºC')
print()
u_x_i = [0.5,0.5,0.5]


def lr(x,m,b):
    return m*x+b

coefg_lr, covg_lr = curve_fit(lr,xg_i,np.log(pg_i),sigma=0.001)
coefp_lr, covp_lr = curve_fit(lr,xp_i,np.log(pp_i),sigma=0.001)


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
print('Regressió lineal del logaritme de les mitjanes temporals de temperatura:')
print()
y_data = np.log(pg_i)
y_model = [coefg_lr[0]*q + coefg_lr[1] for q in xg_i]
res = y_data - y_model
ss_res = np.sum(res**2)
ss_tot = np.sum((y_data - np.mean(y_data))**2)
r2= 1 -(ss_res/ss_tot)
print('Valors ajustats m i b de per a la barra gran:', coefg_lr, 'amb errors ',[float(np.sqrt(covg_lr[0][0])),float(np.sqrt(covg_lr[1][1]))])
print('Coeficient r2 per la regressió de la barra gran: ',r2)
print()
y_data = np.log(pp_i)
y_model = [coefp_lr[0]*q + coefp_lr[1] for q in xp_i]
res = y_data - y_model
ss_res = np.sum(res**2)
ss_tot = np.sum((y_data - np.mean(y_data))**2)
r2= 1 -(ss_res/ss_tot)
print('Valors ajustats m i b de per a la barra petita:', coefp_lr, 'amb errors ',[float(np.sqrt(covp_lr[0][0])),float(np.sqrt(covp_lr[1][1]))])
print('Coeficient r2 per la regressió de la barra petita: ',r2)
print()

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

coefg_lr, covg_lr = curve_fit(lr,xg_i,np.log(amplituds_gran),sigma=0.001)
plt.errorbar(xg_i,np.log(amplituds_gran),xerr=u_x_i,color='k',linestyle='',capsize=5,elinewidth=0.7,markersize=4,marker='D')
plt.plot(xg_lr,coefg_lr[0]*xg_lr+coefg_lr[1],color='darkolivegreen',linestyle='--',label='Barra gran')

coefp_lr, covp_lr = curve_fit(lr,xp_i,np.log(amplituds_petit),sigma=0.001)
plt.errorbar(xp_i,np.log(amplituds_petit),xerr=u_x_i,color='k',linestyle='',capsize=5,elinewidth=0.7,markersize=4,marker='D')
plt.plot(xp_lr,coefp_lr[0]*xp_lr+coefp_lr[1],color='firebrick',linestyle='--',label='Barra petita')

plt.xlabel('$x_i$ [cm]')
plt.ylabel('$\ln{(a_i)}$')
plt.minorticks_on()
plt.tick_params(which= 'major', direction='in',top = True,right =True,size = 10)
plt.tick_params(which= 'minor', direction='in',top = True,right =True,size = 5)
plt.legend(fontsize='12')
plt.grid(linestyle='--')
plt.savefig('graphs/practica_Ia/plots/reg_ampli.png',dpi=300)

#Calculem r2
print('Valors de la regressió de les amplituds: ')
print()
y_data = np.log(amplituds_gran)
y_model = [coefg_lr[0]*q + coefg_lr[1] for q in xg_i]
res = y_data - y_model
ss_res = np.sum(res**2)
ss_tot = np.sum((y_data - np.mean(y_data))**2)
r2= 1 -(ss_res/ss_tot)
print('Valors ajustats m i b de per a la barra gran:', coefg_lr, 'amb error de ',[float(np.sqrt(covg_lr[0][0])),float(np.sqrt(covg_lr[1][1]))])
print('Coeficient r2 per la regressió de l\'amplitud de la barra gran: ',r2)

y_data = np.log(amplituds_petit)
y_model = [coefp_lr[0]*q + coefp_lr[1] for q in xp_i]
res = y_data - y_model
ss_res = np.sum(res**2)
ss_tot = np.sum((y_data - np.mean(y_data))**2)
r2= 1 -(ss_res/ss_tot)
print('Valors ajustats m i b de per a la barra gpetitaran:', coefp_lr, 'amb error de ',[float(np.sqrt(covp_lr[0][0])),float(np.sqrt(covp_lr[1][1]))])
print('Coeficient r2 per la regressió de l\'amplitud de la barra petita: ',r2)
print()
#print('Amb pendents: ',coefg_lr[0],coefp_lr[0])

#Periode de les oscil·lacions
def periode(maxims,temps):
    t = temps.iloc[:,0].tolist()
    a = np.array([t[m] for m in maxims])
    b = np.array([t[m] for m in maxims])[:-1]
    b = np.insert(b,0,0.0)
    tau = a-b
    ntau = tau[1:]
    tau_promig = np.mean(ntau)
    return ntau,tau_promig

print()
print('Periodes per a la barra gran:')
print('Periode d\'oscil·lació per a x = 0: ',float(periode(maxims_t_g0,temps)[1]))
print('Periode d\'oscil·lació per a x = 10: ',float(periode(maxims_t_g10,temps)[1]))
print('Periode d\'oscil·lació per a x = 15: ',float(periode(maxims_t_g15,temps)[1]))
print()
print('Periodes per a la barra gran:')
print('Periode d\'oscil·lació per a x = 0: ',float(periode(maxims_t_p0,temps)[1]))
print('Periode d\'oscil·lació per a x = 10: ',float(periode(maxims_t_p10,temps)[1]))
print('Periode d\'oscil·lació per a x = 20: ',float(periode(maxims_t_p20,temps)[1]))

per_general = np.mean([float(periode(maxims_t_g0,temps)[1]),float(periode(maxims_t_g10,temps)[1]),float(periode(maxims_t_g15,temps)[1]),float(periode(maxims_t_p0,temps)[1]),float(periode(maxims_t_p10,temps)[1]),float(periode(maxims_t_p20,temps)[1])])

per_gran = np.mean([float(periode(maxims_t_g0,temps)[1]),float(periode(maxims_t_g10,temps)[1]),float(periode(maxims_t_g15,temps)[1])])
                   
per_petit = np.mean([float(periode(maxims_t_p0,temps)[1]),float(periode(maxims_t_p10,temps)[1]),float(periode(maxims_t_p20,temps)[1])])

print()
print('Periode d\'oscil·lació barra gran: ', per_gran,' [s], amb incertesa igual a la del temps')
print('Periode d\'oscil·lació barra petita: ', per_petit,' [s], amb incertesa igual a la del temps')
print('Periode d\'oscil·lació general: ', per_general,' [s], amb incertesa igual a la del temps')

#càlcul per atrobar m segons el guió

def trobar_m(amplitud_i,amplitud_j,i,j):
    u_T = 0.000001
    u_x = 0.5
    a = np.array(amplitud_i)
    b = np.array(amplitud_j)
    m_list = (np.log(a) - np.log(b))/(j-i)
    u_m_list = (u_T**2)/((i-j)**2)*(1/(a**2) + 1/(b**2)) + u_x**2*(2*(np.log(a/b))**2)/((i-j)**4)
    return m_list,np.mean(m_list),np.mean(u_m_list)

print()
print('Per a la barra gran')
print('Valor de m per el parell x_i=0 y x_j=10 de la barra gran: ', float(trobar_m(a_g0,a_g10,0,10)[1]),'amb incertesa ',float(trobar_m(a_g0,a_g10,0,10)[2]))
print('Valor de m per el parell x_i=10 y x_j=15 de la barra gran: ', float(trobar_m(a_g0,a_g10,0,10)[1]),'amb incertesa ',float(trobar_m(a_g10,a_g15,10,15)[2]))
print('Valor de m per el parell x_i=0 y x_j=15 de la barra gran: ', float(trobar_m(a_g0,a_g15,0,15)[1]),'amb incertesa ',float(trobar_m(a_g0,a_g15,0,15)[2]))
print()
print('Per a la barra petita')
print('Valor de m per el parell x_i=0 y x_j=10 de la barra petita: ', float(trobar_m(a_p0,a_p10,0,10)[1]),'amb incertesa ',float(trobar_m(a_p0,a_p10,0,10)[2]))
print('Valor de m per el parell x_i=10 y x_j=20 de la barra petita: ', float(trobar_m(a_p10,a_p20,10,20)[1]),'amb incertesa ',float(trobar_m(a_p10,a_p20,10,20)[2]))
print('Valor de m per el parell x_i=0 y x_j=20 de la barra petita: ', float(trobar_m(a_p0,a_p20,0,20)[1]),'amb incertesa ',float(trobar_m(a_p0,a_p20,0,20)[2]))
print()
print('Valor de m per a la barra gran general: ', np.mean([float(trobar_m(a_g0,a_g10,0,10)[1]),float(trobar_m(a_g0,a_g10,0,10)[1]),float(trobar_m(a_g0,a_g15,0,15)[1]),]), 'amb una incertesa de ', max([float(trobar_m(a_g0,a_g10,0,10)[2]),float(trobar_m(a_g10,a_g15,10,15)[2]),float(trobar_m(a_g0,a_g15,0,15)[2])]))
print()
print('Valor de m per a la barra petita general: ', np.mean([float(trobar_m(a_p0,a_p10,0,10)[1]),float(trobar_m(a_p10,a_p20,10,20)[1]),float(trobar_m(a_p0,a_p20,0,20)[1])]), 'amb una incertesa de ', max([float(trobar_m(a_p0,a_p10,0,10)[2]),float(trobar_m(a_p10,a_p20,10,20)[2]),float(trobar_m(a_p0,a_p20,0,20)[2])]))

#Càlcul de h

def trobar_h(i,j,maxim_i,maxim_j,temp):
    t = temps.iloc[:,0].tolist()
    u_t = 0.001
    u_x = 0.5
    per = per_general
    phi_i = np.array([t[a] for a in maxim_i])
    phi_j = np.array([t[a] for a in maxim_j])
    delta_t = phi_i - phi_j
    desfase = (phi_i - phi_j)*2*np.pi/per_general
    delta_x = i-j
    h = desfase/delta_x
    u_h = np.sqrt(u_t**2 * ((2*np.pi/(delta_x*per**2))**2 + (2*np.pi/(per*delta_x))**2) + u_x**2 * (2*np.pi*delta_t/(per*delta_x**2))**2)
    return float(np.mean(h)),max(u_h)

print()
print('Mesures de h per a la barra gran')
print('Valor de h segons calcular sobre 0 i 10 de la barra gran: ', trobar_h(10,0,maxims_t_g10,maxims_t_g0,temps)[0],'amb una incertesa de ',trobar_h(10,0,maxims_t_g10,maxims_t_g0,temps)[1])
print('Valor de h segons calcular sobre 10 i 15 de la barra gran: ', trobar_h(15,10,maxims_t_g15,maxims_t_g10,temps)[0],'amb una incertesa de ',trobar_h(15,10,maxims_t_g15,maxims_t_g10,temps)[1])
print('Valor de h segons calcular sobre 0 i 15 de la barra gran: ', trobar_h(15,0,maxims_t_g15,maxims_t_g0,temps)[0],'amb una incertesa de ',trobar_h(15,0,maxims_t_g15,maxims_t_g0,temps)[1])
gran_general =[np.mean([trobar_h(10,0,maxims_t_g10,maxims_t_g0,temps)[0],trobar_h(15,10,maxims_t_g15,maxims_t_g10,temps)[0],trobar_h(15,0,maxims_t_g15,maxims_t_g0,temps)[0]]),max([trobar_h(10,0,maxims_t_g10,maxims_t_g0,temps)[1],trobar_h(15,10,maxims_t_g15,maxims_t_g10,temps)[1],trobar_h(15,0,maxims_t_g15,maxims_t_g0,temps)[1]])]
print('Valor general: ',gran_general[0], 'amb una incertesa de ',gran_general[1])

print()
print('Mesures de h per a la barra petita')
print('Valor de h segons calcular sobre 0 i 10 de la barra petita: ', trobar_h(10,0,maxims_t_p10,maxims_t_p0,temps)[0],'amb una incertesa de ',trobar_h(10,0,maxims_t_p10,maxims_t_p0,temps)[1])
print('Valor de h segons calcular sobre 10 i 20 de la barra petita: ', trobar_h(20,10,maxims_t_p20,maxims_t_p10,temps)[0],'amb una incertesa de ',trobar_h(20,10,maxims_t_p20,maxims_t_p10,temps)[1])
print('Valor de h segons calcular sobre 0 i 20 de la barra petita: ', trobar_h(20,0,maxims_t_p20,maxims_t_p0,temps)[0],'amb una incertesa de ',trobar_h(20,0,maxims_t_p20,maxims_t_p0,temps)[1])
petit_general = [ np.mean([trobar_h(10,0,maxims_t_p10,maxims_t_p0,temps)[0],trobar_h(20,10,maxims_t_p20,maxims_t_p10,temps)[0],trobar_h(20,0,maxims_t_p20,maxims_t_p0,temps)[0]]),max([trobar_h(10,0,maxims_t_p10,maxims_t_p0,temps)[1],trobar_h(20,10,maxims_t_p20,maxims_t_p10,temps)[1],trobar_h(20,0,maxims_t_p20,maxims_t_p0,temps)[1]])]
print('Valor general: ',petit_general[0], 'amb una incertesa de ', petit_general[1] )

#Càlcul de lambda
m_gran = np.mean([float(trobar_m(a_g0,a_g10,0,10)[1]),float(trobar_m(a_g0,a_g10,0,10)
[1]),float(trobar_m(a_g0,a_g15,0,15)[1]),])
u_m_gran = max([float(trobar_m(a_g0,a_g10,0,10)[2]),float(trobar_m(a_g10,a_g15,10,15)[2]),float(trobar_m(a_g0,a_g15,0,15)[2])])

m_petita = np.mean([float(trobar_m(a_p0,a_p10,0,10)[1]),float(trobar_m(a_p10,a_p20,10,20)[1]),float(trobar_m(a_p0,a_p20,0,20)[1])])
u_m_petita = max([float(trobar_m(a_p0,a_p10,0,10)[2]),float(trobar_m(a_p10,a_p20,10,20)[2]),float(trobar_m(a_p0,a_p20,0,20)[2])])

K = 2.37
r_gran = 5.1*0.5
r_petit = 3*0.5
u_r = 0.1
lambda_gran = 0.5*(K*r_gran*(m_gran**2 - gran_general[0]**2))
lambda_petita = 0.5*(K*r_petit*(m_petita**2 - petit_general[0]**2))

u_lambda_gran = np.sqrt(u_r**2 * (0.5*K*(m_gran**2-gran_general[0]**2))**2 + (u_m_gran*K*m_gran*r_gran)**2 + (gran_general[1]*K*r_gran*gran_general[0])**2)
u_lambda_petita = np.sqrt(u_r**2 * (0.5*K*(m_petita**2-petit_general[0]**2))**2 + (u_m_petita*K*m_petita*r_petit)**2 + (petit_general[1]*K*r_petit*petit_general[0])**2)

print()
print('El valor de lambda per a la barra gran és de ', float(lambda_gran), 'amb un error de ', float(u_lambda_gran))

print('El valor de lambda per a la barra petita és de ', float(lambda_petita), 'amb un error de ', float(u_lambda_petita))
