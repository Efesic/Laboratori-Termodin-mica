import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


matplotlib.rcParams.update({'font.size': 20})
df = pd.read_excel('graphs/practica_Ia/dades_estacionari.xlsx',usecols=('distancia','alumini','llauto','ferro','ambient','d_alumini','d_llauto','d_ferro'))

r_Al = 0.5*np.mean(df.iloc[0:3,5].tolist())
r_Ll = 0.5*np.mean(df.iloc[0:3,6].tolist())
r_Fe = 0.5*np.mean(df.iloc[0:3,7].tolist())

print('Els radis de cada barra són:')
print()
print(r_Al,'[cm] per a la d\'alumini')
print(r_Ll,'[cm] per a la de llautó')
print(r_Fe,'[cm] per a la de ferro')
print()
print('La incertesa de totes les mesures és de ', 0.001,'[cm]')

dist = df.iloc[:,0].tolist()
T_amb = np.mean(df.iloc[0:3,4].tolist())

print()
print('El valor de la T_amb és: ',T_amb)

T_Al = np.array(df.iloc[:,1].tolist())
theta_Al = T_Al - T_amb

T_Ll = np.array(df.iloc[:,2].tolist())
theta_Ll = T_Ll - T_amb

T_Fe = np.array(df.iloc[:,3].tolist())
theta_Fe = T_Fe - T_amb

#Corba a la que ajustar
def exponencial(x,a,b):
    return a*np.exp(b*x)

x= np.linspace(min(dist),max(dist),100)

coef_Al, cov_Al = curve_fit(exponencial,dist,theta_Al, p0=(100,0.01))
coef_Ll, cov_Ll = curve_fit(exponencial,dist,theta_Ll, p0=(80,0.1))
coef_Fe, cov_Fe = curve_fit(exponencial,dist,theta_Fe, p0=(63,0.01))
#Gràfic thetas
plt.figure(figsize=(8,6))
plt.scatter(dist,theta_Al,marker='D',color='firebrick',s=25,label='Alumini')
plt.scatter(dist,theta_Ll,marker='D',color='limegreen',s=25,label='Llauto')
plt.scatter(dist,theta_Fe,marker='D',color='royalblue',s=25,label='Ferro')

plt.plot(x,coef_Al[0]*np.exp(coef_Al[1]*x),color='firebrick',linestyle='--')
plt.plot(x,coef_Ll[0]*np.exp(coef_Ll[1]*x),color='limegreen',linestyle='--')
plt.plot(x,coef_Fe[0]*np.exp(coef_Fe[1]*x),color='royalblue',linestyle='--')

plt.xlabel('$d$ [m]')
plt.ylabel('$\\theta$ [$^{\circ}$C]')
plt.minorticks_on()
plt.tick_params(which= 'major', direction='in',top = True,right =True,size = 10)
plt.tick_params(which= 'minor', direction='in',top = True,right =True,size = 5)
plt.grid(linestyle='--')

plt.legend(title='Punts experimentals',fontsize=14)

plt.tight_layout()

plt.savefig('graphs/practica_Ia/plots/theta_vs_d_estacionaria.png', dpi=300)

print()
print('Coeficients de la regressió exponencial:')
print()
print('Per a l\'alumini: ',[float(coef_Al[0]),float(coef_Al[1])], 'amb un error de', [float(np.sqrt(cov_Al[0][0])),float(np.sqrt(cov_Al[1][1]))])
print('Per al llautó: ',[float(coef_Ll[0]),float(coef_Ll[1])], 'amb un error de', [float(np.sqrt(cov_Ll[0][0])),float(np.sqrt(cov_Ll[1][1]))])
print('Per al ferro: ',[float(coef_Fe[0]),float(coef_Fe[1])], 'amb un error de', [float(np.sqrt(cov_Fe[0][0])),float(np.sqrt(cov_Fe[1][1]))])


#Regresió ln(theta)
def lr(x,m,b):
    return m*x +b

coef_Al, cov_Al = curve_fit(lr,dist,np.log(theta_Al))
coef_Ll, cov_Ll = curve_fit(lr,dist,np.log(theta_Ll))
coef_Fe, cov_Fe = curve_fit(lr,dist,np.log(theta_Fe))

plt.figure(figsize=(8,6))
plt.scatter(dist,np.log(theta_Al),marker='D',color='firebrick',s=25,label='Alumini')
plt.scatter(dist,np.log(theta_Ll),marker='D',color='limegreen',s=25,label='Llauto')
plt.scatter(dist,np.log(theta_Fe),marker='D',color='royalblue',s=25,label='Ferro')

plt.plot(x,coef_Al[0]*x +coef_Al[1],color='firebrick',linestyle='--')
plt.plot(x,coef_Ll[0]*x +coef_Ll[1],color='limegreen',linestyle='--')
plt.plot(x,coef_Fe[0]*x +coef_Fe[1],color='royalblue',linestyle='--')

plt.xlabel('$d$ [m]')
plt.ylabel('$\ln{(\\theta)}$')
plt.minorticks_on()
plt.tick_params(which= 'major', direction='in',top = True,right =True,size = 10)
plt.tick_params(which= 'minor', direction='in',top = True,right =True,size = 5)
plt.grid(linestyle='--')

plt.legend(title='Punts experimentals',fontsize=14,loc='lower left')

plt.tight_layout()

plt.savefig('graphs/practica_Ia/plots/reg_estacionaria.png', dpi=300)

print()
print('Coeficients de la regressió lineal de ln(theta):')
print()
print('Per a l\'alumini: ',[float(coef_Al[0]),float(coef_Al[1])], 'amb un error de', [float(np.sqrt(cov_Al[0][0])),float(np.sqrt(cov_Al[1][1]))])
print('Per al llautó: ',[float(coef_Ll[0]),float(coef_Ll[1])], 'amb un error de', [float(np.sqrt(cov_Ll[0][0])),float(np.sqrt(cov_Ll[1][1]))])
print('Per al ferro: ',[float(coef_Fe[0]),float(coef_Fe[1])], 'amb un error de', [float(np.sqrt(cov_Fe[0][0])),float(np.sqrt(cov_Fe[1][1]))])

#Càlcul de r2 de la regressió lineal:
print()
print('Regressió lineal del logaritme de la temperatura:')
print()
y_data = np.log(theta_Al)
y_model = [coef_Al[0]*q + coef_Al[1] for q in dist]
res = y_data - y_model
ss_res = np.sum(res**2)
ss_tot = np.sum((y_data - np.mean(y_data))**2)
r2= 1 -(ss_res/ss_tot)
print('r2 per al alumini: ',r2)

print()
y_data = np.log(theta_Ll)
y_model = [coef_Ll[0]*q + coef_Ll[1] for q in dist]
res = y_data - y_model
ss_res = np.sum(res**2)
ss_tot = np.sum((y_data - np.mean(y_data))**2)
r2= 1 -(ss_res/ss_tot)
print('r2 per al llautó: ',r2)

print()
y_data = np.log(theta_Fe)
y_model = [coef_Fe[0]*q + coef_Fe[1] for q in dist]
res = y_data - y_model
ss_res = np.sum(res**2)
ss_tot = np.sum((y_data - np.mean(y_data))**2)
r2= 1 -(ss_res/ss_tot)
print('r2 per al ferro: ',r2)
#Càlcul de les K/lambda

K_Al = 2/(r_Al*(coef_Al[0])**2)
u_K_Al = np.sqrt((2*0.001/(coef_Al[0]**2 * r_Al**2))**2 + (4*np.sqrt(cov_Al[0][0])/r_Al * coef_Al[0]**3)**2)

K_Ll = 2/(r_Al*(coef_Ll[0])**2)
u_K_Ll = np.sqrt((2*0.001/(coef_Ll[0]**2 * r_Ll**2))**2 + (4*np.sqrt(cov_Ll[0][0])/r_Ll * coef_Ll[0]**3)**2)

K_Fe = 2/(r_Al*(coef_Fe[0])**2)
u_K_Fe = np.sqrt((2*0.001/(coef_Fe[0]**2 * r_Fe**2))**2 + (4*np.sqrt(cov_Fe[0][0])/r_Fe * coef_Fe[0]**3)**2)

print()
print('Valor de K_Al: ',K_Al,'amb una incertesa de ',u_K_Al)
print('Valor de K_Ll: ',K_Ll,'amb una incertesa de ',u_K_Ll)
print('Valor de K_Fe: ',K_Fe,'amb una incertesa de ',u_K_Fe)

#Càlculs patró ferro

K_pat_Fe = 0.802 # [W/Kcm]

K_pf_Al = (K_pat_Fe*r_Fe*coef_Fe[0]**2)/(r_Al*coef_Al[0]**2)
u_K_pf_Al = np.sqrt( (0.001*K_pat_Fe*coef_Fe[0]**2 / (r_Al*coef_Al[0]**2))**2 + (np.sqrt(cov_Fe[0][0])*K_pat_Fe*r_Fe*2*coef_Fe[0] /(r_Al*coef_Al[0]**2))**2 + (0.001*K_pat_Fe*r_Fe*coef_Fe[0]**2/(r_Al**2 * coef_Al[0]**2))**2 + (cov_Al[0][0]*2*K_pat_Fe*coef_Fe[0]**2 / (r_Al*coef_Al[0]**3))**2)

K_pf_Ll = (K_pat_Fe*r_Fe*coef_Fe[0]**2)/(r_Ll*coef_Ll[0]**2)
u_K_pf_Ll = np.sqrt( (0.001*K_pat_Fe*coef_Fe[0]**2 / (r_Ll*coef_Ll[0]**2))**2 + (np.sqrt(cov_Fe[0][0])*K_pat_Fe*r_Fe*2*coef_Fe[0] /(r_Ll*coef_Ll[0]**2))**2 + (0.001*K_pat_Fe*r_Fe*coef_Fe[0]**2/(r_Ll**2 * coef_Ll[0]**2))**2 + (cov_Ll[0][0]*2*K_pat_Fe*coef_Fe[0]**2 / (r_Ll*coef_Ll[0]**3))**2)

print()
print('Patró ferro: ',K_pat_Fe)
print('Patró ferro, alumini: ',K_pf_Al, 'amb una incertesa de ',u_K_pf_Al)
print('Patró ferro, llautó: ',K_pf_Ll, 'amb una incertesa de ',u_K_pf_Ll)

#Càlculs patró llautó

K_pat_Ll = 1.25 # [W/Kcm]

K_pl_Al = (K_pat_Ll*r_Ll*coef_Ll[0]**2)/(r_Al*coef_Al[0]**2)
u_K_pl_Al = np.sqrt( (0.001*K_pat_Ll*coef_Ll[0]**2 / (r_Al*coef_Al[0]**2))**2 + (np.sqrt(cov_Ll[0][0])*K_pat_Ll*r_Ll*2*coef_Ll[0] /(r_Al*coef_Al[0]**2))**2 + (0.001*K_pat_Ll*r_Ll*coef_Ll[0]**2/(r_Al**2 * coef_Al[0]**2))**2 + (cov_Al[0][0]*2*K_pat_Ll*coef_Ll[0]**2 / (r_Al*coef_Al[0]**3))**2)

K_pl_Fe = (K_pat_Ll*r_Ll*coef_Ll[0]**2)/(r_Fe*coef_Fe[0]**2)
u_K_pl_Fe = np.sqrt( (0.001*K_pat_Ll*coef_Ll[0]**2 / (r_Fe*coef_Fe[0]**2))**2 + (np.sqrt(cov_Ll[0][0])*K_pat_Ll*r_Ll*2*coef_Ll[0] /(r_Fe*coef_Fe[0]**2))**2 + (0.001*K_pat_Ll*r_Ll*coef_Ll[0]**2/(r_Fe**2 * coef_Fe[0]**2))**2 + (cov_Fe[0][0]*2*K_pat_Ll*coef_Ll[0]**2 / (r_Fe*coef_Fe[0]**3))**2)

print()
print('Patró llautó: ',K_pat_Ll)
print('Patró llautó, alumini: ',K_pl_Al, 'amb una incertesa de ',u_K_pl_Al)
print('Patró llautó, ferro: ',K_pl_Fe, 'amb una incertesa de ',u_K_pl_Fe)

#Càlculs patró alumini

K_pat_Al = 2.37 # [W/Kcm]

K_pa_Ll = (K_pat_Al*r_Al*coef_Al[0]**2)/(r_Ll*coef_Ll[0]**2)
u_K_pa_Ll = np.sqrt( (0.001*K_pat_Al*coef_Al[0]**2 / (r_Ll*coef_Ll[0]**2))**2 + (np.sqrt(cov_Al[0][0])*K_pat_Al*r_Al*2*coef_Al[0] /(r_Ll*coef_Ll[0]**2))**2 + (0.001*K_pat_Al*r_Al*coef_Al[0]**2/(r_Ll**2 * coef_Ll[0]**2))**2 + (cov_Ll[0][0]*2*K_pat_Al*coef_Al[0]**2 / (r_Ll*coef_Ll[0]**3))**2)

K_pa_Fe = (K_pat_Al*r_Al*coef_Al[0]**2)/(r_Fe*coef_Fe[0]**2)
u_K_pa_Fe = np.sqrt( (0.001*K_pat_Al*coef_Al[0]**2 / (r_Fe*coef_Fe[0]**2))**2 + (np.sqrt(cov_Al[0][0])*K_pat_Al*r_Al*2*coef_Al[0] /(r_Fe*coef_Fe[0]**2))**2 + (0.001*K_pat_Al*r_Al*coef_Al[0]**2/(r_Fe**2 * coef_Fe[0]**2))**2 + (cov_Fe[0][0]*2*K_pat_Al*coef_Al[0]**2 / (r_Fe*coef_Fe[0]**3))**2)

print()
print('Patró alumini: ',K_pat_Al)
print('Patró alumini, llautó: ',K_pa_Ll, 'amb una incertesa de ',u_K_pa_Ll)
print('Patró alumini, ferro: ',K_pa_Fe, 'amb una incertesa de ',u_K_pa_Fe)