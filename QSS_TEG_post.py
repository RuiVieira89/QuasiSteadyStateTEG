import numpy as np 
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import inspect



dx = 0.01  # [m] slicing for calculations
number_of_coolers = 5  # number of coolers


select = 1
if select == 1:
    data = pd.read_csv(
        "AE.csv",
        names=['Exhaust Power [kW]', 'Exhaust Mass [kg/s]', 'T_EVO [K]'])
    N_time_steps = 1147
elif select == 2:
    data = pd.read_csv(
        "WLTP.csv",
        names=['Exhaust Power [kW]', 'Exhaust Mass [kg/s]', 'T_EVO [K]'])
    N_time_steps = 1801

data['Exhaust Mass [kg/s]'] *= 1e-3
M_air = data['Exhaust Mass [kg/s]']
E_exhaust = ['Exhaust Power [kW]']

N_time_steps = len(data) -1

Twater_in = 30 + 273.15  # [k] water inlet temperature

## AIR HEAT EXCHANGER MODULE (OFFSET FINS) -- inlet conditions ####################
#geometry definition
s = 0.003  # [m] fin spacing
h = 0.0189  # [m] fin heigth
thick = 0.0005  # [m] fin width
l = 0.0531  # [m] offset length

number_of_channels = 100  # [-] number of channels
number_of_offsets = 13  # [-] number of offsets

HE_length = number_of_offsets * l  # [m] system total length

# Correcting the dx to get an integer number of slices
N_slices = 1 + HE_length / dx  # number of slices
N_slices = round(N_slices)
dx = HE_length / N_slices
N_HE = N_slices
# of correction

N_slices_cooler = N_slices / number_of_coolers  # number of dx slices per cooler length

## TEG MODULE

n_pares = 49
largura_elemento = 4e-3
L_TEG = 1e-3
A_elemento = largura_elemento**2
A_modulo = (63e-3)**2

## POST PROCESSING ###################################

print("Post-Processign")

# variable declaration
aa = 10000
T = np.zeros((aa,N_time_steps))
Cp_air = np.zeros((aa,N_time_steps))
Cp_air_av = np.zeros(N_time_steps)
effct = np.zeros(N_time_steps)
Q_hot_acc = np.zeros(N_time_steps)

R_air = np.zeros(N_slices)
R_cooler = np.zeros(N_slices)
R_total = np.zeros(N_slices)
Q_to_cooler = np.zeros(N_slices)
Q_balance = np.zeros(N_slices)
Q_TEGslice = np.zeros((N_slices, N_time_steps))

T_H_TEG_av = np.zeros((2*number_of_coolers,N_time_steps))
T_C_TEG_av = np.zeros((2*number_of_coolers,N_time_steps))
V0_par = np.zeros((2*number_of_coolers,N_time_steps))
V0_modulo = np.zeros((2*number_of_coolers,N_time_steps))
I = np.zeros((2*number_of_coolers,N_time_steps))
Q_joulle = np.zeros((2*number_of_coolers,N_time_steps))
Q_hot_TEG = np.zeros((2*number_of_coolers,N_time_steps))
Q_cold_TEG = np.zeros((2*number_of_coolers,N_time_steps))
Q_hot_peltier = np.zeros((2*number_of_coolers,N_time_steps))
Q_cold_peltier = np.zeros((2*number_of_coolers,N_time_steps))
Q1 = np.zeros((2*number_of_coolers,N_time_steps))
Q2 = np.zeros((2*number_of_coolers,N_time_steps))
P_elect = np.zeros((2*number_of_coolers,N_time_steps))
eff_TEG = np.zeros((2*number_of_coolers,N_time_steps))

P_elect_total = np.zeros(N_time_steps)


## load the data

with open('var_dir.txt', 'r') as f:
    NAME = f.read()


pikle_in = open(f"{NAME}/T_H_TEG.pikle","rb")
T_H_TEG = pickle.load(pikle_in)

pikle_in = open(f"{NAME}/T_C_TEG.pikle","rb")
T_C_TEG = pickle.load(pikle_in)

pikle_in = open(f"{NAME}/Th.pikle","rb")
Th = pickle.load(pikle_in)

pikle_in = open(f"{NAME}/Tc.pikle","rb")
Tc = pickle.load(pikle_in)

pikle_in = open(f"{NAME}/Q_deficit.pikle","rb")
Q_deficit = pickle.load(pikle_in)

pikle_in = open(f"{NAME}/Q_acumulado.pikle","rb")
Q_acumulado = pickle.load(pikle_in)

pikle_in = open(f"{NAME}/Q_hot.pikle","rb")
Q_hot = pickle.load(pikle_in)

pikle_in = open(f"{NAME}/Q_excess.pikle","rb")
Q_excess = pickle.load(pikle_in)

VAR = [T_H_TEG, T_C_TEG, Th, Tc, Q_deficit, Q_acumulado, Q_hot, Q_excess]

def propTEG_N(Temp):
    T = Temp # all temperatures must be in [K]
    ## Hi-z data

    # Seebeck coeficient [V/K]
    alfa_Bi2Te3N = 0.00007423215 - 0.0000015018 * T + 0.0000000029361 * T**2 - 0.000000000002499 * T**3 + 0.000000000000001361 * T**4 # [V/K]

    # electrical resistivity [ohm.m]
    rho_Bi2Te3N = -0.00195922 + 0.00001791526 * T - 0.00000003818 * T**2 + 0.000000000049186 * T**3 - 0.0000000000000298 * T**4# ohm.cm
    rho_Bi2Te3N = rho_Bi2Te3N/100 # ohm.m

    # thermal conductivity [W/mK]
    k_Bi2Te3N = 1.425785 + 0.006514882 * T - 0.00005162 * T**2 + 0.00000011246 * T**3 - 0.000000000076 * T**4 # W/mK
    
    prop_N = [alfa_Bi2Te3N, rho_Bi2Te3N, k_Bi2Te3N] 
    
    return prop_N

def propTEG_P(Temp):
    T = Temp # all temperatures must be in [K]
    ## Hi-z data

    # Seebeck coeficient [V/K]
    alfa_Bi2Te3P = -0.0002559037 + 0.0000023184 * T - 0.000000003181 * T**2 + 0.0000000000009173 * T**3 - 0.000000000000000488 * T**4 # [V/K]

    # electrical resistivity [ohm.m]
    rho_Bi2Te3P = -0.002849603 + 0.00001967684 * T - 0.00000003317 * T**2 + 0.000000000034733 * T**3 - 0.000000000000019 * T**4 # ohm.cm
    rho_Bi2Te3P = rho_Bi2Te3P/100 # ohm.m

    # thermal conductivity [W/mK]
    k_Bi2Te3P = 6.9245746 - 0.05118914 * T + 0.000199588 * T**2 - 0.0000003891 * T**3 + 0.00000000030382 * T**4 # W/mK
    
    prop_P = [alfa_Bi2Te3P, rho_Bi2Te3P, k_Bi2Te3P] 
    
    return prop_P
    

## effectiveness calculation
#Cp_air_av(t)=sum(air_specific_heat(1:N_slices,t))/N_slices # [J/kgK] average air specific heat in the time step t

#average Cp
for t in range(len(data)-1):
    T[0][t]=Th[0][t]
    d=0

    while(T[d][t]>Twater_in):
        
        d=d+1
        
        if d==1:
            
            T[0][t]=Th[0][t]
            
        else:
            
            T[d][t]=T[d-1][t]-1
            
        Cp_air[d-1][t]=(((1.3864*10**-13)*T[d][t]**4)-((6.4747*10**-10)*T[d][t]**3)+((1.0234*10**-6)*T[d][t]**2)-(4.3282*10**-4)*T[d][t]+1.0613)*1000
    
    for kk in range(d):
        Cp_air_av[t]=(Cp_air[kk][t])/((d-1)+1) + Cp_air_av[t]
    
    for k in range(N_slices):
        Q_hot_acc[t] = Q_hot[k][t] + Q_hot_acc[t]

    effct[t]=(Q_hot_acc[t])/(0.5* M_air[t]*Cp_air_av[t]*(Th[0][t]-Twater_in)) # effectiveness



    ## TEG mean cold and hot side temperatures

    i =0
    for ss in range(2*number_of_coolers):
        
        i = ss

        T_H_TEG_av[ss][t] = np.average(T_H_TEG[int((i)*N_slices_cooler/2):int( (i+1) *N_slices_cooler/2),t])
        T_C_TEG_av[ss][t] = np.average(T_C_TEG[int((i)*N_slices_cooler/2):int( (i+1) *N_slices_cooler/2),t])


        Temp = (T_H_TEG_av[ss][t] + T_C_TEG_av[ss][t])/2
        prop_N = propTEG_N(Temp)
        alfa_Bi2Te3N = prop_N[0]
        rho_Bi2Te3N = prop_N[1]
        k_Bi2Te3N = prop_N[2]

        prop_P = propTEG_P(Temp)
        alfa_Bi2Te3P = prop_P[0]
        rho_Bi2Te3P = prop_P[1]
        k_Bi2Te3P = prop_P[2]
    
        Spn = alfa_Bi2Te3P - alfa_Bi2Te3N
        
        k_av = (k_Bi2Te3N+k_Bi2Te3P)/2
        rho_av = (rho_Bi2Te3N+rho_Bi2Te3P)/2   
        
        R_e_par = (2*rho_av*L_TEG)/A_elemento
        R_e_modulo = n_pares*R_e_par
        
        V0_par[ss][t] = Spn*(T_H_TEG_av[ss][t]-T_C_TEG_av[ss][t])
        V0_modulo[ss][t] = n_pares*V0_par[ss][t]
        
        I[ss][t] = V0_modulo[ss][t]/(R_e_modulo+R_e_modulo) # R_load = R_e_modulo
        
        Q_joulle[ss][t] = (L_TEG*rho_av*I[ss][t]**2)/A_elemento # [W]
        Q_hot_TEG[ss][t] = 2*A_elemento*((k_av/L_TEG)*(T_H_TEG_av[ss][t] - T_C_TEG_av[ss][t]) - Q_joulle[ss][t])
        Q_cold_TEG[ss][t] = 2*A_elemento*((k_av/L_TEG)*(T_H_TEG_av[ss][t] - T_C_TEG_av[ss][t]) + Q_joulle[ss][t])
        
        Q_hot_peltier[ss][t] = I[ss][t]*Spn*T_H_TEG_av[ss][t]
        Q_cold_peltier[ss][t] = I[ss][t]*Spn*T_C_TEG_av[ss][t]
        
        Q1[ss][t] = Q_hot_TEG[ss][t] + Q_hot_peltier[ss][t]
        Q2[ss][t] = Q_cold_TEG[ss][t] + Q_cold_peltier[ss][t]
        
        P_elect[ss][t] = Q1[ss][t] - Q2[ss][t]
        
        eff_TEG[ss][t] = P_elect[ss][t]/Q1[ss][t] 
        
    P_elect_total[t] = 2*4*number_of_coolers*np.sum(P_elect[:,t])

print('Average: ', np.average(P_elect_total),'[W]','\n')

plt.figure(1)
xx = np.arange(0,N_time_steps,1)
plt.scatter(xx,P_elect_total)

plt.figure(2)
xx = np.arange(0,ss+1,1)
plt.scatter(xx,P_elect[:,495])

plt.figure(3)
xx = np.arange(0,ss+1,1)
plt.scatter(xx,eff_TEG[:,495])


plt.show()


'''
plt.figure(1)
xxx = np.arange(0, 2*number_of_coolers, 1)
plt.scatter(xxx,T_H_TEG_av[:,0])


plt.figure(1)
xxx = np.arange(0, 2*number_of_coolers, 1)
plt.scatter(xxx,T_C_TEG_av[:,0])

plt.show()


plt.figure(1)
xxx = np.arange(0, N_time_steps, 1)
plt.scatter(xxx,effct)
plt.show()
    

plt.figure(1)
xxx = np.arange(0, N_slices, 1)
plt.scatter(xxx,Th[:,0])


plt.figure(1)
xxx = np.arange(0, N_slices, 1)
plt.scatter(xxx,Tc[:,0])

plt.figure(1)
xxx = np.arange(0, N_slices, 1)
plt.scatter(xxx,T_H_TEG[:,0])

plt.figure(1)
xxx = np.arange(0, N_slices, 1)
plt.scatter(xxx,T_C_TEG[:,0])

plt.show()
'''
