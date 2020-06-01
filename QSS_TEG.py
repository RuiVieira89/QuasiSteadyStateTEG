
import time

start_time = time.time()

import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import numpy as np

print("QSS_TEG.py", "\n")


## funtions declarations ##################################

k_Al = 190

m_cooler = 0.0207916666666667 ## [kg/s] cooling water mass flow rate per plate

import numba

elapsed_time = time.time() - start_time
timeElapsed1_old = elapsed_time

def ETA(start_time,time_par,final_time_par,error_par,it):
    
    global timeElapsed1_old
    
    elapsed_time = time.time() - start_time
    write_increment = 1 # set your time write increment
    
    print('\n', 'Writing step: ', str(time_par), '/',str(final_time_par), ' [s]', '\n')
    print('Error = ',str(error_par))

    if write_increment == time_par+1:
        print('in ', str(it),' iterations; ',str( -((elapsed_time)*( 1/write_increment)*(time_par-final_time_par))/(60*60) ),
              'hrs remaining (estimated)','\n')
    else:
        print('in ', str(it),' iterations for ',str(elapsed_time-timeElapsed1_old),'s; ETA: ',
              str( -((elapsed_time-timeElapsed1_old)*( 1/write_increment)*(time_par-final_time_par))/(60*60) ),'hrs','\n')
    
    
    print('Clock Time = ', str(elapsed_time),' s')
    
    timeElapsed1_old = elapsed_time


@numba.jit(nopython=True)
def Water_Properties(Temp):

    ## water properties
    T = Temp
 
    # [Pa.s] LIQUID water viscosity (ZGGRAFOS,1987)
    visc_liq = 3.8208*(10**-2)*(T-252.33)**-1 

    # [kg/m3] LIQUID water volumic mass 1atm (ZGGRAFOS,1987)
    rho_liq = -3.0115*(10**-6)*(T**3)+9.6272*(10**-4)*(T**2)-0.11052*T+1022.4 

    # [W/mK] water thermal conductivity
    k_water = 4.2365*(10**-9)*(T**3)-1.144*(10**-5)*(T**2)+7.1959*(10**-3)*T-0.63262 

    # [J/kgK] water cp (Zografos, 1986)
    Cp_water = 1000*(1.785*(10**-7)*(T**3)-1.9149*(10**-4)*(T**2)+6.7953*(10**-2)*T-3.7559)
    
    properties = [visc_liq, rho_liq, k_water,Cp_water]

    return properties

@numba.jit(nopython=True)
def R_cooler_water (rho_water, k_water,k_Al, Cp_water,visc_water, m_cooler, dx, Nc,fin_widt,fin_heigt,channel_length):

    #WATER COOLER MODULE
    #the cooling plates have small channels where the water flows 
    #the correlation for laminar flow can be found in tese: Rui Vieira, 2017

    #geometry definition
    fin_area_coller=2*fin_heigt*channel_length*Nc # [m2] Af finned area
    fin_perimeter_coller=2*(fin_widt+channel_length) # [m] fin tip perimeter
    non_fin_area=Nc*fin_widt*channel_length # [m2] Ab area of the base with no fin
    total_HT_area=fin_area_coller+non_fin_area # [m2] At=Ab+Af total heat tranfer area
    flow_area=fin_widt*fin_heigt # [m2] single channel flow area
    D_charact_cooler=(flow_area)**0.5 # [m] characteristic lenght A**1/2

    #Cooler convection model constants, check tese Rui Vieira 2017

    Pr_water =(Cp_water *visc_water )/k_water 

    C1=3.24 # Uniform heat power
    C2=1.5
    C3=0.409 # Uniform heat power
    C4=2
    Par_forma=0.1 # Par de forma (>=90)
    ee=fin_widt/fin_heigt # channel aspect ratio e=a/b of a rectangule

    fPr =0.564/((1+(1.664*Pr_water **(1/6))**(9/2))**(2/9)) # Uniform heat power
    m_coef_cool =2.27+1.65*Pr_water **(1/3) # m coefficient 


    # Flow conditions
    #            m_cooler_channel=m_cooler/Nc # [kg/s] cooling water mass flow per channel

    # [m3/s] cooling water volumic flow per channel
    Q_water_channel =(m_cooler/rho_water )/Nc 

    # [m/s] water velocity in the channels
    u_water =Q_water_channel /flow_area 


    #water Reynolds number
    Re_water =(rho_water *u_water *D_charact_cooler)/visc_water  

    L_coef =0.058*D_charact_cooler*Re_water  # [m] L' entry region
    Z_plus =((L_coef /channel_length)**-1)/(Re_water ) # Z+ coefficient
    Z_star =((channel_length/D_charact_cooler)/(Re_water *Pr_water )) # Z* coefficient

    # fRe(A**0.5) coefficient
    fRe =((12/((ee**0.5)*(1+ee)*(1-((192*ee)/(np.pi**5))*np.tanh(np.pi/(2*ee)))))**2+(3.44/Z_plus **0.5)**2)**0.5 

    #Nusselt number
    Nu_water =((((C4*fPr )/(Z_star **0.5))**m_coef_cool )+((((C2*C3*(fRe /Z_star )**(1/3))**5)+\
        (C1*(fRe /(8*(np.pi**0.5)*(ee**Par_forma))))**5)**(m_coef_cool /5)))**(1/m_coef_cool ) 
    # [Wm2/K] heat transfer coefficient
    h_water =(Nu_water *k_water )/D_charact_cooler 

    # m fin efficiency coeficient
    m_coef_water = ((h_water *fin_perimeter_coller)/(k_Al*non_fin_area))**0.5 
    # fin efficiency
    fin_eff_water =(np.tanh(m_coef_water *fin_heigt))/(m_coef_water *fin_heigt) 
    # group fin efficiency
    fin_eff_water_group =1-(fin_area_coller/total_HT_area)*(1-fin_eff_water ) 


    #Cooler thermal resistance [K/W]
    R_cooler = (1/(h_water *(total_HT_area/(channel_length/dx))*fin_eff_water_group ))/2
    
    return R_cooler

@numba.jit(nopython=True)
def Prop_air (Temp):

    T=Temp

    #AIR PROPERTIES IN S.I. (Zografos, 1986) (equações na tese Rui Vieira, 2017 no ultimo anexo)
    #Calculates the air resistance for Th 

    #viscosity as a function of hot inlet temperature [Pa.s]
    air_viscosity=((2.5914*10**(-15))*T**3)-((1.4346*10**(-11))*T**2)+((5.0523*10**(-8))*T)+4.113*10**(-6) #[Pa.s]
            
    #density as a function of hot inlet temperature [kg/m3]
    air_density=101325/((8314.4598/28.97)*T) #[kg/m3]
            
    #air conductivity [W/mK]
    air_conductivity=((0.000000000015207)*T**3) - ((0.000000048574)*T**2 )+( (0.00010184)*T)- 0.00039333
            
    #Specific Heat [J/kg.K]
    air_specific_heat=(((1.3864*10**-13)*T**4)-((6.4747*10**-10)*T**3)+((1.0234*10**-6)*T**2)-(4.3282*10**-4)*T+1.0613)*1000
    
    properties = [air_viscosity,air_density,air_conductivity,air_specific_heat]
    
    return properties

@numba.jit(nopython=True)
def R_Exhaust (air_viscosity,air_conductivity,air_specific_heat,m_air, N_HE):


    #geometry definition

    s=0.003   # [m] fin spacing
    h=0.0189 # [m] fin heigth
    tick=0.0005 # [m] fin width
    l=0.0531 # [m] offset length

    number_of_channels=100 # [-] number of channels 
    number_of_offsets=10 # [-] number of offsets


    N_slices = N_HE #1+HE_length/dx # number of slices 

    k_ss=20 # [W/mK] thermal conductivity for the offset fins
    #20 W/mK average value for the thermal conductivity of the stainless steel

    #estas areas são para o PC todo, para os calculos das alhetas,ao
    # dividir o PC a meio, é necessário dividir por 2!
    total_area=2*(number_of_channels*number_of_offsets)*(s*l+h*l) #[m2] total heat transfer area 

    #hidraulic diameter for offset fins

    alfa_geo=s/h
    sigma_geo=tick/l
    gama_geo=tick/s

    D_hidraulic=(4*s*h*l)/(2*(s*l+h*l+tick*h)+tick*s) # [m] hydrauslic dyameter

    #Dynamic and themal parameters 
    #Reynolds Number
    Re=(4/np.pi)*(((m_air/number_of_channels))/(air_viscosity*D_hidraulic))

    #Colburn factor (j)
    j=(0.6522*(Re**-0.5403)*(alfa_geo**-0.1541)*(sigma_geo**0.1499)*(gama_geo**-0.0678))*\
        (1+(5.269*10**-5)*(Re**1.34)*(alfa_geo**0.504)*(sigma_geo**0.456)*(gama_geo**-1.055))**0.1

    #Prandtl number
    Pr=(air_specific_heat*air_viscosity)/air_conductivity

    #Nusselt number
    Nu=j*Re*(Pr)**(1/3)

    #Heat tranfer coefficient [W/m2K]
    h_air=(air_conductivity*Nu)/D_hidraulic

    m_fin=((h_air*2*(tick+l))/(k_ss*tick*l))**0.5

    #Fin efficiency 
    fin_eff=(np.tanh(m_fin*h/2))/(m_fin*h/2)

    #Fin group efficiency  
    fin_eff_group=1-((2*h/2)/(2*h/2+s))*(1-fin_eff)

    #Exhaust heat exchanger thermal resistance [K/W]
    R_air=(1/(((total_area/2)/N_slices)*h_air*fin_eff_group))
    
    Exhaust_therm = [R_air,h_air]

    return Exhaust_therm

##  function declarations #################################


select = 2
if select == 1:
    data = pd.read_csv(
        "AE.csv",
        names=['Exhaust Power [kW]', 'Exhaust Mass [kg/s]', 'T_EVO [K]'])
    N_time_steps = 1147
elif select == 2:
    data = pd.read_csv(
        "WLTP.csv",
        names=['Exhaust Power [kW]', 'Exhaust Mass [kg/s]', 'T_EVO [K]'])
    N_time_steps = 1800

data['Exhaust Mass [kg/s]'] *= 1e-3

## intial parameters

max_it = 100000  #maximum number of iterations
dx = 0.01  # [m] slicing for calculations
number_of_coolers = 5  # number of coolers

T_TEG_MAX = 250 + 273.15  # [K] maximum allowed temperature on the hot TEG face
Twater_in = 30 + 273.15  # [k] water inlet temperature

## cooler geometry #################################################################
Nc = 70  # number of channels
channel_widt = 0.001  # channel width [m]
fin_widt = 0.001  # fin widt [m]
fin_heigt = 0.01  # fin heigt [m]
channel_length = 0.13  # channel length [m]
lateral_length = 0.14  # lateral length [m] comprimento transversal
Metal_tick = 0.01  # [m] metal thickness joined by the fins

fin_area_coller = 2 * fin_heigt * channel_length * Nc  # [m2] Af finned area
fin_perimeter_coller = 2 * (fin_widt + channel_length)  # [m] fin tip perimeter
non_fin_area = Nc * fin_widt * channel_length  # [m2] Ab area of the base with no fin
total_HT_area = fin_area_coller + non_fin_area  # [m2] At=Ab+Af total heat tranfer area
flow_area = fin_widt * fin_heigt  # [m2] single channel flow area
Dh_cooler = (4 * flow_area) / (2 * (fin_widt + fin_heigt)
                               )  # [m] hidraulic diameter 4A/P
D_charact_cooler = (flow_area)**0.5  # [m] characteristic lenght A**1/2
###################################################################################

## TEG MODULE MODULE ##############################################################
#TEG module goemetric parameters and properties
TEG_thick = 0.0053  # [m] TEG module thickness

# [W/mK] TEG module thermal conuctivity, determined from experimental data->f(Q,T,A)
k_TEG = (304.6 * TEG_thick) / ((250 - 30) * ((62**2) * 10**-6))  # [W/mK]

TEG_area = (62 * 10**-3)**2  # [m2] TEG module area

R_TEG = TEG_thick / (k_TEG * TEG_area / (
    (62 * 10**-3) / dx))  # [K/W] ONE! TEG module thermal resistance for dx
R_TEG_four = R_TEG / 4  # [K/W] two! TEG module thermal resistance, in parallel for dx
###################################################################################

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

k_ss = 20  # [W/mK] thermal conductivity for the offset fins
#20 W/mK average value for the thermal conductivity of the stainless steel

#estas areas são para o PC todo, para os calculos das alhetas,ao
# dividir o PC a meio, é necessário dividir por 2!
total_area = 2 * (number_of_channels * number_of_offsets) * (
    s * l + h * l)  #[m2] total heat transfer area
finned_area = 2 * number_of_channels * number_of_offsets * h * l  # [m2] area added by the fins
planar_area = 2 * number_of_channels * number_of_offsets * (
    thick + s
) * l  #[m2] heat transfer area without the fins {na tese devia dividir por 2!!}

alfa_geo = s / h
sigma_geo = thick / l
gama_geo = thick / s

D_hidraulic = (4 * s * h * l) / (2 * (s * l + h * l + thick * h) + thick * s
                                 )  # [m] hydrauslic dyameter
###################################################################################

## Heat Pipe MODULE ###############################################################
k_HP = 189  # [W/mK] HP material thermal conductivity, typical for alloy Copper

# [W/mK] HP thermal resistance (before phase change) T_TEG_HOT<250ºC
#for the length of a section (see Rui Vieira, 2017)
R_HP_not_work = 0.0165300185136207  # [W/mK]

# [W/mK] thermal resistance for T_TEG_HOT<250ºC for the 8 HP in parallel for the lenght of a section
R_HP_not_work_all = 0.5 * (R_HP_not_work / 8) / (HE_length / dx)  # [W/mK]

# [W/mK] 8 working HP thermal resistance (after phase change) T_TEG_HOT=250ºC
#for the length of a section (see Rui Vieira, 2017)
R_HP_work = 0.00114862278067739 / (HE_length / dx)  # [W/mK]
###################################################################################

## Constact resistance MODULE #####################################################
h_silicone=75000 # heat transfer coefficient of thermal grease(siliconeoil),w/m**2K

R_contact_TEG = 1/( h_silicone * ( number_of_offsets*l*number_of_channels*s ) / N_slices )
###################################################################################

## variable declations ############################################################
Tc = np.zeros((N_slices, N_time_steps))
R_air = np.zeros(N_slices)
R_cooler = np.zeros(N_slices)
R_total = np.zeros(N_slices)
Q_hot = np.zeros((N_slices, N_time_steps))
#T_H_TEG = np.zeros((N_slices, N_time_steps))
T_H_TEG = np.full([N_slices, N_time_steps], Twater_in)
Q_to_cooler = np.zeros(N_slices)
Q_balance = np.zeros(N_slices)
Q_excess = np.zeros((N_slices, N_time_steps))
#T_C_TEG = np.zeros((N_slices, N_time_steps))
T_C_TEG = np.full([N_slices, N_time_steps], Twater_in)
Q_acumulado = np.zeros((N_slices, N_time_steps))
Q_TEGslice = np.zeros((N_slices, N_time_steps))
Q_deficit = np.zeros((N_slices, N_time_steps))

Q_TEGmodule = np.zeros((2*number_of_coolers, N_time_steps))

Q_total = np.zeros(N_time_steps)
Q_it = np.zeros(max_it+1)

M_air = data['Exhaust Mass [kg/s]']

Th = np.array(data['T_EVO [K]'])
Zero = np.zeros((N_slices-1, N_time_steps+1))
Th = np.append(Th[None,:], Zero, axis=0)


###################################################################################


for t in range(len(data)-1): # -1 accounts for the title 
    it = 0
    error_it = 1

    while (error_it > 1e-4 and it < max_it and np.isnan(error_it) ==0):
        it += 1

        for x in range(N_slices-1):  # distance discretization  range([start], stop[, step])


            if (Tc[x][t] < 273 or Tc[x+1][t] < 273 or np.isreal(Tc[x][t]) == 0) and it <= 2.1:
                
                if x == 0:
                    Tc[x][t]=311
                Tc[x + 1][t] = Tc[x][t] -0.1
            
            if it <= 2.1:
                if x == 1:
                    
                    Tc[x][t] = 311
                                   
                Th[x + 1][t] = Th[x][t] - 5
                
            if Th[x][t]<0 or np.isreal(Th[x][t])==0 or Th[x][t]>Th[0][t]:
                
                Th[x][t]=Th[0][t] - 5*x
                
            if ((x + 1) % N_slices_cooler) == 0:
                
                # water inlet conditions
                Tc[x][t]=Twater_in

            
            ## Properties calculation #######################
            Temp_hot = Th[x][t]
            [air_viscosity,air_density,air_conductivity,air_specific_heat] = Prop_air(Temp_hot)
            m_air = M_air[t]
            
            [R_air_out, h_air] = R_Exhaust (air_viscosity,air_conductivity,air_specific_heat,m_air, N_HE)
            R_air[x] = R_air_out

            Temp_cold = Tc[x][t]
            [visc_liq, rho_liq, k_water,Cp_water] = Water_Properties(Temp_cold)
            rho_water = rho_liq
            visc_water = visc_liq
            R_cooler[x] = R_cooler_water (rho_water, k_water,k_Al, Cp_water,visc_water, m_cooler, dx, Nc,fin_widt,
                                          fin_heigt,channel_length)
            ##################################################

            #### POWER CALCULATIONS ##########################
            # [K/W] total thermal resistance
            R_total[x]=(R_air[x]+R_cooler[x]+R_TEG_four+2*R_HP_not_work_all + 2*R_contact_TEG) # [K/W] 
            # [W] energy lost by the exhaust gases to the HE
            Q_hot[x][t]=(Th[x][t]-Tc[x][t])/R_total[x] # [W] 
            
            # [K] Temperature of the TEG hot face
            T_H_TEG[x][t]=Th[x][t]-Q_hot[x][t]*(2*R_HP_not_work_all+R_air[x] + R_contact_TEG) # [K] 
            
            # Excess or deficit to the optimum point --> T_H_TEG[x][t]=T_TEG_MAX
            Q_balance[x]=(T_H_TEG[x][t]-T_TEG_MAX)/(R_air[x]+R_HP_not_work_all + R_contact_TEG) # [W]
            
            
            if T_H_TEG[x][t]>(T_TEG_MAX): # [K] determining if the HP is working
                
                #  [K/W] total thermal resistance
                R_total[x]=R_air[x]+R_cooler[x]+R_TEG_four+2*R_HP_not_work_all + 2*R_contact_TEG # [K/W] 
                
                # [W] energy lost by the exhaust gases
                Q_hot[x][t]=(Th[x][t]-T_TEG_MAX)/(R_air[x]+R_HP_not_work_all) # [W] 
                
                # [W] power dissipated by the cooler
                Q_to_cooler[x]=((T_TEG_MAX)-Tc[x][t])/(2*R_contact_TEG+R_TEG_four+R_cooler[x]+R_HP_not_work_all) # [W] 
                
                # [W] excess heat that is transfered by the HP to the next slice
                Q_excess[x][t]=Q_hot[x][t]-Q_to_cooler[x] # [W]  
                
                
                #cooler calculation
                if ((x + 1) % N_slices_cooler) == 0: # (x +1 # N_slices_cooler) == 0: porque o array começa em 0
                    
                    # water inlet conditions
                    Tc[x][t]=Twater_in
                    
                else:
                    
                    # [K] cold temperature update
                    Tc[x][t]=(Q_to_cooler[x]/(2*m_cooler*Cp_water))+Tc[x+1][t] # [K] 
                    
                
                
                # [K] Temperature of the TEG cold face
                T_C_TEG[x][t]=(Tc[x][t])+(Q_to_cooler[x]*(R_contact_TEG + R_cooler[x])) # [K] 
                
                # [K] Temperature of the TEG cold face
                T_H_TEG[x][t]=T_TEG_MAX # [K] temperature set by the HP
                
                
                # [K] hot exhaust gases temperature update
                Th[x+1][t]=Th[x][t]-(Q_hot[x][t]/(0.5*m_air*air_specific_heat))
                
                
                if (Q_excess[x][t])>0:
                    for i in range(N_slices-1):
                        #determining the excess acumulated
                        Q_acumulado[i][t]= Q_acumulado[i][t] + Q_excess[i][t]
                    
                else:
                    
                    #there is no change 
                    Q_acumulado[x+1][t]=Q_acumulado[x][t]
                    
                
                
                #  of the EXCESS calculation procedure
                
            else: # no excess generation in this slice
                
                Q_excess[x][t]=0
                
                if x==0: # if there is no execess in this dt
                    Q_acumulado[x][t]=0
                else:
                    Q_acumulado[x][t]=Q_acumulado[x-1][t]
                
                
                
                if Q_acumulado[x][t]>0 and Q_acumulado[x][t]>abs(Q_balance[x]): #  excess heat available?
                    
                    # Q_acumulado update (subtrair o que foi gasto)
                    Q_acumulado[x][t]=Q_acumulado[x-1][t]-abs(Q_balance[x]) # [W]
                    
                    # TEG Hot side temperature 
                    T_H_TEG[x][t]=T_TEG_MAX # [K]
                    
                    # [W] power dissipated by the cooler
                    Q_to_cooler[x]=((T_TEG_MAX)-Tc[x][t])/(2*R_contact_TEG + R_TEG_four+R_cooler[x]) # [W] 
                    
                    # Energy lost from the exhaust gases to cooler
                    Q_hot[x][t]=(Th[x][t]-T_TEG_MAX)/(R_air[x]+R_HP_not_work_all) # [K]
                    
                    
                    #cooler calculations
                    if ((x + 1) % N_slices_cooler) == 0:
                        
                        # water inlet conditions
                        Tc[x][t]=Twater_in
                        
                    else:
                        
                        # [K] cold temperature update
                        Tc[x][t]=(Q_to_cooler[x]/(2*m_cooler*Cp_water))+Tc[x+1][t] # [K] 
                    
                    
                    # [K] hot exhaust gases temperrature update
                    Th[x+1][t]=Th[x][t]-(Q_hot[x][t]/(0.5*m_air*air_specific_heat))
                    
                    # [K] Temperature of the TEG cold face
                    T_C_TEG[x][t]=(Tc[x][t])+(Q_to_cooler[x]*(R_contact_TEG+R_cooler[x])) # [K] 
                    
                else: # NO excess heat available
                    
                    if Q_acumulado[x][t]>0:
                        
                        Q_hot[x][t]=Q_hot[x][t]+Q_acumulado[x][t]
                        
                        Q_acumulado[x][t]=0
                        
                        Q_to_cooler[x]=0
                        
                        # [K] hot exhaust gases temperrature update
                        Th[x+1][t]=Th[x][t]-(Q_hot[x][t]/(0.5*m_air*air_specific_heat))
                    
                        
                        #cooler calculations
                        if ((x + 1) % N_slices_cooler) == 0:
                            
                            # water inlet conditions
                            Tc[x][t]=Twater_in
                            
                        else:
                        
                            # [K] cold temperature update
                            Tc[x][t]=(Q_hot[x][t]/(2*m_cooler*Cp_water))+Tc[x+1][t] # [K] 
                        
                
                        # [K] Temperature of the TEG cold face
                        T_C_TEG[x][t]=(Tc[x][t])+(Q_hot[x][t]*(R_contact_TEG+R_cooler[x])) # [K] 
                        
                    else:
                    
                        Q_acumulado[x][t]=0 # !!
                        Q_to_cooler[x]=0
                    
                        # [K] hot exhaust gases temperrature update
                        Th[x+1][t]=Th[x][t]-(Q_hot[x][t]/(0.5*m_air*air_specific_heat))
                    
                        
                        # cooler calculations
                        if ((x + 1) % N_slices_cooler) == 0:
                            
                            # water inlet conditions
                            Tc[x][t]=Twater_in
                            
                        else:
                        
                            # [K] cold temperature update
                            Tc[x][t]=(Q_hot[x][t]/(2*m_cooler*Cp_water))+Tc[x+1][t] # [K] 
                        
                
                        # [K] Temperature of the TEG cold face
                        T_C_TEG[x][t]=(Tc[x][t])+(Q_hot[x][t]*(R_contact_TEG+R_cooler[x])) # [K] 

            if it>max_it: # maximum allowed iterations number
                
                T_H_TEG[x][t]=T_TEG_MAX
                print("skip_T")
                print(t)
                print(error_it)
                break
                    
            # Power traversing the TEG module per dx
            if T_H_TEG[x][t]==T_TEG_MAX:
                
                Q_TEGslice[x][t]=Q_to_cooler[x]
                
            else:
                
                Q_TEGslice[x][t]=Q_hot[x][t]
                            
            # Q_deficit calculation
            if Q_balance[x]>=0:
                
                Q_deficit[x][t]=0
                
            else:
                
                Q_deficit[x][t]=Q_balance[x]
                
        #energy traversing each TEG module
        
        for jj in range(N_slices):
            
            if jj == 0:
                mod = 0
                for i in range(2*number_of_coolers):
                    Q_TEGmodule[i][t] = 0 
                    
                Q_total[t] = 0
                
            if (jj == (mod+1)*(N_slices/number_of_coolers)/2 and jj != 0): 
                mod += 1

            Q_TEGmodule[mod][t] = Q_TEGmodule[mod][t] + Q_TEGslice[jj][t]
            
            # [W] total heat transferred from the hot exhaust gases to the HE
            Q_total[t] = Q_total[t] + Q_hot[jj][t] + Q_to_cooler[jj] # [W]

        ## Residuals calculation
        Q_it[it]=Q_total[t]
        error_it = abs(Q_it[it]-Q_it[it-1])/abs(Q_it[it])
        
        
        if it>max_it: # maximum allowed iterations number
            
            print("skip_t")
            print(t)
            print(error_it(it))
            break
    
    if ((t + 1) % 100) == 0:
        elapsed_time = time.time() - start_time
        ETA(start_time,t,N_time_steps,error_it,it)
        print(t+1, " it: ", it, " error: ", error_it,"\n", "Elapsed_time: ",time.strftime("%H:%M:%S", time.gmtime(elapsed_time)),
              " hh:mm:ss \n")
        

    

print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))," hh:mm:ss \n")

## DATA SAVE
print("saving data")

import pickle
import os
import inspect

path = os.getcwd() # current directory
NAME = f"{path}/Results-{select}-{int(time.time())}"
os.mkdir(NAME) # create directory 

VAR = [T_H_TEG, T_C_TEG, Th, Tc, Q_deficit, Q_acumulado, Q_hot, Q_excess]

def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]

for var in VAR:
    
    VAR_NAME_all = retrieve_name(var)
    VAR_NAME = VAR_NAME_all[0]
    
    pickle_out = open(f"{NAME}/{VAR_NAME}.pikle", "wb")
    pickle.dump(var, pickle_out)
    pickle_out.close()

text_file = open("var_dir.txt", "w")
text_file.write(NAME)
text_file.close()



print("pickling done","\n")

elapsed_time = time.time() - start_time
print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))," hh:mm:ss", "\n")
print("ran in: ", elapsed_time,"\n")
print("")

print("End")
