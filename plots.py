import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.image as image
from matplotlib import patches
import scipy.optimize as opt
import pickle

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

#plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=15)     # fontsize of the axes title
plt.rc('axes', labelsize=15)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)

import constants
import delay_model

def plot1(inclination_in = 90):
    phases = np.linspace(0,1,1000)
    yy_3d = np.zeros(len(phases))
    vv_3d = np.zeros(len(phases))
    for i in range(len(phases)):
        yy_3d[i], vv_3d[i] = delay_model.timeDelay_3d(phases[i], plot = False, Q = 25, inclination_in = inclination_in)
    yy_cm = delay_model.timeDelay_sp(phases,setting='cm',inclination_in = inclination_in)
    yy_egg = delay_model.timeDelay_sp(phases,setting='egg',inclination_in = inclination_in)
    yy_plav = delay_model.timeDelay_sp(phases,setting='plav',inclination_in = inclination_in)
    
    
    v_cm = delay_model.radialVelocity(phases,setting='cm',inclination_in = inclination_in)
    v_egg = delay_model.radialVelocity(phases,setting='egg',inclination_in = inclination_in)
    v_plav = delay_model.radialVelocity(phases,setting='plav',inclination_in = inclination_in)
    v_md = delay_model.radialVelocity(phases,setting='md',inclination_in = inclination_in)
    
    fig, axs = plt.subplots(2)
    
    axs[0].plot(phases,yy_3d,'g-',label='3d')
    axs[0].plot(phases,yy_cm,'b:',label='cm')
    axs[0].plot(phases,yy_egg,'r--',label='egg')
    axs[0].plot(phases,yy_plav,'k-.',label='plav')
    
    axs[1].plot(phases,v_md,linestyle='-',color='purple',label='Munez Darias',linewidth = 4, alpha = 0.7)
    axs[1].plot(phases,v_cm,'b:',label='cm')
    axs[1].plot(phases,v_egg,'r--',label='egg')
    axs[1].plot(phases,v_plav,'k-.',label='plav')
    axs[1].plot(phases,vv_3d,'g-',label='3D')
    
    axs[0].legend()
    axs[1].legend()
    
    axs[1].set_xlabel('Orbital Phase')
    axs[0].set_ylabel('Time Delay (s)')
    axs[1].set_ylabel('Radial Velocity (km/s)')
    plt.show()

def plotRochePotential(m1_in=1.4, m2_in=0.7, period_in=.787, res1d = 1002):
    m1 = m1_in * constants.M_Sun
    m2 = m2_in * constants.M_Sun
    q = m2_in/m1_in
    period = period_in * constants.day
    
    a = (constants.G*(m1+m2)*(period/(2*np.pi))**2)**(1/3)
    L2 = (a*(.5+.227*np.log(q)/np.log(10)))
    
    xx, yy = np.linspace(-1.5*a,2*a, res1d), np.linspace(-1.75*a,1.75*a, res1d)
    XX, YY = np.meshgrid(xx, yy)
    
    RR, TT = np.sqrt(XX**2+YY**2), np.arctan2(YY,XX)
    
    VV = delay_model.rochePotential(RR,TT,m1=m1,m2=m2,period=period,beta=np.pi/2)
    vv2 = delay_model.rochePotential(L2,0,m1=m1,m2=m2,period=period,beta=np.pi/2)
    
    VV[VV<vv2] = 0
    
    fig, ax = plt.subplots()
    ax.imshow(-1*VV,extent=[-1.5*a,2*a,-1.75*a,1.75*a],vmin=1.2e11,vmax=1.8e11)
    ax.scatter([L2],[0],c='red')
    ax.set_aspect(1)
    plt.show(block = False)
    
    plt.figure()
    rr = np.linspace(0,1.5*a, res1d)
    vv = delay_model.rochePotential(rr,0,m1=m1,m2=m2,period=period,beta=0)
    
    r2 = m1/(m1+m2)*a
    eff_force = constants.G*m1/(a-rr)**2 - constants.G*m2/rr**2 - (2*np.pi/period)**2*(r2-rr)
    
    # ~ plt.plot(rr,vv)
    plt.plot(rr,eff_force,color='red')
    plt.axvline(x=L2)
    plt.axhline(y=0)
    plt.show()

def plot2(m1_in = 1.4, m2_in = 1.3, inclination_in = 45, period_in=.787):
    phases = np.linspace(0,1,100)
    alphas = np.array(np.arange(0,24,2))
    peak_point = []
    for ai in range(len(alphas)):
        yy_3d = np.zeros(len(phases))
        for i in range(len(phases)):
            yy_3d[i] = delay_model.timeDelay_3d(phases[i],m1_in=m1_in,m2_in=m2_in,inclination_in=inclination_in,period_in=period_in,plot = False, Q = 100, disk_angle_in = alphas[ai])
        peak_point.append(max(yy_3d))
        plt.plot(phases,yy_3d, label = r'$\alpha$ = ' + str(alphas[ai]) + ' deg', alpha = 0.9, c = cm.jet(alphas[ai]/max(alphas)))
        print(f'ai = {ai}')
    plt.legend()
    plt.xlabel('Orbital Phase')
    plt.ylabel('Time Delay (s)')
    
    plt.show()
    
    plt.plot(alphas, peak_point, 'k-')
    plt.xlabel(r'Disk Shielding Angle (deg)')
    plt.ylabel('Time Delay @ 0.5 phase')
    plt.show()
    
def plot3A(m1_in = 1, inclination_in = 90):
    data_dir = r'C:\Users\thoma\Documents\GitHub\echoxrb\data\mz_data'
    m2_in = 0.6*m1_in
    period_in = 1.5
    phases = np.linspace(0,1,20)
    yy_3d = np.zeros(len(phases))
    vv_3d = np.zeros(len(phases))
    scale = 1
    
    for i in range(len(phases)):
        vv_3d[i] = delay_model.timeDelay_3d_full(phases[i], plot = False, Q = 100, m1_in = m1_in, m2_in = m2_in, inclination_in = inclination_in, disk_angle_in = 0, period_in = period_in)
    plt.plot(phases,vv_3d*scale)
    
    data = np.loadtxt(f"{data_dir}//alpha0.txt")
    plt.scatter(data[:,0],data[:,1],label=r'$\alpha$ = 0')
    
    for i in range(len(phases)):
        vv_3d[i] = delay_model.timeDelay_3d_full(phases[i], plot = False, Q = 50, m1_in = m1_in, m2_in = m2_in, inclination_in = inclination_in, disk_angle_in = 8, period_in = period_in)
    plt.plot(phases,vv_3d*scale)
    
    data = np.loadtxt(f"{data_dir}//alpha8.txt")
    plt.scatter(data[:,0],data[:,1],label=r'$\alpha$ = 8')
    
    for i in range(len(phases)):
        vv_3d[i] = delay_model.timeDelay_3d_full(phases[i], plot = False, Q = 50, m1_in = m1_in, m2_in = m2_in, inclination_in = inclination_in, disk_angle_in = 14, period_in = period_in)
    plt.plot(phases,vv_3d*scale)
    
    data = np.loadtxt(f"{data_dir}//alpha14.txt")
    plt.scatter(data[:,0],data[:,1],label=r'$\alpha$ = 14')
    
    for i in range(len(phases)):
        vv_3d[i] = delay_model.timeDelay_3d_full(phases[i], plot = False, Q = 50, m1_in = m1_in, m2_in = m2_in, inclination_in = inclination_in, disk_angle_in = 18, period_in = period_in)
    plt.plot(phases,vv_3d*scale)
    
    data = np.loadtxt(f"{data_dir}//alpha18.txt")
    plt.scatter(data[:,0],data[:,1],label=r'$\alpha$ = 18')
    
    plt.xlabel('Orbital Phase')
    plt.ylabel('Radial Velocity (km/s)')
    plt.legend()
    plt.show()

def plot3B(m1_in = 1.5):
    data_dir = r'C:\Users\thoma\Documents\GitHub\echoxrb\data\mz_data'
    m2_in = 0.6*m1_in
    period_in = 0.60
    phases = np.linspace(0,1,20)
    yy_3d = np.zeros(len(phases))
    vv_3d = np.zeros(len(phases))
    scale = 0.9
    
    for i in range(len(phases)):
        vv_3d[i] = delay_model.timeDelay_3d_full(phases[i], plot = False, Q = 100, m1_in = m1_in, m2_in = m2_in, inclination_in = 90, disk_angle_in = 0, period_in = period_in)
    plt.plot(phases,vv_3d*scale)
    
    data = np.loadtxt(f"{data_dir}//i90.txt")
    plt.scatter(data[:,0],data[:,1],label=r'i = 90')
    
    for i in range(len(phases)):
        vv_3d[i] = delay_model.timeDelay_3d_full(phases[i], plot = False, Q = 100, m1_in = m1_in, m2_in = m2_in, inclination_in = 40, disk_angle_in = 0, period_in = period_in)
    plt.plot(phases,vv_3d*scale)
    
    data = np.loadtxt(f"{data_dir}//i40.txt")
    plt.scatter(data[:,0],100*data[:,1],label=r'i = 40')
    
    for i in range(len(phases)):
        vv_3d[i] = delay_model.timeDelay_3d_full(phases[i], plot = False, Q = 100, m1_in = m1_in, m2_in = m2_in, inclination_in = 20, disk_angle_in = 0, period_in = period_in)
    plt.plot(phases,vv_3d*scale)
    
    data = np.loadtxt(f"{data_dir}//i20.txt")
    plt.scatter(data[:,0],data[:,1],label=r'i = 20')
    plt.axvline(x = 0.25)
    plt.xlabel('Orbital Phase')
    plt.ylabel('Radial Velocity (km/s)')
    plt.legend()
    plt.show()
 
def plot4(inclination_in = 40, alpha_in = 0, period_in = 0.5):
    inclination = inclination_in*np.pi/180

    G = constants.G
    period = period_in * constants.day
    c = constants.c

    N0, N1, N2, N3, N4 = 0.886, -1.132, 1.523, -1.892, 0.867
    
    
    #i90,alpha0
    # ~ 0.888, -1.291, 1.541, -1.895, 0.861
    #i40,alpha0
    # ~ 0.886, -1.132, 1.523, -1.892, 0.867
    #i90,alpha8
    # ~ 0.982, -1.430, 2.552, -3.284, 1.542
    #i40,alpha8
    # ~ 0.975, -1.559, 2.828, -3.576, 1.651
    #i90,alpha16
    # ~ 1.326, -1.871, 1.745, -0.782, 0.000
    #i40,alpha16
    # 1.428, -2.362, 2.421, -1.122, 0.000
     
        
        
    q = np.linspace(0,1,100)
    
    yy = (N0+N1*q+N2*q**2+N3*q**3+N4*q**4)
    
    plt.plot(q,yy)
    
    phases = np.linspace(0,1,100)
    phases_tight_window = np.linspace(0.2,0.3,5)
    yy_3d = np.zeros(np.shape(phases))
    vv_3d = np.zeros(np.shape(phases))
    
    m1_in = 1
    m1 = m1_in * constants.M_Sun
    
    for m2_in in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:    
        m2 = m2_in * constants.M_Sun
        # Compute the semi-major axis
        a = (G*(m1+m2)*period**2/(4*np.pi**2))**(1/3)
        
        for i in range(len(phases_tight_window)):
            vv_3d[i] = delay_model.timeDelay_3d_full(phases_tight_window[i], plot = False, Q = 50, m1_in = m1_in, m2_in = m2_in, inclination_in = inclination_in, disk_angle_in = alpha_in, period_in = period_in)
        
        K2 = 1e-3*np.sin(2*np.pi*0.25)*np.sin(inclination)*2*np.pi*a/period*(1/(1+m2_in))
       
        print("k2",K2)
        
        # ~ plt.scatter([m2_in],[max(vv_3d)/K2],c='black')
        # ~ yy_3d_25_90, vv_3d_25_90 = delay_model.timeDelay_radialVelocity_3d(0.25, plot = False, Q = 100, m1_in = m1_in, m2_in = m2_in, inclination_in = inclination_in, disk_angle_in = alpha_in, period_in = period_in, beta_approx = 90)
        # ~ yy_3d_25_0, vv_3d_25_0 = delay_model.timeDelay_radialVelocity_3d(0.25, plot = False, Q = 100, m1_in = m1_in, m2_in = m2_in, inclination_in = inclination_in, disk_angle_in = alpha_in, period_in = period_in, beta_approx = 0)
        # ~ plt.scatter([m2_in/m1_in],[vv_3d_25_90/K2],c='red')
        # ~ plt.scatter([m2_in/m1_in],[vv_3d_25_0/K2],c='blue')
        
        plt.scatter([m2_in/m1_in],[max(vv_3d)/K2],c='purple')
        
    plt.title(r'i = ' + str(inclination_in) + r'$, \alpha$ = ' + str(alpha_in))
    plt.xlabel('$q$')
    plt.ylabel('$K_{em}/K_2$')
    plt.show()
    
def plot5(inclination_in = 44, disk_angle_in = 5, m2_in = 0.7, m1_in = 1.4, period_in = 0.787):
    phases = np.linspace(0,1,25)
    yy_3d_0 = np.zeros(len(phases))
    yy_3d_45 = np.zeros(len(phases))
    yy_3d_90 = np.zeros(len(phases))
    yy_3d_full = np.zeros(len(phases))
    
    with open('asra_beta.pickle', 'rb') as f:
        asra_beta = pickle.load(f)
    print(asra_beta)
    
    for i in range(len(phases)):
        yy_3d_0[i] = delay_model.timeDelay_3d_asra(phases[i], Q = 50, inclination_in = inclination_in, asra_beta = 0, disk_angle_in = disk_angle_in, m2_in = m2_in, m1_in = m1_in, period_in = period_in)
        yy_3d_90[i] = delay_model.timeDelay_3d_asra(phases[i], Q = 50, inclination_in = inclination_in, asra_beta = 90, disk_angle_in = disk_angle_in, m2_in = m2_in, m1_in = m1_in, period_in = period_in)
        yy_3d_45[i] = delay_model.timeDelay_3d_asra(phases[i], Q = 50, inclination_in = inclination_in, asra_beta = asra_beta, disk_angle_in = disk_angle_in, m2_in = m2_in, m1_in = m1_in, period_in = period_in)
        
    for i in range(len(phases)):
        print(i)
        yy_3d_full[i] = delay_model.timeDelay_3d_full(phases[i], Q = 50, inclination_in = inclination_in, disk_angle_in = disk_angle_in, m2_in = m2_in, m1_in = m1_in, period_in = period_in)
        
    # ~ yy_cm = delay_model.timeDelay_sp(phases,setting='cm',inclination_in = inclination_in, disk_angle_in = disk_angle_in)
    # ~ yy_egg = delay_model.timeDelay_sp(phases,setting='egg',inclination_in = inclination_in)
    # ~ yy_plav = delay_model.timeDelay_sp(phases,setting='plav',inclination_in = inclination_in)
    
    
    fig, axs = plt.subplots()
    
    axs.plot(phases,yy_3d_full,'b-',label='3d full')
    axs.plot(phases,yy_3d_0,'g:',label=r'3d symmetry $\beta = 0$')
    axs.plot(phases,yy_3d_90,'g--',label=r'3d symmetry $\beta = 90$')
    axs.plot(phases,yy_3d_45,'k-',label=r'3d symmetry calibration func',linewidth=4,alpha=0.5)
    
    # ~ axs.plot(phases,0.5*(yy_3d_90+yy_3d_0),'r-',label=r'mean 0 and 90')
    # ~ axs.plot(phases,yy_cm,'b:',label='cm')
    # ~ axs.plot(phases,yy_egg,'r--',label='egg')
    # ~ axs.plot(phases,yy_plav,'k-.',label='plav')
    
    axs.legend()
    
    axs.set_xlabel('Orbital Phase')
    axs.set_ylabel('Time Delay (s)')
    
    plt.show()

def plot6(m2_in = 5*1.4, m1_in = 1.4, period_in = 0.787):
    Q = 500
    
    # Convert to SI units
    m1 = m1_in * constants.M_Sun
    m2 = m2_in * constants.M_Sun
    
    q = m2_in/m1_in
    
    period = period_in * constants.day
    
    # Compute the orbital semi-major axis
    a = (constants.G*(m1+m2)*period**2/(4*np.pi**2))**(1/3)
    r2 = m1/(m1+m2)*a
    
    # Compute the location of the L1 Lagrange point
    roots = np.roots([(2*np.pi/period)**2,-r2*(2*np.pi/period)**2-2*a*(2*np.pi/period)**2,2*a*r2*(2*np.pi/period)**2 + a**2*(2*np.pi/period)**2,constants.G*(m1-m2)-r2*(2*np.pi/period)**2*a**2,2*a*constants.G*m2,-a**2*constants.G*m2])
    L1 = np.real(roots[np.argmin(np.abs(np.imag(roots)))]) # There should only be one purely real solution in the 5 complex solutions, use that one
    
        
    # Compute the value of the effective gravitational potential at the L1 point
    potential_at_L1 = delay_model.rochePotential(L1,0,0,m1=m1,m2=m2,period=period)
    
    psi_roche = np.linspace(-np.pi, np.pi, num=Q, endpoint=True)
    r_roche_90 = np.zeros(len(psi_roche))
    r_roche_45 = np.zeros(len(psi_roche))
    r_roche_0 = np.zeros(len(psi_roche))
    # Solve for r_roche as a function of psi_roche at fixed beta (this is the azimuthally symmetric approximation, we fix beta at a value - usually 45 deg)
    for i in range(len(psi_roche)):
        r_roche_90[i] = opt.brentq(delay_model.rochePotential,L1*1e-3,L1,args=(psi_roche[i],np.radians(90),m1,m2,period,potential_at_L1),maxiter=100)
        r_roche_0[i] = opt.brentq(delay_model.rochePotential,L1*1e-3,L1,args=(psi_roche[i],np.radians(0),m1,m2,period,potential_at_L1),maxiter=100)
        r_roche_45[i] = opt.brentq(delay_model.rochePotential,L1*1e-3,L1,args=(psi_roche[i],np.radians(45),m1,m2,period,potential_at_L1),maxiter=100)
        
    fig, ax = plt.subplots()
    
    plt.axhline(y=0,c='k',linestyle='--', zorder= -5)
    plt.axvline(x=0,c='k',linestyle='--', zorder = -5)
    
    ax.plot(r_roche_90*np.cos(psi_roche), r_roche_90*np.sin(psi_roche), linewidth = 1, color = 'green', label = r'$\beta$ = 90 deg')
    ax.plot(r_roche_0*np.cos(psi_roche), r_roche_0*np.sin(psi_roche), linewidth = 1, linestyle = '-', color = 'blue', label = r'$\beta$ = 0 deg')
    ax.plot(r_roche_45*np.cos(psi_roche), r_roche_45*np.sin(psi_roche), linewidth = 1, linestyle = '-', color = 'k', label = r'$\beta$ = 45 deg')
    
    
    egg_radiusDonor = a*(.49*q**(.667))/(.6*q**(.667)+np.log(1+q**(.333)))
    Egg_radiusDonor = egg_radiusDonor*np.ones(np.shape(psi_roche))
    ax.plot(Egg_radiusDonor*np.cos(psi_roche), Egg_radiusDonor*np.sin(psi_roche), linewidth = 1, linestyle = '--', color = 'red', label = r'Eggleton')
    
    res = 10*(r_roche_90 - r_roche_0)
    ax.plot(res*np.cos(psi_roche), res*np.sin(psi_roche), linewidth = 1, color = 'purple', label = r'10X Residual')
    
    ax.scatter([a],[0],s=150,c='black')
    
    ax.text(x=0.9*a,y=0.05*a,s='Compact\nObject')
    ax.text(x=L1,y=0.05*L1,s=r'$L_1$')
    
    print(f'Eggleton Volume: {4/3*np.pi*egg_radiusDonor**3}')
    # ~ print(f'beta 0 Volume: {2*np.pi*np.sum(r_roche_0**2*np.sin(psi_roche))*np.pi/len(psi_roche)}')
    

    # ~ ax.set_rmax(2)
    # ~ ax.set_rticks([0.5, 1, 1.5, 2])  # Less radial ticks
    L = 1.1*max(r_roche_0)
    plt.xlim([-L,1.2*a])
    plt.ylim([-L,L])
    ax.set_aspect('equal')
    plt.legend()
    plt.show()
        
def plot7(m2_in = 0.7, m1_in = 1.4, period_in = 0.1):
    i_arr = np.linspace(0,90,5)
    d_arr = np.linspace(0,15,20)
    with open('asra_beta.pickle', 'rb') as f:
        asra_beta = pickle.load(f)
        
    yy_0 = np.zeros([len(i_arr),len(d_arr)])
    yy_90 = np.zeros([len(i_arr),len(d_arr)])
    yy_cal = np.zeros([len(i_arr),len(d_arr)])
    yy_full = np.zeros([len(i_arr),len(d_arr)])
    
    for j in range(len(i_arr)):
        for k in range(len(d_arr)):
            # ~ yy_0[j, k] = delay_model.timeDelay_3d_asra(0.5, Q = 50, inclination_in = i_arr[j], asra_beta = 0, disk_angle_in = d_arr[k], m2_in = m2_in, m1_in = m1_in, period_in = period_in)
            # ~ yy_90[j, k] = delay_model.timeDelay_3d_asra(0.5, Q = 50, inclination_in = i_arr[j], asra_beta = 90, disk_angle_in = d_arr[k], m2_in = m2_in, m1_in = m1_in, period_in = period_in)
            yy_cal[j, k] = delay_model.timeDelay_3d_asra(0.5, Q = 50, inclination_in = i_arr[j], asra_beta = asra_beta, disk_angle_in = d_arr[k], m2_in = m2_in, m1_in = m1_in, period_in = period_in)
            yy_full[j, k] = delay_model.timeDelay_3d_full(0.5, Q = 50, inclination_in = i_arr[j], disk_angle_in = d_arr[k], m2_in = m2_in, m1_in = m1_in, period_in = period_in)
    
    
    # ~ for k in range(len(d_arr)):
        # ~ plt.plot(i_arr, yy_cal[:,k],label = r'ASRA $\alpha$ = ' + f'{d_arr[k]} deg')
        # ~ plt.scatter(i_arr, yy_full[:,k])
    
    # ~ plt.xlabel('Inclination (deg)')
    # ~ plt.ylabel('Time Delay (s) @ 0.5 phase')
    # ~ plt.legend()
    # ~ plt.show()
    
    for j in range(len(i_arr)):
        plt.plot(d_arr, yy_cal[j,:],label = r'ASRA i = ' + f'{i_arr[j]} deg')
        plt.scatter(d_arr, yy_full[j,:])
    
    plt.xlabel('Disk Shielding Angle (deg)')
    plt.ylabel('Time Delay (s) @ 0.5 phase')
    plt.legend()
    plt.show()

def fig8(inclination_in = 90, disk_angle_in = 17.5, m2_in = 0.7, m1_in = 1.4, period_in = 0.787):
    phases = np.linspace(0,1,50)
    yy_3d_asra = np.zeros(len(phases))
    yy_cm = delay_model.radialVelocity(phases,setting='cm',inclination_in = inclination_in)
    yy_egg = delay_model.radialVelocity(phases,setting='egg',inclination_in = inclination_in)
    yy_plav = delay_model.radialVelocity(phases,setting='plav',inclination_in = inclination_in)
    
    with open('asra_beta.pickle', 'rb') as f:
        asra_beta = pickle.load(f)
    print(asra_beta)
    
    for i in range(len(phases)):
        yy_3d_asra[i] = delay_model.timeDelay_3d_asra(phases[i], Q = 50, inclination_in = inclination_in, asra_beta = asra_beta, disk_angle_in = disk_angle_in, m2_in = m2_in, m1_in = m1_in, period_in = period_in, mode='return_rv')
        
    phases_full = np.linspace(0,1,15)
    yy_3d_full = np.zeros(len(phases_full))
    for i in range(len(phases_full)):
        print(f'{i}/{len(phases_full)}')
        yy_3d_full[i] = delay_model.timeDelay_3d_full(phases_full[i], Q = 50, inclination_in = inclination_in, disk_angle_in = disk_angle_in, m2_in = m2_in, m1_in = m1_in, period_in = period_in, mode = 'return_rv')
    
    fig, axs = plt.subplots()
    lw = 2
    axs.scatter(phases_full,yy_3d_full,fc='green',ec='black',label='3D peak',zorder=5,marker='^')
    axs.plot(phases,yy_3d_asra,'g-',label=r'3D centroid',linewidth=lw)
    
    axs.plot(phases,yy_cm,'b:',label='cm',linewidth=lw)
    axs.plot(phases,yy_egg,'r--',label='egg',linewidth=lw)
    axs.plot(phases,yy_plav,'k-.',label='plav',linewidth=lw)
    
    axs.legend()
    
    axs.set_xlabel('Orbital Phase')
    axs.set_ylabel('Time Delay (s)')
    
    # ~ im = image.imread(r'data\xrb_schematic.png')
    # ~ axs.imshow(im,aspect='auto', extent=(0.5-0.25, 0.5+0.25, 0.5, 0.5+7), alpha = 1)
    
    # ~ axs.set_xlim([0,1])
    # ~ axs.set_ylim([0,20])
    
    plt.show()




def figA(inclination_in = 45, disk_angle_in = 0, m2_in = 0.5*1.4, m1_in = 1.4, period_in = 0.787):
    phases = np.linspace(0,1,50)
    yy_3d_asra = np.zeros(len(phases))
    yy_cm = delay_model.timeDelay_sp(phases,setting='cm',inclination_in = inclination_in)
    yy_egg = delay_model.timeDelay_sp(phases,setting='egg',inclination_in = inclination_in)
    yy_plav = delay_model.timeDelay_sp(phases,setting='plav',inclination_in = inclination_in)
    
    asra_lt = delay_model.ASRA_LT_model()
    
    for i in range(len(phases)):
        yy_3d_asra[i], _ = asra_lt.evaluate(phases[i], Q = 50, inclination_in = inclination_in, disk_angle_in = disk_angle_in, m2_in = m2_in, m1_in = m1_in, period_in = period_in)
        
    phases_full = np.linspace(0,1,15)
    yy_3d_full = np.zeros(len(phases_full))
    for i in range(len(phases_full)):
        print(f'{i}/{len(phases_full)}')
        yy_3d_full[i] = delay_model.timeDelay_3d_full(phases_full[i], Q = 50, inclination_in = inclination_in, disk_angle_in = disk_angle_in, m2_in = m2_in, m1_in = m1_in, period_in = period_in)
    
    fig, axs = plt.subplots()
    lw = 2
    axs.scatter(phases_full,yy_3d_full,fc='green',ec='black',label='Full 3D',zorder=5)
    axs.plot(phases,yy_3d_asra,'g-',label=r'ASRA 3D',linewidth=lw)
    
    axs.plot(phases,yy_cm,'b:',label='cm',linewidth=lw)
    axs.plot(phases,yy_egg,'r--',label='egg',linewidth=lw)
    axs.plot(phases,yy_plav,'k-.',label='plav',linewidth=lw)
    
    axs.legend()
    
    axs.set_xlabel('Orbital Phase')
    axs.set_ylabel('Time Delay (s)')
    
    # ~ im = image.imread(r'data\xrb_schematic.png')
    # ~ axs.imshow(im,aspect='auto', extent=(0.5-0.25, 0.5+0.25, 0.5, 0.5+7), alpha = 1)
    
    # ~ axs.set_xlim([0,1])
    # ~ axs.set_ylim([0,20])
    
    plt.show()

def figB(inclination_in = 2.5, disk_angle_in = 0, m2_in = 0.5*1.4, m1_in = 1.4, period_in = 0.787):
    phases = np.linspace(0,1,50)
    yy_3d_asra = np.zeros(len(phases))
    yy_cm = delay_model.radialVelocity(phases,setting='cm',inclination_in = inclination_in)
    yy_egg = delay_model.radialVelocity(phases,setting='egg',inclination_in = inclination_in)
    yy_plav = delay_model.radialVelocity(phases,setting='plav',inclination_in = inclination_in)
    
    asra_lt = delay_model.ASRA_LT_model()
    
    for i in range(len(phases)):
        _, yy_3d_asra[i] = asra_lt.evaluate(phases[i], Q = 50, inclination_in = inclination_in, disk_angle_in = disk_angle_in, m2_in = m2_in, m1_in = m1_in, period_in = period_in)
        
    phases_full = np.linspace(0,1,15)
    yy_3d_full = np.zeros(len(phases_full))
    for i in range(len(phases_full)):
        print(f'{i}/{len(phases_full)}')
        yy_3d_full[i] = delay_model.timeDelay_3d_full(phases_full[i], Q = 50, inclination_in = inclination_in, disk_angle_in = disk_angle_in, m2_in = m2_in, m1_in = m1_in, period_in = period_in, mode = 'return_rv')
    
    fig, axs = plt.subplots()
    lw = 2
    axs.scatter(phases_full,yy_3d_full,fc='green',ec='black',label='Full 3D',zorder=5)
    # ~ axs.plot(phases,-yy_3d_asra,'g-',linewidth=lw)
    axs.plot(phases,yy_3d_asra,'g-',label=r'ASRA 3D',linewidth=lw)
    
    axs.plot(phases,yy_cm,'b:',label='cm',linewidth=lw)
    axs.plot(phases,yy_egg,'r--',label='egg',linewidth=lw)
    axs.plot(phases,yy_plav,'k-.',label='plav',linewidth=lw)
    
    axs.legend()
    
    axs.set_xlabel('Orbital Phase')
    axs.set_ylabel('Radial Velocity (km/s)')
    
    # ~ im = image.imread(r'data\xrb_schematic.png')
    # ~ axs.imshow(im,aspect='auto', extent=(0.5-0.25, 0.5+0.25, 0.5, 0.5+7), alpha = 1)
    
    # ~ axs.set_xlim([0,1])
    # ~ axs.set_ylim([0,20])
    
    plt.show()

def figC(m2_in = 0.7, m1_in = 1.4, period_in = 0.787, inclination_in = 44):
    Q = 500
    
    inclination = inclination_in*np.pi/180
    # Convert to SI units
    m1 = m1_in * constants.M_Sun
    m2 = m2_in * constants.M_Sun
    
    q = m2_in/m1_in
    
    period = period_in * constants.day
    
    # Compute the orbital semi-major axis
    a = (constants.G*(m1+m2)*period**2/(4*np.pi**2))**(1/3)
    r2 = m1/(m1+m2)*a
    
    # Compute the location of the L1 Lagrange point
    roots = np.roots([(2*np.pi/period)**2,-r2*(2*np.pi/period)**2-2*a*(2*np.pi/period)**2,2*a*r2*(2*np.pi/period)**2 + a**2*(2*np.pi/period)**2,constants.G*(m1-m2)-r2*(2*np.pi/period)**2*a**2,2*a*constants.G*m2,-a**2*constants.G*m2])
    L1 = np.real(roots[np.argmin(np.abs(np.imag(roots)))]) # There should only be one purely real solution in the 5 complex solutions, use that one
    
        
    # Compute the value of the effective gravitational potential at the L1 point
    potential_at_L1 = delay_model.rochePotential(L1,0,0,m1=m1,m2=m2,period=period)
    
    psi_roche = np.linspace(-np.pi, np.pi, num=Q, endpoint=True)
    r_roche_0 = np.zeros(len(psi_roche))
    # Solve for r_roche as a function of psi_roche at fixed beta (this is the azimuthally symmetric approximation, we fix beta at a value - usually 45 deg)
    for i in range(len(psi_roche)):
        r_roche_0[i] = opt.brentq(delay_model.rochePotential,L1*1e-3,L1,args=(psi_roche[i],np.radians(0),m1,m2,period,potential_at_L1),maxiter=100)
        
    fig, ax = plt.subplots()
    
    ax.plot(r_roche_0*np.cos(psi_roche), r_roche_0*np.sin(psi_roche), linewidth = 1, linestyle = '-', color = 'black', label = r'$\beta$ = 0 deg')
    # ~ ax.plot(r_roche_45*np.cos(psi_roche), r_roche_45*np.sin(psi_roche), linewidth = 1, linestyle = '-', color = 'k', label = r'$\beta$ = 45 deg')
    
    
    # ~ egg_radiusDonor = a*(.49*q**(.667))/(.6*q**(.667)+np.log(1+q**(.333)))
    # ~ Egg_radiusDonor = egg_radiusDonor*np.ones(np.shape(psi_roche))
    # ~ ax.plot(Egg_radiusDonor*np.cos(psi_roche), Egg_radiusDonor*np.sin(psi_roche), linewidth = 1, linestyle = '--', color = 'red', label = r'Eggleton')

    ax.scatter([a],[0],s=150,c='black')
    ax.scatter([0],[0],s=150,c='black',marker='+')
    ax.scatter([r2],[0],s=150,marker='+',c='black')
    ax.text(x=0.95*r2,y=-0.075*a,s='CM')


    ax.arrow(x=r2, y=0, dx=0.15*a*np.sin(inclination), dy=0.15*a*np.cos(inclination),head_width=0.025*a,fc='black')
    ax.arrow(x=r2, y=0, dx=0, dy=0.15*a,head_width=0.025*a,fc='black')
    ax.arrow(x=r2, y=0, dy=0, dx=0.15*a,head_width=0.025*a,fc='black')
    
    index = 300
    scale = 0.86
    ax.arrow(x=0, y=0, dx=scale*r_roche_0[index]*np.cos(psi_roche[index]), dy=scale*r_roche_0[index]*np.sin(psi_roche[index]),head_width=0.025*a,fc='black')
    
    ax.plot([0,a],[0,0],'k--')
    
    ax.text(x=0.96*a,y=-0.075*a,s='CM1')
    ax.text(x=0,y=-0.075*a,s='CM2')
    ax.text(x=L1,y=0.05*L1,s=r'$L_1$')
    
    ax.text(x=2.573e9,y=4.77e8,s=r'$\hat{e}$')
    
    ax.text(x=2.1e9,y=6.85e8,s=r'$\hat{z}$')
    ax.text(x=2.61e9,y=-1.9e8,s=r'$\hat{x}$')
    
    arc_size = 1.75e8
    arc0 = patches.Arc((1.77e9,-3e8),width=arc_size,height=arc_size,theta1=0, theta2=360)
    ax.add_patch(arc0)
    ax.scatter([1.77e9],[-3e8],s=95,marker='x',color='black')
    ax.text(x=1.6e9,y=-2.7e8,s=r'$\hat{y}$')
    
    arc1 = patches.Arc((r2,0),width=3.2e8,height=3.2e8,theta1=90-inclination_in, theta2=90)
    ax.add_patch(arc1)
    ax.text(x=2.2e9,y=2.16e8,s=r'$i$')
    
    arc2 = patches.Arc((0,0),width=3.2e8,height=3.2e8,theta1=0, theta2=180/np.pi*psi_roche[index])
    ax.add_patch(arc2)
    ax.text(x=2.1e8,y=6.5e7,s=r'$\psi$')
    
    ax.text(x=3.7e8,y=4.38e8,s=r'$\vec{p}_2$')
    
    scale = 0.92
    ax.arrow(x=a, y=0, dx=scale*(r_roche_0[index]*np.cos(psi_roche[index])-a), dy=scale*r_roche_0[index]*np.sin(psi_roche[index]),head_width=0.025*a,fc='blue',ec='blue',alpha=1)
    
    ax.text(x=1.410e9,y=5.35e8,s=r'$\vec{p}_1$')
    
    scale = 0.15
    ax.arrow(x=r_roche_0[index]*np.cos(psi_roche[index]), y=r_roche_0[index]*np.sin(psi_roche[index]),dx=scale*a*np.sin(inclination),dy=scale*a*np.cos(inclination),head_width=0.025*a,fc='red',ec='red',alpha=1)
    ax.text(x=1.29e9,y=1.1e9,s=r'$\hat{e}$')
    
    arc3 = patches.Arc((r_roche_0[index]*np.cos(psi_roche[index]),r_roche_0[index]*np.sin(psi_roche[index])),width=1e8,height=1e8,theta1=0, theta2=360)
    ax.add_patch(arc3)
    
    ax.plot([0,0],[-0.1*a,-0.45*a],'k--')
    ax.plot([a,a],[-0.1*a,-0.45*a],'k--')
    ax.plot([r2,r2],[-0.1*a,-0.35*a],'k--')
    ax.arrow(x=0.9*r2/2, y=-0.35*a,dx=-.74*r2/2,dy=0,head_width=0.025*a,fc='black',ec='black',alpha=1)
    ax.arrow(x=1.1*r2/2, y=-0.35*a,dx=.74*r2/2,dy=0,head_width=0.025*a,fc='black',ec='black',alpha=1)
    
    ax.arrow(x=0.93*a/2, y=-0.45*a,dx=-.83*a/2,dy=0,head_width=0.025*a,fc='black',ec='black',alpha=1)
    ax.text(x=1.025e9,y=-1.15e9,s=r'$r_2$')
    
    ax.arrow(x=1.07*a/2, y=-0.45*a,dx=.83*a/2,dy=0,head_width=0.025*a,fc='black',ec='black',alpha=1)
    ax.text(x=1.55e9,y=-1.47e9,s=r'$a$')
    
    ax.text(x=8.19e8,y=7.4e8,s=r'$P$')
    
    # ~ print(f'Eggleton Volume: {4/3*np.pi*egg_radiusDonor**3}')
    # ~ print(f'beta 0 Volume: {2*np.pi*np.sum(r_roche_0**2*np.sin(psi_roche))*np.pi/len(psi_roche)}')
    

    # ~ ax.set_rmax(2)
    # ~ ax.set_rticks([0.5, 1, 1.5, 2])  # Less radial ticks
    L = 1.1*max(r_roche_0)
    plt.xlim([-L,1.2*a])
    plt.ylim([-L,L])
    ax.set_aspect('equal')
    # ~ plt.legend()
    plt.show()



def convert_latex_python():
    with open(r'data\roche_expression.txt', 'r') as f:
        s = f.read().replace('\n', '')
    
    s = s.replace(r'\left','')
    s = s.replace(r'\right','')
    
    s = s.replace(r'm_{2}','m2')
    s = s.replace(r'm_{1}','m1')
    s = s.replace(r'r_{2}','r2')
    s = s.replace(r'\omega','omega')
    s = s.replace(r'r_{2}','r2')
    s = s.replace(r'V_{0}','potential_at_L1')
    
    s = s.replace(r'\sin{(\psi )}','np.sin(psi_roche[i])')
    s = s.replace(r'\cos{(\psi )}','np.cos(psi_roche[i])')
    s = s.replace(r'\sin{(\beta )}','np.sin(asra_beta)')
    s = s.replace(r'\cos{(\beta )}','np.cos(asra_beta)')
    
    # ~ s = s.replace(r'p_{2}',f'---------TERM{0}'+'\n \n')
    for i in range(10):
        s = s.replace(r'p_{2}^{' + str(i) + r'}',f'---------TERM{i}'+'\n \n') 
    
    
    for i in range(2,13):
        s = s.replace(r'\sin^{' + str(i) + r'}{(\beta )}','np.sin(asra_beta)**' + str(i))
        s = s.replace(r'\cos^{' + str(i) + r'}{(\beta )}','np.cos(asra_beta)**' + str(i))
        s = s.replace(r'\sin^{' + str(i) + r'}{(\psi )}','np.sin(psi_roche[i])**' + str(i))
        s = s.replace(r'\cos^{' + str(i) + r'}{(\psi )}','np.cos(psi_roche[i])**' + str(i))
        s = s.replace('^{'+str(i)+'}',f'**{i}')
        

    s = s.replace(' + ','+')
    s = s.replace(' +','+')
    s = s.replace('+ ','+')
    s = s.replace(' - ','-')
    s = s.replace('- ','-')
    s = s.replace(' -','-')
    s = s.replace(' ','*')
    
    print(s)


    # ~ fig = corner.corner(samples)
    # ~ plt.show()

 
# ~ figB(inclination_in = 45, m1_in = 1.4)
# ~ plot6()
# ~ figB()
# ~ fig1()

# ~ plotRochePotential()

# ~ convert_latex_python()

# ~ figA()
# ~ figC()
figB(inclination_in = 90)

