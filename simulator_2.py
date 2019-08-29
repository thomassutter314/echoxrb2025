#
# simulator_2.py
#

# Authored by TS, minor improvements by WIC

### 2018-04-28 WIC - I have done a bit of tidying to avoid any
### dependence on globally-set values. My comments begin with
### triple-comment symbol like this: "###" and optionally an actual
### comment

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.integrate import quad
import scipy.optimize as opt
import constants
import emcee
import corner ### 2018-04-28 WIC - it's better to be here.
import scipy.optimize as op
import matplotlib.path as path
import matplotlib.patches as patches
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import pickle
from scipy.interpolate import RegularGridInterpolator as RGI
from tkinter.filedialog import askopenfilename
from tkinter import Tk

# import the priors object (now exported to an external module)
import echoPriors

### 2018-04-28 WIC - a couple of useful utilities
import time, os

### 2018-04-28 WIC - brought the G value up here.
G = constants.G


### 2018-04-28 WIC - if we want to reproduce Jonker et al. 2004 Figure
### 4
import jonkerMass
from delaySignalRoche import *

# A couple of global parameters for plotting
plt.rc('font', family='serif')
matplotlib.rcParams.update({'font.size': 18})

### 2018-04-29 WIC - implemented priors in a new class
### class PriorsOLDER(object):  # renamed for safe testing

### 2018-05-04 REPLACED with echoPriors, imported above.

def RocheModelDelay(phase = .5, m1_in = 1.4, m2_in = .7, period_in = .787, \
                    inclination = np.radians(44.), eccentricity = 0, omega = np.radians(90.),\
                     binNumber = 100, Q = 120, disk = True, diskShieldingAngle = np.radians(0), intKind = 'quadratic', \
                    radialVelocity = False, innerLagrange = False):
    y = delaySignal(phase = phase, m1_in = m1_in, m2_in = m2_in, period_in = period_in, inclination = inclination, eccentricity = eccentricity, \
                        omega = omega, binNumber = binNumber, Q = Q, disk = disk, diskShieldingAngle = diskShieldingAngle, intKind = intKind, \
                        radialVelocity = radialVelocity, innerLagrange = innerLagrange, plot = False)
    return y




def configureData(R = 50):
    Q = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5]
    #DATA = np.zeros([160+160*18+10*160*18,3])
    #CORR = np.zeros([160+160*18+10*160*18])
    CORR = np.zeros([15,18,7998])
    #print(np.shape(DATA))
    corrFile = "correction_q=%s.pickle" % Q[0]
    file = open(corrFile,'rb')
    data = pickle.load(file)
    I = data['inclination']
    PHI = data['phase']

    for q in range(len(Q)):
        corrFile = "correction_q=%s.pickle" % Q[q]
        file = open(corrFile,'rb')
        data = pickle.load(file)
        #print(np.shape(data['corrections']))
        for i in range(len(I)):
            for phi in range(len(PHI)):
                CORR[q][i][phi] = data['corrections'][i][phi]
        
        #print(data.keys())
        #print(len(data['phase'][::R]))
        #print(np.shape(data['corrections']))
        #for i2 in range(18):
            #print("index",i2*160+i*160*18)
        #    for i3 in range(160):
        #        index = i3+i2*160+i*160*18
        #        DATA[index][0] = A[i]
        #        DATA[index][1] = data['inclination'][i2]
        #        DATA[index][2] = data['phase'][i3]
                #DATA[index][3] = data['corrections'][i2][i3]
        #        CORR[index] = data['corrections'][i2][i3]
        #print(DATA)
        #file.close()
    FUNC = RGI((Q, I, PHI), CORR, bounds_error = False)
    return FUNC


def initialPseudoConfig():
    FUNC_CORRECTION = configureData()
    np.save('Func_Correction.npy',FUNC_CORRECTION)
    #DOut = {'func':FUNC_CORRECTION}
    #fileName = 'Func_Correction.pickle'
    #pickle.dump(DOut, open(fileName, 'wb'))
try:
    funcARY = np.load('Func_Correction.npy')
    FUNC_CORRECTION = funcARY.item()
except:
    print("Pseudo 3D correction is missing. Run: initialPseudoConfig() and restart.")




def parameterDependence(m1 = 1.4, m2 = 0.7, \
                              eccentricity = 0, \
                              period = .787, iDeg = 44., \
                              omegaDeg = 90., \
                              parameter = 4, \
                              N = 20,\
                              cmap = cm.cool,\
                              radialVelocity = False, \
                              rocheLobe = True):

    """Wrapper to plot the delay curve dependence on inclination for
    constant longitude of periastron"""
    #parameterList = [r"$m_1$",r"$m_2$","e","P","i",r"$\omega$"]
    parameterList = [r"$m_1$",r"$m_2$","e","P (days)",r"i$(^\circ)$",r"$\omega(^\circ)$"]
    figName='figDelay_' + str(parameterList[parameter]) + 'DegVary.png'
    i = np.radians(iDeg)
    omega = np.radians(omegaDeg)
    tt = np.linspace(0,1,1000)
    yy = [0]*10
    Vals = [m1,m2,eccentricity,period,i,omega]
    parameterRanges = [[.01,3],[0,3],[.01,.99],[0.1,25],[0,.5*np.pi],[0,.5*np.pi]]
    #fig = plt.figure("Dependences")
    fig, ax = plt.subplots()
    plt.clf()
    for k in range(N):
        rgba_colors = np.zeros(4)
        rgba_colors[0] = .9*k/N
        rgba_colors[2] = .8
        rgba_colors[3] = 1
        rgba_colors[1] = 0
        Vals[parameter] = parameterRanges[parameter][0] + (parameterRanges[parameter][1]-parameterRanges[parameter][0])*k/N
        #delaySignal(plot = False)
        yy = timeDelay(tt,*Vals,radialVelocity=radialVelocity, rocheLobe = rocheLobe)
        plt.plot(tt[1:-1],yy[1:-1],color = cmap(k/N),linewidth = 1)
    #Y = timeDelay(tt,m1,m2,eccentricity,period,0,omega)
    #plt.plot(tt[1:-1],Y[1:-1],c='r',linewidth = 5)
    #Y = timeDelay(tt,m1,m2,eccentricity,period,np.radians(90),omega)
    #plt.plot(tt[1:-1],Y[1:-1],c='b')
    plt.xlabel("Orbital Phase")
    plt.ylabel("Echo Delay (s)")
    low = parameterRanges[parameter][0]
    high = parameterRanges[parameter][1]
    if parameter == 4 or parameter == 5:
        low = np.degrees(parameterRanges[parameter][0])
        high = np.degrees(parameterRanges[parameter][1])
        
    data = np.array([[low,high],[low,high]])
    #data = np.linspace(-1,1,62500)
    #data = np.zeros([250,250])
    cax = ax.imshow(data, interpolation='nearest', cmap = cmap)
    cbar = fig.colorbar(cax, ticks=[low, high], orientation='vertical')    
    cbar.ax.set_title(parameterList[parameter])
    #plt.title(parameterList[parameter])
    plt.show(block=False)
    ### 2018-04-28 WIC - save the figure to disk
    #plt.savefig(figName)

def inclinationDependence(m1 = 1.4, m2 = 0.3, \
                              eccentricity = .45, \
                              period = 16.3, iDeg = 60., \
                              omegaDeg = 90., \
                              figName='figDelay_iDegVary.png'):

    """Wrapper to plot the delay curve dependence on inclination for
    constant longitude of periastron"""

    ### 2018-04-28 WIC - Switched the defaults back to numerical values
    ### since we're no longer inheriting from globally-defined special
    ### values.
    
    i = np.radians(iDeg)
    omega = np.radians(omegaDeg)
    plt.figure(3)
    plt.clf()
    tt = np.linspace(0,1,1000)
    yy = [0]*10
    for k in range(18):
        yy = timeDelay(tt,m1,m2,eccentricity,period,np.radians(5*k),omega)
        plt.plot(tt[1:-1],yy[1:-1],'-g',linewidth = 1)
    Y = timeDelay(tt,m1,m2,eccentricity,period,0,omega)
    plt.plot(tt[1:-1],Y[1:-1],c='r',linewidth = 5)
    Y = timeDelay(tt,m1,m2,eccentricity,period,np.radians(90),omega)
    plt.plot(tt[1:-1],Y[1:-1],c='b')
    plt.xlabel("Orbital Phase")
    plt.ylabel("Echo Delay (s)")
    plt.show(block=False)

    ### 2018-04-28 WIC - save the figure to disk
    plt.savefig(figName)

def omegaDependence(m1 = 1.4, m2 = 0.3, \
                        eccentricity = 0.45, \
                        period = 16.3, \
                        iDeg = 60., \
                        omegaDeg = 90., \
                        figName='figDelay_omegaVary.png'):

    """Wrapper to plot the delay curve dependence on the longitude of
    periastron for constant inclination"""

    ### 2018-04-28 WIC - Switched the defaults back to numerical values
    ### since we're no longer inheriting from globally-defined special
    ### values.

    ### i = 90
    i = np.radians(iDeg)
    omega = np.radians(omegaDeg)
    plt.figure(3)
    plt.clf()
    tt = np.linspace(0,1,1000)
    yy = [0]*10
    for k in range(18):
        yy = timeDelay(tt,m1,m2,eccentricity,period,i,np.radians(5*k))
        plt.plot(tt[1:-1],yy[1:-1],c='g')
    Y = timeDelay(tt,m1,m2,eccentricity,period,i,0)
    plt.plot(tt[1:-1],Y[1:-1],c='r',linewidth = 5)
    Y = timeDelay(tt,m1,m2,eccentricity,period,i,np.radians(90))
    plt.plot(tt[1:-1],Y[1:-1],c='b')
    plt.xlabel("Orbital Phase")
    plt.ylabel("Echo Delay (s)")
    plt.show(block=False)

    ### 2018-04-28 WIC - save the figure
    plt.savefig(figName)

def run(m1 = 1.4, m2 = 0.3, eccentricity = 0.45, \
            period = 16.3, iDeg = 60., omegaDeg = 90., \
            figDelay='figExample_delay.png', \
            figSepar='figExample_separ.png'):

    """Compute and plot a delay curve"""

    ### 2018-04-28 WIC - changed the defaults back to numerical values
    ### since we're no longer inheriting from global values. Added
    ### figure names for the delay and separation curves.

    i = np.radians(iDeg)
    omega = np.radians(omegaDeg)
    plt.figure(1)
    plt.clf()
    tt = np.linspace(0,1,1000)
    yy1,yy2 = separationFunc(tt,m1,m2,eccentricity,period,i,omega)
    yy1 = timeDelay(tt,m1,m2,eccentricity,period,i,omega)
    yy3,yy4 = separationFunc(tt,m1,m2,eccentricity,period,i,omega)
    yy3 = timeDelay(tt,m1,m2,eccentricity,period,i+.1,omega)
    
    #plt.plot(yy1[1:-1],tt[1:-1],yy3[1:-1],tt[1:-1])
    plt.ylabel("Orbital Phase")
    plt.xlabel("Delay (s)")
    plt.plot(yy1[1:-1],tt[1:-1])
    plt.show(block=False)
    plt.savefig(figSepar)

    plt.figure(2)
    plt.clf()
    tt = np.linspace(0,1,1000)
    #yy = separationFunc(tt,m1,m2,eccentricity,period,i,omega)
    plt.ylabel("Orbital Phase")
    plt.xlabel("Binary Separation (m)")
    plt.plot(yy2[1:-1],tt[1:-1],yy4[1:-1],tt[1:-1])
    plt.show(block=False)
    plt.savefig(figDelay)

def integrand(r):

    ### 2018-04-28 WIC - calls global constants
    A = 2/mu*E+2*constants.G*(m1+m2)/r-L**2/(mu**2*r**2)
    A = 1/np.sqrt(A)
    return A

def F_simple(r):
    #print(r)
    r_0 = (1-e)*a #initial position is set to periastron
    I = quad(integrand,r_0,r)[0]
    #print(I)
    return I
#def F(r):
    #print(r)
#    r_0 = (1-e)*a #initial position is set to periastron
#    I = [0]*len(r)
#    #print(I)
#    for i in range(len(r)):
#        I[i] = quad(integrand,r_0,r[i])[0]
#    #print(I)
#    return I

def rangeTimeDelay(phase=np.array([]), \
                  m1_in=1.4, m2_in=0.3, \
                  eccentricity=0.45, period_in=16.3,\
                  inclination=np.radians(60.), omega=np.radians(90.), \
                  gamma=0, rocheLobe = True):

    """Returns the time delay as a function of orbital phase."""
    
    ### 2018-05-06 WIC - I think this is our place to work in radial
    ### velocities. 

    ### 2018-04-28 WIC - converted the defaults to numerical values
    ### (so that this can be imported separately

    #print("eccentricity",eccentricity)
    #for i in range(len(phase)):
    #    if phase[i] == 0:
    #        phase[i] = .001
    #    if phase[i] == .5:
    #        phase[i] = 5.001
    #    if phase[i] == 1:
    #        phase[i] = .999
    phase = np.mod(phase,1) #Performance Issues?
    m1 = m1_in * constants.M_Sun
    m2 = m2_in * constants.M_Sun
    q = m2_in/m1_in
    period = period_in * constants.day
    tolerancePhaseTransitionPercent = .0005
    Rs = period*tolerancePhaseTransitionPercent
    e = eccentricity
    phase = np.array(phase)

    ### 2018-04-28 WIC - ensure G is locally set.
    G = constants.G

    ### 2018-04-28 WIC - return gracefully if zero-length input given
    if np.size(phase) < 1:
        return np.array([])

    #opt.brentq(t-separationFunc_INV(y,m1,m2,eccentricity,period),-10,10)
    #mu = m1*m2/(m1+m2)
    a = (G*(m1+m2)*period**2/(4*np.pi**2))**(1./3.)

    if rocheLobe == True:
        radiusDonor = a*(.49*q**(.667))/(.6*q**(.667)+np.log(1+q**(.333)))
        #print(radiusDonor/a)
    else:
        radiusDonor = 0
        
    #print("a",a)
    #E = (-G*m1*m2)/(2*a)
    #print("E",E)
    
    #L = m1*m2*np.sqrt(a*G*(1-e**2)/(m1+m2))
    #L_ang=mu*np.sqrt(G*(m1+m2)*a*(1-e*e))
    #print("L_ang",L_ang)
    #print("L",L)
    #print("S insides",L**2/(2*E*mu)+a**2)
    #S = np.sqrt(L**2/(2*E*mu)+a**2)
    RotR = np.sqrt(G*(m1+m2)/(a**3))
    #print("S",S)
    ymin = a*(1-e)
    ymax = a*(1+e)
    #tmin = -np.sqrt(-mu/(2*E))*a*np.pi*0.5
    tmin = -period/4
    #tmin = -np.sqrt(-mu/(2*E))*np.sqrt(S**2-(ymin-a)**2)+np.sqrt(-mu/(2*E))*a*np.arcsin((ymin-a)/S)
    #print("tmin",tmin)
    #tmax = -np.sqrt(-mu/(2*E))*np.sqrt(S**2-(ymax-a)**2)+np.sqrt(-mu/(2*E))*a*np.arcsin((ymax-a)/S)
    #tmax = np.sqrt(-mu/(2*E))*a*np.arcsin((ymax-a)/S)
    #tmax = np.sqrt(-mu/(2*E))*a*np.pi*0.5
    tmax = period/4
    #print("tmax",tmax)
    t = 2 * phase * (tmax - tmin) #converts orbital phase into actual times
    #print(t)
    #print(ymax)
    #print(a)
    #print(S)
    #print((ymax-a)/S)
    #tCrit = -np.sqrt(-mu/(2*E))*np.sqrt(S**2-(ymax-a)**2)+np.sqrt(-mu/(2*E))*a*np.arcsin((ymax-a)/S) - tmin
    tCrit = period/2
    #tCrit = 0.5*period
    #print(tCrit,tCrit/period)
    #print("tCrit",tCrit)
    a1 = ymin
    b1 = ymax
    separation = [0]*len(t)
    delta = [0]*len(t)
    deltaMAX = [0]*len(t)
    deltaMIN = [0]*len(t)
    vRVC = [0]*len(t)
    for i in range(len(t)):
        #if t[i] == tmax:
        #    "tmax Hit"
        #if t[i] == tmin:
        #    "tmin Hit"
        if e > .0001:
            if t[i] < (tCrit-Rs) and t[i] > Rs:
                def separationFunc_INV(y):
                    #conTerm = y-a
                    proTerm = (y-a)/a
                    #if conTerm > S:
                    #    conTerm = S
                    #if conTerm < -S:
                    #    conTerm = -S
                    if proTerm > e:
                        proTerm = e
                    if proTerm < -e:
                        proTerm = -e
                        
                    #print("proTerm: " + str(proTerm))
                    #print("e**2: " + str(e**2))
                    #T = -np.sqrt(-mu/(2*E))*np.sqrt(S**2-(conTerm)**2)+np.sqrt(-mu/(2*E))*a*np.arcsin(conTerm/S) - tmin
                    #print("model 1:" + str(T))
                    T = -period/(2*np.pi)*np.sqrt(e**2-(proTerm)**2)+period/(2*np.pi)*np.arcsin(proTerm/e) - tmin
                    #print("model 2:" + str(T))
                    return T-t[i]
                try:
                    separation[i] = opt.brentq(separationFunc_INV,a1,b1, maxiter=100)
                    #separation[i] = opt.brenth(separationFunc_INV,a1,b1, maxiter=100)
                    #separation[i] = opt.ridder(separationFunc_INV,a1,b1,maxiter=100)
                    #separation[i] = opt.bisect(separationFunc_INV,a1,b1,maxiter=100)
                    #separation[i] = opt.newton(separationFunc_INV,.5*(b1+a1),maxiter=100)
                except ValueError:
                    print("timeDelay, t <  tCrit - Oops!  That was no valid number: t[i] = %.2e. a1, b1=%.2e, %.2e. Try again..." % (phase[i], a1, b1))
                    ### print(t[i]) ### Merged into the above line.
                    separation[i] = 10000

                except RuntimeError:
                    print("timeDelay t < tCrit WARN: runtime error in brent: a1=%.2e, b1=%.2e: " % (a1, b1))
                    separation[i] = 10000

                    # print the parameters
                    print("timeDelay t < tCrit WARN: a=%.2e, S=%.2e, m1=%.2e, m2=%.2e, e=%.2f" % (a,S,m1,m2,e))
                cto = a*(1-e**2)/(e*separation[i])-1/e
                sto = np.sqrt(1-cto**2)
            if t[i] > (tCrit+Rs) and t[i] < (period-Rs):
                def separationFunc_INV(y):
                    #conTerm = y-a
                    proTerm = (y-a)/a
                    #if conTerm > S:
                    #    conTerm = S
                    #if conTerm < -S:
                    #    conTerm = -S
                    if proTerm > e:
                        proTerm = e
                    if proTerm < -e:
                        proTerm = -e
                    #T = 2*tCrit+np.sqrt(-mu/(2*E))*np.sqrt(S**2-(conTerm)**2)-np.sqrt(-mu/(2*E))*a*np.arcsin((conTerm)/S) + tmin
                    T = 2*tCrit+period/(2*np.pi)*np.sqrt(e**2-(proTerm)**2)-period/(2*np.pi)*np.arcsin(proTerm/e) + tmin
                    #print("model 2:" + str(T))
                    return T-t[i]
                try:
                    separation[i] = opt.brentq(separationFunc_INV,a1,b1, maxiter=100)
                    #separation[i] = opt.brenth(separationFunc_INV,a1,b1, maxiter=100)
                    #separation[i] = opt.ridder(separationFunc_INV,a1,b1,maxiter=100)
                    #separation[i] = opt.bisect(separationFunc_INV,a1,b1,maxiter=100)
                    #separation[i] = opt.newton(separationFunc_INV,.5*(b1+a1),maxiter=100)
                except ValueError:
                    print("timeDelay, t >= tCrit - Oops!  That was no valid number. t[i] = %.2e, a1, b1=%.2e, %.2e.  Try again..." % (phase[i], a1, b1))
                    ### print(t[i]) ### merged into previous statement.
                    separation[i] = 10000

                except RuntimeError:
                    print("timeDelay t >= tCrit WARN: runtime error in brent: a1=%.2e, b1=%.2e: " % (a1, b1))
                    separation[i] = 10000    
                    # print the parameters
                    print("timeDelay t >= tCrit WARN: a=%.2e, S=%.2e, m1=%.2e, m2=%.2e, e=%.2f" % (a,S,m1,m2,e))
                cto = a*(1-e**2)/(e*separation[i])-1/e
                sto = -np.sqrt(1-cto**2)
            if t[i] >= (tCrit-Rs) and t[i] <= (tCrit+Rs):
                #print('hit')
                separation[i] = a*(1+e)
                cto = -1
                sto = 0
                #theta = 0 is defined as the periastron position of the system; thus, at the critical time we may set theta = 180.
                #This is the angle of apastron passage which is the moment of t = tcrit
            if (t[i] >= 0 and t[i] <= (Rs)) or (t[i] >= (period-Rs) and t[i] <= period):
                #print('hit')
                separation[i] = a*(1-e)
                cto = 1
                sto = 0
                #theta = 0 is defined as the periastron position of the system.
            psi = np.linspace(0,np.radians(90),1000)
            deltaMAX[i] = np.sqrt(separation[i]**2+radiusDonor**2-2*separation[i]*radiusDonor*np.cos(psi)) \
                           -(separation[i]-radiusDonor*np.cos(psi))*np.sin(inclination)*(sto*np.cos(omega)+cto*np.sin(omega))

            
            deltaMIN[i] = np.sqrt(separation[i]**2+radiusDonor**2-2*separation[i]*radiusDonor*np.cos(psi)) \
                           -(separation[i]-radiusDonor*np.cos(psi))*np.sin(inclination)*(sto*np.cos(omega)+cto*np.sin(omega)) \
                           -radiusDonor*np.sin(psi)*np.sqrt(1-(np.sin(inclination)*(sto*np.cos(omega)+cto*np.sin(omega)))**2)

            deltaMAX[i]=deltaMAX[i]/constants.c
            deltaMIN[i]=deltaMIN[i]/constants.c
            

            plt.plot(np.degrees(psi),deltaMIN[i],'g-',np.degrees(psi),deltaMAX[i],'r-')
            print("delaypsi=0",(separation[i]-radiusDonor)*(1-np.sin(inclination)*(sto*np.cos(omega)+cto*np.sin(omega)))/constants.c)
            print("radiusDonor",radiusDonor/constants.c)
            print("separation", separation[i]/constants.c)
            return np.degrees(psi), deltaMIN, deltaMAX
                           
        else:
            delta[i] = (a-radiusDonor)*(1-np.sin(inclination)*(np.sin(RotR*t[i]+omega)))/constants.c
    
def timeDelay(phase=np.array([]), \
                  m1_in=1.4, m2_in=0.7, \
                  eccentricity=0.45, period_in=16.3,\
                  inclination=np.radians(60.), omega=np.radians(90.), \
                  gamma=0, radialVelocity = False, rocheLobe = True, radDonor = True, simpleKCOR = True, separationReturn = False, \
                  innerLagrange = False, ellipticalCORR = False, pseudo3D = True):
    
    if ellipticalCORR == True:
        simpleKCOR = False
        innerLagrange = False
    if pseudo3D == True:
        rocheLobe = True
        innerLagrange = False
        


    #High inclination case for alpha = 0
    N0 = .888
    N1 = -1.291
    N2 = 1.541
    N3 = -1.895
    N4 = .861

    """Returns the time delay as a function of orbital phase."""
    
    ### 2018-05-06 WIC - I think this is our place to work in radial
    ### velocities. 

    ### 2018-04-28 WIC - converted the defaults to numerical values
    ### (so that this can be imported separately

    #print("eccentricity",eccentricity)
    #for i in range(len(phase)):
    #    if phase[i] == 0:
    #        phase[i] = .001
    #    if phase[i] == .5:
    #        phase[i] = 5.001
    #    if phase[i] == 1:
    #        phase[i] = .999
    phase = np.mod(phase,1)
    m1 = m1_in * constants.M_Sun
    m2 = m2_in * constants.M_Sun
    q = m2_in/m1_in
    period = period_in * constants.day
    tolerancePhaseTransitionPercent = .0005
    Rs = period*tolerancePhaseTransitionPercent
    e = eccentricity
    phase = np.array(phase)

    ### 2018-04-28 WIC - ensure G is locally set.
    G = constants.G

    ### 2018-04-28 WIC - return gracefully if zero-length input given
    if np.size(phase) < 1:
        return np.array([])

    #opt.brentq(t-separationFunc_INV(y,m1,m2,eccentricity,period),-10,10)
    #mu = m1*m2/(m1+m2)
    a = (G*(m1+m2)*period**2/(4*np.pi**2))**(1./3.)

    if rocheLobe == True:
        if innerLagrange == False:
            radiusDonor = a*(1-eccentricity)*(.49*q**(.667))/(.6*q**(.667)+np.log(1+q**(.333)))
            #Radius of the donor at periastron passage using the Eggleton radius
        if innerLagrange == True:
            radiusDonor = a*(1-eccentricity)*(.5+.227*np.log(q)/np.log(10))
    else:
        radiusDonor = 0
        
    #print("a",a)
    #E = (-G*m1*m2)/(2*a)
    #print("E",E)
    
    #L = m1*m2*np.sqrt(a*G*(1-e**2)/(m1+m2))
    #L_ang=mu*np.sqrt(G*(m1+m2)*a*(1-e*e))
    #print("L_ang",L_ang)
    #print("L",L)
    #print("S insides",L**2/(2*E*mu)+a**2)
    #S = np.sqrt(L**2/(2*E*mu)+a**2)
    RotR = np.sqrt(G*(m1+m2)/(a**3))
    #print("S",S)
    ymin = a*(1-e)
    ymax = a*(1+e)
    #tmin = -np.sqrt(-mu/(2*E))*a*np.pi*0.5
    tmin = -period/4
    #tmin = -np.sqrt(-mu/(2*E))*np.sqrt(S**2-(ymin-a)**2)+np.sqrt(-mu/(2*E))*a*np.arcsin((ymin-a)/S)
    #print("tmin",tmin)
    #tmax = -np.sqrt(-mu/(2*E))*np.sqrt(S**2-(ymax-a)**2)+np.sqrt(-mu/(2*E))*a*np.arcsin((ymax-a)/S)
    #tmax = np.sqrt(-mu/(2*E))*a*np.arcsin((ymax-a)/S)
    #tmax = np.sqrt(-mu/(2*E))*a*np.pi*0.5
    tmax = period/4
    #print("tmax",tmax)
    t = 2 * phase * (tmax - tmin) #converts orbital phase into actual times
    #print(t)
    #print(ymax)
    #print(a)
    #print(S)
    #print((ymax-a)/S)
    #tCrit = -np.sqrt(-mu/(2*E))*np.sqrt(S**2-(ymax-a)**2)+np.sqrt(-mu/(2*E))*a*np.arcsin((ymax-a)/S) - tmin
    tCrit = period/2
    #tCrit = 0.5*period
    #print(tCrit,tCrit/period)
    #print("tCrit",tCrit)
    a1 = ymin
    b1 = ymax
    separation = [0]*len(t)
    delta = [0]*len(t)
    vRVC = [0]*len(t)
    for i in range(len(t)):
        #if t[i] == tmax:
        #    "tmax Hit"
        #if t[i] == tmin:
        #    "tmin Hit"
        if e > .0001:
            if t[i] < (tCrit-Rs) and t[i] > Rs:
                def separationFunc_INV(y):
                    #conTerm = y-a
                    proTerm = (y-a)/a
                    #if conTerm > S:
                    #    conTerm = S
                    #if conTerm < -S:
                    #    conTerm = -S
                    if proTerm > e:
                        proTerm = e
                    if proTerm < -e:
                        proTerm = -e
                        
                    #print("proTerm: " + str(proTerm))
                    #print("e**2: " + str(e**2))
                    #T = -np.sqrt(-mu/(2*E))*np.sqrt(S**2-(conTerm)**2)+np.sqrt(-mu/(2*E))*a*np.arcsin(conTerm/S) - tmin
                    #print("model 1:" + str(T))
                    T = -period/(2*np.pi)*np.sqrt(e**2-(proTerm)**2)+period/(2*np.pi)*np.arcsin(proTerm/e) - tmin
                    #print("model 2:" + str(T))
                    return T-t[i]
                try:
                    separation[i] = opt.brentq(separationFunc_INV,a1,b1, maxiter=100)
                    #separation[i] = opt.brenth(separationFunc_INV,a1,b1, maxiter=100)
                    #separation[i] = opt.ridder(separationFunc_INV,a1,b1,maxiter=100)
                    #separation[i] = opt.bisect(separationFunc_INV,a1,b1,maxiter=100)
                    #separation[i] = opt.newton(separationFunc_INV,.5*(b1+a1),maxiter=100)
                except ValueError:
                    print("timeDelay, t <  tCrit - Oops!  That was no valid number: t[i] = %.2e. a1, b1=%.2e, %.2e. Try again..." % (phase[i], a1, b1))
                    ### print(t[i]) ### Merged into the above line.
                    separation[i] = 10000

                except RuntimeError:
                    print("timeDelay t < tCrit WARN: runtime error in brent: a1=%.2e, b1=%.2e: " % (a1, b1))
                    separation[i] = 10000

                    # print the parameters
                    print("timeDelay t < tCrit WARN: a=%.2e, S=%.2e, m1=%.2e, m2=%.2e, e=%.2f" % (a,S,m1,m2,e))
                cto = a*(1-e**2)/(e*separation[i])-1/e
                sto = np.sqrt(1-cto**2)
            if t[i] > (tCrit+Rs) and t[i] < (period-Rs):
                def separationFunc_INV(y):
                    #conTerm = y-a
                    proTerm = (y-a)/a
                    #if conTerm > S:
                    #    conTerm = S
                    #if conTerm < -S:
                    #    conTerm = -S
                    if proTerm > e:
                        proTerm = e
                    if proTerm < -e:
                        proTerm = -e
                    #T = 2*tCrit+np.sqrt(-mu/(2*E))*np.sqrt(S**2-(conTerm)**2)-np.sqrt(-mu/(2*E))*a*np.arcsin((conTerm)/S) + tmin
                    T = 2*tCrit+period/(2*np.pi)*np.sqrt(e**2-(proTerm)**2)-period/(2*np.pi)*np.arcsin(proTerm/e) + tmin
                    #print("model 2:" + str(T))
                    return T-t[i]
                try:
                    separation[i] = opt.brentq(separationFunc_INV,a1,b1, maxiter=100)
                    #separation[i] = opt.brenth(separationFunc_INV,a1,b1, maxiter=100)
                    #separation[i] = opt.ridder(separationFunc_INV,a1,b1,maxiter=100)
                    #separation[i] = opt.bisect(separationFunc_INV,a1,b1,maxiter=100)
                    #separation[i] = opt.newton(separationFunc_INV,.5*(b1+a1),maxiter=100)
                except ValueError:
                    print("timeDelay, t >= tCrit - Oops!  That was no valid number. t[i] = %.2e, a1, b1=%.2e, %.2e.  Try again..." % (phase[i], a1, b1))
                    ### print(t[i]) ### merged into previous statement.
                    separation[i] = 10000

                except RuntimeError:
                    print("timeDelay t >= tCrit WARN: runtime error in brent: a1=%.2e, b1=%.2e: " % (a1, b1))
                    separation[i] = 10000    
                    # print the parameters
                    print("timeDelay t >= tCrit WARN: a=%.2e, S=%.2e, m1=%.2e, m2=%.2e, e=%.2f" % (a,S,m1,m2,e))
                cto = a*(1-e**2)/(e*separation[i])-1/e
                sto = -np.sqrt(1-cto**2)
            if t[i] >= (tCrit-Rs) and t[i] <= (tCrit+Rs):
                #print('hit')
                separation[i] = a*(1+e)
                cto = -1
                sto = 0
                #theta = 0 is defined as the periastron position of the system; thus, at the critical time we may set theta = 180.
                #This is the angle of apastron passage which is the moment of t = tcrit
            if (t[i] >= 0 and t[i] <= (Rs)) or (t[i] >= (period-Rs) and t[i] <= period):
                #print('hit')
                separation[i] = a*(1-e)
                cto = 1
                sto = 0
                #theta = 0 is defined as the periastron position of the system.
            delta[i] = (separation[i]-radiusDonor)*(1-np.sin(inclination)*(sto*np.cos(omega)+cto*np.sin(omega)))/constants.c
        else:
            delta[i] = (a-radiusDonor)*(1-np.sin(inclination)*(np.sin(RotR*t[i]+omega)))/constants.c
    if separationReturn == True:
        if e > .0001:
            return separation
        else:
            return [a]*len(phase)
    if radialVelocity == False:
        if pseudo3D == True:
            #print(delta)
            CORR = np.zeros(len(phase))
            for i in range(len(delta)):
                #print("q",q)
                #print("inclination",inclination)
                #print("ttt[i]",ttt[i])
                CORR[i] = FUNC_CORRECTION([q,inclination,phase[i]])
                #print("CORR",CORR)
                delta[i] = delta[i]*CORR[i]
        if ellipticalCORR == True:
            A = 1
        return delta
    else:
        for i in range(len(t)):
            if e > .0001:
                #print(vels,"vels")
                #theta = np.arccos(a*(1-e**2)/(e*separation[i])-1/e)
                cto = a*(1-e**2)/(e*separation[i])-1/e
                if t[i] < tCrit:
                    sto = np.sqrt(1-cto**2)
                else:
                    sto = -np.sqrt(1-cto**2)
                #vPerLON = vels*(e*np.sin(theta)*np.sin(theta-omega)+(1+e*np.cos(theta))*np.cos(theta-omega))/(np.sqrt(1+2*e*np.cos(theta)+e**2))
                vPerLON = [0,0]
                vPerLON[0] = np.sin(inclination)*(m1/(m1+m2))*(2*np.pi*a)/(period*np.sqrt(1-e**2))*(e*np.cos(omega)+cto*np.cos(omega)-sto*np.sin(omega)) #radialVelocity for mass 2
                #vPerLON[0] = np.sin(inclination)*(m1/(m1+m2))*vels*(e*sto*(sto*np.cos(omega)-cto*np.sin(omega))+(1+e*cto)*(cto*np.cos(omega)+sto*np.sin(omega)))/(np.sqrt(1+2*e*cto+e**2)) #radialVelocity for mass 2
                vPerLON[1] = -np.sin(inclination)*(m2/(m1+m2))*(2*np.pi*a)/(period*np.sqrt(1-e**2))*(e*np.cos(omega)+cto*np.cos(omega)-sto*np.sin(omega)) #radial Velocity for mass 1
                #vPerLON[1] = -np.sin(inclination)*(m2/(m1+m2))*vels*(e*sto*(sto*np.cos(omega)-cto*np.sin(omega))+(1+e*cto)*(cto*np.cos(omega)+sto*np.sin(omega)))/(np.sqrt(1+2*e*cto+e**2)) #radial Velocity for mass 1
                #print(vPerLON,"vPerLON")

                if (rocheLobe == True) and (simpleKCOR == True):
                    vPerLON[0] = vPerLON[0]*(m1*separation[i]-(m1+m2)*radiusDonor)/(m1*separation[i]) #Changes radial velocity signal so that it originates from the side of the donor facing the accretor.
                    #vPerLON[0] = vPerLON[0]/(N0+N1*q+N2*q**2+N3*q**3+N4*q**4)

                
            else:
                vels = np.sqrt(G*(m1+m2)*(1/a))
                vPerLON = [0,0]
                vPerLON[0] = np.sin(inclination)*(m1/(m1+m2))*vels*(np.cos(RotR*t[i]+omega))
                vPerLON[1] = -np.sin(inclination)*(m2/(m1+m2))*vels*(np.cos(RotR*t[i]+omega))

                if (rocheLobe == True) and (simpleKCOR == True):
                    vPerLON[0] = vPerLON[0]*(m1*a-(m1+m2)*radiusDonor)/(m1*a) #Changes radial velocity signal so that it originates from the side of the donor facing the accretor.
                    #vPerLON[0] = vPerLON[0]/(N0+N1*q+N2*q**2+N3*q**3+N4*q**4)

                
            if radDonor == True:
                vRVC[i] = vPerLON[0] #Only stores the first mass values
            else:
                vRVC[i] = vPerLON[1] #Only stores the second mass values
        vRVC = np.add(vRVC,gamma)
        return vRVC

def timeDelayPlot(radialVelocity = True, m1_in=1.4, m2_in=0.3, \
                  eccentricity=.5, period_in=16.3,\
                  inclination=np.radians(90.), omega=np.radians(90.), \
                  rocheLobe = True, radDonor = True, pseudo3D = False):
    tt = np.linspace(0,1,5000)
    yy = timeDelay(tt,m1_in = m1_in, m2_in = m2_in, eccentricity = eccentricity, period_in = period_in, inclination = inclination, omega = omega, \
                   radialVelocity = radialVelocity, rocheLobe = rocheLobe, radDonor = radDonor, pseudo3D = pseudo3D)
    plt.figure(1)
    plt.ylabel("Echo Delay (s)")
    if radialVelocity == True:
        #plt.ylabel(r"Velocity $\frac{m}{s}$")
        plt.ylabel(r"Velocity ($\frac{km}{s}$)")
        yy = .001*np.array(yy)
    plt.plot(tt,yy)
    plt.xlabel("Phase")
    m1 = m1_in * constants.M_Sun
    m2 = m2_in * constants.M_Sun
    period = period_in*constants.day
    a = (G*(m1+m2)*period**2/(4*np.pi**2))**(1./3.)/constants.c
    q = m2_in/m1_in
    print("a = " + str(a))
    print("Donor Radius = " + str(a*(.49*q**(.667))/(.6*q**(.667)+np.log(1+q**(.333)))))
    plt.show(block=False)

def orbit(m1_in=1.4, m2_in=0.3, eccentricity=0.45, period_in=16.3, \
              figOrbit='figExample_orbit.png'):

    """Convenience method to plot the orbit"""

    ### 2018-04-28 WIC - converted defaults to numerical
    ### values. Renamed m1, m2 to m1_in, m2_in for consistency with
    ### terms elsewhere (though this is only convenience, variables
    ### m1_in, m2_in are only defined within the scope of orbit() ).
    G = constants.G
    e = eccentricity
    P = period_in*constants.day
    m1 = m1_in * constants.M_Sun
    m2 = m2_in * constants.M_Sun
    a = (G*(m1+m2)*P**2/(4*np.pi**2))**(1/3)
    L = m1*m2*np.sqrt(a*G*(1-e**2)/(m1+m2))
    mu = m1_in * m2_in / (m1_in + m2_in)
    e = eccentricity
    E = (-G*m1*m2)/(2*a)
    w = 0.5*G*m1*m2/E
    S = np.sqrt(L**2/(2*E*mu)+a**2)
    #E = -1.
    #S = 1.11803398875
    #w = -1.5
    #print(a)
    #print("Orbit INFO: E=%.2e, S=%.2e, w=%.2e" % (E,S,w))
    #print(S)
    #print(w)
    ymin = -1*(S+w)
    ymax = S-w
    y = np.linspace(ymin,ymax,1000)
    #t = separationFunc_INV(y,m1,m2,eccentricity,period)
    tmin = -np.sqrt(-mu/(2*E))*np.sqrt(S**2-(ymin+w)**2)-np.sqrt(-mu/(2*E))*w*np.arcsin((ymin+w)/S)
    #print("Orbit INFO: tmin: %.3e" % (tmin))
    t = -np.sqrt(-mu/(2*E))*np.sqrt(S**2-(y+w)**2)-np.sqrt(-mu/(2*E))*w*np.arcsin((y+w)/S) -tmin
    #t = -np.sqrt(-mu/(2*E))*np.sqrt(S**2-(y+w)**2)-np.sqrt(-mu/(2*E))*w*np.arcsin((y+w)/S)
    #print("Orbit INFO: t %.2e" % (t))
    #tCrit = separationFunc_INV(ymax,m1,m2,eccentricity,period)
    tCrit = -np.sqrt(-mu/(2*E))*np.sqrt(S**2-(ymax+w)**2)-np.sqrt(-mu/(2*E))*w*np.arcsin((ymax+w)/S) - tmin
    #tmin = separationFunc_INV(ymin,m1,m2,eccentricity,period)
    T = 2*tCrit-t
    #print(y)
    plt.figure()
    plt.plot(y,t)
    plt.plot(y,T)
    plt.show(block=False)
    
    # Write the orbit figure to disk
    #plt.savefig(figOrbit)

def go(e=0.45, a=6.0e9, figName='figTMP_go.png'):   

    """Plots f_simple for given eccentricity and semimajor axis"""

    ### 2018-04-28 WIC - this seems to refer to a function F(r) that
    ### has been completely commented out (near line 230 or so)... is
    ### this method now obsolete? 
    ###
    ### Either way, I have added in some default values.

    ### WARN - this may not work! I think "integrand" expects x,y to
    ### be defined globally...

    rr = np.linspace((1-e)*a,(1+e)*a,1000)
    #print(len(rr))
    plt.xlabel("time")
    plt.ylabel("separation")
    tt = np.array(F(rr))
    knot = F_simple((1+e)*a)
    print("orbit INFO: knot:", knot)
    plt.plot(tt,rr,2*knot-tt,rr)
    plt.show(block=False)
    plt.savefig(figName)

def simpleFit(X=np.array([]), Y=np.array([]), yerr=np.array([]), \
                  func=timeDelay, \
                  xlabel = 'xlabel', ylabel = 'ylabel', \
                  figNum=4, \
                  plotFit=True, figPlot= 'fig_simpleFit.png', radialVelocity = False, rocheLobe = True):

    """Does a fit to a delay curve, returning the parameters and
    plotting the delay curve with the fitted function."""
    def FUNC(x,m1_in,m2_in,eccentricity,period_in,inclination,omega,t0,radialVelocity=radialVelocity,rocheLobe=rocheLobe):
        x_phase = phaseFromTimes(x,t0=t0,period_in=period_in)
        y = timeDelay(phase=x_phase, \
                  m1_in=m1_in, m2_in=m2_in, \
                  eccentricity=eccentricity, period_in=period_in,\
                  inclination=inclination, omega=omega,radialVelocity=radialVelocity,rocheLobe=rocheLobe)
        return y
    func = FUNC
    ### 2018-04-28 WIC - put in default values so that this will work
    ### stand-alone if needed. Exit "gracefully" if input not
    ### given. Plotting is now an option.
    if np.size(X) < 1 or np.size(Y) < 1:
        return np.array([])

    ### 2018-04-28 WIC: brought the bouunds up to a separate array
    bounds = \
        ([1.38 , 0, 0, 0.1, 0, -np.pi, -5], [1.42, 50, 1, 1.35, np.pi, 2*np.pi,5])
    #bounds = \
    #    ([1.28 , 0, 0, 10.25, 0, 0, -5], [1.48, 50, 1, 20.35, 2*np.pi, 2*np.pi,5])
    pGuess = [1.4, .4, .1, .78, np.radians(27), -1*np.radians(87), 0]

    #popt, pcov = opt.curve_fit(func,X,Y,bounds=([.001 , .000001], [.56, .02]),p0 = [ 0.1,  .00003] )
    ### popt, pcov = opt.curve_fit(func,X,Y,bounds=([1.38 , 0, 0, 16.25, 0, 0], [1.42, 10, 1, 16.35, 2*np.pi, 2*np.pi]),p0 = [1.4, 1, .7, 16.3, 1, .2])

    popt, pcov = opt.curve_fit(func,X,Y,bounds=bounds, p0=pGuess)
    return popt

### 2018-05-04 routines to transform to and from the
### reparameterization go here
def parsToReparam(parsIn=np.array([]), nMin=7, degrees=False):

    """Translates the parameters (mass, angles, etc.) into a
    reparameterization that will allow convenient sampling of periodic
    variables"""

    # Initialise the output to the input params. That way, anything we
    # don't explicitly alter is left unchanged.
    parsRepar = np.copy(parsIn)

    # if the input parameters aren't what we expect, return them
    # unchanged.
    if np.size(parsIn) < nMin:
        return parsRepar

    # We assume the following physical parameters:
    # 
    # [m1, m2, e, P, i, w, others...]

    # and use the following reparameterization (with m1s = sqrt(m1)):
    
    # [m1s cos(i), m1s sin(i), e, P, m2s cos(w), m2s sin(w), others... ]

    m1sqrt = np.sqrt(parsIn[0])
    m2sqrt = np.sqrt(parsIn[1])

    iRad = np.copy(parsIn[4])
    wRad = np.copy(parsIn[5])
    if degrees:
        iRad = np.radians(parsIn[4])
        wRad = np.radians(parsIn[5])

    # update the pieces 
    parsRepar[0] = m1sqrt * np.cos(iRad)
    parsRepar[1] = m1sqrt * np.sin(iRad)
    parsRepar[4] = m2sqrt * np.cos(wRad)
    parsRepar[5] = m2sqrt * np.sin(wRad)

    return parsRepar

def reparamToPars(parsRepar=np.array([]), nMin=7, degrees=False):

    """Transforms the reparameterized parameters back to the original
    parameterization"""

    ### 2018-05-04 WIC: currently assuming the angles are in degrees. 

    # initialize
    parsOut = np.copy(parsRepar)
    if np.size(parsRepar) < nMin:
        return parsOut

    # m1 and m2
    parsOut[0] = parsRepar[0]**2 + parsRepar[1]**2
    parsOut[1] = parsRepar[4]**2 + parsRepar[5]**2

    # i, w
    parsOut[4] = np.arctan2(parsRepar[1],parsRepar[0]) 
    parsOut[5] = np.arctan2(parsRepar[5],parsRepar[4])

    if degrees:
        parsOut[4] = np.degrees(parsOut[4])
        parsOut[5] = np.degrees(parsOut[5]) 

    return parsOut

def lnlike(theta, x, y, y_err, obsRealTime = True, radialVelocity = False, rocheLobe = True, radDonor = True):
    m1F, m2F, eF, pF, iF, wF, t0F  = theta #model parameters, N = 7
    if obsRealTime == True:
        x = phaseFromTimes(x, t0 = t0F, period_in = pF)
    model = timeDelay(x,m1F,m2F,eF,pF,iF,wF,rocheLobe = rocheLobe,radialVelocity = radialVelocity, radDonor = radDonor)
    inv_sigma2 = 1.0/(y_err**2)
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

def phaseFromTimes(time, t0 = 0, period_in = 16.3):
    period = period_in * constants.day
    time = np.array(time)
    phase = np.mod((time-t0)/period,1)
    return phase
def timesFromPhase(phi, t0 = 0, period_in = 16.3):
    period = period_in * constants.day
    phi = np.array(phi)
    time = phi*period+t0
    return time

def priorUninform(theta=np.array([]), pars=np.array([])):

    """Uninformative prior"""

    return 0.

def priorFlat(theta=np.array([]), 
                  bounds=np.array([ [1.35, 0., .001, 0., -np.pi, -np.pi, -np.inf], \
                  [1.45, 8., .999, 30.45, np.pi, np.pi, np.inf] ])):

    """Returns a 1/0 using flat priors in the parameters"""

    # initialize the return value
    thisPrior = 0.

    # if any of the parameters are outside bounds, return -np.inf,
    # else return 0.
    bOut = (theta <= bounds[0]) | (theta > bounds[1])
    if np.sum(bOut) > 0:
        thisPrior = -np.inf

    return thisPrior

def lnprior(theta, priorFunc=priorFlat, hyperPars=np.array([]) ):

    """Computes the log(prior) for a given set of parameters. Given
    parameters theta, this evaluates function priorFunc on parameters
    theta using hyperparameters hyperPars."""

    ### WIC - I have written function priorFlat to do what these lines
    ### used to do, but in a more uniform sense. Note that I now think
    ### this method is obsolete, and have copied the line below into
    ### lnProb().
    return priorFunc(theta, hyperPars)

###    m1F, m2F, eF, pF, iF, wF = theta
###    if 1.35 < m1F < 1.45 and 0 < m2F < 8 and .001 < eF < .999 and 0 < pF < 30.45 and -1*np.pi <= iF < np.pi and -1*np.pi <= wF < np.pi:
###        return 0.0
###    return -np.inf

def lnProbObj(thetaIn, x_echo, y_echo, y_err_echo, x_radial, y_radial, y_err_radial, x_radialCompact, y_radialCompact, y_err_radialCompact, priorObj=None, \
                  useReparam=True, \
                  obsRealTime=False, \
                  enforcePositive=True, \
                  rocheLobe=True):
    
    """ Like lnprob but accepts an instance of the Priors() class to
    evaluate the prior probability. EnforcePositive returns -np.inf if
    any of the first five parameters are less than zero.
    if useReparam is set, then the parameters used by the sampler were
    "reparameterized," and so we convert them back into physical
    parameters. """
    
    ### 2018-05-06 WIC - radial velocity data would be taken in here
    ### as optional arguments.

    ### 2018-05-04 WIC - If we're using a reparameterization, then
    ### convert the reparameters into the physical parameters expected
    ### by the prior and by the likelihood function.

    if useReparam:
        theta = reparamToPars(thetaIn)
    else:
        theta = np.copy(thetaIn)

    ### WIC 2018-05-06 - this enforcePositive call may be redundant,
    ### since the Priors object now enforces positivity in the first
    ### FOUR arguments (not the inclination). 

    ### Added this constraint, which might be already too
    ### restrictive. This stops emcee from even trying to sample any
    ### jump for which any of the first five parameters is
    ### negative. At the cost of extra specialization, this does
    ### prevent the module from even attempting to evaluate the ln
    ### likielihood for values that will produce NaN (and thus require
    ### expensive solver evaluations that are doomed from the start).
    bNeg = theta[0:5] < 0
    if np.sum(bNeg) > 0:
        if enforcePositive:
            return -np.inf

    lp = priorObj.evaluate(theta)+lQ(theta)
    #print(lp)
    
    if not np.isfinite(lp): 
        return -np.inf

    ### 2018-04-31 WIC - if a proposed stretch produced NaN lnProb,
    ### return -np.inf instead (so that the proposed stretch is
    ### rejected rather than returning NaN and bringing the whole
    ### process to a halt).
    lnPro = lp + lnlike(theta, x_echo, y_echo, y_err_echo, radialVelocity = False, obsRealTime = obsRealTime, rocheLobe = rocheLobe) + \
            lnlike(theta, x_radial, y_radial, y_err_radial, radialVelocity = True, obsRealTime = obsRealTime) + \
            lnlike(theta, x_radialCompact, y_radialCompact, y_err_radialCompact, radialVelocity = True, obsRealTime = obsRealTime, radDonor = False)
    if np.isnan(lnPro):
        lnPro = -np.inf
    
    return lnPro


#def lnprob(theta, x, y, y_err, priorFunc=priorFlat, priorPars=np.array([]) ):
#
#    """Computes ln(prob). Arguments:
#    theta = the test parameters
#    x,y,y_err = phase, delay, delay_error
#    priorFunc = the name of a method that returns lnprior(theta,
#                priorPars) . If that method is not defined, or behaves
#                strangely, then the results of lnprob will be difficult
#                to understand.
#    priorPars = the hyperparameters of the prior. 
#    """
#
#    #lp = lnprior(theta, priorFunc, priorPars)
#
#    lp = priorFunc(theta, priorPars)
#
#    if not np.isfinite(lp): ### Is this clause needed?
#        return -np.inf
#    return lp + lnlike(theta, x, y, y_err)
def runSequence(runList='runList.txt', timeFile='timeFile.txt'):
    """Runs full simulation multiple times in accordance to the command file (runsList) which dictates how many runs and the parameters for each run."""
    ### Blank argument dictionary
    dArgs = {}
    if not os.access(runList, os.R_OK):
        print("runSimsFromPars WARN - parameter file not readable: %s" \
                  % (runList))
    with open(runList, 'r') as rObj:
        for thisLine in rObj:
            if thisLine.find('#') > -1:
                continue
            vLine = thisLine.strip().split()
            thisArg = vLine[0]
            thisVal = [0]*(len(vLine)-1)
            for k in range(len(vLine)-1):
                thisVal[k] = vLine[k+1] ### at the moment still a string
            dArgs[thisArg] = thisVal
            
    parsFilUsed = list()
    parsFilUsed.append(runList)
    dArgs['parsFilUsed'] = parsFilUsed
    tStartInitial = time.time()
    runNumber = int(dArgs['runNumber'][0])
    tStarts = [0]*runNumber
    tEnds = [0]*runNumber
    runDuration = [0]*runNumber
    del dArgs['runNumber'] #removes the run number from the dictionary so that it can be passed to the runSims function
    print("Start Time = " + time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
    print("Lauching for " + dArgs['nSteps'][0] + " steps with " + dArgs['nWalkers'][0] + " walkers for " + str(runNumber) + " iterations.")
    index = 0
    C = []
    with open(timeFile, 'r') as rObj:
        for thisLine in rObj:
            index += 1
            if thisLine.find('Run_Parameters_(runNumber,Steps,Walkers):') > -1:
                vLine = thisLine.replace('Run_Parameters_(runNumber,Steps,Walkers):','')
                Val = eval(vLine)
                totalEvaluations = np.exp(sum(np.log(Val)))
            if thisLine.find('Durations:') > -1:
                vLine = thisLine.replace('Durations:','')
                Val = eval(vLine)
                totalTime = sum(Val)
                C.append(totalTime/totalEvaluations)
    try:
        Const = np.mean(C)/60 #Measured Proportionality between timeDelay evaluation number and run duration in minutes.
    except:
        Const = 0.000104285714286
    estimatedRunTime = runNumber*int(dArgs['nSteps'][0])*int(dArgs['nWalkers'][0])*Const
    print("Estimated Run Time: " + str(estimatedRunTime) + " minutes")
    keys = list(dArgs.keys())
    for j in range(runNumber):
        print('j',j)
        Params = {}
        for i in range(len(keys)):
            try:
                Params[keys[i]] = dArgs[keys[i]][j]
            except:
                Params[keys[i]] = dArgs[keys[i]][0]
                
            Params[keys[i]]
            if Params[keys[i]].find('.') > -1:
                try:
                    Params[keys[i]] = float(Params[keys[i]]) * 1.0
                except:
                    Params[keys[i]] = Params[keys[i]][:]
            else:
                try:
                    Params[keys[i]] = int(Params[keys[i]]) * 1
                except:
                    Params[keys[i]] = Params[keys[i]][:]
        directory = "run_" + str(j)
        if not os.path.exists(directory):
            os.makedirs(directory)
        os.chdir(directory)
        #changes to a new directory where the results from runSims can be dumped.
        tStarts[j] = time.time()
        print(Params)
        runSims(**Params)
        tEnds[j] = time.time()
        runDuration[j] = tEnds[j]-tStarts[j]
        os.chdir("..")
        #changes back to the original directory.
    with open(timeFile, 'a') as wFile:
        #wFile.write("run time durations of simulator_2.py using runSequence routine. \n")
        wFile.write("Run_Parameters_(runNumber,Steps,Walkers): " + str([runNumber,int(dArgs['nSteps'][0]),int(dArgs['nWalkers'][0])]) + "\n")
        wFile.write("Start_Times: " + str(tStarts) + "\n")
        wFile.write("End_Times: " + str(tEnds) + "\n")
        wFile.write("Durations: " + str(runDuration) + "\n")

def loadParsAndRun(parsIn='parsInp.txt', runAnyway=True, TEST=True):

    """Runs simulations taking input from parameter file. Arguments:
    parsIn = text file from which parameters will be taken.
    
    loadParsAndRun() will produce a default parameter file.
    Prior parameters can be specified by file in the parameter file."""

    ### Blank argument dictionary
    dArgs = {}

    if len(parsIn) < 3:
        print("runSimsFromPars WARN - bad input parameter filename: %s" \
                  % (parsIn))

        if not runAnyway:
            print("runSimsFromPars WARN - runAnyway=False. Returning.")
            return

        parsOutDummy = 'parsDEFAULT.txt'
        print("runSimsFromPars INFO - running anyway, parameters will write to file %s" % (parsOutDummy))

        runSims(parsFilOut=parsOutDummy, EMCEE=False, TEST=TEST)

        if os.access(parsOutDummy, os.R_OK):
            print("runSimsFromPars INFO - Default parameters in file %s" \
                      % (parsOutDummy))

        return ### Closes the conditional on input filename zero length

    if not os.access(parsIn, os.R_OK):
        print("runSimsFromPars WARN - parameter file not readable: %s" \
                  % (parsIn))

        if not runAnyway:
            print("runSimsFromPars WARN - runAnyway=False. Returning.")
            return

        print("runSimsFromPars WARN - running anyway with defaults")
        print("runSimsFromPars WARN - (This may not be what you want!)")
        
        ## Here we run anyway with only the default arguments
        runSims(EMCEE=False,TEST=TEST)
        return  ### Closes conditional on param file not found

    ### 2018-04-28 WIC - now we take advantage of python's
    ### argument-passing structure. We read in the parameter file as a
    ### dictionary, then use the arguments found to launch
    ### runSims(). The advantage of this approach is that the
    ### parameter file can be as incomplete as we wish, anything not
    ### specified will use default value.

    ### So, this is a quick and dirty hack thrown together with
    ### reinvented parsing (ugh), which won't work on multi-valued
    ### input parameters (like lists). There's also no checking that
    ### the input pargument names actually have correspondence to the
    ### names expected by runSims(). That could all come later...

    dArgs = {}
    with open(parsIn, 'r') as rObj:
        for thisLine in rObj:
            if thisLine.find('#') > -1:
                continue
            vLine = thisLine.strip().split()
            if len(vLine) != 2:
                continue

            # OK now we parse the [argument, value] pair. This is
            # going to be a bit of a hack but let's go there anyway
            thisArg = vLine[0]
            thisRaw = vLine[1] ### at the moment still a string
            if thisRaw.find('.') > -1:
                try:
                    thisVal = float(thisRaw) * 1.0
                except:
                    thisVal = thisRaw[:]
            else:
                try:
                    thisVal = int(thisRaw) * 1
                except:
                    thisVal = thisRaw[:]
            
            # at this point, pass the argument/value pair to the
            # dictionary
            dArgs[thisArg] = thisVal

    # Finally, ensure runSims writes out the correct filename of the
    # parameter-file used!
    dArgs['parsFilUsed'] = parsIn

    tStart = time.time()
    if 'nSteps' in dArgs.keys(): ### Print some useful timing info
        print("loadParsAndRun INFO - launching sims for %.2e steps." \
              % (dArgs['nSteps']))
    runSims(**dArgs,TEST=TEST)

    tEnd = time.time()
    print("loadParsAndRun INFO - Run took %.2e minutes." \
          % ((tEnd-tStart)/60.) )

def runSims(nObs_echo = 5, nObs_radial = 0, nObs_radialCompact = 0, \
                m1_true=1.4, m2_true=.3, \
                eccentricity_true = .45, \
                period_true = 16.3, \
                iDeg_true = 60., \
                omegaDeg_true=90., \
                t0_true=0., \
                cannedPhases_echo=0, \
                cannedPhases_radial=0, \
                cannedPhases_radialCompact=0, \
                nFineTimes=1000, \
                obsUncty_echo = 10, obsUncty_radial = 10000, obsUncty_radialCompact = 10000, \
                obsSetting_echo=0, \
                obsSetting_radial=0, \
                obsSetting_radialCompact=0, \
                preserveOldFigures=False, \
                closeFigsAfter=False, \
                writePars=True, \
                parsFilOut='parsRan.txt', \
                dataFile='dataFile.txt', \
                parsFilUsed='', \
                priorName='binaryBounded', \
                priorFile='parsPrior.txt', \
                useGuessFile=True, \
                guessFile='cannedGuess.txt', \
                guessRangeFile='cannedRangeGuess.txt', \
                useOO=True, \
                fewerFigs=True, \
                EMCEE=False, \
                nSteps=3500, \
                nWalkers=100, \
                useReparam=True, \
                Verbose=True, \
                obsRealTime = True, \
                TEST = True, \
                rocheLobe = True, \
                genWith3D = True):
    
    """Run the simulations. A few notes on variables whose names might
    not be self-explanatory:
    parsFilOut = text file name to store all the arguments.
    parsFilUsed = passed-in text file name for any user-specified
    parameters (used only as a record; a higher-level routine would
    have read in the contents of this file to send the arguments to
    runSims)."""

    ### 2018-04-28 WIC - added this method. We may want to think about
    ### how we might improve the inputting of parameter values.

    ### Close all open matplotlib windows before running anything
    if not preserveOldFigures:
        killPlotWindows(Verbose=False)

    if nObs_echo < 1:
        print("runSims WARN - Low number of echo observations: nObs = %i" \
                  % (nObs_echo))

    ### do some conversions
    inclination_true = np.radians(iDeg_true)
    omega_true = np.radians(omegaDeg_true)

    ### Set up the true parameters array
    tt = np.linspace(0., 1., nFineTimes)
    if TEST == True:
        true_parameters = [m1_true, m2_true, eccentricity_true, \
                               period_true, inclination_true, \
                               omega_true, t0_true]

        if Verbose:
            print("runSims INFO - True parameters:", true_parameters)

        # OK now set up the observations, with measurement uncertainty.
        x_echo = obsPhases(nObs_echo, canned=cannedPhases_echo, setting=obsSetting_echo, obsRealTime=obsRealTime, t0=t0_true, period_in = period_true)
        x_radial = obsPhases(nObs_radial, canned = cannedPhases_radial, setting=obsSetting_radial, obsRealTime=obsRealTime, t0=t0_true, period_in = period_true)
        x_radialCompact = obsPhases(nObs_radialCompact, canned = cannedPhases_radialCompact, setting=obsSetting_radialCompact, obsRealTime=obsRealTime, t0=t0_true, period_in = period_true)
        if obsRealTime == True:
            x_echo_phases = phaseFromTimes(x_echo, t0=t0_true, period_in=period_true)
            x_radial_phases = phaseFromTimes(x_radial, t0=t0_true, period_in=period_true)
            x_radialCompact_phases = phaseFromTimes(x_radialCompact, t0=t0_true, period_in=period_true)
        else:
            x_echo_phases = x_echo
            x_radial_phases = x_radial
            x_radialCompact_phases = x_radialCompact

        #Generate Hypothetical Observations from the true 3d Model    
        yOrig_echo = np.zeros(len(x_echo))
        if genWith3D == True:
            for i in range(len(yOrig_echo)):
                yOrig_echo[i] = RocheModelDelay(x_echo_phases[i], m1_true, m2_true, period_true, inclination_true, eccentricity_true, \
                                    omega_true, radialVelocity = False)
        else:
            yOrig_echo = timeDelay(x_echo_phases, m1_true, m2_true, eccentricity_true, \
                           period_true, inclination_true, omega_true, rocheLobe = rocheLobe, radialVelocity = False)
            

        
        yOrig_radial = timeDelay(x_radial_phases, m1_true, m2_true, eccentricity_true, \
                              period_true, inclination_true, omega_true, radialVelocity = True)
        yOrig_radialCompact = timeDelay(x_radialCompact_phases, m1_true, m2_true, eccentricity_true, \
                              period_true, inclination_true, omega_true, radialVelocity = True, radDonor = False)

        # Rather than re-using nObs, force the uncty to have the same
        # length as the phases.
        y_err_echo = np.random.normal(0., obsUncty_echo, np.size(x_echo))

        #Manually Setting Errors for Determinism
        y_err_echo = np.array([.75,-.75,1.2,-.75,.75])
        #y_err_echo = np.array([])



        

        
        y_echo = yOrig_echo + np.array([.55,-.35,1,-.73,.3])
        #y_echo = yOrig_echo
        y_err_radial = np.random.normal(0., obsUncty_radial, np.size(x_radial))


        #Manually Setting Errors for Determinism
        y_err_radial = np.array([500,-500,500,-500,500,-500,500,-500,500,-500, \
                                 500,-500,500,-500,500,-500,500,-500,500,-500])*15
        #y_err_radial = np.array([])

        
        y_radial = yOrig_radial + y_err_radial
        y_err_radialCompact = np.random.normal(0., obsUncty_radialCompact, np.size(x_radialCompact))
        y_radialCompact = yOrig_radialCompact + y_err_radialCompact

        # Now produce the fine-grained grid
        yy_echo = timeDelay(tt, m1_true, m2_true, eccentricity_true, \
                           period_true, inclination_true, omega_true, rocheLobe = rocheLobe, radialVelocity = False)
        yy_radial = timeDelay(tt, m1_true, m2_true, eccentricity_true, \
                           period_true, inclination_true, omega_true, radialVelocity = True)
        yy_radialCompact = timeDelay(tt, m1_true, m2_true, eccentricity_true, \
                           period_true, inclination_true, omega_true, radialVelocity = True, radDonor = False)
    if TEST == False:
        #SCO X-1
        #x_echo_phases = np.array([.02,.16]) + .5
        #y_echo = np.array([13.5,8.5])
        #y_err_echo = np.array([3,1.5])
        #x_echo_phases = np.array([.5])
        #y_echo = np.array([14])
        #y_err_echo = np.array([1])
        x_echo_phases = np.array([.22,.85,.57])
        y_echo = np.array([1.8,.9,4.4])
        y_err_echo = np.array([.15,.15,.15])
        ###x_echo_phases = np.array([.907,.802,.934,.461,.204,.986])
        ###y_echo = np.array([2.8,3.3,2.7,1.9,2.8,1.2])
        ###y_err_echo = np.array([.4,.4,.3,1.0,.4,.2])

        #x_echo_phases = np.array([])
        #y_echo = np.array([])
        #y_err_echo = np.array([])
        

        x_radial_phases = np.linspace(0,1,16)
        y_radial = -277*10**3*np.array(np.sin(2*np.pi*x_radial_phases))
        #y_radial = timeDelay(x_radial_phases,eccentricity = 0,m1_in=1.4,m2_in=.7,period_in=.787,inclination=np.radians(44),radialVelocity = True) #SCO X-1
        
        vLine = []
        #try:
        #    with open('donorRadialData.txt', 'r') as rObj:
        #        for thisLine in rObj:
        #            print(thisLine)
        #            vLine.append(thisLine)
        #    x_radial_phases = np.mod(np.array(eval(vLine[0])) + .5,1)
        #    y_radial = np.array(eval(vLine[2]))*1000
        #except:
        #    x_radial_phases = []
        #    y_radial = []
        #
        vLine = []
        try:
            with open('compactRadialData.txt', 'r') as rObj:
                for thisLine in rObj:
                    print(thisLine)
                    vLine.append(thisLine)
            x_radialCompact_phases = np.mod(np.array(eval(vLine[0])),1)
            y_radialCompact = np.array(eval(vLine[2]))*1000
            #plt.scatter(x_radial_phases,y_radial,'y--')
        except:
            x_radialCompact_phases = []
            y_radialCompact = [] #Manually turning the compact object radial velocity off

            #x_radial_phases = [] #Manually turning the donor object radial velocity off
            #y_radial = []



        #x_radial_phases = []
        #y_radial = []
        #y_err_radial = []
        #y_err_radial = np.array([0]*len(y_radial))
        y_err_radial = .25*np.array(y_radial)+1000
        #y_err_radial = .10*y_radial+1000
        y_err_radialCompact = np.array([0]*len(y_radialCompact))
        x_echo = timesFromPhase(x_echo_phases, t0=t0_true, period_in=period_true)
        x_radial = timesFromPhase(x_radial_phases, t0=t0_true, period_in=period_true)
        x_radialCompact = timesFromPhase(x_radialCompact_phases, t0=t0_true, period_in=period_true)
        nObs_echo = len(y_echo)
        nObs_radial = len(y_radial)
        nObs_radialCompact = len(y_radialCompact)

        
    ### Moved the prior-object generation up here so that we can use
    ### it to take a guess with the prior if needed.
    
        
    ## Can get the initial state from disk if we have a canned set
    initialState1 = np.array([])
    labelGuess = 'Initial guess'

    if useGuessFile and np.size(initialState1) < 1:
        initialState1 = loadInitialGuess(guessFile, Verbose=Verbose)
        if Verbose:
            print("runSims INFO - trying to load guess from %s" \
                      % (guessFile))

    # IF we didn't already read the guess from file, generate the
    # prior state by fitting to the observed delay curve. Use numpy's
    # "around" function rather than looping through to do the
    # rounding.
    if np.size(initialState1) < 7:
        if nObs_echo > 2:
            if Verbose:
                print("runSims INFO - fitting delay curve for initial guess")
            initialState1 = simpleFit(x_echo, y_echo, y_err_echo, timeDelay, \
                                          xlabel = "Orbital Phase", \
                                          ylabel = "Delay (s)", \
                                          plotFit=True, radialVelocity = False, rocheLobe = rocheLobe)
            labelGuess = 'Initial fit'
        else:
            if Verbose:
                print("runSims INFO - fitting radial velocity curve for initial guess")
            initialState1 = simpleFit(x_radial, y_radial, y_err_radial, timeDelay, \
                                          xlabel = "Orbital Phase", \
                                          ylabel = "Velocity (m/s)", \
                                          plotFit=True, radialVelocity = True)
            

    initialState = np.around(initialState1, 3)
    nDim = np.size(initialState)

    ### plot the initial state. We do the plotting here because there
    ### are so many variables passed to the plotter otherwise that
    ### it's just inconvenient to specify them all

    print("PHASES:",x_echo_phases)
    
    plt.figure(5)
    plt.clf()
    if TEST == True:
        plt.plot(tt,yy_echo,'b-', lw=2, label='"Truth"') ### red --> blue
    tFine = np.linspace(0,1,500) ## called tt something else
    #print(initialState1)
    initialState2 = initialState1[:-1]
    #print(initialState2)
    plt.plot(tFine,timeDelay(tFine,*initialState2,radialVelocity = False,rocheLobe=rocheLobe),'--g', \
                 label=labelGuess)
    plt.errorbar(x_echo_phases, y_echo, yerr=y_err_echo, fmt="ok",c = 'c', zorder=5, \
                     ecolor='k', mec='k',mfc = 'w',ms=5, \
                     label='"Observations"')
    plt.xlabel('Orbital phase')
    plt.ylabel('Delay (s)')
    plt.title("Echo Delay Initial Guess")
    plt.legend(loc=0, fontsize=10)
    plt.savefig('fig_initialGuess_echo.png')
    
    plt.figure(6)
    plt.clf()
    if TEST == True:
        plt.plot(tt,yy_radial,'b-', lw=2, label='"Truth"') ### red --> blue
    tFine = np.linspace(0,1,500) ## called tt something else
    plt.plot(tFine,timeDelay(tFine,*initialState2,radialVelocity = True),'--g', \
                 label=labelGuess)
    plt.errorbar(x_radial_phases, y_radial, yerr=y_err_radial, fmt="ok",c = 'c', zorder=5, \
                     ecolor='k', mec='k', mfc = 'w',ms=5, \
                     label='"Observations"')
    plt.xlabel('Orbital phase')
    plt.ylabel('Velocity (m/s)')
    plt.title("Radial Velocity Initial Guess")
    plt.legend(loc=0, fontsize=10)
    plt.savefig('fig_initialGuess_radial.png')
    
    plt.figure("velocityDelayMap")
    plt.xlabel('Delay (s)')
    plt.ylabel('Velocity (m/s)')
    if TEST == True:
        plt.plot(yy_echo, yy_radial,'b-', lw=2, label='"Truth"')
    plt.plot(timeDelay(tFine,*initialState2,radialVelocity = False,rocheLobe=rocheLobe),timeDelay(tFine,*initialState2,radialVelocity = True),'--g', \
                 label=labelGuess)
    plt.savefig('fig_velocityDelayMap.png')


    plt.figure("radialCompact")
    if TEST == True:
        plt.plot(tt,yy_radialCompact,'b-', lw=2, label='"Truth"') ### red --> blue
    tFine = np.linspace(0,1,500) ## called tt something else
    #print(initialState1)
    initialState2 = initialState1[:-1]
    #print(initialState2)
    plt.plot(tFine,timeDelay(tFine,*initialState2,radialVelocity = True,rocheLobe=rocheLobe,radDonor=False),'--g', \
                 label=labelGuess)
    plt.errorbar(x_radialCompact_phases, y_radialCompact, yerr=y_err_radialCompact, fmt="ok",c = 'c', zorder=5, \
                     ecolor='k', mec='k', mfc = 'w',ms=5, \
                     label='"Observations"')
    plt.xlabel('Orbital phase')
    plt.ylabel('Velocity (m/s)')
    plt.title("Radial Velocity Initial Guess (Compact)")
    plt.legend(loc=0, fontsize=10)
    plt.savefig('fig_initialGuess_radialCompact.png')
    
    plt.show(block=True)

    

    

    if EMCEE:
        # run emcee, supplying the quantities that used to be
        # top-level
        
        if TEST == True:
            truths=[m1_true, m2_true, eccentricity_true, \
                        period_true, inclination_true, omega_true, t0_true]
        else:
            truths=[0,0,0,0,0,0,0]
        labels=[r'$m_1$', r'$m_2$', r'eccentricity', \
                    r'period (days)',r'inclination (rad)',r'$\omega$ (rad)',r'$t_0$ (s)']

        # filename-friendly labels
        lFiles = ['m1', 'm2', 'e', 'P', 'i', 'w', 't0']

        ### ndim is computed from the length of the prior state here
        ### before calling emcee. While I don't see a reason ndim
        ### would ever differ from np.size(initialState), for the moment
        ### I keep the two separate just in case there is a reason for
        ### them to differ.
        samplesMCMC = runEmcee(ndim=nDim, \
                                   initialState=initialState, \
                                   x_echo=x_echo, y_echo=y_echo, y_err_echo=y_err_echo, \
                                   x_radial=x_radial, y_radial=y_radial, y_err_radial=y_err_radial, \
                                   x_radialCompact=x_radialCompact, y_radialCompact=y_radialCompact, y_err_radialCompact=y_err_radialCompact, \
                                   nSteps=nSteps, \
                                   nwalkers=nWalkers, \
                                   varNames=labels, \
                                   priorName=priorName, \
                                   priorFile=priorFile, \
                                   useReparam=useReparam, \
                                   useOO=useOO, \
                                   obsRealTime = obsRealTime, \
                                   guessRangeFile = guessRangeFile)

        ### 2018-05-03 WIC: added some logic to return gracefully if
        ### runEmcee terminated without returning a populated samples
        ### array.

        try:
            shSamples = np.shape(samplesMCMC)
            print("runSims INFO - samples shape: ", shSamples)
        except:
            print("runSims WARN - Bad samples resulted from Emcee.")
            print("runSims WARN - Not proceeding to plot.")
            return
    
        if np.size(shSamples) < 2:
            print("runSims WARN - Samples < two dimensional!")
            print("runSims WARN - Not proceeding to plot.")
            return

        ### 2018-05-04 WIC - if we reparameterized, convert back to
        ### physical parameters
        if useReparam:
            print("runSims INFO - transforming reparam to physical")
            samplesPhys = reparamToPars(samplesMCMC.T)
            samplesMCMC = samplesPhys.T

        showCorner(samplesMCMC, labels=labels, truths=truths, \
                       x_echo=x_echo_phases, y_echo=y_echo, y_err_echo=y_err_echo, \
                       x_radial=x_radial_phases, y_radial=y_radial, y_err_radial=y_err_radial, \
                       x_radialCompact=x_radialCompact_phases, y_radialCompact=y_radialCompact, y_err_radialCompact=y_err_radialCompact, \
                       lFiles=lFiles, \
                       fewerFigs=fewerFigs, \
                       TEST=TEST, \
                       Verbose=Verbose, \
                       rocheLobe=rocheLobe)
        
        DATA_STORAGE = [samplesMCMC, labels, truths, \
                       x_echo_phases, y_echo, y_err_echo, \
                       x_radial_phases, y_radial, y_err_radial, \
                       x_radialCompact, y_radialCompact, y_err_radialCompact]
        DATA_NAMES = ["samplesMCMC","labels","truths","x_echo","y_echo","y_err_echo","x_radial_phases",\
                          "y_radial", "y_err_radial","x_radialCompact", "y_radialCompact", "y_err_radialCompact"]
        with open(dataFile, 'w') as wObj:
            sTime = time.strftime("%a, %d %b %Y %H:%M:%S %Z", time.localtime())
            wObj.write('TIME: ' + str(sTime) + '\n')

            for i in range(len(DATA_NAMES)):
                wObj.write(DATA_NAMES[i]+': ')
                for element in DATA_STORAGE[i]:
                    wObj.write('%s ' % element)
                wObj.write('\n')
            #wObj.write('samplesMCMC  %i\n' % (obsUncty_echo))
            #wObj.write('samplesMCMC  %.2f\n' % (obsUncty_echo))
            
    ### 2018-04-28 WIC - Write simulation parameters to disk. If we
    ### refactor this all into an OO approach then we'd set a separate
    ### method. For the moment, just write out here. We use separate
    ### variables for the filename and whether to write (rather than
    ### just testing on filename length) because we might want to
    ### auto-generate the output file name later.

    ### Strictly speaking we probably want to do the writing BEFORE
    ### calling emcee, so that we can note the parameters for a failed
    ### run too. However our quick-but-robust reading/writing methods
    ### are so long that it would break up the program flow. Later we
    ### can do this correctly in OO when/if we get to refactoring.

    if not writePars or len(parsFilOut) < 1:
        killPlotWindows(Verbose=Verbose)
        return

    ### Use python's "with" syntax to ensure the file handle is closed
    ### on exit.
    with open(parsFilOut, 'w') as wObj:
        sTime = time.strftime("%a, %d %b %Y %H:%M:%S %Z", time.localtime())
        wObj.write('# simulator_2.runSims ran on %s\n' % (sTime))

        # If a parameter file was passed in, and it exists, write a
        # record of this too.
        if len(parsFilUsed) > 0 and os.access(parsFilUsed, os.R_OK):
            wObj.write('# Using parameters from file %s\n' % (parsFilUsed))

        ### This is where an explicitly OO approach would really come
        ### into its own... we could just set a list of attributes and
        ### loop through them. For the moment, we'll just list the
        ### parameters one by one to aid with debugging later... The
        ### "\n" at the end of each write statement is a carriage
        ### return terminating the output line.

        # Simulated data variables
        wObj.write('### Simulated data generation ###\n')
        wObj.write('nObs_echo  %i \n' % (nObs_echo))
        wObj.write('nObs_radial  %i \n' % (nObs_radial))
        wObj.write('nObs_radialCompact  %i \n' % (nObs_radialCompact))
        wObj.write('obsSetting_echo  %s\n' % (obsSetting_echo))
        wObj.write('obsSetting_radial  %s\n' % (obsSetting_radial))
        wObj.write('obsSetting_radialCompact  %s\n' % (obsSetting_radialCompact))
        wObj.write('cannedPhases_echo  %i\n' % (cannedPhases_echo))
        wObj.write('cannedPhases_radial  %i\n' % (cannedPhases_radial))
        wObj.write('cannedPhases_radialCompact  %i\n' % (cannedPhases_radialCompact))
        wObj.write('nFineTimes  %i\n' % (nFineTimes))
        wObj.write('obsUncty_echo  %.2f\n' % (obsUncty_echo))
        wObj.write('obsUncty_radial  %.2f\n' % (obsUncty_radial))
        wObj.write('obsUncty_radialCompact  %.2f\n' % (obsUncty_radialCompact))

        # EMCEE control variables
        wObj.write('### Emcee control parameters ### \n')
        wObj.write('EMCEE  %i\n' % (EMCEE))
        wObj.write('nSteps  %i\n' % (nSteps))
        wObj.write('nWalkers %i\n' % (nWalkers))
        ### wObj.write('nDim  %i\n' % (nDim)) ### computed after initialState

        # Program control variables
        wObj.write('### Program control parameters ### \n')
        wObj.write('preserveOldFigures  %i\n' % (preserveOldFigures) )
        wObj.write('fewerFigs  %i\n' % (fewerFigs))
        wObj.write('parsFilOut  %s\n' % (parsFilOut))
        wObj.write('Verbose  %i\n' % (Verbose))
        wObj.write('rocheLobe  %i\n' % (rocheLobe))

        # Prior information
        wObj.write("### Prior function choices\n")
        wObj.write('useOO  %i\n' % (useOO))
        if useOO:
            if len(priorFile) > 2 and os.access(priorFile, os.R_OK):
                wObj.write("priorFile  %s\n" % (priorFile))
            else:
                wObj.write("priorName  %s\n" % (priorName))

        # Parameters used
        wObj.write('### True model parameters ### \n')
        wObj.write('m1_true  %.3f\n' % (m1_true))
        wObj.write('m2_true  %.3f\n' % (m2_true))
        wObj.write('eccentricity_true  %.3f\n' % (eccentricity_true))
        wObj.write('period_true  %.3f\n' % (period_true))
        wObj.write('iDeg_true  %.3f\n' % (iDeg_true))
        wObj.write('omegaDeg_true  %.3f\n' % (omegaDeg_true))
        wObj.write('t0_true  %.3f\n' % (t0_true))

        # Canned initial guess?
        wObj.write("### Reading initial guess from file?\n")
        wObj.write('useGuessFile  %i\n' % (useGuessFile))
        wObj.write('guessFile %s\n' % (guessFile))

        # Using our reparameterization?
        wObj.write("### Using reparameterization?\n")
        wObj.write("useReparam  %i\n" % (useReparam))

    if closeFigsAfter:
        killPlotWindows(Verbose=Verbose)

def loadInitialGuess(filGuess='', Verbose=True):

    """Loads initial-guess parameters from file"""

    retPars = np.array([])

    if len(filGuess) < 1:
        return retPars

    if not os.access(filGuess, os.R_OK):
        if Verbose:
            print("loadInitialGuess WARN - cannot read initial-guess file %s" \
                      % (filGuess))
        return retPars

    retPars = np.genfromtxt(filGuess)
    retPars[4] = np.radians(retPars[4])
    retPars[5] = np.radians(retPars[5])
    return retPars

def killPlotWindows(Verbose=True):

    """Utility to close all open plot windows"""

    if Verbose:
        print("killPlotWindows INFO - Closing plot windows.")
        print("killPlotWindows INFO - See fig*png for plots.")
    plt.close('all')

def obsPhases(nObs=4, canned=0, setting='uniform', obsRealTime=True, period_in=0.787, t0=0):

    """Sets up observation phases. 
    If "canned" is set, this returns a pre-built set of observation
    phases (useful if we want to use the exact same phases when
    testing)."""
    period = period_in * constants.day
    phiObs = np.random.uniform(size=nObs)
    if setting == 'uniform':
        phiObs = np.linspace(0.01, 1., num=nObs, endpoint=False)
    if setting == 'early':
        phiObs = np.random.uniform(size=nObs)
        for j in range(len(phiObs)):
            if phiObs[j] > .5:
                phiObs[j] += (1-phiObs[j])*.6
            else:
                phiObs[j] += -phiObs[j]*.6
    if setting == 'late':
        phiObs = .3+.4*np.random.uniform(size=nObs)
    #pre-set observations
    if canned == 1:
        phiObs = np.array([.3,.4,.5,.6,.7])
    if canned == 2:
        phiObs = np.array([.3,.7])
    if canned == 3:
        phiObs = np.array([.25,.5,.75])
    if canned == 4:
        phiObs = np.array([0.01   , 0.03475, 0.0595 , 0.08425, 0.109  , 0.13375, 0.1585 , \
       0.18325, 0.208  , 0.23275, 0.2575 , 0.28225, 0.307  , 0.33175, \
       0.3565 , 0.38125, 0.406  , 0.43075, 0.4555 , 0.48025, 0.505  , \
       0.52975, 0.5545 , 0.57925, 0.604  , 0.62875, 0.6535 , 0.67825, \
       0.703  , 0.72775, 0.7525 , 0.77725, 0.802  , 0.82675, 0.8515 , \
       0.87625, 0.901  , 0.92575, 0.9505 , 0.97525])
    if canned == 5:
        phiObs = np.linspace(0,1,20)

    if obsRealTime == True:
        phiObs = phiObs*period + t0
    return phiObs

def runEmcee(ndim = 7, \
                 nwalkers=100, nSteps=350, \
                 initialState=np.array([]), \
                 x_echo=np.array([]), y_echo=np.array([]), y_err_echo=np.array([]), \
                 x_radial=np.array([]), y_radial=np.array([]), y_err_radial=np.array([]), \
                 x_radialCompact=np.array([]), y_radialCompact=np.array([]), y_err_radialCompact=np.array([]), \
                 EMCEE=True, \
                 varNames=[], \
                 priorName='binaryBounded', \
                 priorFile='', \
                 guessRangeFile='', \
                 useOO=True, \
                 useReparam=True, \
                 showPlot=True, \
                 Verbose=True, \
                 obsRealTime=True, \
                 rocheLobe=True):
    
    """Refactored the MCMC into a separate method"""

    # WIC - this turns out to be very easy, since this was already all
    # bundled into a conditional. I've left variable EMCEE in as a
    # legacy in case you find it useful. Otherwise we can just junk
    # it.

    # bad return initialized
    samplesNone = np.array([])

    # legacy argument
    if not EMCEE:
        return samplesNone
    
    # Defensive programming again... check the inputs were set OK. We
    # probably gould be slick and turn this into a loop through
    # conditions, but for the moment I don't mind just including the
    # separate conditions.
    if np.size(initialState) < 1:
        if Verbose:
            print("runEmcee WARN - initial state not given. Returning.")
        return samplesNone

    if np.size(x_echo) < 1:
        if Verbose:
            print("runEmcee WARN - echo phase array has zero size")

    if np.size(y_echo) < 1:
        if Verbose:
            print("runEmcee WARN - echo delay array has zero size")

    if np.size(y_err_echo) < 1:
        if Verbose:
            print("runEmcee WARN - echo uncertainty array has zero size")
        
    #pos = initialState + initialVariability

    if Verbose:
        print("runEmcee INFO: check point 1")

    ### 2018-04-28 WIC - are the args() our method for getting
    ### arguments through to lnPrior? Test by setting up a
    ### hyperparameters array that looks like Thomas' bounds array
    ### hardcoded into lnprior.
    #pHyper = np.array([[0., 1.,], [0., 1]])

    ### Assemble the bounds for the flat prior out of what used to be
    ### in lnPrior:
    priorFunc = priorFlat
    boundsLo = np.array( [0., 0., .001, 0., -np.pi, -np.pi, -np.inf] )
    boundsHi = np.array( [50., 50., .999, 30.45, np.pi, np.pi, np.inf] )
    pHyper = np.vstack(( boundsLo, boundsHi ))

    ### 2018-04-29 UPDATE - set up a prior object and pass that in to
    ### lnprob as an argument. The argument-passing syntax here is
    ### still just a bit awkward, since we're passing the same names
    ### all the way through to the sampler. The PRIORS object reads in
    ### the prior file in preference to the string supplied on the
    ### command line.
    PRIORS = echoPriors.Priors(namePrior=priorName, \
                                   filParams=priorFile)
    PRIORS.wrapPlotPriors()
    if Verbose:
        ### Report the arguments to runEmcee
        print("runEmcee INFO - using OO priors.")
        print("runEmcee INFO - arguments: namePrior=%s, filParams=%s" \
                % (priorName, priorFile))

    ### 2018-05-03 WIC - evaluate the initial guess using the
    ### prior object. If it's not finite, warn and return.
    ### We do this to the initial state expressed as physical
    ### parameters

    #initialState = [2, 2, .7, 16, 1, .2]
    #pos = [initialState + .001*np.random.randn(ndim) for i in range(nwalkers)]
    boundsMeansSigmas = []
    pos = [np.zeros(ndim) for i in range(nwalkers)]
    posPhysical = [np.zeros(ndim) for i in range(nwalkers)]
    with open(guessRangeFile, 'r') as rObj:
        for sLine in rObj:
            if sLine.find('#') > -1:
                continue
            if sLine.find('NAME') > -1:
                continue
            vLine = sLine.strip().split()
            boundsMeansSigmas.append(vLine)
    #boundsMeansSigmas = boundsMeansSigmas[:-3]
    print(boundsMeansSigmas)
    for i in range(len(boundsMeansSigmas)):
        if boundsMeansSigmas[i][0] == 'gaussianOne':
            for j in range(nwalkers):
                pos[j][i] = np.random.normal(float(boundsMeansSigmas[i][1]),float(boundsMeansSigmas[i][2]))
        if boundsMeansSigmas[i][0] == 'binaryBoundedOne':
            for j in range(nwalkers):
                pos[j][i] = RAND(float(boundsMeansSigmas[i][1]),float(boundsMeansSigmas[i][2]))
        if i == 4 or i == 5:
            for j in range(nwalkers):
                pos[j][i] = np.radians(pos[j][i])
            
    for i in range(ndim):
        plt.figure()
        A = np.zeros([nwalkers,ndim])
        for j in range(nwalkers):
            A[j] = pos[j]
        plt.hist(A[:,i],bins = 15)
        plt.title(varNames[i])
        #print("pos(" + str(i) + "): " + str(pos[:][i]))
        plt.savefig("InitializationState_" + str(i) + ".png")
    
    posPhysical = pos
    if useReparam:
        for i in range(nwalkers):
            pos[i] = parsToReparam(posPhysical[i])
    #pos = [initialState + .1*np.random.randn(ndim) for i in range(nwalkers)] 
    ## ndim, nwalkers = 6, 100
    #pos = [2.1+.0001*np.random.randn(ndim) for i in range(nwalkers)]
    #initialState = [m1 m2 eccentricity]
    
    ### 2018-05-04 WIC - whatever parameterization we use for the
    ### sampler, we'll want to keep the initial guess in physical
    ### parameters.
    initialStatePhysical = np.copy(initialState)
    if useReparam:
        initialState = parsToReparam(initialStatePhysical, degrees=False)
    #print("pos: ")
    #print(pos)
    
    print("initial state - ",initialStatePhysical)
    lp_initial = PRIORS.evaluate(initialStatePhysical)
    print("lp_initial = ",lp_initial)
    if not np.isfinite(lp_initial):
        print("runEmcee WARN - initial state has bad prior probability.")
        print("runEmcee WARN - halting.")
        return None
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnProbObj, \
                                        args=(x_echo,y_echo,y_err_echo,x_radial,y_radial,y_err_radial,x_radialCompact,y_radialCompact,y_err_radialCompact, PRIORS, \
                                                     useReparam,obsRealTime,rocheLobe) )
    if Verbose:
        print("runEmcee INFO: check point 2")
    #sampler.reset()
    Pos,Prob,State = sampler.run_mcmc(pos, nSteps)
    if Verbose:
        print("runEmcee INFO: check point 3")

    # discard the initial 250 steps so that the walkers are well distributed.
    samples = sampler.chain[:, 250:, :].reshape((-1, ndim))

    ### 2018-04-28 WIC - since all the plotting steps below refer to
    ### the samples array in some way, and also need the truth
    ### paramters, I prefer to refactor them into a separate
    ### method. So, we return the samples from this method
    ### ("runEmcee") and pass them into a separate method that uses
    ### the samples.
    
    if not showPlot:
        return samples

    ### Moved the plot that calls the sampler up here.
    for i in range(np.shape(samples)[-1]-1):

        ### 2018-04-28 WIC - condition trap for zero dynamic range
        yVal = sampler.chain[:,:,i].T
        if np.abs(np.max(yVal)-np.min(yVal)) < 1e-5:
            continue
        A=plt.figure(10 + i)  ### 2018-04-28 WIC - constrain the figure number
        plt.clf()
        plt.title(varNames[i])
        res=plt.plot(reparamToPars(sampler.chain[:,:,i].T), '-', color='k', alpha=0.15)
        A.savefig("walkerPath_" + str(i) + ".png")

    return samples

def showCorner(samples=np.array([]), \
               truths=[], labels=[], \
               x_echo=np.array([]), y_echo=np.array([]), y_err_echo=np.array([]), \
               x_radial=np.array([]), y_radial=np.array([]), y_err_radial=np.array([]), \
               x_radialCompact=np.array([]), y_radialCompact=np.array([]), y_err_radialCompact=np.array([]), \
               quantiles=[0.16, 0.5, 0.84], \
               nFine=100, \
               figCorner='fig_triangle.png', \
               figSamplesEcho='fig_sampleDeltas_echo.png', \
               figSamplesRadial='fig_sampleVels_radial.png', \
               figSamplesRadialCompact='fig_sampleVels_radialCompact.png', \
               lFiles=[], \
               fewerFigs=True, \
               eachPairOnce=True, \
               Verbose=True, \
               parsResultsFile = "MCMC_results.txt", \
               TEST=True, \
               rocheLobe=True,\
               loaded = False):

    

    if loaded == False:
        saveRunResults(samples,truths,labels,x_echo,y_echo,\
                       y_err_echo,x_radial,y_radial,\
                       y_err_radial,TEST,lFiles)
        print("Saving run results")


    """Does the corner-plot with the samples"""

    ### new argument eachPairOnce: only plot hist2d for each pair once
    ### (e.g. only plot (m2, e) not BOTH (m2,e) and (e,m2)
    
    # exit gracefully if no samples were provided
    if np.size(samples) < 1:
        return
    
    ### fig = corner.corner(samples, labels=['$m_1$', '$m_2$', 'eccentricity','period (days)','inclination (rad)','$\omega$ (rad)'],
###                      truths=[m1_true, m2_true, eccentricity_true, period_true, inclination_true, omega_true],
###                        quantiles=[0.16, 0.5, 0.84],
###                        show_titles=True, title_kwargs={"fontsize": 12})

    #fig = plt.figure(5)
    if loaded == False:
        if TEST == True:
            fig = corner.corner(samples, labels=labels, truths=truths, \
                                quantiles=quantiles, show_titles=True, \
                                #fig=fig, \
                                title_kwargs={"fontsize": 12})
        else:
            fig = corner.corner(samples, labels=labels, \
                                quantiles=quantiles, show_titles=True, \
                                #fig=fig, \
                                title_kwargs={"fontsize": 12})
        fig.savefig(figCorner)


    ### 2018-04-28 WIC - at this point we can unpack the truth values
    ### from the "truths" array that was passed to this method. We may
    ### want to think about some more elegant ways of doing this
    ### rather than passing lots of arguments back and forth (I
    ### suspect we're going to end up with an object-oriented
    ### refactoring), but that can wait till after paper
    ### submission. For the moment, let's unpack the values here from
    ### "truths:"
    m1_true, m2_true, eccentricity_true, \
        period_true, inclination_true, omega_true, t0_true = truths

    # generate the finer grid of phases
    x1 = np.linspace(0,1,nFine)
    
    #axes = np.array(fig.axes).reshape((ndim, ndim))
    #ax.axvline(initialState[i], color="g")

    if Verbose:
        print("showCorner INFO - printing!!!!")

    # Now set up the second figure
    plt.figure(2)
    plt.clf()
    plt.xlabel("Orbital Phase")
    plt.ylabel("Time Delay (s)")
    #plt.title("Delay Curve")
    for m1F,m2F,eF,pF,iF,wF,t0F in \
            samples[np.random.randint(len(samples), size=100)]:
        yy = timeDelay(x1,m1F,m2F,eF,pF,iF,wF,radialVelocity=False,rocheLobe=rocheLobe)
        plt.plot(x1,yy,color = "0.25",alpha = 0.1, zorder=1)
        #plt.plot(xl, m*xl**2, color="k", alpha=0.1)
        #print(M2)
    #samples[:, 0] = np.exp(samples[:, 0])

    # 2018-04-28 WIC actually I think it might also work to just call
    # this as timeDelay(x1, *truths)... perhaps for later testing.
    if TEST == True:
        Y = timeDelay(x1, m1_true, m2_true, eccentricity_true, \
                          period_true,inclination_true,omega_true,radialVelocity=False,rocheLobe=rocheLobe)

        ### changed plot color to blue for red/green colorblind viewers
        plt.plot(x1,Y,color = "b",alpha = 0.8, lw=2, zorder=2)
    #plt.scatter(x,y,c = 'green',marker = "o")
    plt.errorbar(x_echo, y_echo, yerr=y_err_echo, fmt='ok', c = 'c', zorder=5, \
                 ecolor='k', mec='k',mfc = 'w', ms=5)
    #plt.figure(3)

    ### 2018-04-30 WIC - I like this figure so much that I want to
    ### save it
    plt.savefig(figSamplesEcho)
    ############################# Does another plot for the radial velocity curves
    plt.figure(3)
    plt.clf()
    plt.xlabel("Orbital Phase")
    plt.ylabel("Velocity (km/s)")
    #plt.title("Delay Curve")
    for m1F,m2F,eF,pF,iF,wF,t0F in \
            samples[np.random.randint(len(samples), size=100)]:

        yy = timeDelay(x1,m1F,m2F,eF,pF,iF,wF,radialVelocity = True)
        plt.plot(x1,yy/1000,color = "0.25",alpha = 0.1, zorder=1)
        #plt.plot(xl, m*xl**2, color="k", alpha=0.1)
        #print(M2)
    #samples[:, 0] = np.exp(samples[:, 0])

    # 2018-04-28 WIC actually I think it might also work to just call
    # this as timeDelay(x1, *truths)... perhaps for later testing.
    if TEST == True:
        Y = timeDelay(x1, m1_true, m2_true, eccentricity_true, \
                          period_true,inclination_true,omega_true,radialVelocity = True)

        ### changed plot color to blue for red/green colorblind viewers
        plt.plot(x1,Y/1000,color = "b",alpha = 0.8, lw=2, zorder=2)
    #plt.scatter(x,y,c = 'green',marker = "o")
    plt.errorbar(x_radial, y_radial/1000, yerr=y_err_radial/1000, fmt='ok', c = 'c', zorder=5, \
                 ecolor='k', mec='k',mfc = 'w', ms=5)
    #plt.figure(3)

    ### 2018-04-30 WIC - I like this figure so much that I want to
    ### save it
    plt.savefig(figSamplesRadial)

    plt.figure("figSamplesRadialCompact")
    plt.clf()
    plt.xlabel("Orbital Phase")
    plt.ylabel("Velocity (m/s)")
    #plt.title("Delay Curve")
    for m1F,m2F,eF,pF,iF,wF,t0F in \
            samples[np.random.randint(len(samples), size=100)]:

        yy = timeDelay(x1,m1F,m2F,eF,pF,iF,wF,radialVelocity = True,radDonor = False)
        plt.plot(x1,yy,color = "0.25",alpha = 0.1, zorder=1)
        #plt.plot(xl, m*xl**2, color="k", alpha=0.1)
        #print(M2)
    #samples[:, 0] = np.exp(samples[:, 0])

    # 2018-04-28 WIC actually I think it might also work to just call
    # this as timeDelay(x1, *truths)... perhaps for later testing.
    if TEST == True:
        Y = timeDelay(x1, m1_true, m2_true, eccentricity_true, \
                          period_true,inclination_true,omega_true,radialVelocity = True,radDonor = False)

        ### changed plot color to blue for red/green colorblind viewers
        plt.plot(x1,Y,color = "b",alpha = 0.8, lw=2, zorder=2)
    #plt.scatter(x,y,c = 'green',marker = "o")
    plt.errorbar(x_radialCompact, y_radialCompact, yerr=y_err_radialCompact, fmt='ok', c = 'c', zorder=5, \
                 ecolor='k', mec='k',mfc = 'w', ms=5)
    #plt.figure(3)

    ### 2018-04-30 WIC - I like this figure so much that I want to
    ### save it
    plt.savefig(figSamplesRadialCompact)
    
    ### 2018-04-28 WIC - I suspect this can be taken from the "labels"
    ### array... isn't it in the same order?
    ### varNames = [r'$m_1$',r'$m_2$',r'eccentricity',r'period (days)',r'inclination (rad)',r'$\omega$ (rad)']
    varNames = labels[:]

    m1_mcmc, m2_mcmc, e_mcmc, p_mcmc, i_mcmc, w_mcmc, t0_mcmc = \
        map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), \
                zip(*np.percentile(samples, [16, 50, 84], \
                                       axis=0)))
    with open(parsResultsFile, 'w') as wObj:
        sTime = time.strftime("%a, %d %b %Y %H:%M:%S %Z", time.localtime())
        wObj.write('# simulator_2.runSims ran on %s\n' % (sTime))
        wObj.write('m_1 = ' + (str(m1_mcmc))+ '\n')
        wObj.write('m_2 = ' + (str(m2_mcmc))+ '\n')
        wObj.write('e = ' + (str(e_mcmc))+ '\n')
        wObj.write('p = ' + (str(p_mcmc))+ '\n')
        wObj.write('i = ' + (str(i_mcmc))+ '\n')
        wObj.write('w = ' + (str(w_mcmc))+ '\n')
        wObj.write('t0 = ' + (str(t0_mcmc))+ '\n')
    

    if Verbose:
        print("showCorner INFO:", list(m1_mcmc),list(m2_mcmc),list(e_mcmc),list(p_mcmc),list(i_mcmc),list(w_mcmc),list(t0_mcmc))

    ### print("showCorner INFO -- ", np.shape(samples))
    ### ndim = np.shape(samples)[-1]
    ### for i in range(ndim-1):
    ###    plt.figure(10 + i)  ### 2018-04-28 WIC - constrain the figure number
    ###    plt.clf()
    ###    plt.title(varNames[i])
    ###    res=plt.plot(sampler.chain[:,:,i].T, '-', color='k', alpha=0.15)
    #np.random.seed(19680801)
    #Posterier distribution of m1+m2
    #plt.figure(900)
    #plt.xlabel("Mass ($M_\odot$)")
    #plt.ylabel("Probability Density")
    #plt.title("Posterier Distribution")
    #data = samples[:,0]+samples[:,1]
    #plt.hist(data,bins = 80,normed = True)

    plt.figure(901)
    plt.xlabel(r"$m_2$ ($M_\odot$)")
    plt.ylabel("Probability Density")
    plt.xlim([0,3])
    data = samples[:,1]
    plt.hist(data,bins = 80,normed = True)
    if TEST == True:
        plt.axvline(x=m2_true,c = 'black')

    plt.figure(902)
    plt.clf()
    plt.xlabel("Inclination (Deg.)")
    plt.ylabel("Probability Density")
    data = np.degrees(samples[:,4])
    plt.hist(data,bins = 80,normed = True)
    if TEST == True:
        plt.axvline(x=np.degrees(inclination_true),c = 'black')
    #n, bins = np.histogram(data, 100)
    #left = np.array(bins[:-1])
    #right = np.array(bins[1:])
    #bottom = np.zeros(len(left))
    #top = bottom + n
    #XY = np.array([[left, left, right, right], [bottom, top, top, bottom]]).T
    #barpath = path.Path.make_compound_path_from_polys(XY)
    #patch = patches.PathPatch(barpath)
    #ax.add_patch(patch)
    #ax.set_xlim(left[0], right[-1])
    #ax.set_ylim(bottom.min(), top.max())
    if loaded == False:
        plt.savefig('fig_massPosterior.png')
    
    #2D projection of the posterier with 1 of the dimensions set to m1+m2
    #for i in range(4):
    #    plt.figure(50+i)
    #    plt.clf()
    #    plt.xlabel("Mass")     
    #    plt.ylabel(varNames[i+2])     
    #    corner.hist2d(data,samples[:,i+2],bins = 50,levels=[.7,.95])
    #    plt.savefig('fig_mass_' + lFiles[i+2] + '2D_Posterior.png')

    plt.figure(60)
    plt.clf()
    plt.xlabel("Inclination (Deg.)")
    plt.ylabel(r"$m_2$ ($M_\odot$)")
    corner.hist2d(np.degrees(samples[:,4]),samples[:,1],bins = 50,\
                      levels=[.68,.95],plot_datapoints = False,smooth = True)
    plt.xlim([0,90])
    plt.ylim([0,2.8])
    if TEST == True:
        plt.axvline(x=np.degrees(inclination_true),c = 'black')
        plt.axhline(y=m2_true,c = 'black')
    if loaded == False:
        plt.savefig('fig_cornerHist_i.png')


    
    #plt.figure(61)
    #plt.clf()
    #plt.xlabel(r"$\omega$")
    #plt.ylabel(varNames[2])
    #corner.hist2d(np.degrees(samples[:,5]),samples[:,2],bins = 50, \
    #                  levels=[.68,.95],plot_datapoints = False,smooth = True)
    #plt.savefig('fig_cornerHist_omega.png')

    ### 2018-04-28 WIC - OK here's where we loop through all parameter
    ### combinations... 

    ### 2018-04-28 WIC - I think the array "true_parameters" has the
    ### same information as the "truths" array, no??
        
    ### 2018-04-28 WIC - allow argument that puts fewer figures on the
    ### screen
    if fewerFigs:
        return

    # labels for filenames
    if len(lFiles) < 1:
        lFiles = labels[:]

    if loaded == False:
        for q in range(len(varNames)): ### 2018-04-28 WIC - scale with variables
            fig, ax = plt.subplots()
            plt.xlabel(varNames[q])
            plt.ylabel("Probability Density")
            plt.title("Posterier Distribution")
            #np.random.seed(19680801)
            data = samples[:,q]
            truthsTemp = truths
            truthsTemp[4] = np.degrees(truthsTemp[4])
            truthsTemp[5] = np.degrees(truthsTemp[5])
            if q == 4 or q == 5:
                data = np.degrees(data)
            plt.hist(data,bins = 80,normed = True)
            if TEST == True:
                plt.axvline(x=truthsTemp[q],c = 'black')
     
            # save the figure to disk
            figNam = 'fig_post_%s.png' % (lFiles[q])
            plt.savefig(figNam)

            #n, bins = np.histogram(data, 50,)
            #left = np.array(bins[:-1])
            #right = np.array(bins[1:])
            #bottom = np.zeros(len(left))
            #top = bottom + n
            #XY = np.array([[left, left, right, right], [bottom, top, top, bottom]]).T
            #barpath = path.Path.make_compound_path_from_polys(XY)
            #patch = patches.PathPatch(barpath)
            #ax.add_patch(patch)
            #ax.set_xlim(left[0], right[-1])
            #ax.set_ylim(bottom.min(), top.max())
            r = 0
            for r in range(len(varNames)):

                ### 2018-04-28 WIC - whatever we're choosing, plotting x,x
                ### is indeed uninformative.
                if q == r:
                    continue
                
                ### 2018-04-28 WIC - actually I think plotting only if q >
                ### r avoids double-plotting.
                if q > r and eachPairOnce:
                    continue
                
                plt.figure(10*q+r+100)
                plt.clf()
                plt.xlabel(varNames[q])
                plt.ylabel(varNames[r])
                corner.hist2d(samples[:,q],samples[:,r],bins = 50,levels=[.69,.95],plot_datapoints = False,smooth = True)
            if TEST == True:
                plt.axvline(x=truths[q],c = 'black')
                plt.axhline(y=truths[r],c = 'black')
                plt.scatter(truths[q],truths[r],marker = 's')
    
            ### 2018-04-28 WIC - save the figure to disk
            figName = 'fig_twoPlot_%s_v_%s.png' % (lFiles[r], lFiles[q])
            plt.savefig(figName)
            
    if Verbose == True:
        plt.show(block=False)
    #samples[:, 0] = np.exp(samples[:, 0])
    


### 2018-04-29 WIC: test routines for the class Priors() follow.
def TestPriors(funcname='', parsName=''):

    """Test routine for the Priors() class"""
    
    PR = echoPriors.Priors(namePrior=funcname, filParams=parsName)

    # try sampling from the prior
    if PR.namePrior.find('ixed') > -1:
        guess = PR.sampleMixedPrior(size=1)
        print(guess)
    
    # try setting up the plot ranges
    PR.wrapPlotPriors()
    
    
def TestDefaultGuess(filPrior='parsPrior.txt', \
                         filOut='tmp_PriorSample.txt'):

    """Draws a sample from the prior described by parsPrior.txt"""

    PRI =  echoPriors.Priors(filParams=filPrior)
    if PRI.namePrior.find('ixed') < 0:
        return

    guess = PRI.sampleMixedPrior(size=1)
    sCommen = 'Sample from prior file %s' % (filPrior)
    np.savetxt(filOut, guess, header=sCommen, fmt='%.3f')

def RAND(a,b):
    return a+(b-a)*np.random.random()
def genRadVsignal(K = 50, N = 20):
    phi = np.linspace(0,1,N)
    V = K*np.sin(2*np.pi*phi)
    return phi, V




def hypoRun():
    loadParsAndRun()



def quick(N = 10, m1_in = 1.4, m2_in = .7, period_in = .787, \
                    inclination = np.radians(44.), omega = np.radians(90.), eccentricity = 0, \
                    binNumber = 100, Q = 120, diskShieldingAngle = np.radians(0), intKind = 'cubic', radialVelocity = False,u=.6, plot = True):
    tt = np.linspace(0,1,5000)
    T1 = timeDelay(tt, m1_in=m1_in,m2_in=m2_in,period_in=period_in,inclination=inclination, \
                       omega=omega,eccentricity=eccentricity, \
                       radialVelocity=radialVelocity,rocheLobe=False, \
                       ellipticalCORR = True,pseudo3D = False)
    T2 = timeDelay(tt, m1_in=m1_in,m2_in=m2_in,period_in=period_in,inclination=inclination, \
                       omega=omega,eccentricity=eccentricity,radialVelocity=radialVelocity,rocheLobe=True,innerLagrange=False,pseudo3D = False)
        
    plt.xlabel("Phase")
    plt.ylabel("Velocity (km/s)")
    #plt.scatter(phi,T,marker = 's',c='black', s = 10)
    #plt.plot(tt,yy, 'g-',label='3D model')
    plt.plot(tt,np.array(T1)/1000, 'b:',label='CM model')
    #plt.plot(ttt,TP, 'k-.',label='Pseudo 3D model')
    q = m2_in/m1_in
    N0 = .886
    N1 = -1.132
    N2 = 1.523
    N3 = -1.892
    N4 = .867
    
    T4 = np.array(T1)*(N0+N1*q+N2*q**2+N3*q**3+N4*q**4)/1000
    
    plt.plot(tt,np.array(T4), 'k-',label='Munez Darias')
    
    plt.plot(tt,np.array(T2)/1000, 'r--',label='SP model')

    plt.legend()

def saveRunResults(samples,truths,labels,x_echo,y_echo,\
                   y_err_echo,x_radial,y_radial,y_err_radial,\
                   TEST,lFiles,File = 'runResults.pickle'):
    DOut = {'samples':samples, 'truths':truths, 'labels':labels,\
            'x_echo':x_echo, 'y_echo':y_echo, 'y_err_echo':y_err_echo,\
            'x_radial':x_radial, 'y_radial': y_radial,\
            'y_err_radial':y_err_radial, 'TEST':TEST, 'lFiles':lFiles}
    pickle.dump(DOut, open(File, 'wb'))

def loadRunResults(plot = True,quickfigs = False):
    #Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    #filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
    #print(filename)
    filename = 'runResults.pickle'
    file = open(filename,'rb')
    data = pickle.load(file)
    #print(data['TEST'])
    if quickfigs == True:
        A = data['samples']
        corner.hist2d(A[:,0],A[:,1],levels=[.68,.95],bins=50,smooth=True,plot_datapoints=False)
        tt = np.linspace(0,max(A[:,0]),3)
        plt.plot(tt,.20*tt,'g--')
        plt.plot(tt,.34*tt,'g--')
        plt.xlabel(r"$m_1$ ($M_\odot$)")
        plt.ylabel(r"$m_2$ ($M_\odot$)")
        plt.text(1.19, .5, "q = 0.34", fontsize=18)
        plt.text(1.65756, .303339, "q = 0.21", fontsize=18)
        plt.show()
        
    if plot == True:
        showCorner(samples=data['samples'],\
                   truths=data['truths'],\
                   labels=data['labels'],\
                   x_echo=data['x_echo'],\
                   y_echo=data['y_echo'],\
                   y_err_echo=data['y_err_echo'],\
                   x_radial=data['x_radial'],\
                   y_radial=data['y_radial'],\
                   y_err_radial=data['y_err_radial'],\
                   TEST=data['TEST'],\
                   lFiles=data['lFiles'],\
                   loaded = True,\
                   fewerFigs = True)
    if plot == False:
        return data['samples']
def makePlotCorner(supress = True):
    A = loadRunResults(plot = False)
    A[:,3] = A[:,3]*24
    A[:,4] = np.degrees(A[:,4])
    labelsLoc = np.array([r'$m_1$ ($M_\odot$)', r'$m_2$ ($M_\odot$)', \
                    r'P (Hr)',r'i (Deg.)'])
    data = np.array([A[:,0],A[:,1],A[:,3],A[:,4]])
    data = np.transpose(data)
    figure = corner.corner(data,labels = labelsLoc,levels = [.68,.95], \
                           show_titles=True, title_kwargs={"fontsize": 10}, \
                           truths = [1.4,.7,.787*24,44],plot_datapoints=False,smooth=True)
    figure.savefig('corner.png')
    if supress == False:
        for j in range(5):
            for k in range(j+1):
                if (j != 2 and k!= 2):
                    if j != k:
                        plt.figure(str(j) + ',' + str(k))
                        corner.hist2d(A[:,j],A[:,k],bins=50,levels=[.68,.95],plot_datapoints=False,smooth = True)
                        plt.xlabel(labelsLoc[j])
                        plt.ylabel(labelsLoc[k])
                    else:
                        plt.figure()
                        plt.hist(A[:,j],bins=80,normed=True)
                        plt.ylabel("Probability Density")
                        plt.xlabel(labelsLoc[j])


def lQ(theta,q1=.21,q2=.34):
    #print(theta)
    q = float(theta[1]/theta[0])
    #print(q)
    #print(q)
    if q > q1 and q < q2:
        out = -np.log(q2-q1)
    else:
        out = -np.inf
        #print('else')
    #print(out)
    return out
                
    
    
    
    
    
    
    
