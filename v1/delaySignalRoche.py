#3D model of donor reprocessing of dirac_delta signal from the compact object

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib import cm
import simulator_2
from simulator_2 import *
from scipy import interpolate as scpint
import time
import sys
import math
from scipy.interpolate import RegularGridInterpolator as RGI

#from astropy.table import Table
import pickle

plt.rc('font', family='serif')
matplotlib.rcParams.update({'font.size': 18})


# ~ try:
    # ~ funcARY = np.load('Func_Correction.npy')
    # ~ FUNC_CORRECTION = funcARY.item()
# ~ except:
    # ~ print("Pseudo 3D correction is missing. Run: initialPseudoConfig() and restart.")
    # ~ #Attempts to load the corrections for the pseudo 3D model,
    # ~ #if the correction file does not exist, the user must recalibrate the pseudo 3D model
    # ~ #This takes ~1 hour on my system



def genRadialVelocityMap(phase, \
                  m1_in=1.4, m2_in=0.7, period_in=.787, eccentricity = 0,\
                  inclination=np.radians(44.), omega=np.radians(90.), cmap = cm.jet, plot = True, Q = 25, RETURN = False):

    m1 = m1_in * constants.M_Sun
    m2 = m2_in * constants.M_Sun
    q = m2_in/m1_in
    period = period_in * constants.day
    omega_angV = 2*np.pi/period
    G = constants.G
    a = (G*(m1+m2)*period**2/(4*np.pi**2))**(1./3.)
    r2 = m1_in/(m1_in+m2_in)*a
    rcm = m1_in/(m1_in+m2_in)*a
    radiusDonor = a*(.49*q**(.667))/(.6*q**(.667)+np.log(1+q**(.333)))
    #print(radiusDonor)
    separation = simulator_2.timeDelay([phase],m1_in=m1_in,m2_in=m2_in,eccentricity=eccentricity,period_in=period_in,inclination=inclination,omega=omega,separationReturn=True)
    separation = separation[0]
    theta_ORBIT = 2*np.pi*phase #Only Valid For Circular Orbits!!!!
    (n, m) = (Q, Q)
    # Meshing a unit sphere according to n, m 
    #theta = np.linspace(0, 2*np.pi, num=n, endpoint=False)
    #phi = np.linspace(np.pi * (-0.5 + 1./(m+1)), np.pi*0.5, num=m, endpoint=False)
    #phi = np.linspace(0, np.pi*0.5, num=m, endpoint=False)

    
    L2 = a*(.5+.227*np.log(q)/np.log(10))
    #print("L2 = " + str(L2))
    rr = m1/(m1+m2)*a-L2
    Uconst = 0
    
    def potentialFunc(p2,theta2):
        #phiU = -G*m1/(np.sqrt((m2/(m1+m2)*S+r*np.cos(theta))**2+(r*np.sin(theta))**2))- \
         #     G*m2/(np.sqrt((m1/(m1+m2)*S-r*np.cos(theta))**2+(r*np.sin(theta))**2))- \
         #     .5*omega**2*r**2
        r2 = m1_in/(m1_in+m2_in)*a
        phiU = -G*m1/np.sqrt(a**2+p2**2+2*a*p2*np.cos(theta2))-G*m2/p2 \
                   -.5*omega_angV**2*(r2**2+p2**2+2*r2*p2*np.cos(theta2))
        return phiU-Uconst
    Uconst = potentialFunc(L2,np.pi)
    #print(Uconst)
    
    def potentialFuncINV(psi):
        psi = np.add(psi,np.pi)
        r = np.zeros(len(psi))
        a1 = L2/1000
        b1 = L2
        for i in range(len(psi)):
            r[i] = opt.brentq(potentialFunc,a1,b1,args=(psi[i]),maxiter=100)
        return r
    # Meshing a unit sphere according to n, m 
    theta = np.linspace(0, 2*np.pi, num=n, endpoint=True)
    #phi = np.linspace(np.pi * (-0.5 + 1./(m+1)), np.pi*0.5, num=m, endpoint=False)
    if plot == False:
        phi = np.linspace(0, np.pi*0.5, num=m, endpoint=True)
    if plot == True:
        phi = np.linspace(-np.pi*0.5, np.pi*0.5, num=m, endpoint=True)
    theta, phi = np.meshgrid(theta, phi)
    theta, phi = theta.ravel(), phi.ravel()
    if plot == True:
        mesh_x, mesh_y = ((np.pi*0.5 - phi)*np.cos(theta), (np.pi*0.5 - phi)*np.sin(theta))
        triangles = mtri.Triangulation(mesh_x, mesh_y).triangles
    psi = np.radians(90) - phi
    alpha_angle = theta
    #return psi
    Rfunc = np.array(potentialFuncINV(psi))
    #RfuncPlot = Rfunc/max(Rfunc)
    x, y, z = Rfunc*np.cos(phi)*np.cos(theta), Rfunc*np.cos(phi)*np.sin(theta), Rfunc*np.sin(phi)
    
    #xv = G*m1*x/(a**2+2*a*z+Rfunc**2)**(1.5)+G*m2*x/Rfunc**3-.5*omega**2*x/np.sqrt(r2**2+2*r2*z+Rfunc**2)
    #yv = G*m1*y/(a**2+2*a*z+Rfunc**2)**(1.5)+G*m2*y/Rfunc**3-.5*omega**2*y/np.sqrt(r2**2+2*r2*z+Rfunc**2)
    #zv = G*m1*(z+a)/(a**2+2*a*z+Rfunc**2)**(1.5)+G*m2*z/Rfunc**3-.5*omega**2*(z+r2)/np.sqrt(r2**2+2*r2*z+Rfunc**2)

    #xv,yv,zv = xv/np.sqrt(xv**2+yv**2+zv**2),yv/np.sqrt(xv**2+yv**2+zv**2),zv/np.sqrt(xv**2+yv**2+zv**2)

    
    #Vcm = timeDelay([phase],m1_in=m1_in,m2_in=m2_in,period_in=period_in,eccentricity=eccentricity,inclination=inclination,omega=omega,\
      #                  radialVelocity=True,rocheLobe=False,radDonor=True)
    #vals = Vcm*(m1_in+m2_in)/(m1_in*separation)*np.sqrt((Rfunc*np.cos(psi)-(m1_in)/(m1_in+m2_in)*separation)**2+(Rfunc*np.sin(psi)*np.sin(alpha))**2)
    #vals = Vcm*(m1_in+m2_in)/(m1_in*separation)*np.sqrt(Rfunc**2-2*Rfunc*np.cos(psi)*(m1_in*separation)/(m1_in+m2_in)+((m1_in*separation)/(m1_in+m2_in))**2)

    OmgRot1 = m1_in/(m1_in+m2_in)*np.sqrt(constants.G*(m1+m2)*(2/separation-1/a))
    #print(OmgRot1)
    #print("separation",separation)
    omgRot = OmgRot1/separation*(m1_in+m2_in)/m1_in #Instantaneous Rotation Rate
    #print(omgRot)
    vals = -omgRot*(Rfunc*np.cos(psi)-separation*m1_in/(m1_in+m2_in))*np.sin(inclination)*np.cos(theta_ORBIT+omega) + \
           -omgRot*Rfunc*np.sin(psi)*np.sin(alpha_angle)*np.sin(inclination)*np.sin(theta_ORBIT+omega)

    #vals = -omega_angV*np.sin(2*np.pi*phase)*np.sin(inclination)*(rcm-Rfunc*np.cos(psi))
    #print(rcm)
    #print(Rfunc)
    
    if plot == True:
        #colors = np.mean(vals[triangles], axis=1)

        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        #ax.scatter(0,0,r2,s=35)
        #ax = fig.gca(projection='3d')

        ax.set_xlim3d(-1,1)
        ax.set_ylim3d(-1,1)
        #ax.set_zlim3d(-1,1)


        #Calculation of time delay at many points over the surface of the star
        colors = np.mean(vals[triangles], axis=1)


        sm = mpl.cm.ScalarMappable(cmap=cmap)
        sm.set_array([min(colors),max(colors)])
        ticks = np.linspace(min(colors),max(colors),10)
        plt.colorbar(sm,ticks=ticks)
        soa = np.array([[0, 0, 1, 0, 0, 1], [0, 0, 1, np.cos(inclination), np.sin(inclination)*np.cos(theta_ORBIT+omega), -np.sin(inclination)*np.sin(theta_ORBIT+omega)]])
        X, Y, Z, U, V, W = zip(*soa)
        a = .75
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        U,V,W = a*U,a*V,a*W
        labels = ['Accretor', 'Observer']
        for i in range(len(labels)):
            ax.text(X[i]+U[i], Y[i]+V[i], Z[i]+W[i], labels[i])
        quiv = ax.quiver(X, Y, Z, U, V, W)
        
         
        #quivNormal = ax.quiver(x/max(Rfunc), y/max(Rfunc), z/max(Rfunc), .3*xv, .3*yv, .3*zv, zorder = 0)
        #ax.scatter(x/max(Rfunc), y/max(Rfunc), z/max(Rfunc),s = 50, c='black', marker='^',zorder = 100)

        #soa = np.array([[0, 0, 1, 0, 0, 1], [0, 0, 1, np.cos(inclination), np.sin(inclination)*np.cos(theta_ORBIT+omega), -np.sin(inclination)*np.sin(theta_ORBIT+omega)]])
        #soa = np.array([[0, 0, 1, 0, 0, 1], [np.cos(inclination), np.sin(inclination)*np.cos(theta_ORBIT+omega), -np.sin(inclination)*np.sin(theta_ORBIT+omega), np.cos(inclination), np.sin(inclination)*np.cos(theta_ORBIT+omega), -np.sin(inclination)*np.sin(theta_ORBIT+omega)]])
        #X, Y, Z, U, V, W = zip(*soa)
        #labels = ['Accretor', 'Observer']
        #for i in range(len(labels)):
        #    ax.text(X[i]+U[i], Y[i]+V[i], Z[i]+W[i], labels[i])
        #quiv = ax.quiver(X, Y, Z, U, V, W)
        #data = np.array([[low,high],[low,high]])
        #cax = ax.imshow(data, interpolation='nearest', cmap = cmap)
        #cbar = fig.colorbar(cax, ticks=[low, high], orientation='vertical')
        
        triang = mtri.Triangulation(x/max(Rfunc), y/max(Rfunc), triangles)
        collec = ax.plot_trisurf(triang, z/max(Rfunc), cmap=cmap, shade=False, linewidth=0.)
        collec.set_array(colors)
        collec.autoscale()
        ax.scatter(0,0,r2/max(Rfunc))
        ax.scatter(0,0,L2/max(Rfunc))
        print('10^(' + str(np.log(max(Rfunc))/np.log(10))+")")
        #collec.set_array(colors)
        #collec.autoscale()
        plt.show(block = False)
    if RETURN == True:
        return psi, alpha_angle, np.array(vals)


def genDisplacementMap(phase, \
                  m1_in=1.4, m2_in=0.7, period_in=.787, eccentricity = 0,\
                  inclination=np.radians(44.), omega=np.radians(90.), cmap = cm.spring, plot = True, Q = 25, RETURN = False):

    m1 = m1_in * constants.M_Sun
    m2 = m2_in * constants.M_Sun
    q = m2_in/m1_in
    period = period_in * constants.day
    omega_angV = 2*np.pi/period
    G = constants.G
    a = (G*(m1+m2)*period**2/(4*np.pi**2))**(1./3.)
    r2 = m1_in/(m1_in+m2_in)*a
    rcm = m1_in/(m1_in+m2_in)*a
    radiusDonor = a*(.49*q**(.667))/(.6*q**(.667)+np.log(1+q**(.333)))
    #print(radiusDonor)
    separation = simulator_2.timeDelay([phase],m1_in=m1_in,m2_in=m2_in,eccentricity=eccentricity,period_in=period_in,inclination=inclination,omega=omega,separationReturn=True)
    separation = separation[0]
    theta_ORBIT = 2*np.pi*phase #Only Valid For Circular Orbits!!!!
    (n, m) = (Q, Q)
    # Meshing a unit sphere according to n, m 
    #theta = np.linspace(0, 2*np.pi, num=n, endpoint=False)
    #phi = np.linspace(np.pi * (-0.5 + 1./(m+1)), np.pi*0.5, num=m, endpoint=False)
    #phi = np.linspace(0, np.pi*0.5, num=m, endpoint=False)

    
    L2 = a*(.5+.227*np.log(q)/np.log(10))
    #print("L2 = " + str(L2))
    rr = m1/(m1+m2)*a-L2
    Uconst = 0
    
    def potentialFunc(p2,theta2):
        #phiU = -G*m1/(np.sqrt((m2/(m1+m2)*S+r*np.cos(theta))**2+(r*np.sin(theta))**2))- \
         #     G*m2/(np.sqrt((m1/(m1+m2)*S-r*np.cos(theta))**2+(r*np.sin(theta))**2))- \
         #     .5*omega**2*r**2
        r2 = m1_in/(m1_in+m2_in)*a
        phiU = -G*m1/np.sqrt(a**2+p2**2+2*a*p2*np.cos(theta2))-G*m2/p2 \
                   -.5*omega_angV**2*(r2**2+p2**2+2*r2*p2*np.cos(theta2))
        return phiU-Uconst
    Uconst = potentialFunc(L2*1,np.pi)
    #print(Uconst)
    
    def potentialFuncINV(psi):
        psi = np.add(psi,np.pi)
        r = np.zeros(len(psi))
        a1 = L2/1000
        b1 = L2
        for i in range(len(psi)):
            r[i] = opt.brentq(potentialFunc,a1,b1,args=(psi[i]),maxiter=100)
        return r
    # Meshing a unit sphere according to n, m 
    theta = np.linspace(0, 2*np.pi, num=n, endpoint=True)
    #phi = np.linspace(np.pi * (-0.5 + 1./(m+1)), np.pi*0.5, num=m, endpoint=False)
    if plot == False:
        phi = np.linspace(0, np.pi*0.5, num=m, endpoint=True)
    if plot == True:
        phi = np.linspace(-np.pi*0.5, np.pi*0.5, num=m, endpoint=True)
    theta, phi = np.meshgrid(theta, phi)
    theta, phi = theta.ravel(), phi.ravel()
    if plot == True:
        mesh_x, mesh_y = ((np.pi*0.5 - phi)*np.cos(theta), (np.pi*0.5 - phi)*np.sin(theta))
        triangles = mtri.Triangulation(mesh_x, mesh_y).triangles
    psi = np.radians(90) - phi
    alpha_angle = theta
    #return psi
    Rfunc = np.array(potentialFuncINV(psi))
    #RfuncPlot = Rfunc/max(Rfunc)
    x, y, z = Rfunc*np.cos(phi)*np.cos(theta), Rfunc*np.cos(phi)*np.sin(theta), Rfunc*np.sin(phi)
    
    #xv = G*m1*x/(a**2+2*a*z+Rfunc**2)**(1.5)+G*m2*x/Rfunc**3-.5*omega**2*x/np.sqrt(r2**2+2*r2*z+Rfunc**2)
    #yv = G*m1*y/(a**2+2*a*z+Rfunc**2)**(1.5)+G*m2*y/Rfunc**3-.5*omega**2*y/np.sqrt(r2**2+2*r2*z+Rfunc**2)
    #zv = G*m1*(z+a)/(a**2+2*a*z+Rfunc**2)**(1.5)+G*m2*z/Rfunc**3-.5*omega**2*(z+r2)/np.sqrt(r2**2+2*r2*z+Rfunc**2)

    #xv,yv,zv = xv/np.sqrt(xv**2+yv**2+zv**2),yv/np.sqrt(xv**2+yv**2+zv**2),zv/np.sqrt(xv**2+yv**2+zv**2)

    
    #Vcm = timeDelay([phase],m1_in=m1_in,m2_in=m2_in,period_in=period_in,eccentricity=eccentricity,inclination=inclination,omega=omega,\
      #                  radialVelocity=True,rocheLobe=False,radDonor=True)
    #vals = Vcm*(m1_in+m2_in)/(m1_in*separation)*np.sqrt((Rfunc*np.cos(psi)-(m1_in)/(m1_in+m2_in)*separation)**2+(Rfunc*np.sin(psi)*np.sin(alpha))**2)
    #vals = Vcm*(m1_in+m2_in)/(m1_in*separation)*np.sqrt(Rfunc**2-2*Rfunc*np.cos(psi)*(m1_in*separation)/(m1_in+m2_in)+((m1_in*separation)/(m1_in+m2_in))**2)

    OmgRot1 = m1_in/(m1_in+m2_in)*np.sqrt(constants.G*(m1+m2)*(2/separation-1/a))
    #print(OmgRot1)
    #print("separation",separation)
    omgRot = OmgRot1/separation*(m1_in+m2_in)/m1_in #Instantaneous Rotation Rate
    #print(omgRot)
    vals = r2-Rfunc*np.cos(psi)
    #print(vals)
    #vals = -omega_angV*np.sin(2*np.pi*phase)*np.sin(inclination)*(rcm-Rfunc*np.cos(psi))
    #print(rcm)
    #print(Rfunc)
    
    if plot == True:
        #colors = np.mean(vals[triangles], axis=1)

        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        #ax.scatter(0,0,r2,s=35)
        #ax = fig.gca(projection='3d')

        ax.set_xlim3d(-1,1)
        ax.set_ylim3d(-1,1)
        #ax.set_zlim3d(-1,1)


        #Calculation of time delay at many points over the surface of the star
        colors = np.mean(vals[triangles], axis=1)


        sm = mpl.cm.ScalarMappable(cmap=cmap)
        sm.set_array([min(colors),max(colors)])
        ticks = np.linspace(min(colors),max(colors),10)
        plt.colorbar(sm,ticks=ticks)
        soa = np.array([[0, 0, 1, 0, 0, 1], [0, 0, 1, np.cos(inclination), np.sin(inclination)*np.cos(theta_ORBIT+omega), -np.sin(inclination)*np.sin(theta_ORBIT+omega)]])
        X, Y, Z, U, V, W = zip(*soa)
        a = .75
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        U,V,W = a*U,a*V,a*W
        labels = ['Accretor', 'Observer']
        for i in range(len(labels)):
            ax.text(X[i]+U[i], Y[i]+V[i], Z[i]+W[i], labels[i])
        quiv = ax.quiver(X, Y, Z, U, V, W)
        
         
        #quivNormal = ax.quiver(x/max(Rfunc), y/max(Rfunc), z/max(Rfunc), .3*xv, .3*yv, .3*zv, zorder = 0)
        #ax.scatter(x/max(Rfunc), y/max(Rfunc), z/max(Rfunc),s = 50, c='black', marker='^',zorder = 100)

        #soa = np.array([[0, 0, 1, 0, 0, 1], [0, 0, 1, np.cos(inclination), np.sin(inclination)*np.cos(theta_ORBIT+omega), -np.sin(inclination)*np.sin(theta_ORBIT+omega)]])
        #soa = np.array([[0, 0, 1, 0, 0, 1], [np.cos(inclination), np.sin(inclination)*np.cos(theta_ORBIT+omega), -np.sin(inclination)*np.sin(theta_ORBIT+omega), np.cos(inclination), np.sin(inclination)*np.cos(theta_ORBIT+omega), -np.sin(inclination)*np.sin(theta_ORBIT+omega)]])
        #X, Y, Z, U, V, W = zip(*soa)
        #labels = ['Accretor', 'Observer']
        #for i in range(len(labels)):
        #    ax.text(X[i]+U[i], Y[i]+V[i], Z[i]+W[i], labels[i])
        #quiv = ax.quiver(X, Y, Z, U, V, W)
        #data = np.array([[low,high],[low,high]])
        #cax = ax.imshow(data, interpolation='nearest', cmap = cmap)
        #cbar = fig.colorbar(cax, ticks=[low, high], orientation='vertical')
        
        triang = mtri.Triangulation(x/max(Rfunc), y/max(Rfunc), triangles)
        collec = ax.plot_trisurf(triang, z/max(Rfunc), cmap=cmap, shade=False, linewidth=0.)
        collec.set_array(colors)
        collec.autoscale()
        #triang = mtri.Triangulation(x, y, triangles)
        #collec = ax.plot_trisurf(triang, z, cmap=cmap, shade=False, linewidth=0.)
        ax.scatter(0,0,r2/max(Rfunc))
        ax.scatter(0,0,L2/max(Rfunc))
        print(max(Rfunc))
        #collec.set_array(colors)
        #collec.autoscale()
        plt.show(block = False)
    if RETURN == True:
        return psi, alpha_angle, np.array(vals)

    

def genTimeDelayMap(phase, \
                  m1_in=1.4, m2_in=0.7, period_in=.787, eccentricity = 0,\
                  inclination=np.radians(44.), omega=np.radians(90.), cmap = cm.cool, plot = True, Q = 25, RETURN = False):

    m1 = m1_in * constants.M_Sun
    m2 = m2_in * constants.M_Sun
    q = m2_in/m1_in
    period = period_in * constants.day
    omega_angV = 2*np.pi/period
    G = constants.G
    a = (G*(m1+m2)*period**2/(4*np.pi**2))**(1./3.)
    s = a*(1-eccentricity)
    #s=a
    r2 = m1_in/(m1_in+m2_in)*s
    radiusDonor = s*(.49*q**(.667))/(.6*q**(.667)+np.log(1+q**(.333)))
    separation = simulator_2.timeDelay([phase],m1_in=m1_in,m2_in=m2_in,eccentricity=eccentricity,period_in=period_in,inclination=inclination,omega=omega,separationReturn=True)
    separation = separation[0]
    theta_ORBIT = 2*np.pi*phase
    (n, m) = (Q, Q)
    #print("separation",separation)
    #print("radiusDonor",radiusDonor)
    # Meshing a unit sphere according to n, m 
    #theta = np.linspace(0, 2*np.pi, num=n, endpoint=False)
    #phi = np.linspace(np.pi * (-0.5 + 1./(m+1)), np.pi*0.5, num=m, endpoint=False)
    #phi = np.linspace(0, np.pi*0.5, num=m, endpoint=False)

    
    L2 = s*(.5+.227*np.log(q)/np.log(10))
    #print("L2 = " + str(L2))
    rr = m1/(m1+m2)*s-L2
    Uconst = 0
    
    def potentialFunc(p2,theta2):
        #phiU = -G*m1/(np.sqrt((m2/(m1+m2)*S+r*np.cos(theta))**2+(r*np.sin(theta))**2))- \
        #      G*m2/(np.sqrt((m1/(m1+m2)*S-r*np.cos(theta))**2+(r*np.sin(theta))**2))- \
        #      .5*omega**2*r**2
        r2 = m1_in/(m1_in+m2_in)*s
        phiU = -G*m1/np.sqrt(s**2+p2**2+2*s*p2*np.cos(theta2))-G*m2/p2 \
                   -.5*omega_angV**2*(r2**2+p2**2+2*r2*p2*np.cos(theta2))
        return phiU-Uconst
    Uconst = potentialFunc(L2,np.pi)
    #print(Uconst)
    
    def potentialFuncINV(psi):
        psi = np.add(psi,np.pi)
        r = np.zeros(len(psi))
        a1 = L2/1000
        b1 = L2
        for i in range(len(psi)):
            r[i] = opt.brentq(potentialFunc,a1,b1,args=(psi[i]),maxiter=100)
        return r
    # Meshing a unit sphere according to n, m 
    theta = np.linspace(0, 2*np.pi, num=n, endpoint=True)
    #phi = np.linspace(np.pi * (-0.5 + 1./(m+1)), np.pi*0.5, num=m, endpoint=False)
    if plot == False:
        phi = np.linspace(0, np.pi*0.5, num=m, endpoint=True)
    if plot == True:
        phi = np.linspace(-np.pi*0.5, np.pi*0.5, num=m, endpoint=True)
    theta, phi = np.meshgrid(theta, phi)
    theta, phi = theta.ravel(), phi.ravel()
    if plot == True:
        mesh_x, mesh_y = ((np.pi*0.5 - phi)*np.cos(theta), (np.pi*0.5 - phi)*np.sin(theta))
        triangles = mtri.Triangulation(mesh_x, mesh_y).triangles
    psi = np.radians(90) - phi
    alpha_angle = theta
    #return psi
    Rfunc = np.array(potentialFuncINV(psi))
    #RfuncPlot = Rfunc/max(Rfunc)
    x, y, z = Rfunc*np.cos(phi)*np.cos(theta), Rfunc*np.cos(phi)*np.sin(theta), Rfunc*np.sin(phi)
    
    #xv = G*m1*x/(a**2+2*a*z+Rfunc**2)**(1.5)+G*m2*x/Rfunc**3-.5*omega**2*x/np.sqrt(r2**2+2*r2*z+Rfunc**2)
    #yv = G*m1*y/(a**2+2*a*z+Rfunc**2)**(1.5)+G*m2*y/Rfunc**3-.5*omega**2*y/np.sqrt(r2**2+2*r2*z+Rfunc**2)
    #zv = G*m1*(z+a)/(a**2+2*a*z+Rfunc**2)**(1.5)+G*m2*z/Rfunc**3-.5*omega**2*(z+r2)/np.sqrt(r2**2+2*r2*z+Rfunc**2)

    #xv,yv,zv = xv/np.sqrt(xv**2+yv**2+zv**2),yv/np.sqrt(xv**2+yv**2+zv**2),zv/np.sqrt(xv**2+yv**2+zv**2)

    
    vals = np.array(np.sqrt(Rfunc**2+separation**2-2*Rfunc*separation*np.cos(psi))+\
            -separation*np.sin(inclination)*np.sin(omega+theta_ORBIT)+\
            Rfunc*np.cos(psi)*np.sin(inclination)*np.sin(omega+theta_ORBIT)+\
           -Rfunc*np.sin(psi)*np.cos(alpha_angle)*np.cos(inclination)+\
           -Rfunc*np.sin(psi)*np.sin(alpha_angle)*np.sin(inclination)*np.cos(omega+theta_ORBIT))
    vals = vals/constants.c
    
    if plot == True:
        #colors = np.mean(vals[triangles], axis=1)

        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        #ax = fig.gca(projection='3d')

        ax.set_xlim3d(-1,1)
        ax.set_ylim3d(-1,1)
        ax.set_zlim3d(-1,1)
        #ax.set_xticks([]) 
        #ax.set_yticks([]) 
        #ax.set_zticks([])


        #Calculation of time delay at many points over the surface of the star
        colors = np.mean(vals[triangles], axis=1)


        sm = mpl.cm.ScalarMappable(cmap=cmap)
        sm.set_array([min(colors),max(colors)])
        ticks = np.linspace(min(colors),max(colors),10)   
        plt.colorbar(sm,ax=ax,ticks=ticks)
        #ax.set_title("Time Delay (s)",y=1.15,x = 1.073)
        ax.set_title("Time Delay (s)",y=1.15)
        soa = np.array([[0, 0, 1, 0, 0, 1], [0, 0, 1, np.cos(inclination), np.sin(inclination)*np.cos(theta_ORBIT+omega), -np.sin(inclination)*np.sin(theta_ORBIT+omega)]])
        X, Y, Z, U, V, W = zip(*soa)
        a = .75
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        U,V,W = a*U,a*V,a*W
        labels = ['Accretor', 'Observer']
        for i in range(len(labels)):
            ax.text(X[i]+U[i], Y[i]+V[i], Z[i]+W[i], labels[i])
        quiv = ax.quiver(X, Y, Z, U, V, W)
        
         
        #quivNormal = ax.quiver(x/max(Rfunc), y/max(Rfunc), z/max(Rfunc), .3*xv, .3*yv, .3*zv, zorder = 0)
        #ax.scatter(x/max(Rfunc), y/max(Rfunc), z/max(Rfunc),s = 50, c='black', marker='^',zorder = 100)

        #soa = np.array([[0, 0, 1, 0, 0, 1], [0, 0, 1, np.cos(inclination), np.sin(inclination)*np.cos(theta_ORBIT+omega), -np.sin(inclination)*np.sin(theta_ORBIT+omega)]])
        #soa = np.array([[0, 0, 1, 0, 0, 1], [np.cos(inclination), np.sin(inclination)*np.cos(theta_ORBIT+omega), -np.sin(inclination)*np.sin(theta_ORBIT+omega), np.cos(inclination), np.sin(inclination)*np.cos(theta_ORBIT+omega), -np.sin(inclination)*np.sin(theta_ORBIT+omega)]])
        #X, Y, Z, U, V, W = zip(*soa)
        #labels = ['Accretor', 'Observer']
        #for i in range(len(labels)):
        #    ax.text(X[i]+U[i], Y[i]+V[i], Z[i]+W[i], labels[i])
        #quiv = ax.quiver(X, Y, Z, U, V, W)
        #data = np.array([[low,high],[low,high]])
        #cax = ax.imshow(data, interpolation='nearest', cmap = cmap)
        #cbar = fig.colorbar(cax, ticks=[low, high], orientation='vertical')
        
        triang = mtri.Triangulation(x/max(Rfunc), y/max(Rfunc), triangles)
        collec = ax.plot_trisurf(triang, z/max(Rfunc), cmap=cmap, shade=False, linewidth=0.)
        collec.set_array(colors)
        collec.autoscale()
        ax.view_init(38, 0)
        #ax.set_xticks([]) 
        #ax.set_yticks([]) 
        #ax.set_zticks([])
        #ax.set_axis_off()
        #for idx, angle in enumerate(np.linspace(0, 360, 100)):
        #    ax.view_init(38, angle)
        #    plt.draw()
        #    plt.savefig('m4_l4-%04d.png' % idx)
        plt.show()
    if RETURN == True:
        return psi, alpha_angle, np.array(vals)


def genApparentIntensityMap(phase, \
                  m1_in=1.4, m2_in=0.7, period_in=.787, eccentricity = 0,\
                  inclination=np.radians(44.), omega=np.radians(90.), cmap = cm.hot, plot = True, Q = 25, disk = False, RETURN = False, alpha = np.radians(5), u = .6):

    m1 = m1_in * constants.M_Sun
    m2 = m2_in * constants.M_Sun
    q = m2_in/m1_in
    period = period_in * constants.day
    omega_angV = 2*np.pi/period
    G = constants.G
    a = (G*(m1+m2)*period**2/(4*np.pi**2))**(1./3.)
    s = a*(1-eccentricity)
    #s = a
    r2 = m1_in/(m1_in+m2_in)*s
    radiusDonor = a*(.49*q**(.667))/(.6*q**(.667)+np.log(1+q**(.333)))
    separation = simulator_2.timeDelay([phase],m1_in=m1_in,m2_in=m2_in,eccentricity=eccentricity,period_in=period_in,inclination=inclination,omega=omega,separationReturn=True)
    separation = separation[0]
    theta_ORBIT = 2*np.pi*phase
    (n, m) = (Q, Q)
    # Meshing a unit sphere according to n, m 
    #theta = np.linspace(0, 2*np.pi, num=n, endpoint=False)
    #phi = np.linspace(np.pi * (-0.5 + 1./(m+1)), np.pi*0.5, num=m, endpoint=False)
    #phi = np.linspace(0, np.pi*0.5, num=m, endpoint=False)

    
    L2 = s*(.5+.227*np.log(q)/np.log(10))
    #print("L2 = " + str(L2))
    rr = m1/(m1+m2)*s-L2
    Uconst = 0
    
    def potentialFunc(p2,theta2):
        #phiU = -G*m1/(np.sqrt((m2/(m1+m2)*S+r*np.cos(theta))**2+(r*np.sin(theta))**2))- \
        #      G*m2/(np.sqrt((m1/(m1+m2)*S-r*np.cos(theta))**2+(r*np.sin(theta))**2))- \
        #      .5*omega**2*r**2
        r2 = m1_in/(m1_in+m2_in)*s
        phiU = -G*m1/np.sqrt(s**2+p2**2+2*s*p2*np.cos(theta2))-G*m2/p2 \
                   -.5*omega_angV**2*(r2**2+p2**2+2*r2*p2*np.cos(theta2))
        return phiU-Uconst
    Uconst = potentialFunc(L2,np.pi)
    #print(Uconst)
    
    def potentialFuncINV(psi):
        psi = np.add(psi,np.pi)
        r = np.zeros(len(psi))
        a1 = L2/1000
        b1 = L2
        for i in range(len(psi)):
            r[i] = opt.brentq(potentialFunc,a1,b1,args=(psi[i]),maxiter=100)
        return r
    # Meshing a unit sphere according to n, m 
    theta = np.linspace(0, 2*np.pi, num=n, endpoint=True)
    #phi = np.linspace(np.pi * (-0.5 + 1./(m+1)), np.pi*0.5, num=m, endpoint=False)
    if plot == False:
        phi = np.linspace(0, np.pi*0.5, num=m, endpoint=True)
    if plot == True:
        phi = np.linspace(-np.pi*0.5, np.pi*0.5, num=m, endpoint=True)
    theta, phi = np.meshgrid(theta, phi)
    theta, phi = theta.ravel(), phi.ravel()
    if plot == True:
        mesh_x, mesh_y = ((np.pi*0.5 - phi)*np.cos(theta), (np.pi*0.5 - phi)*np.sin(theta))
        triangles = mtri.Triangulation(mesh_x, mesh_y).triangles
    psi = np.radians(90) - phi
    alpha_angle = theta
    #return psi
    Rfunc = np.array(potentialFuncINV(psi))
    #RfuncPlot = Rfunc/max(Rfunc)
    x, y, z = Rfunc*np.cos(phi)*np.cos(theta), Rfunc*np.cos(phi)*np.sin(theta), Rfunc*np.sin(phi)

    
    xv = G*m1*x/(s**2+2*s*z+Rfunc**2)**(1.5)+G*m2*x/Rfunc**3-.5*omega**2*x/np.sqrt(r2**2+2*r2*z+Rfunc**2)
    yv = G*m1*y/(s**2+2*s*z+Rfunc**2)**(1.5)+G*m2*y/Rfunc**3-.5*omega**2*y/np.sqrt(r2**2+2*r2*z+Rfunc**2)
    zv = G*m1*(z+a)/(s**2+2*s*z+Rfunc**2)**(1.5)+G*m2*z/Rfunc**3-.5*omega**2*(z+r2)/np.sqrt(r2**2+2*r2*z+Rfunc**2)

    xv,yv,zv = xv/np.sqrt(xv**2+yv**2+zv**2),yv/np.sqrt(xv**2+yv**2+zv**2),zv/np.sqrt(xv**2+yv**2+zv**2)

    
    #T_0 = 4.894762604*10**18.
    T_0 = 1.
    A1 = (Rfunc**2+separation**2-2*z*separation)**(-1) #Distance Attenuation
    A2 = -np.sqrt(A1)*(x*xv+y*yv+(z-separation))*zv #Projected area toward accretor
    #A3 = xv*np.cos(inclinatio/n)+yv*np.sqrt(1-(np.cos(inclination))**2-(np.sin(inclination)*np.sin(theta_ORBIT+omega))**2)-zv*np.sin(inclination)*np.sin(theta_ORBIT+omega)
    A3 = xv*np.cos(inclination)-zv*np.sin(inclination)*np.sin(theta_ORBIT+omega)+yv*np.sin(inclination)*np.cos(theta_ORBIT+omega) #Projected area toward observer
    #A4 = (1-u+u*A3)/(1+u/3)#Linear Limb Darkening
    vals = T_0*A1*A2*A3
    
    for i in range(len(vals)):
        if vals[i] < 0 or A2[i] < 0 or A3[i] < 0:
            vals[i] = 0
        if disk == True:
            sinDelta = radiusDonor*(np.sin(psi[i])*np.cos(alpha_angle[i]))/np.sqrt(radiusDonor**2+separation**2-2*separation*radiusDonor*np.cos(psi[i]))
            if abs(np.arcsin(sinDelta)) < alpha:
                vals[i] = 0
    
    if plot == True:
        #colors = np.mean(vals[triangles], axis=1)

        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        #ax = fig.gca(projection='3d')

        ax.set_xlim3d(-1,1)
        ax.set_ylim3d(-1,1)
        ax.set_zlim3d(-1,1)
        #ax.set_xticks([-1,1]) 
        #ax.set_yticks([-1,1]) 
        #ax.set_zticks([-1,1])


        #Calculation of time delay at many points over the surface of the star
        colors = np.mean(vals[triangles], axis=1)


        sm = mpl.cm.ScalarMappable(cmap=cmap)
        sm.set_array([min(colors),max(colors)])
        ticks = np.linspace(min(colors),max(colors),10)
        plt.colorbar(sm,ax=ax,ticks=ticks)
        ax.set_title("Intensity (Arbitrary Units)",y=1.15)
        soa = np.array([[0, 0, 1, 0, 0, 1], [0, 0, 1, np.cos(inclination), np.sin(inclination)*np.cos(theta_ORBIT+omega), -np.sin(inclination)*np.sin(theta_ORBIT+omega)]])
        X, Y, Z, U, V, W = zip(*soa)
        a = .75
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        U,V,W = a*U,a*V,a*W
        labels = ['Accretor', 'Observer']
        for i in range(len(labels)):
            ax.text(X[i]+U[i], Y[i]+V[i], Z[i]+W[i], labels[i])
        quiv = ax.quiver(X, Y, Z, U, V, W)
        
         
        #quivNormal = ax.quiver(x/max(Rfunc), y/max(Rfunc), z/max(Rfunc), .3*xv, .3*yv, .3*zv, zorder = 0)
        #ax.scatter(x/max(Rfunc), y/max(Rfunc), z/max(Rfunc),s = 50, c='black', marker='^',zorder = 100)

        #soa = np.array([[0, 0, 1, 0, 0, 1], [0, 0, 1, np.cos(inclination), np.sin(inclination)*np.cos(theta_ORBIT+omega), -np.sin(inclination)*np.sin(theta_ORBIT+omega)]])
        #soa = np.array([[0, 0, 1, 0, 0, 1], [np.cos(inclination), np.sin(inclination)*np.cos(theta_ORBIT+omega), -np.sin(inclination)*np.sin(theta_ORBIT+omega), np.cos(inclination), np.sin(inclination)*np.cos(theta_ORBIT+omega), -np.sin(inclination)*np.sin(theta_ORBIT+omega)]])
        #X, Y, Z, U, V, W = zip(*soa)
        #labels = ['Accretor', 'Observer']
        #for i in range(len(labels)):
        #    ax.text(X[i]+U[i], Y[i]+V[i], Z[i]+W[i], labels[i])
        #quiv = ax.quiver(X, Y, Z, U, V, W)
        #data = np.array([[low,high],[low,high]])
        #cax = ax.imshow(data, interpolation='nearest', cmap = cmap)
        #cbar = fig.colorbar(cax, ticks=[low, high], orientation='vertical')
        
        triang = mtri.Triangulation(x/max(Rfunc), y/max(Rfunc), triangles)
        collec = ax.plot_trisurf(triang, z/max(Rfunc), cmap=cmap, shade=False, linewidth=0.)
        collec.set_array(colors)
        collec.autoscale()
        plt.show()
    if RETURN == True:
        vals = vals*np.sin(psi)*Rfunc**2 #Jacobian Determinate for Spherical Coordinates
        return psi, alpha_angle, vals





def plot1():
    alphas = np.linspace(0,15,5)
    TrueDelays = np.zeros(len(alphas))
    PseudoDelays = np.zeros(len(alphas))
    for i in range(len(alphas)):
        TrueDelays[i] = delaySignal(alpha = np.radians(alphas[i]),plot = False)
        PseudoDelays[i] = (timeDelay(np.array([.5]), m1_in=1.4, m2_in=0.7, period_in=0.788,\
                  inclination=np.radians(44.),eccentricity = 0, omega=np.radians(90.), alpha=np.radians(alphas[i]), \
                  gamma=0, radialVelocity = False, rocheLobe = True, radDonor = True, simpleKCOR = True, separationReturn = False, \
                  ellipticalCORR = False, pseudo3D = True, SP_setting = 'egg'))[0]
    plt.plot(alphas,TrueDelays,label='true')
    plt.plot(alphas,PseudoDelays,label='pseudo')
    plt.legend()
    plt.xlabel(r"$\alpha$ $(^{\circ{}})}$")
    plt.ylabel("Delay (s)")
    plt.show()

def delaySignalFast(phase = .5, m1_in = 1.4, m2_in = .7, period_in = .787, \
                    inclination = np.radians(44.), eccentricity = 0, omega = np.radians(90.), alpha = np.radians(0), \
                    binNumber = 100, Q = 120, plot = True, disk = True, intKind = 'quadratic', \
                    radialVelocity = False, u = .6, pseudo3DTEST = True, outputData = False):
    
    sigFunc = genTimeDelayMap
    
    psiECHO, alphaECHO, T = sigFunc(phase,m1_in=m1_in,m2_in=m2_in,eccentricity=eccentricity,period_in=period_in,\
                                            inclination=inclination,omega=omega,plot = False,Q=Q,RETURN = True)
    
    psiINT, alphaINT, I = genApparentIntensityMap(phase,m1_in=m1_in,m2_in=m2_in,eccentricity=eccentricity,period_in=period_in,\
                                            inclination=inclination,omega=omega,plot = False,Q=Q,RETURN = True,disk=disk,alpha = alpha,u=u)
    BIN = [[]]*binNumber
    maxT = max(T)
    minT = min(T)
    delta = (maxT-minT)/binNumber
    
    for i in range(len(T)):
        binIndex = int(np.ceil((T[i]-minT)/delta))
        #print(binIndex)
        if binIndex > binNumber:
            binIndex = binNumber
        BIN[binIndex-1] = BIN[binIndex-1] + [i]
        
    delayList = np.linspace(min(T),max(T),binNumber)
    
    intensityList = np.zeros(binNumber)
    for i in range(binNumber):
        for j in range(len(BIN[i])):
            intensityList[i] += I[BIN[i][j]]
    #intensityList = intensityList/max(intensityList)
    
    A = []
    B = []
    for i in range(len(intensityList)):
        if intensityList[i] > .33:
            B.append(intensityList[i])
            A.append(delayList[i])
    try:
        f = scpint.UnivariateSpline(A, B)
        tt = np.linspace(min(A),max(A),500)
        yy = f(tt)
        output = tt[np.argmax(yy)]
        return output
    except:
        print("Warning: Bad Fit -> Low Intensity")
        return np.inf
        

    
    output = tt[np.argmax(yy)] 
    plt.axvline(x=output,c='black',label='Output')
    plt.plot(tt,yy,'k-')
    plt.scatter(delayList,intensityList,marker='<')
    plt.show()


    
    
    

def delaySignal(phase = .5, m1_in = 1.4, m2_in = .7, period_in = .787, \
                    inclination = np.radians(44.), eccentricity = 0, omega = np.radians(90.), alpha = np.radians(0), \
                    binNumber = 100, Q = 120, plot = True, disk = True, intKind = 'quadratic', \
                    radialVelocity = False, u = .6, pseudo3DTEST = True, outputData = False):
    if radialVelocity == True:
        sigFunc = genRadialVelocityMap
    else:
        sigFunc = genTimeDelayMap

    psiECHO, alphaECHO, T = sigFunc(phase,m1_in=m1_in,m2_in=m2_in,eccentricity=eccentricity,period_in=period_in,\
                                            inclination=inclination,omega=omega,plot = False,Q=Q,RETURN = True)
    #negativeV = False
    #if T[int((len(T)-1)/2)] < 0:
    #    negativeV = True
    #T = abs(T)


    psiINT, alphaINT, I = genApparentIntensityMap(phase,m1_in=m1_in,m2_in=m2_in,eccentricity=eccentricity,period_in=period_in,\
                                            inclination=inclination,omega=omega,plot = False,Q=Q,RETURN = True,disk=disk,alpha = alpha,u=u)
    if plot == True:
        sigFunc(phase,m1_in=m1_in,m2_in=m2_in,period_in=period_in,eccentricity=eccentricity,\
                                            inclination=inclination,omega=omega,plot = True,Q=25,RETURN = False)
        genApparentIntensityMap(phase,m1_in=m1_in,m2_in=m2_in,period_in=period_in,eccentricity=eccentricity,\
                                            inclination=inclination,omega=omega,plot = True,Q=25,RETURN = False,disk=disk,alpha=alpha,u=u)

    #print("Check Point 1")  
    #if psiECHO == psiINT and alphaECHO == alphaINT:
    #    psi,alpha = psiECHO,alphaECHO
    BIN = [[]]*binNumber
    if radialVelocity == True:
        delta = (max(T)-min(T))/binNumber
    else:
        delta = max(T)/binNumber
    #print(len(T))
    for i in range(len(T)):
        #print(i)
        #print("Check Point " + str(i+2))
        #print(T[i]/delta)
        if radialVelocity == True:
            binIndex = int(np.ceil((T[i]-min(T))/delta))
        else:
            binIndex = int(np.ceil((T[i])/delta))
        if binIndex > binNumber:
            binIndex = binNumber
        #print("binIndex = " + str(binIndex))
        BIN[binIndex-1] = BIN[binIndex-1] + [i]
    #print("Check Point 2")
    #print(BIN)
    if radialVelocity == True:
        delayList = np.arange(min(T),max(T),delta) + delta/2
    else:
        delayList = np.arange(0,max(T),delta) + delta/2
    if len(delayList) > binNumber:
        delayList = delayList[:-1]
    intensityList = np.zeros(binNumber)
    #print(delayList)
    #print(max(T))
    for i in range(binNumber):
        #print(len(BIN[i]))
        for j in range(len(BIN[i])):
            intensityList[i] += I[BIN[i][j]]
    #print("len(delayList) = " + str(len(delayList)))
    #print("len(intensityList) = " + str(len(intensityList)))
    #print(delayList)
    #print(intensityList)
    if radialVelocity == True:
        deg = 6
        A = np.polyfit(delayList,intensityList,deg = deg)
        print(A)
        def f(t,power = deg):
            R = 0
            for i in range(power+1):
                R += A[i]*t**(power-i)
            return R
    else:
        f = scpint.interp1d(delayList,intensityList,kind=intKind)
    tt = np.linspace(min(delayList),max(delayList),5000)
    yy = f(tt)
    #print("Check Point 3 ")  
    if plot == True:
        delaySIM2 = np.array(simulator_2.timeDelay([phase],m1_in=m1_in,m2_in=m2_in,eccentricity=eccentricity,period_in=period_in,inclination=inclination, \
                                           omega=omega,alpha=alpha,radialVelocity=radialVelocity, pseudo3D = pseudo3DTEST))
        print("delaySIM2",delaySIM2)
        if radialVelocity == True:
            T = T/1000
            delayList = np.array(delayList)/1000
            tt = tt/1000
            delaySIM2 = delaySIM2/1000
        plt.figure()
        plt.axvline(x=delaySIM2,c='black',label='pseudo 3D')
        plt.axvline(x=sum(tt*yy)/sum(yy),c='black',linestyle='dashed',label='centroid')
        plt.scatter(delayList,intensityList, s=50, facecolors='none', edgecolors='b')
        plt.plot(tt,yy,'g-')
        if radialVelocity == True:
            plt.xlabel("Velocity (km/s)")
        else:
            plt.xlabel("Time (s)")
        plt.ylabel("Intensity (Arbitrary Units)")
        plt.legend()
        plt.show(block = False)
    #output = delayList[np.argmax(intensityList)]
    if max(yy) < .1:
        output = 0.0001
        print("V = 0, Intensity Too Low")
    else:
        output = tt[np.argmax(yy)] #This chooses the peak intensity. This is not the best way to compute a delay to output most likely...
        #output = sum(tt*yy)/sum(yy) #This corresponds to the weighted average (centroid) of the distribution
    #if negativeV == True:
    #    output = -output
    if outputData == False:
        return output
    else:
        return delayList,intensityList

def effectiveRadius(phi = .5, m1_in = 1.4, m2_in = .7, period_in = .787,\
                    inclination = np.radians(44.), omega = np.radians(90.), eccentricity = 0, \
                    binNumber = 100, Q = 120, alpha = np.radians(0), intKind = 'cubic', \
                     radialVelocity = False,u=.6, plot = False , p3d = True, outputData = False, \
                     # FUNC = FUNC_CORRECTION, \
                    verbose = False):

    # 2020-04-23 WIC - removed the unused call to FUNC
    
    q = m2_in/m1_in
    period = period_in * constants.day
    m1 = m1_in * constants.M_Sun
    m2 = m2_in * constants.M_Sun
    a = (G*(m1+m2)*period**2/(4*np.pi**2))**(1./3.)
    #Tmatch = simulator_2.timeDelay([phi], m1_in=m1_in,m2_in=m2_in,period_in=period_in,inclination=inclination, \
    #                   omega=omega,eccentricity=eccentricity,radialVelocity=radialVelocity,rocheLobe=True,pseudo3D = True,SP_setting = 'egg')
    #Tmatch = Tmatch[0] #The effective radius is computed by matching a spherical model to this delay time.
    #print('pseudo 3D',Tmatch)
    Tmatch = delaySignal(plot = False,phase=phi,m1_in=m1_in,m2_in=m2_in,period_in=period_in,inclination=inclination,\
                         omega=omega,eccentricity=eccentricity,Q = 150, alpha = alpha)
    print('full 3D',Tmatch)
    R_eff = a - constants.c*Tmatch/(1-np.sin(inclination)*np.cos(2*np.pi*phi))
    R_egg = a*(1-eccentricity)*(.49*q**(.667))/(.6*q**(.667)+np.log(1+q**(.333)))
    R_plav = a*(1-eccentricity)*(.5+.227*np.log(q)/np.log(10))

    if verbose == True:
        print('separation = ' + str(round(a/constants.R_Sun,2))+str(' Solar Radii'))
        print('Eggleton Radius = ' + str(round(a/constants.R_Sun*(1-eccentricity)*(.49*q**(.667))/(.6*q**(.667)+np.log(1+q**(.333))),2))+str(' Solar Radii'))
        print('Plavec Radius = ' + str(round(a/constants.R_Sun*(1-eccentricity)*(.5+.227*np.log(q)/np.log(10)),2))+str(' Solar Radii'))
        print('matching delay = ' + str(Tmatch) + ' s')
        print('R effective = ' + str(round(R_eff/constants.R_Sun,2)) + str(' Solar Radii'))
    if plot == True:
        tt = np.linspace(0,1,50)
        
        T_egg = simulator_2.timeDelay(tt, m1_in=m1_in,m2_in=m2_in,period_in=period_in,inclination=inclination, \
                           omega=omega,eccentricity=eccentricity,radialVelocity=radialVelocity,rocheLobe=True,pseudo3D = False,SP_setting = 'egg') #Eggleton SP

        T_plav = simulator_2.timeDelay(tt, m1_in=m1_in,m2_in=m2_in,period_in=period_in,inclination=inclination, \
                           omega=omega,eccentricity=eccentricity,radialVelocity=radialVelocity,rocheLobe=True,pseudo3D = False,SP_setting = 'plav') #Plavec SP

        TP = simulator_2.timeDelay(tt, m1_in=m1_in,m2_in=m2_in,period_in=period_in,inclination=inclination, \
                           omega=omega,eccentricity=eccentricity,radialVelocity=radialVelocity,rocheLobe=True,pseudo3D = True,SP_setting = 'egg') #Pseudo 3D
        
        plt.plot(tt,T_egg,'-ks',label = 'SP Eggleton')
        plt.plot(tt,T_plav,'-ko', label = 'SP Plavec')
        plt.plot(tt,TP,'k-.', label = '3D')
        plt.legend()
        plt.show()
    if verbose == False and plot == False:
        return R_eff,R_egg,R_plav

def Reff_plot(phase = 0.5, N = 10, plotType = 'm2', variationRange = [.2,2], alpha = np.radians(0)):
    #Uses the default parameters of the function effectiveRadius.
    #These parameters should be set to the parameters of Sco X-1.
    #The variable specified by 'plotType' is then varied over
    #the specified range for N iterations
    
    vR = variationRange
    valVec = np.zeros(N)
    parVec = np.zeros(N)
    eggVec = np.zeros(N)
    plavVec = np.zeros(N)
    iterLen = (vR[1]-vR[0])/(N-1)
    if plotType == 'm2':
        XLABEL = r"Donor Mass (M$_\odot$)"
        for i in range(N):
            print(str(i+1)+'/'+str(N))
            m2_in = vR[0] + iterLen*i
            eff = effectiveRadius(m2_in=m2_in,phi=phase,alpha=alpha)
            valVec[i] = eff[0]/constants.R_Sun
            parVec[i] = m2_in
            eggVec[i] = eff[1]/constants.R_Sun
            plavVec[i] = eff[2]/constants.R_Sun
            print('x,y',parVec[i],valVec[i])
        #m,b = np.polyfit(parVec,valVec,1)
        #print('(m,b)',m,b)
        #xx = np.linspace(min(parVec),max(parVec),100)
        #yy = m*xx+b
        #plt.plot(xx,yy,'g--',label='linear fit')
    if plotType == 'inclination':
        XLABEL = r"Inclination (Deg.)"
        for i in range(N):
            print(str(i+1)+'/'+str(N))
            inclination = vR[0] + iterLen*i
            eff = effectiveRadius(inclination=np.radians(inclination),phi=phase,alpha=alpha)
            tt = np.linspace(0,1,100)
            yy = timeDelay(tt,inclination = np.radians(inclination))
            plt.plot(tt,yy,label=inclination)
            valVec[i] = eff[0]/constants.R_Sun
            parVec[i] = inclination
            eggVec[i] = eff[1]/constants.R_Sun
            plavVec[i] = eff[2]/constants.R_Sun
            print(parVec[i],valVec[i])
        plt.legend()
        plt.show(block=False)
        plt.figure() 
        
     
    plt.plot(parVec,eggVec,'sr--',label='Eggleton')
    plt.plot(parVec,plavVec,'or--',label='Plavec')
    plt.plot(parVec,valVec,'^k-',markersize=8,label='3D effective')
    plt.ylabel(r"Effective Donor Radius (R$_\odot$)")
    plt.xlabel(XLABEL)
    plt.title('phase = ' + str(phase) + r', $\alpha = $' +  str(round(np.degrees(alpha),2)) + r'$^\circ$')
    plt.legend()
    plt.show()
    

def delaySignalOrbit(N = 10, m1_in = 1.4, m2_in = .7, period_in = .787, \
                    inclination = np.radians(44.), omega = np.radians(90.), eccentricity = 0, \
                    binNumber = 100, Q = 120, alpha = np.radians(0), intKind = 'cubic', \
                     radialVelocity = False,u=.6, plot = True , p3d = True, outputData = False, \
                     # FUNC = FUNC_CORRECTION, \
                     FUNC = None, \
                     fast = False, verbose = False):

    # 2020-04-23 WIC - handle the case where FUNC_CORRECTION hasn't been set yet
    
    #Make sure that Q = 120 for high precision
    q = m2_in/m1_in
    period = period_in * constants.day
    m1 = m1_in * constants.M_Sun
    m2 = m2_in * constants.M_Sun
    a = (G*(m1+m2)*period**2/(4*np.pi**2))**(1./3.)
    if verbose == True:
        print('separation = ' + str(round(a/constants.R_Sun,2))+str(' Solar Radii'))
        print('Eggleton Radius = ' + str(round(a/constants.R_Sun*(1-eccentricity)*(.49*q**(.667))/(.6*q**(.667)+np.log(1+q**(.333))),2))+str(' Solar Radii'))
        print('Plavec Radius = ' + str(round(a/constants.R_Sun*(1-eccentricity)*(.5+.227*np.log(q)/np.log(10)),2))+str(' Solar Radii'))
    
    q = m2_in/m1_in
    T = np.zeros(N)
    phi = np.zeros(N)
    tt = np.linspace(0,1,10000)
    timeInitial = time.time()
    for i in range(N):
        #sys.stdout.write("%s/%s" % (i+1,N),end='\r')
        #sys.stdout.flush()
        #print("%s/%s" % (i+1,N), end = '\r', flush = True)
        if verbose == True:
            print("%s/%s" % (i+1,N))
        phi[i] = i/(N-1)
        if fast == False:
            T[i] = delaySignal(phase = phi[i], m1_in=m1_in,m2_in=m2_in,eccentricity=eccentricity,period_in=period_in,inclination=inclination,omega=omega, \
                        binNumber=binNumber,Q=Q,alpha=alpha,disk=True,plot=False,radialVelocity=radialVelocity,u=u)
        if fast == True:
            T[i] = delaySignalFast(phase = phi[i], m1_in=m1_in,m2_in=m2_in,eccentricity=eccentricity,period_in=period_in,inclination=inclination,omega=omega, \
                        binNumber=binNumber,Q=Q,alpha=alpha,disk=True,plot=False,radialVelocity=radialVelocity,u=u)
    timeFinal = time.time()
    print("Time Per Evaulation = %.3f s" % ((timeFinal-timeInitial)/N))
        
    f = scpint.interp1d(phi,T,kind=intKind)
    
    yy = f(tt)
    T1 = simulator_2.timeDelay(tt, m1_in=m1_in,m2_in=m2_in,period_in=period_in,inclination=inclination, \
                       omega=omega,eccentricity=eccentricity,radialVelocity=radialVelocity,rocheLobe=False,ellipticalCORR = True,pseudo3D = False)

    #xxT4, T4 = corrCoef(yy = T1, q = q, i = inclination)
    #print(np.shape(T4))
    #print(np.shape(tt))
    
    T2 = simulator_2.timeDelay(tt, m1_in=m1_in,m2_in=m2_in,period_in=period_in,inclination=inclination, \
                       omega=omega,eccentricity=eccentricity,radialVelocity=radialVelocity,rocheLobe=True,pseudo3D = False,SP_setting = 'egg') #Eggleton SP

    T3 = simulator_2.timeDelay(tt, m1_in=m1_in,m2_in=m2_in,period_in=period_in,inclination=inclination, \
                       omega=omega,eccentricity=eccentricity,radialVelocity=radialVelocity,rocheLobe=True,pseudo3D = False,SP_setting = 'plav') #Plavec SP

    ttt = np.linspace(0,1,100)
    CORR = np.zeros(len(ttt))
    # ~ TP = simulator_2.timeDelay(ttt, m1_in=m1_in,m2_in=m2_in,period_in=period_in,inclination=inclination, \
                       # ~ omega=omega,eccentricity=eccentricity,radialVelocity=radialVelocity,rocheLobe=True,pseudo3D = True, \
                               # ~ SP_setting = 'egg',alpha=alpha) #Pseudo 3D

    # 2020-04-23 this call to FUNC (the only one in this method) was
    # commented out. I have inserted some syntax to gracefully handle
    # the case where FUNC hasn't yet been set.

    #if FUNC is not None:
    #    for i in range(len(ttt)):
    #        CORR[i] = FUNC([q,inclination,ttt[i]])
    #        TP[i] = TP[i]*CORR[i]

    if plot == True:
        plt.figure(1)

    #A = 0
    #B = 0
    #for i in range(N):
    #    #print(int(np.ceil(phi[i]*(len(tt)-1))))
    #    A += T[i]*T1[int(np.floor(phi[i]*(len(tt)-1)))]
    #    B += (T1[int(np.floor(phi[i]*(len(tt)-1)))])**2
    #fHat = A/B
    #print("fHat = " + str(fHat))
        Arr_CM = np.zeros(N)
        A = 0
        B = 0
        for i in range(N):
            #print(int(np.ceil(phi[i]*(len(tt)-1))))
            indeX = int(np.floor(phi[i]*(len(tt)-1)))
            Arr_CM[i] = T1[indeX]
            A += T[i]*T1[indeX]
            B += (T1[indeX])**2
            #A += (T2[indeX]-T[i])*(T[i]-T1[indeX])
            #B += (T[i]-T1[indeX])**2
        #fHat = A/B
        #print("fHat = " + str(fHat))
        #Fcm = scpint.interp1d(tt,T1)
        #Fsp = scpint.interp1d(tt,T2)
        #print("check point 1")
        #def funcLinearCombination(tt,A,B):
        #    y = A*Fcm(tt)+B*Fsp(tt)
        #    return y
        #popt = LSFit(func=funcLinearCombination,X=phi,Y=T)
        #A = popt[0]
        #B = popt[1]
        #print("popt = " + str(popt))
        if plot == True:
            plt.xlabel("Phase")
            plt.ylabel("Time (s)")
            plt.scatter(phi,T,marker = 's',c='black', s = 10)
            plt.plot(tt,yy, 'g-',label=r'3D model')
            plt.plot(tt,T1, 'b:',label='CM model')
            plt.plot(tt,T2, 'r--',label='SP model (Egg)')
            plt.plot(tt,T3, 'r-.', label='SP model (Plav)')
            # ~ if p3d == True:
                # ~ plt.plot(ttt,TP, 'k-.',label='Pseudo 3D model')
            #plt.plot(tt,A*np.array(T1)+B*np.array(T2), 'k-.',label='SCALE{CM model}')
            #plt.plot(tt,fHat*np.array(T1), 'k-.',label='SCALE{CM model}')
            #plt.plot(tt,T4, 'k--',label='Corr')
            #plt.plot(tt,np.array(T2), 'r--',label='SP model')
            #plt.plot(tt,fHat*np.array(T1)/np.array(yy), 'r--',label='SP model')
            #plt.plot(tt,T2-np.array([1]*len(T2)), 'r--',label='SP Roche model')
            #plt.plot(tt,T2, 'r--',label='SP Roche model')
            #plt.plot(tt,T3, 'r:',label='SP Lagrange model')
    else:
        A = 0
        B = 0
        for i in range(N):
            #print(int(np.ceil(phi[i]*(len(tt)-1))))
            A += T[i]*T1[int(np.floor(phi[i]*(len(tt)-1)))]
            B += (T1[int(np.floor(phi[i]*(len(tt)-1)))])**2
        fHat = A/B
        #print("fHat = " + str(fHat))
        #if plot == True:
        #    plt.xlabel("Phase")
        #    plt.ylabel("Velocity (km/s)")
        #    plt.scatter(phi,T/1000,marker = 's',c='black', s = 10)
        #    plt.plot(tt,yy/1000, 'g-',label='3D model')
        #    plt.plot(tt,T1/1000, 'b:',label='CM model')
        #    plt.plot(tt,T2/1000, 'r:',label='SP model')
            #plt.plot(tt,fHat*np.array(T1)/1000, 'k-.',label='SCALE{CM model}')
            #plt.plot(tt,T2/1000, 'r--',label='SP model')
            #plt.plot(tt,T3/1000, 'r:',label='SP Lagrange model')
            #plt.plot(tt,(T2+2*T1+T3)/4, 'k--',label='(SP,CM) model')
            #plt.plot(tt,(T2+T1)/2000, 'k-.',label='avg(SP,CM) model')
        #N0 = .886
        #N1 = -1.132
        #N2 = 1.523
        #N3 = -1.892
        #N4 = .867
        #T4 = T1*(N0+N1*q+N2*q**2+N3*q**3+N4*q**4)/1000
        #if plot == True:
        #    plt.plot(tt,T4, 'k--',label='Munez Darias')
            #plt.plot(tt,(T1/abs(T1))*(T2*T1**2)**(1/3), 'k--',label='(SP,CM) model')
            #plt.plot(tt,(T3+T1)/2, 'k:',label='(SP Lagrange,CM) model')
                


            #centerShift = np.array([(T2[int(len(T2)/2)]-yy[int(len(yy)/2)])]*len(T1))
            #centerShift = np.array([(T1[int(len(T1)/2)]-(T2[int(len(T2)/2)]))]*len(T1))
            #T3 = np.array(T1)*(((.5*T1[int(len(T1)/2)]+.5*T2[int(len(T2)/2)]))/(T1[int(len(T1)/2)]))-.5*centerShift
            #T3 = np.array(T1)*0.6792909626482754
            #print((T2[int(len(T2)/2)]/T1[int(len(T1)/2)]))
            #plt.plot(tt,T3, 'k--',label='CM,SP Blend')
            #plt.plot(tt,(T2-yy),'g--',label='SP-3D')
            #plt.plot(tt,yy/T2,'r--',label='3D/SP')
            #plt.legend()
            #plt.figure(2)
            #symmetryCheck = np.zeros(int(round(len(yy)/2)))
            #for i in range(len(symmetryCheck)):
            #    symmetryCheck[i] = yy[i]-yy[len(yy)-1-i]
            #plt.plot(tt[:-int(len(tt)/2)],symmetryCheck)
            #print(np.shape(symmetryCheck[0]))
            #print("Central Shift  = %.3f" % (T2[int(len(T2)/2)]-yy[int(len(yy)/2)]))

            
    if plot == True:
        plt.legend()
        plt.show()
    if outputData == False:
        OUTPUT1 = (np.array(yy[int(.1*len(yy)):int(-.1*len(yy))])/np.array(T2[int(.1*len(T2)):int(-.1*len(T2))]))
        OUTPUT2 = tt[int(.1*len(tt)):int(-.1*len(tt))]
    else:
        OUTPUT1 = tt
        OUTPUT2 = yy
    
    return OUTPUT1,OUTPUT2

def delaySignalOrbit_diskAngles(N = 10, m1_in = 1.4, m2_in = .7, period_in = .787, \
                    inclination = np.radians(44.), omega = np.radians(90.), eccentricity = 0, \
                    binNumber = 100, Q = 120, alpha = np.radians(0), intKind = 'cubic',\
                    radialVelocity = False,u=.6, plot = True , p3d = True, cmap = plt.cm.cool):
    alphas_deg = np.arange(0,20,5)
    for k in range(6):
        phi = np.zeros(N)
        T = np.zeros(N)
        TP = np.zeros(N)
        tt = np.linspace(0,1,10000)
        timeInitial = time.time()
        for i in range(N):
            #sys.stdout.write("%s/%s" % (i+1,N),end='\r')
            #sys.stdout.flush()
            #print("%s/%s" % (i+1,N), end = '\r', flush = True)
            print("%s/%s" % (i+1,N))
            phi[i] = i/(N-1)
            T[i] = delaySignal(phase = phi[i], m1_in=m1_in,m2_in=m2_in,eccentricity=eccentricity,period_in=period_in,inclination=inclination,omega=omega, \
                            binNumber=binNumber,Q=Q,alpha=np.radians(alphas_deg[k]),disk=True,plot=False,radialVelocity=radialVelocity,u=u)
        TP = simulator_2.timeDelay(tt, m1_in=m1_in,m2_in=m2_in,period_in=period_in,inclination=inclination, \
                    omega=omega,eccentricity=eccentricity,radialVelocity=radialVelocity,rocheLobe=True,pseudo3D = True, \
                            SP_setting = 'egg',alpha=np.radians(alphas_deg[k])) #Pseudo 3D 
        timeFinal = time.time()
        print("Time Per Evaulation = %.3f s" % ((timeFinal-timeInitial)/N))
        f = scpint.interp1d(phi,T,kind=intKind)
        yy = f(tt)
        plt.scatter(phi,T,marker = 's',c='black', s = 10)
        plt.plot(tt,yy, color = cmap(alphas_deg[k]/max(alphas_deg)),label=r'$\alpha$ = ' + str(int(alphas_deg[k])) + r'$^{\circ{}}$')
        plt.plot(tt,TP,'g--')
        plt.legend()
        plt.show(block = False)
    
    

def correctionInclination(Ni ,Na ,i0Deg = 5, ifDeg = 90, alphaDeg_i = 0, alphaDeg_f = 18 ,corrFile='correction_', cmap = plt.cm.cool, \
                              m1_in = 1, m2_in = 1, period_in = .787, radialVelocity = False):

    
    #Ni is the number of inclination corrections, and Na is the number of alpha corrections
    #Generates the correction files for a given mass ratio q.

    
    
    A = np.zeros([Ni,10000])
    q = m2_in/m1_in

    # create arrays to store the phase, correction and inclination
    vIncl = np.zeros(Ni)
    vPhase= np.array([])
    aCorr = np.zeros([Na,Ni,7998])
    aCorr_i = np.array([])
    vAlpha = np.zeros(Na)
    
    #low = 0
    #high = 1
    #data = np.array([[low,high],[low,high]])
    #fig, ax = plt.subplots()
    #cax = ax.imshow(data, interpolation='nearest', cmap = cmap)
    #cbar = fig.colorbar(cax,ticks=[low, high], orientation='vertical')


    #sm = mpl.cm.ScalarMappable(cmap=cmap)
    #sm.set_array([i0Deg,ifDeg])
    #ticks = np.linspace(i0Deg,ifDeg,10)
    #plt.colorbar(sm,ticks=ticks)
    for l in range(Na):
        alphaDeg = alphaDeg_i + (alphaDeg_f-alphaDeg_i)*(l/(Na-1))
        print("alpha = " + str(alphaDeg))
        alpha = np.radians(alphaDeg)
        for k in range(Ni):
            inclination = i0Deg + (ifDeg-i0Deg)*((k)/(Ni-1))
            print("inclination = " + str(inclination))
            inclination = np.radians(inclination)
            corr_curve,tt = delaySignalOrbit(inclination = inclination,plot = False, \
                                             m1_in = m1_in, m2_in = m2_in, period_in = period_in, \
                                             radialVelocity = radialVelocity, alpha = alpha)
            corr_curve = corr_curve[1:-1]
            tt = tt[1:-1]
            
            #plt.plot(tt,corr_curve, c = cmap(k/N))

            # construct return arrays
            vIncl[k] = inclination
            if np.size(aCorr_i) < 1:
                vPhase = np.copy(tt)
                aCorr_i = np.copy(corr_curve)
            else:
                aCorr_i = np.vstack((aCorr_i, corr_curve))
                print(np.shape(aCorr_i))
        vAlpha[l] = alpha
        #puts the numbers into a rank 3 array
        aCorr[l,:,:] = aCorr_i
        #clears the temporary rank 2 array for repeated use
        aCorr_i = np.array([])
        
    #plt.show()
    print("correctionInclination INFO:", np.shape(vAlpha), np.shape(vIncl), np.shape(vPhase), np.shape(aCorr))
        
    # wrap our collection into a dictionary and serialize
    DOut = {'phase':vPhase, 'inclination':vIncl, 'alpha':vAlpha ,'corrections':aCorr}
    corrFile += ("q=" + str(round(q,1)))
    if radialVelocity == True:
        corrFile += ("RadVq=" + str(round(q,1)))
    corrFile += ".pickle"
    pickle.dump(DOut, open(corrFile, 'wb'))


def go(radialVelocity = False,Ni=30,Na=10,fast=True):
    timeInitial = time.time()
    timeTest = delaySignalOrbit(plot = False,fast=fast)
    timeFinal = time.time()
    delta_T = timeFinal-timeInitial
    Q = np.arange(.1,1.6,.1)
    runDurationEstimate = delta_T*Ni*Na*len(Q)
    
    hours = int(round(runDurationEstimate/3600))
    mins = int(round((runDurationEstimate-3600*hours)/60))
    print("Estimated Run Time = " + str(hours) + " hours " + str(mins) + " minutes")
    for i in Q:
        print("q = " + str(i))
        correctionInclination(Ni = Ni, Na = Na ,m2_in = i, radialVelocity = radialVelocity)
        
    

def loadData(corrFile='tmp_correction.pickle', cmap = plt.cm.cool, R = 10):
    
    file = open(corrFile,'rb')
    pickleFile = pickle.load(file)
    file.close()
    plt.figure(1)
    i0Deg = pickleFile[0]['inclination'][0]
    N = len(pickleFile['inclination'])
    ifDeg = pickleFile[0]['inclination'][N-1]
    #sm = mpl.cm.ScalarMappable(cmap=cmap)
    #sm.set_array([np.degrees(i0Deg),np.degrees(ifDeg)])
    #ticks = np.linspace(np.degrees(i0Deg),np.degrees(ifDeg),10)
    #plt.colorbar(sm,ticks=ticks)

    #def func(x,A,B,C,D,E):
    #    y = A*x**4+B*x**3+C*x**2+D*x+E
    #    return y

    #def func(x,A,B,C):
    #    B = .5
    #    y = -A+B*np.sqrt(1-((x-0.5)/C)**2)
    #    #y = A*np.sin(B*x+C)
    #    return y
    YY = np.zeros([N,len(pickleFile['phase'])])
    POPT = np.zeros([N,3])
    for i in range(N):
        plt.plot(pickleFile['phase'],pickleFile[0]['corrections'][i], c = cmap(i/N))
        #popt = LSFit(func = func, X = pickleFile['phase'][::R], Y = pickleFile['corrections'][i][::R])
        #POPT[i] = popt
        #print("popt",popt)
        #YY[i] = func(pickleFile['phase'],*popt)
        #plt.plot(pickleFile['phase'],YY[i], color=cmap(i/N), linestyle = 'dashed')
    #plt.figure(2)
    #sm = mpl.cm.ScalarMappable(cmap=cmap)
    #sm.set_array([np.degrees(i0Deg),np.degrees(ifDeg)])
    #ticks = np.linspace(np.degrees(i0Deg),np.degrees(ifDeg),10)
    #plt.colorbar(sm,ticks=ticks)
    #for i in range(N):
    #    plt.plot(pickleFile['phase'],pickleFile['corrections'][i]-YY[i], color = cmap(i/N))
    #NAMES = ['A','B','C']
    #for i in range(3):
    #    MEAN = np.mean(POPT[:,i])
    #    print("Mean(" + NAMES[i] + ") = " + str(MEAN))
    #    plt.figure(NAMES[i])
    #    plt.title(NAMES[i])
    #    plt.plot(np.degrees(pickleFile['inclination']),POPT[:,i])
    #print("A_slope = " + str((POPT[:,0][len(POPT[:,0])-1]-POPT[:,0][0])/np.radians(20)))
    #print("A_10 = " + str(POPT[:,0][0]))
    #print("C_slope = " + str((POPT[:,2][len(POPT[:,2])-1]-POPT[:,2][0])/np.radians(20)))
    #print("C_10 = " + str(POPT[:,2][0]))
    
    return pickleFile, POPT

#Deprecated: This function does not contain alpha! use the newer version in simulator_2.py
def configureData(R = 50):
    Q = [0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5]
    #DATA = np.zeros([160+160*18+10*160*18,3])
    #CORR = np.zeros([160+160*18+10*160*18])
    CORR = np.zeros([11,18,7998])
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
    
    
    
def LSFit(func, X=np.array([]), Y=np.array([]), bounds = ([-15,0,0],[.1,10,10]), pGuess = [0,1,1]):

    

    #bounds = ([0,0],[2,2])

    #pGuess = [1,1]


    popt, pcov = opt.curve_fit(func,X,Y,bounds=bounds, p0=pGuess)
    return popt

def corrCoef(yy,q,i,xx = 0):
    if xx == 0:
        xx = np.linspace(0,1,len(yy))
    def g1(ic):
        if ic >= 0 and ic <= np.pi/12:
            y = 1
        if ic > np.pi/12 and ic <= np.pi/4:
            y = 1-6/np.pi*(ic-np.pi/12)
        if ic > np.pi/4:
            y = 0
        return y
            
    def g2(ic):
        if ic >= np.pi/12 and ic <= 5*np.pi/12:
                y = 1-6/np.pi*abs(ic-np.pi/4)
        else:
            y = 0
        return y

    def g3(ic):
        if ic >= 5*np.pi/12 and ic <= np.pi/2:
            y = 1
        if ic > np.pi/4 and ic <= 5*np.pi/12:
            y = 1+6/np.pi*(ic-5*np.pi/12)
        if ic < np.pi/4:
            y = 0
        return y
    G1 = g1(ic=i)
    G2 = g2(ic=i)
    G3 = g3(ic=i)
    print(G1+G2+G3)
    corr_1 = 0
    corr_2 = 0
    corr_3 = 0
    yy3 = np.zeros(len(yy))
    if G1 > 0:
        A = .2308*q-.3244
        B = .08167*q+.4189
        C = .4134
        corr_1 = -A+B*np.sqrt(abs(1-((xx-0.5)/C)**2))
        yy3 += G1*corr_1*yy
    if G2 > 0:
        A = 0
        B = -.1573*q+.6993
        C = -.1129*q+3.598-.4371*i
        corr_2 = -A+B*np.sqrt(abs(1-((xx-0.5)/C)**2))
        yy3 += G2*corr_2*yy
    if G3 > 0:
        A = -.1045*i+.1608*q-.1468
        B = .5
        C = -1.268*i+1.301
        corr_3 = -A+B*np.sqrt(abs(1-((xx-0.5)/C)**2))
        yy3 += G3*corr_3*yy
        
    return xx,yy3
    

def orbitPlot(m1_in = 1.4, m2_in = .7, period_in = .787, omega = np.radians(90.), eccentricity = 0, N = 10):
    period = period_in*constants.day
    m1 = m1_in * constants.M_Sun
    m2 = m2_in * constants.M_Sun
    a = (G*(m1+m2)*period**2/(4*np.pi**2))**(1./3.)
    theta = np.linspace(0,2*np.pi,10000)
    separation = a*(1-eccentricity**2)/(1+eccentricity*np.cos(theta))
    yy1 = m2_in/(m2_in+m1_in)*separation*np.sin(theta)
    xx1 = m2_in/(m2_in+m1_in)*separation*np.cos(theta)

    yy2 = -m1_in/(m2_in+m1_in)*separation*np.sin(theta)
    xx2 = -m1_in/(m2_in+m1_in)*separation*np.cos(theta)
    fig, ax = plt.subplots()
    
    ax.plot(xx1,yy1,xx2,yy2)
    A1 = []
    B1 = []
    A2 = []
    B2 = []
    n = []
    for i in range(N):
        A1.append(xx1[int(i*(len(xx1)-1)/N)])
        A2.append(yy1[int(i*(len(xx1)-1)/N)])
        B1.append(xx2[int(i*(len(xx1)-1)/N)])
        B2.append(yy2[int(i*(len(xx1)-1)/N)])
        n.append(str(i))
    plt.scatter(A1,A2)
    plt.scatter(B1,B2)
    #print(A)
    for i, txt in enumerate(n):
        ax.annotate(txt, (A1[i],A2[i]))
    for i, txt in enumerate(n):
        ax.annotate(txt, (B1[i],B2[i]))
    plt.show()
def generateRochePotential(m1_in = 1.4,m2_in = .7,period_in = .787,FAC=3,N=280, plot = True):
    m1 = m1_in*constants.M_Sun
    m2 = m2_in*constants.M_Sun
    q = m2_in/m1_in
    P = period_in*constants.day
    omega = 2*np.pi/P
    G = constants.G
    S = (G*(m1+m2)*P**2/(4*np.pi**2))**(1/3)
    radiusDonor = S*(.49*q**(.667))/(.6*q**(.667)+np.log(1+q**(.333)))
    rPre = np.linspace(0, 1.5*S, 500)
    thetaPre = np.linspace(0, 2*np.pi, 500)
    r, theta = np.meshgrid(rPre, thetaPre)
    phi = -G*m1/(np.sqrt((m2/(m1+m2)*S+r*np.cos(theta))**2+(r*np.sin(theta))**2))- \
              G*m2/(np.sqrt((m1/(m1+m2)*S-r*np.cos(theta))**2+(r*np.sin(theta))**2))- \
              .5*omega**2*r**2
    if plot == False:
        return phi
    baseLine = phi[0][0]
    #print(baseLine)
    for i in range(len(r)):
        for j in range(len(theta)):
            if abs(phi[i][j]) > FAC*abs(baseLine):
                phi[i][j] = FAC*baseLine
                
    #phi = -G*m1/(np.sqrt((m2/(m1+m2)*S+r*np.cos(theta))**2+(r*np.sin(theta))**2))-G*m2/(np.sqrt((m1/(m1+m2)*S-r*np.cos(theta))**2+(r*np.sin(theta))**2))
    X, Y = r*np.cos(theta), r*np.sin(theta)
    fig = plt.figure()
    ax = fig.add_subplot(111,projection = '3d')
    ax.plot_surface(X, Y, phi, cmap=plt.cm.cool)
    #ax.set_zlim(-.1*10**13, 0)
    ax.set_xlabel(r'$x(m)$')
    ax.set_ylabel(r'$y(m)$')
    ax.set_zlabel(r'$\phi$')
    #plt.figure()
    #plt.imshow(phi, cmap='hot', interpolation='nearest')
    #plt.imshow(phi,cmap = cm.plasma)
    plt.figure()
    circle = plt.Circle((m1/(m1+m2)*S,0),radiusDonor,color='b', fill=True, alpha = .5)
    plt.gcf().gca().add_artist(circle)
    plt.contour(X,Y,phi,N,zorder=1)
    plt.show(block = False)
def polarRocheLobe(m1_in = 1.4,m2_in = 0.7,period_in = .787,Q = 30,N = 30,cmap = cm.jet,plot = True):
    m1 = m1_in*constants.M_Sun
    m2 = m2_in*constants.M_Sun
    q = m2_in/m1_in
    P = period_in*constants.day
    omega = 2*np.pi/P
    G = constants.G
    a = (G*(m1+m2)*P**2/(4*np.pi**2))**(1/3)
    radiusDonor = a*(.49*q**(.667))/(.6*q**(.667)+np.log(1+q**(.333)))
    L2 = (a*(.5+.227*np.log(q)/np.log(10)))
    #print("L2 = " + str(L2))
    rr = m1/(m1+m2)*a-L2
    rcm = m1_in/(m1_in+m2_in)*a
    Uconst = 0
    def potentialFunc(p2,theta2):
        #phiU = -G*m1/(np.sqrt((m2/(m1+m2)*S+r*np.cos(theta))**2+(r*np.sin(theta))**2))- \
        #      G*m2/(np.sqrt((m1/(m1+m2)*S-r*np.cos(theta))**2+(r*np.sin(theta))**2))- \
        #      .5*omega**2*r**2
        r2 = m1_in/(m1_in+m2_in)*a
        phiU = -G*m1/np.sqrt(a**2+p2**2+2*a*p2*np.cos(theta2))-G*m2/p2 \
                   -.5*omega**2*(r2**2+p2**2+2*r2*p2*np.cos(theta2))
        return phiU-Uconst
    Uconst = potentialFunc(L2,np.pi)
    #print(Uconst)
    def potentialFuncINV(psi):
        psi = np.add(psi,np.pi)
        r = np.zeros(len(psi))
        a1 = L2/1000
        b1 = L2
        for i in range(len(psi)):
            r[i] = opt.brentq(potentialFunc,a1,b1,args=(psi[i]),maxiter=100)
        return r
    theta = np.linspace(np.radians(0),np.radians(360),Q)
    #print(theta)\\
    #theta = [np.radians(181)]
    R_Eggleton = [radiusDonor]*len(theta)
    R_Lagrange = [L2]*len(theta)
    r = potentialFuncINV(theta)
    #plt.scatter(r*np.cos(theta),r*np.sin(theta),color = 'blue',marker = 's')


    Y = r*np.sin(theta)
    X = r*np.cos(theta)
    YE = R_Eggleton*np.sin(theta)
    XE = R_Eggleton*np.cos(theta)
    xx,yy,xxE,yyE = [],[],[],[]
    for i in range(len(Y)):
        if Y[i] > 0:
            yy.append(Y[i])
            xx.append(X[i])
        if YE[i] > 0:
            yyE.append(YE[i])
            xxE.append(XE[i])
    xx = np.array(xx)
    yy = np.array(yy)
    xxE = np.array(xxE)
    yyE = np.array(yyE)
    #plt.figure()
    plt.scatter(xx,yy)
    plt.scatter(xxE,yyE)
    L = max(xx)-min(xx)
    LE = max(xxE)-min(xxE)
    V,A,VE,AE = 0,0,0,0
    for k in range(len(xx)-1):
        A += 2*yy[k]*(xx[k]-xx[k+1])
        V += yy[k]**2*np.pi*(xx[k]-xx[k+1])
        AE += 2*yyE[k]*(xxE[k]-xxE[k+1])
        VE += yyE[k]**2*np.pi*(xxE[k]-xxE[k+1])
    print("A =",A)
    print("AE =",AE)
    print("V =",V)
    print("VE =",VE)
    
    #print("total volume = " + str(totV))
    #print("total area = " + str(totA))
    #print("total volume (Eggleton) = " + str(totVE))
    #print("total area (Eggleton) = " + str(totAE))

    V_Eggleton = (4/3)*np.pi*(radiusDonor)**3
    print("V Eggleton = " + str(V_Eggleton))
    A_Eggleton = np.pi*(radiusDonor)**2
    print("A Eggleton = " + str(A_Eggleton))
    #plt.scatter([1.379*10**9],[0],label = 'Lagrange Point (L1)')
    #plt.scatter([a],[0],label = 'Compact Object')
    #plt.scatter([rcm],[0],label = 'CM')
    #plt.scatter([0],[0])
    plt.plot(r*np.cos(theta),r*np.sin(theta),'k-',label = 'Roche Lobe')
    plt.plot(R_Eggleton*np.cos(theta),R_Eggleton*np.sin(theta),'g--', label = 'Eggleton Radius')
    plt.legend()
    #plt.plot(R_Lagrange*np.cos(theta),R_Lagrange*np.sin(theta),'k--')


    
#def polar3DRocheLobe(phase, \
#                  m1_in=1.4, m2_in=0.7, period_in=.787, eccentricity = 0,\
#                  inclination=np.radians(44.), omega=np.radians(90.), cmap = cm.cool, plot = True, Q = 25, RETURN = False):
#    m1 = m1_in * constants.M_Sun
#    m2 = m2_in * constants.M_Sun
#    q = m2_in/m1_in
#    period = period_in * constants.day
#    G = constants.G
#    a = (G*(m1+m2)*period**2/(4*np.pi**2))**(1./3.)
#    S = a
#    radiusDonor = a*(.49*q**(.667))/(.6*q**(.667)+np.log(1+q**(.333)))
#    separation = timeDelay([phase],m1_in=m1_in,m2_in=m2_in,eccentricity=eccentricity,period_in=period_in,inclination=inclination,omega=omega,separationReturn=True)
#    separation = separation[0]
#    theta_ORBIT = 2*np.pi*phase
#    
#    (n, m) = (Q, Q)
#    # Meshing a unit sphere according to n, m 
#    theta = np.linspace(0, 2*np.pi, num=n, endpoint=False)
#    #phi = np.linspace(np.pi * (-0.5 + 1./(m+1)), np.pi*0.5, num=m, endpoint=False)
#    phi = np.linspace(0, np.pi*0.5, num=m, endpoint=False)
#    theta, phi = np.meshgrid(theta, phi)
#    theta, phi = theta.ravel(), phi.ravel()
#    #theta = np.append(theta, [0.]) # Adding the north pole...
#    #phi = np.append(phi, [np.pi*0.5])
#    mesh_x, mesh_y = ((np.pi*0.5 - phi)*np.cos(theta), (np.pi*0.5 - phi)*np.sin(theta))
#    triangles = mtri.Triangulation(mesh_x, mesh_y).triangles
#    
#    
#    L2 = S*(.5+.227*np.log(m2_in/m1_in)/np.log(10))
#    #print("L2 = " + str(L2))
#    rr = m1/(m1+m2)*S-L2
#    Uconst = 0
#    #print(Uconst)
#    def potentialFunc(p2,theta2):
#        #phiU = -G*m1/(np.sqrt((m2/(m1+m2)*S+r*np.cos(theta))**2+(r*np.sin(theta))**2))- \
#        #      G*m2/(np.sqrt((m1/(m1+m2)*S-r*np.cos(theta))**2+(r*np.sin(theta))**2))- \
#        #      .5*omega**2*r**2
#        r2 = m1_in/(m1_in+m2_in)*S
#        phiU = -G*m1/np.sqrt(S**2+p2**2+2*S*p2*np.cos(theta2))-G*m2/p2- \
#                   .5*omega**2*np.sqrt(r2**2+p2**2+2*r2*p2*np.cos(theta2))
#        return phiU-Uconst
#    Uconst = potentialFunc(L2,np.pi)
#    def potentialFuncINV(psi):
#       psi = np.add(psi,np.pi)
#        r = np.zeros(len(psi))
#        a1 = L2/100
#       b1 = L2
#        for i in range(len(psi)):
#            r[i] = opt.brentq(potentialFunc,a1,b1,args=(psi[i]),maxiter=100)
#        return r
#    (n, m) = (Q, Q)
#    # Meshing a unit sphere according to n, m 
#    theta = np.linspace(0, 2*np.pi, num=n, endpoint=False)
#    #phi = np.linspace(np.pi * (-0.5 + 1./(m+1)), np.pi*0.5, num=m, endpoint=False)
#    if plot == False:
#        phi = np.linspace(0, np.pi*0.5, num=m, endpoint=False)
#    if plot == True:
#        phi = np.linspace(-np.pi*0.5, np.pi*0.5, num=m, endpoint=False)
#    theta, phi = np.meshgrid(theta, phi)
#    theta, phi = theta.ravel(), phi.ravel()
#    if plot == True:
#        mesh_x, mesh_y = ((np.pi*0.5 - phi)*np.cos(theta), (np.pi*0.5 - phi)*np.sin(theta))
#        triangles = mtri.Triangulation(mesh_x, mesh_y).triangles
#    psi = np.radians(90) - phi
#    alpha = theta
#    Rfunc = np.array(potentialFuncINV(psi))
#    RfuncPlot = Rfunc/max(Rfunc)
#    
#    x, y, z = RfuncPlot*np.cos(phi)*np.cos(theta), RfuncPlot*np.cos(phi)*np.sin(theta), RfuncPlot*np.sin(phi)
#
#    #vals = np.array(np.sqrt(Rfunc**2+separation**2-2*Rfunc*separation*np.cos(psi))+\
#    #        -separation*np.sin(inclination)*np.sin(omega+theta_ORBIT)+\
#    #        Rfunc*np.cos(psi)*np.sin(inclination)*np.sin(omega+theta_ORBIT)+\
#    #       -Rfunc*np.sin(psi)*np.cos(alpha)*np.cos(inclination)+\
#    #       -Rfunc*np.sin(psi)*np.sin(alpha)*np.sin(inclination)*np.cos(omega+theta_ORBIT))
#    #vals = vals/constants.c
#    
#    if plot == True:
#        #colors = np.mean(vals[triangles], axis=1)
#
#        # Plotting
#        fig = plt.figure()
#        ax = fig.add_subplot(111, projection='3d')
#        #ax = fig.gca(projection='3d')
#
#        ax.set_xlim3d(-1,1)
#        ax.set_ylim3d(-1,1)
#        ax.set_zlim3d(-1,1)
#
#        #cmap = plt.get_cmap('Blues')
#        cmap = cm.cool
#
#        #Calculation of time delay at many points over the surface of the star
#        colors = np.mean(vals[triangles], axis=1)
#
#
#        sm = mpl.cm.ScalarMappable(cmap=cmap)
#        sm.set_array([min(colors),max(colors)])
#        ticks = np.linspace(min(colors),max(colors),10)
#        plt.colorbar(sm,ticks=ticks)
#        soa = np.array([[0, 0, 1, 0, 0, 1], [0, 0, 1, np.cos(inclination), np.sin(inclination)*np.cos(theta_ORBIT+omega), -np.sin(inclination)*np.sin(theta_ORBIT+omega)]])
#        X, Y, Z, U, V, W = zip(*soa)
#        labels = ['Accretor', 'Observer']
#        for i in range(len(labels)):
#            ax.text(X[i]+U[i], Y[i]+V[i], Z[i]+W[i], labels[i])
#        quiv = ax.quiver(X, Y, Z, U, V, W)
#
#        
#
#        #soa = np.array([[0, 0, 1, 0, 0, 1], [0, 0, 1, np.cos(inclination), np.sin(inclination)*np.cos(theta_ORBIT+omega), -np.sin(inclination)*np.sin(theta_ORBIT+omega)]])
#        #soa = np.array([[0, 0, 1, 0, 0, 1], [np.cos(inclination), np.sin(inclination)*np.cos(theta_ORBIT+omega), -np.sin(inclination)*np.sin(theta_ORBIT+omega), np.cos(inclination), np.sin(inclination)*np.cos(theta_ORBIT+omega), -np.sin(inclination)*np.sin(theta_ORBIT+omega)]])
#        #X, Y, Z, U, V, W = zip(*soa)
#        #labels = ['Accretor', 'Observer']
#        #for i in range(len(labels)):
#        #    ax.text(X[i]+U[i], Y[i]+V[i], Z[i]+W[i], labels[i])
#        #quiv = ax.quiver(X, Y, Z, U, V, W)
#        #data = np.array([[low,high],[low,high]])
#        #cax = ax.imshow(data, interpolation='nearest', cmap = cmap)
#        #cbar = fig.colorbar(cax, ticks=[low, high], orientation='vertical')
#        
#        triang = mtri.Triangulation(x, y, triangles)
#        collec = ax.plot_trisurf(triang, z, cmap=cmap, shade=False, linewidth=0.)
#        collec.set_array(colors)
#        collec.autoscale()
#        plt.show(block = False)
#    if RETURN == True:
#        return psi, alpha, np.array(vals)

def delaySignal_burst(phase = .5, m1_in = 1.4, m2_in = .7, period_in = .787, \
                    inclination = np.radians(44.), eccentricity = 0, omega = np.radians(90.), alpha = np.radians(0),\
                    delta = .2, Q = 120, plotAll = True, disk = True, intKind = 'quadratic', \
                    radialVelocity = False, u = .6, pseudo3DTEST = True, outputData = False, timeScale = 55, Nevals = 50):
    
    x_burst = np.linspace(-5,30,Nevals)
    y_burst = xRayBurstModel(x_burst)

    
    plt.figure(500)
    plt.plot(x_burst,y_burst,'r-')
    plt.scatter(x_burst,y_burst)
    plt.xlabel("Time (s)")
    plt.ylabel("Fractional Intensity")
    plt.title("X-ray burst")



    if radialVelocity == True:
        sigFunc = genRadialVelocityMap
    else:
        sigFunc = genTimeDelayMap
    
    #if plot == True:
    #    sigFunc(phase,m1_in=m1_in,m2_in=m2_in,period_in=period_in,eccentricity=eccentricity,\
    #                                        inclination=inclination,omega=omega,plot = True,Q=25,RETURN = False)
    #    genApparentIntensityMap(phase,m1_in=m1_in,m2_in=m2_in,period_in=period_in,eccentricity=eccentricity,\
    #                                        inclination=inclination,omega=omega,plot = True,Q=25,RETURN = False,disk=disk,diskShieldingAngle=diskShieldingAngle,u=u)


    #Generates separate reprocessing curves for each instant in the burst.
    #These separate curves are then summed according to their respective burst intensities

    plt.figure(501)
    YY_ALL = 0
    for zeta in range(len(x_burst)):
        print(str(zeta) + '/' + str(len(x_burst)))
        psiECHO, alphaECHO, T = sigFunc(phase,m1_in=m1_in,m2_in=m2_in,eccentricity=eccentricity,period_in=period_in,\
                                                inclination=inclination,omega=omega,plot = False,Q=Q,RETURN = True)
        psiINT, alphaINT, I = genApparentIntensityMap(phase,m1_in=m1_in,m2_in=m2_in,eccentricity=eccentricity,period_in=period_in,\
                                                inclination=inclination,omega=omega,plot = False,Q=Q,RETURN = True,\
                                                           disk=disk,alpha = alpha,u=u)
        #Scale intensity by instantaneous burst intensity
        I_prev = I
        I = y_burst[zeta]*I_prev
        #print(zeta,y_burst[zeta])
        #Add time that has lapsed from t=0 (start of burst) to echo delay time
        T += x_burst[zeta]

        #delta is the temporal size of each bin
        if radialVelocity == True:
            binNumber = np.ceil((max(T)-min(T))/delta)
        else:
            binNumber = int(np.ceil(max(T)/delta))

        #print("binNumber",binNumber)
        
        BIN = [[]]*binNumber
        
        for i in range(len(T)):
            if radialVelocity == True:
                binIndex = int(np.ceil((T[i]-min(T))/delta))
            else:
                binIndex = int(np.ceil((T[i])/delta))
            if binIndex > binNumber:
                binIndex = binNumber
            BIN[binIndex-1] = BIN[binIndex-1] + [i]
        if radialVelocity == True:
            delayList = np.arange(min(T),max(T),delta) + delta/2
        else:
            delayList = np.arange(0,max(T),delta) + delta/2
        if len(delayList) > binNumber:
            delayList = delayList[:-1]
        intensityList = np.zeros(binNumber)
        for i in range(binNumber):
            for j in range(len(BIN[i])):
                intensityList[i] += I[BIN[i][j]]
        if radialVelocity == True:
            deg = 6
            A = np.polyfit(delayList,intensityList,deg = deg)
            #print(A)
            def f(t,power = deg):
                R = 0
                for i in range(power+1):
                    R += A[i]*t**(power-i)
                return R
        else:
            delayList = np.append(delayList,[timeScale])
            #print(delayList)
            intensityList = np.append(intensityList,[0]) #This puts a zero intensity point at 55 s delay
            #print(intensityList)
            f = scpint.interp1d(delayList,intensityList,kind=intKind)
        #print(min(delayList),max(delayList))
        tt = np.linspace(min(delayList),max(delayList),500)
        yy = f(tt)
        if np.size(YY_ALL) == 1:
            YY_ALL = yy
        else:
            YY_ALL += yy
        if plotAll == True:
            delaySIM2 = np.array(simulator_2.timeDelay([phase],m1_in=m1_in,m2_in=m2_in,eccentricity=eccentricity,period_in=period_in,inclination=inclination, \
                                               omega=omega,radialVelocity=radialVelocity, pseudo3D = False))
            #Add time that has lapsed from t=0 (start of burst) to echo delay time
            delaySIM2 += x_burst[zeta]
            #print("delaySIM2",delaySIM2)
            if radialVelocity == True:
                T = T/1000
                delayList = np.array(delayList)/1000
                tt = tt/1000
                delaySIM2 = delaySIM2/1000
            plt.axvline(x=delaySIM2,c='black')
            plt.scatter(delayList,intensityList, s=50, facecolors='none', edgecolors='b')
            plt.plot(tt,yy,'g-')
            if radialVelocity == True:
                plt.xlabel("Velocity (km/s)")
            else:
                plt.xlabel("Time (s)")
            plt.ylabel("Intensity (Arbitrary Units)")
            plt.show(block = False)
        #output = delayList[np.argmax(intensityList)]
        if max(yy) < 1000:
            output = 0.0001
            print("V = 0, Intensity Too Low")
        else:
            output = tt[np.argmax(yy)]
    YY_ALL = YY_ALL/max(YY_ALL) #This sets the maximum value to 1
    #YY_ALL is the total response from the burst

    arguments = [phase,m1_in,m2_in,period_in,inclination,eccentricity,omega,alpha]
    TT_impulse, YY_impulse = delaySignal(*arguments,plot = False,outputData = True)
    YY_impulse = YY_impulse/max(YY_impulse)
    
    
    plt.figure(502)
    plt.plot(tt,YY_ALL,'g-',label='Echo Response')
    delaySIM2 = np.array(simulator_2.timeDelay([phase],m1_in=m1_in,m2_in=m2_in,eccentricity=eccentricity,period_in=period_in,inclination=inclination, \
                                               omega=omega,radialVelocity=radialVelocity, pseudo3D = False))
    plt.axvline(x=delaySIM2,c='black',label='Pseudo 3D')
    plt.plot(x_burst,y_burst,'r-',label='X-ray burst')
    plt.scatter(x_burst,y_burst)

    plt.plot(TT_impulse,YY_impulse,'k:',label=r'$\chi(t)$')
    
    plt.legend()
    plt.show()


def xRayBurstModel(xx,a = 1, b = .3, c = 0):
        Y = np.zeros(len(xx))
        for i in range(len(xx)):
            if xx[i] > 0:
                Y[i] = a*np.exp(-1*b*xx[i])+c
            else:
                Y[i] = (a+c)/(-min(xx))*(xx[i]-min(xx))
        Y = Y/max(Y)
        return Y



def varyPlots(N = 5, parameter = 0,Q = 10,m1=1.4,m2=.7, \
              period = .787,i=np.radians(44), \
              omega=np.radians(90), eccentricity = 0, \
              cmap = cm.cool):
    fig, ax = plt.subplots()
    plt.clf()
    parameterList = [r"$m_1$ (M$_\odot)$",r"$m_2$ (M$_\odot)$","P (days)",r"i$(^\circ)$",r"$\omega(^\circ)$","e"]
    Vals = [m1,m2,period,i,omega,eccentricity]
    parameterRanges = [[.1,3],[.1,3],[0.1,5],[0,.5*np.pi],[0,.5*np.pi],[.01,.99]]
    for k in range(N):
        rgba_colors = np.zeros(4)
        rgba_colors[0] = .9*k/N
        rgba_colors[2] = .8
        rgba_colors[3] = 1
        rgba_colors[1] = 0
        Vals[parameter] = parameterRanges[parameter][0] + (parameterRanges[parameter][1]-parameterRanges[parameter][0])*k/N
        x,y = delaySignalOrbit(N = 10,\
                               m1_in = Vals[0], m2_in = Vals[1],\
                               period_in = Vals[2], inclination = Vals[3],\
                               omega = Vals[4], eccentricity = Vals[5],\
                               plot = False, outputData = True, Q = Q)
        plt.plot(x,y,color = cmap(k/N),linewidth = 1)
    low = parameterRanges[parameter][0]
    high = parameterRanges[parameter][1]
    if parameter == 3 or parameter == 4:
        low = np.degrees(parameterRanges[parameter][0])
        high = np.degrees(parameterRanges[parameter][1])
        
    data = np.array([[low,high],[low,high]])
    #data = np.linspace(-1,1,62500)
    #data = np.zeros([250,250])
    cax = ax.imshow(data, interpolation='nearest', cmap = cmap)
    cbar = fig.colorbar(cax, ticks=[low, high], orientation='vertical')    
    cbar.ax.set_title(parameterList[parameter],y=1.05)
    plt.xlabel("Orbital Phase")
    plt.ylabel("Echo Delay (s)")
    plt.show()
def m2MidPointVary(N = 5, parameter = 0,Q = 10,m1=1.4,m2=.7, \
              period = .787,i=np.radians(44), \
              omega=np.radians(90), eccentricity = 0, \
              cmap = cm.cool):
    fig, ax = plt.subplots()
    plt.clf()
    parameterList = [r"$m_1$",r"$m_2$","P (days)",r"i$(^\circ)$",r"$\omega(^\circ)$","e"]
    Vals = [m1,m2,period,i,omega,eccentricity]
    parameterRanges = [[.1,3],[.01,3],[0.1,5],[0,.5*np.pi],[0,.5*np.pi],[.01,.99]]
    midPoints = np.zeros(N)
    values = np.zeros(N)
    for k in range(N):
        Vals[parameter] = parameterRanges[parameter][0] + (parameterRanges[parameter][1]-parameterRanges[parameter][0])*k/N
        midPoints[k] = delaySignal(m1_in = Vals[0], m2_in = Vals[1],\
                               period_in = Vals[2], inclination = Vals[3],\
                               omega = Vals[4], eccentricity = Vals[5],\
                               plot = False, Q = Q)
        values[k] = Vals[parameter]
        if parameter == 3 or parameter == 4:
            values[k] = np.degrees(values[k])
    plt.scatter(values,midPoints)
    plt.plot(values,midPoints,'g--')
    plt.xlabel(parameterList[parameter])
    plt.ylabel("Max Echo Delay(s)")
    plt.show()

    plt.figure()
    plt.plot(values,midPoints,'g--')
    plt.xlabel(parameterList[parameter])
    plt.ylabel("Max Echo Delay(s)")
    plt.show()
def figure_1(N = 10, m1_in = 1.4, m2_in = .7, period_in = .787, \
                    inclination = np.radians(44.), omega = np.radians(90.), eccentricity = 0, \
                    binNumber = 100, Q = 120, alpha = np.radians(0), intKind = 'cubic', \
                     radialVelocity = True,u=.6, plot = True , p3d = True, outputData = False):
    q = m2_in/m1_in
    #N0 = .888
    #N1 = -1.291
    #N2 = 1.541
    #N3 = -1.895
    #N4 = .861
    N0 = .886
    N1 = -1.132
    N2 = 1.523
    N3 = -1.892
    N4 = .867
    tt = np.linspace(0,1,500)
    #CM Model
    yy_1 = timeDelay(tt,m1_in=m1_in,m2_in=m2_in,eccentricity=eccentricity,period_in=period_in,inclination=inclination,omega=omega,\
                   radialVelocity = radialVelocity, rocheLobe = False, pseudo3D = False)
    plt.plot(tt,yy_1/1000,'b:',label='CM model')
    #Munez Darias
    yy_2 = yy_1*(N0+N1*q+N2*q**2+N3*q**3+N4*q**4)
    print(N0+N1*q+N2*q**2+N3*q**3+N4*q**4)
    plt.plot(tt,yy_2/1000,'k-',label='Munez Darias')

    yy_3 = timeDelay(tt,m1_in=m1_in,m2_in=m2_in,eccentricity=eccentricity,period_in=period_in,inclination=inclination,omega=omega,\
                   radialVelocity = radialVelocity, rocheLobe = True, pseudo3D = True)
    plt.plot(tt,yy_3/1000,'r--',label='SP model')
    plt.xlabel("Phase")
    plt.ylabel(r"Velocity (km/s)")
    plt.legend()
    
    
# ~ from simulator_2 import timeDelay
# ~ figure_1()
# ~ plt.show()

# ~ genTimeDelayMap(0.5)
# ~ delaySignalOrbit()
delaySignal_burst()
