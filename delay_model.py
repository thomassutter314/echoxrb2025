import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.tri as mtri
import matplotlib as mpl
import scipy.optimize as opt
from scipy import interpolate as scpint

import constants


def timeDelay(phase, \
                m1_in=1.4, m2_in=0.7, period_in=0.787, inclination=np.radians(44.), \
                setting = 'plav'):
    
    phase = np.mod(phase,1)
    m1 = m1_in * constants.M_Sun
    m2 = m2_in * constants.M_Sun
    q = m2_in/m1_in
    G = constants.G
    period = period_in * constants.day
    c = constants.c
    
    # Compute the semi-major axis
    a = (G*(m1+m2)*period**2/(4*np.pi**2))**(1/3)
    
    if setting == 'egg':
        radiusDonor = a*(.49*q**(.667))/(.6*q**(.667)+np.log(1+q**(.333)))
        #Radius of the donor at periastron passage using the Eggleton radius
    if setting == 'plav':
        radiusDonor = a*(.5+.227*np.log(q)/np.log(10))
    if setting == 'cm':
        radiusDonor = 0

    return (a-radiusDonor)*(1-np.cos(2*np.pi*phase)*np.sin(inclination))/c
        
def radialVelocity(phase, \
                m1_in=1.4, m2_in=0.7, period_in=0.787, inclination=np.radians(44.), \
                setting = 'cm'):
                    
    phase = np.mod(phase,1)
    m1 = m1_in * constants.M_Sun
    m2 = m2_in * constants.M_Sun
    q = m2_in/m1_in
    G = constants.G
    period = period_in * constants.day
    c = constants.c
    
    # Compute the semi-major axis
    a = (G*(m1+m2)*period**2/(4*np.pi**2))**(1/3)

    if setting == 'egg':
        radiusDonor = a*(.49*q**(2/3))/(.6*q**(2/3)+np.log(1+q**(1/3)))
        #Radius of the donor at periastron passage using the Eggleton radius
    if setting == 'plav':
        radiusDonor = a*(.5+.227*np.log(q)/np.log(10))
    if setting == 'cm':
        radiusDonor = 0
    
    if setting == 'md':
        N0 = 0.886
        N1 = -1.132
        N2 = 1.523
        N3 = -1.892
        N4 = 0.867
        return -np.sin(2*np.pi*phase)*np.sin(inclination)*2*np.pi*a/period*(1/(1+q))*(N0+N1*q+N2*q**2+N3*q**3+N4*q**4)
        
        
    return -np.sin(2*np.pi*phase)*np.sin(inclination)*2*np.pi*a/period*(1/(1+q))*(1-radiusDonor/a*(1+q))
                    
def genTimeDelayMap(phase, \
                  m1_in=1.4, m2_in=0.7, period_in=.787, inclination=np.radians(44.), cmap = cm.cool, plot = True, Q = 25, RETURN = False):

    m1 = m1_in * constants.M_Sun
    m2 = m2_in * constants.M_Sun
    q = m2_in/m1_in
    period = period_in * constants.day
    omega_angV = 2*np.pi/period
    G = constants.G
    a = (G*(m1+m2)*period**2/(4*np.pi**2))**(1/3)

    r2 = m1_in/(m1_in+m2_in)*a
    radiusDonor = a*(.49*q**(.667))/(.6*q**(.667)+np.log(1+q**(.333)))
    (n, m) = (Q, Q)    
    L2 = a*(.5+.227*np.log(q)/np.log(10))
    rr = m1/(m1+m2)*a-L2
    Uconst = 0
    
    def potentialFunc(p2,theta2):
        r2 = m1_in/(m1_in+m2_in)*a
        phiU = -G*m1/np.sqrt(a**2+p2**2+2*a*p2*np.cos(theta2))-G*m2/p2 - 0.5*omega_angV**2*(r2**2+p2**2+2*r2*p2*np.cos(theta2))
        return phiU-Uconst
        
    Uconst = potentialFunc(L2,np.pi)
    
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
    # If we aren't making a pretty plot, just do half the polar coord system because the other half isn't illuminated
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
    Rfunc = np.array(potentialFuncINV(psi))
    x, y, z = Rfunc*np.cos(phi)*np.cos(theta), Rfunc*np.cos(phi)*np.sin(theta), Rfunc*np.sin(phi)

    vals = np.array(np.sqrt(Rfunc**2+a**2-2*Rfunc*a*np.cos(psi))+\
            -a*np.sin(inclination)*np.sin(2*np.pi*phase)+\
            Rfunc*np.cos(psi)*np.sin(inclination)*np.sin(2*np.pi*phase)+\
           -Rfunc*np.sin(psi)*np.cos(alpha_angle)*np.cos(inclination)+\
           -Rfunc*np.sin(psi)*np.sin(alpha_angle)*np.sin(inclination)*np.cos(2*np.pi*phase))
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
        ax.set_title("Time Delay (s)",y=1.15)
        soa = np.array([[0, 0, 1, 0, 0, 1], [0, 0, 1, np.cos(inclination), np.sin(inclination)*np.cos(2*np.pi*phase), -np.sin(inclination)*np.sin(2*np.pi*phase)]])
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
        
        triang = mtri.Triangulation(x/max(Rfunc), y/max(Rfunc), triangles)
        collec = ax.plot_trisurf(triang, z/max(Rfunc), cmap=cmap, shade=False, linewidth=0.)
        collec.set_array(colors)
        collec.autoscale()
        ax.view_init(38, 0)
        plt.show()
    if RETURN == True:
        return psi, alpha_angle, np.array(vals)

def genApparentIntensityMap(phase, \
                  m1_in=1.4, m2_in=0.7, period_in=.787, eccentricity = 0, inclination=np.radians(44.), cmap = cm.hot, plot = True, Q = 25, disk = False, RETURN = False, alpha = np.radians(5)):
    m1 = m1_in * constants.M_Sun
    m2 = m2_in * constants.M_Sun
    q = m2_in/m1_in
    period = period_in * constants.day
    omega_angV = 2*np.pi/period
    G = constants.G
    a = (G*(m1+m2)*period**2/(4*np.pi**2))**(1./3.)

    r2 = m1_in/(m1_in+m2_in)*a
    radiusDonor = a*(.49*q**(.667))/(.6*q**(.667)+np.log(1+q**(.333)))
    (n, m) = (Q, Q)
    L2 = a*(.5+.227*np.log(q)/np.log(10))
    rr = m1/(m1+m2)*a-L2
    Uconst = 0
    
    def potentialFunc(p2,theta2):
        r2 = m1_in/(m1_in+m2_in)*a
        phiU = -G*m1/np.sqrt(a**2+p2**2+2*a*p2*np.cos(theta2))-G*m2/p2 \
                   -.5*omega_angV**2*(r2**2+p2**2+2*r2*p2*np.cos(theta2))
        return phiU-Uconst
    Uconst = potentialFunc(L2,np.pi)
    
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
    Rfunc = np.array(potentialFuncINV(psi))
    
    x, y, z = Rfunc*np.cos(phi)*np.cos(theta), Rfunc*np.cos(phi)*np.sin(theta), Rfunc*np.sin(phi)
    
    xv = G*m1*x/(a**2+2*a*z+Rfunc**2)**(1.5)+G*m2*x/Rfunc**3
    yv = G*m1*y/(a**2+2*a*z+Rfunc**2)**(1.5)+G*m2*y/Rfunc**3
    zv = G*m1*(z+a)/(a**2+2*a*z+Rfunc**2)**(1.5)+G*m2*z/Rfunc**3

    xv,yv,zv = xv/np.sqrt(xv**2+yv**2+zv**2), yv/np.sqrt(xv**2+yv**2+zv**2), zv/np.sqrt(xv**2+yv**2+zv**2)

    T_0 = 1.
    A1 = (Rfunc**2+a**2-2*z*a)**(-1) #Distance Attenuation
    A2 = -np.sqrt(A1)*(x*xv+y*yv+(z-a))*zv #Projected area toward accretor
    A3 = xv*np.cos(inclination)-zv*np.sin(inclination)*np.sin(2*np.pi*phase)+yv*np.sin(inclination)*np.cos(2*np.pi*phase) #Projected area toward observer
    vals = T_0*A1*A2*A3
    
    for i in range(len(vals)):
        if vals[i] < 0 or A2[i] < 0 or A3[i] < 0:
            vals[i] = 0
        if disk == True:
            sinDelta = radiusDonor*(np.sin(psi[i])*np.cos(alpha_angle[i]))/np.sqrt(radiusDonor**2+a**2-2*a*radiusDonor*np.cos(psi[i]))
            if abs(np.arcsin(sinDelta)) < alpha:
                vals[i] = 0
    
    if plot == True:
        #colors = np.mean(vals[triangles], axis=1)
        

        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlim3d(-1,1)
        ax.set_ylim3d(-1,1)
        ax.set_zlim3d(-1,1)

        #Calculation of time delay at many points over the surface of the star
        colors = np.mean(vals[triangles], axis=1)

        sm = mpl.cm.ScalarMappable(cmap=cmap)
        sm.set_array([min(colors),max(colors)])
        ticks = np.linspace(min(colors),max(colors),10)
        plt.colorbar(sm,ax=ax,ticks=ticks)
        ax.set_title("Intensity (Arbitrary Units)",y=1.15)
        soa = np.array([[0, 0, 1, 0, 0, 1], [0, 0, 1, np.cos(inclination), np.sin(inclination)*np.cos(2*np.pi*phase), -np.sin(inclination)*np.sin(2*np.pi*phase)]])
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
        
        triang = mtri.Triangulation(x/max(Rfunc), y/max(Rfunc), triangles)
        collec = ax.plot_trisurf(triang, z/max(Rfunc), cmap=cmap, shade=False, linewidth=0.)
        collec.set_array(colors)
        collec.autoscale()
        plt.show()
    if RETURN == True:
        vals = vals*np.sin(psi)*Rfunc**2 #Jacobian Determinate for Spherical Coordinates
        return psi, alpha_angle, vals

def delaySignal(phase = .5, \
                    m1_in = 1.4, m2_in = .7, period_in = .787, inclination = np.radians(44.),\
                    binNumber = 100, Q = 120, plot = True, disk = True, intKind = 'quadratic', \
                    pseudo3DTEST = True, outputData = False):
                        
    psiECHO, alphaECHO, T = genTimeDelayMap(phase,m1_in=m1_in,m2_in=m2_in,period_in=period_in,\
                                            inclination=inclination,plot = False,Q=Q,RETURN = True)

    psiINT, alphaINT, I = genApparentIntensityMap(phase,m1_in=m1_in,m2_in=m2_in,period_in=period_in,\
                                            inclination=inclination,plot = False,Q=Q,RETURN = True)
    if plot == True:
        genTimeDelayMap(phase,m1_in=m1_in,m2_in=m2_in,period_in=period_in,inclination=inclination,plot = True,Q=25,RETURN = False)
        genApparentIntensityMap(phase,m1_in=m1_in,m2_in=m2_in,period_in=period_in,inclination=inclination,plot = True,Q=25,RETURN = False)

    BIN = [[]]*binNumber
    delta = max(T)/binNumber
    for i in range(len(T)):
        binIndex = int(np.ceil((T[i])/delta))
        if binIndex > binNumber:
            binIndex = binNumber
        BIN[binIndex-1] = BIN[binIndex-1] + [i]

    delayList = np.arange(0,max(T),delta) + delta/2
    if len(delayList) > binNumber:
        delayList = delayList[:-1]
    intensityList = np.zeros(binNumber)

    for i in range(binNumber):
        for j in range(len(BIN[i])):
            intensityList[i] += I[BIN[i][j]]
    f = scpint.interp1d(delayList,intensityList,kind=intKind)
    tt = np.linspace(min(delayList),max(delayList),5000)
    yy = f(tt)
    if plot == True:
        plav_delaySIM2 = timeDelay(phase,m1_in=m1_in,m2_in=m2_in,period_in=period_in,inclination=inclination,setting='plav')
        egg_delaySIM2 = timeDelay(phase,m1_in=m1_in,m2_in=m2_in,period_in=period_in,inclination=inclination,setting='egg')
        cm_delaySIM2 = timeDelay(phase,m1_in=m1_in,m2_in=m2_in,period_in=period_in,inclination=inclination,setting='cm')
        
        plt.figure()
        plt.axvline(x=plav_delaySIM2,c='green',label='Plavec')
        plt.axvline(x=egg_delaySIM2,c='orange',label='Eggleton')
        plt.axvline(x=cm_delaySIM2,c='blue',label='Center of Mass')
        plt.axvline(x=sum(tt*yy)/sum(yy),c='black',linestyle='dashed',label='centroid')
        plt.scatter(delayList,intensityList, s=50, facecolors='none', edgecolors='b')
        plt.plot(tt,yy,'g-')
        if radialVelocity == True:
            plt.xlabel("Velocity (km/s)")
        else:
            plt.xlabel("Time (s)")
        plt.ylabel("Intensity (Arbitrary Units)")
        plt.legend()
        plt.show()
    if max(yy) < .1:
        output = 0.0001
        print("V = 0, Intensity Too Low")
    else:
        output = tt[np.argmax(yy)] #This chooses the peak intensity. This is not the best way to compute a delay to output most likely...
    if outputData == False:
        return output
    else:
        return delayList,intensityList


delaySignal(phase = 0.5)
# ~ tt = np.linspace(0,1,1000)
# ~ yy_cm = 1e-3*radialVelocity(tt,setting='cm')
# ~ yy_egg = 1e-3*radialVelocity(tt,setting='egg')
# ~ yy_plav = 1e-3*radialVelocity(tt,setting='plav')
# ~ yy_md = 1e-3*radialVelocity(tt,setting='md')

# ~ plt.plot(tt,yy_cm,label='Center of Mass')
# ~ plt.plot(tt,yy_egg,label='Eggleton')
# ~ plt.plot(tt,yy_plav,label='Plavec')
# ~ plt.plot(tt,yy_md,'k--',label='Munoz-Darias')

# ~ plt.ylabel('Velocity (km/s)')
# ~ plt.xlabel('Orbital Phase')
# ~ plt.legend()
# ~ plt.show()
