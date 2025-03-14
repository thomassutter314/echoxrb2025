import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.tri as mtri
import matplotlib as mpl
from numba import jit
import scipy.optimize as opt
from scipy import interpolate as scpint
import time
import pickle

import constants



class ASRA_LT_model():
    def __init__(self):
        # Load the ASRA beta
        with open('asra_beta.pickle', 'rb') as f:
            self.asra_beta = pickle.load(f)
        
    def evaluate(self, phase, m1_in=1.4, m2_in=0.7, inclination_in=44., disk_angle_in = 5, period_in=.787, Q = 25):
        # Convert to SI units
        m1 = m1_in * constants.M_Sun
        m2 = m2_in * constants.M_Sun
        inclination = inclination_in*np.pi/180
        disk_angle = disk_angle_in*np.pi/180
        
        period = period_in * constants.day
        omega = 2*np.pi/period
        
        # Compute the orbital semi-major axis
        a = (constants.G*(m1+m2)*period**2/(4*np.pi**2))**(1/3)
        r2 = m1/(m1+m2)*a
        
        # Compute the location of the L1 Lagrange point
        roots = np.roots([(omega)**2,-r2*(omega)**2-2*a*(omega)**2,2*a*r2*(omega)**2 + a**2*(omega)**2,constants.G*(m1-m2)-r2*(omega)**2*a**2,2*a*constants.G*m2,-a**2*constants.G*m2])
        L1 = np.real(roots[np.argmin(np.abs(np.imag(roots)))]) # There should only be one purely real solution in the 5 complex solutions, use that one
        
        # Compute the value of the effective gravitational potential at the L1 point
        potential_at_L1 = rochePotential(L1,psi=0,beta=0,m1=m1,m2=m2,period=period)
        
        # Solve for the height of the Roche Lobe at the 45 degree angle (45 deg is a good compromise)
        H = opt.brentq(rochePotential,L1*1e-3,L1,args=(np.pi/2,np.pi/4,m1,m2,period,potential_at_L1),maxiter=100)
        
        asra_beta_val = np.pi/180*self.asra_beta(inclination, np.tan(disk_angle)*a/H) # Evaluate the calibrated asra_beta function to get the best asra_beta for these params
        
        psi_roche = np.linspace(0, np.pi/2, num=Q, endpoint=True)
        
        # Azimuthal angle for the Roche Lobe
        beta_roche = np.linspace(0, 2*np.pi, num=Q, endpoint=True)
        
        # Make a meshgrid of the polar angles
        PSI_roche, BETA_roche = np.meshgrid(psi_roche, beta_roche)
    
        # numerically solve the equation V(p2,psi,beta) = V(L1) for p2 given psi and beta. Then loop over all psi and beta to construct the full lobe
        #for i in range(len(r_roche)):
        #   r_roche[i] = opt.brentq(rochePotential,L1*1e-3,L1,args=(psi_roche[i],asra_beta_val,m1,m2,period,potential_at_L1),maxiter=100)
        # ~ r_roche = list(np.zeros(len(psi_roche)))
        
        r_roche = chandrupatla(rochePotential,L1*1e-3,L1,args=(psi_roche,asra_beta_val,m1,m2,period,potential_at_L1),maxiter=100)
        
        R_roche = np.tile(r_roche, (len(beta_roche),1))
        # Now we unravel this into 1D arrays
        R_roche, PSI_roche, BETA_roche = R_roche.ravel(), PSI_roche.ravel(), BETA_roche.ravel()
        
        # Distance from points on surface of Roche lobe to compact object
        P1 = np.sqrt(R_roche**2 + a**2 - 2*R_roche*a*np.cos(PSI_roche))
        
        
        # Angle between p2 (vector from cm of donor to point on surface) and e (vector from cm of binary system to earth)
        p2hat_dot_ehat = -np.cos(PSI_roche)*np.sin(inclination)*np.cos(2*np.pi*phase) +\
                      -np.sin(PSI_roche)*np.sin(BETA_roche)*np.sin(inclination)*np.sin(2*np.pi*phase) +\
                      +np.sin(PSI_roche)*np.cos(BETA_roche)*np.cos(inclination)
                      
        # Angle between psi_hat and e (psi_hat is the polar unit vector in the spherical coordinate system where p2 is the radial coordinate)
        psihat_dot_ehat = np.sin(PSI_roche)*np.sin(inclination)*np.cos(2*np.pi*phase) +\
                  -np.cos(PSI_roche)*np.sin(BETA_roche)*np.sin(inclination)*np.sin(2*np.pi*phase) +\
                  +np.cos(PSI_roche)*np.cos(BETA_roche)*np.cos(inclination)
        
        # Compute the delay times across the Roche Lobe
        delays = P1 - a*np.cos(2*np.pi*phase)*np.sin(inclination) - R_roche * p2hat_dot_ehat
        
        delays = delays/constants.c
        
        # Compute the radial velocity over the Roche Lobe
        radial_velocity = -1*(2*np.pi/period)*np.sin(inclination)*(R_roche*np.sin(PSI_roche)*np.sin(BETA_roche)*np.cos(2*np.pi*phase) + (r2-R_roche*np.cos(PSI_roche))*np.sin(2*np.pi*phase))
        # Convert the radial velocity to km/s from m/s
        radial_velocity *= 1e-3
        
        #################################
        # Now let's compute the apparent intensity over the Roche Lobe, we have 3 terms A1, A2, and A3
        
        #Let's compute A1: Distance Attenuation
        A1 = P1**(-2)
        
        # We will implement the disk shielding angle here by taking all values of A1A2 corresponding to angles below the shielding angle to zero
        A1[R_roche/P1*np.abs(np.sin(PSI_roche)*np.cos(BETA_roche)) < np.sin(disk_angle)] = 0
        
        #Let's compute A2: Projected area on surface of donor towards accretor
        
        # First we need to compute vectors normal to the Roche Lobe surface via a gradient of the potential function
        vecs_normal_roche = np.array(polarGradRochePotential(R_roche,PSI_roche,BETA_roche,m1,m2,period))# returns dV/dp2 vec(p2) + 1/p2*dV/dpsi vec(psi)
        
        # Manage special case of psi = 0 where the gradient function can spuriously evaluate to zero
        psi_zero_indices = (PSI_roche == 0)
        vecs_normal_roche[0, psi_zero_indices] = 1
        vecs_normal_roche[1, psi_zero_indices] = 0
        
        # Normalize the vectors to 1
        vecs_normal_roche = vecs_normal_roche/np.sqrt(vecs_normal_roche[0]**2+vecs_normal_roche[1]**2)
    
        # Take a normalized dot product of roche normal with the -p1 vector (p1 points from accretor to point on surface of donor)
        A2 = (vecs_normal_roche[0]*(a*np.cos(PSI_roche)-R_roche) - vecs_normal_roche[1]*a*np.sin(PSI_roche)) \
             /(np.sqrt(R_roche**2 + a**2 - 2*R_roche*a*np.cos(PSI_roche)))
        A2[A2<0] = 0 # Make strictly positive or zero
        
        #Let's compute A3: Projected area on surface of donor towards observer
        A3 = vecs_normal_roche[0] * p2hat_dot_ehat + vecs_normal_roche[1] * psihat_dot_ehat
        A3[A3<0] = 0 # Make strictly positive or zero
    
        T_0 = 10
        apparent_intensities = T_0*A1*A2*A3
        
        if np.sum(R_roche**2*np.sin(PSI_roche)*apparent_intensities) != 0:
            # The Jacobian is sin(psi)*R**2
            centroid_delays = np.average(delays,weights=R_roche**2*np.sin(PSI_roche)*apparent_intensities) # Compute the centroid of the distribution of delay times weighted to the intensities
            centroid_rv = np.average(radial_velocity,weights=R_roche**2*np.sin(PSI_roche)*apparent_intensities)
        else:
            centroid_delays = 0
            centroid_rv = 0
            
        return centroid_delays, centroid_rv
    
    def evaluate_array(self, phases, m1_in=1.4, m2_in=0.7, inclination_in=44., disk_angle_in = 5, period_in=.787, Q = 25):
        tt = np.empty(len(phases))
        vv = np.empty(len(phases))
        for i in range(len(phases)):
            tt[i], vv[i] = self.evaluate(phases[i], m1_in=m1_in, m2_in=m2_in, inclination_in=inclination_in, disk_angle_in = disk_angle_in, period_in=period_in, Q = Q)
        return tt, vv
    
    def check_eclipse(self, m1_in=1.4, m2_in=0.7, inclination_in=44., period_in=.787, Q = 25, fancy = True):
        # Convert to SI units
        m1 = m1_in * constants.M_Sun
        m2 = m2_in * constants.M_Sun
        inclination = inclination_in*np.pi/180
        period = period_in * constants.day
        omega = 2*np.pi/period
        
        # Compute the orbital semi-major axis
        a = (constants.G*(m1+m2)*period**2/(4*np.pi**2))**(1/3)
        r2 = m1/(m1+m2)*a
        
        # Compute the location of the L1 Lagrange point
        roots = np.roots([(omega)**2,-r2*(omega)**2-2*a*(omega)**2,2*a*r2*(omega)**2 + a**2*(omega)**2,constants.G*(m1-m2)-r2*(omega)**2*a**2,2*a*constants.G*m2,-a**2*constants.G*m2])
        L1 = np.real(roots[np.argmin(np.abs(np.imag(roots)))]) # There should only be one purely real solution in the 5 complex solutions, use that one
        
        # Compute the value of the effective gravitational potential at the L1 point
        potential_at_L1 = rochePotential(L1,psi=0,beta=0,m1=m1,m2=m2,period=period)
        
        if fancy:
            # Fully correct, but also a bit more computationally expensive
            psi_roche = np.linspace(0, np.pi/2, num=Q, endpoint=True)
            r_roche = chandrupatla(rochePotential,L1*1e-3,L1,args=(psi_roche,0,m1,m2,period,potential_at_L1),maxiter=100)
            return np.any(r_roche*np.sin(psi_roche)/(a - r_roche*np.cos(psi_roche)) > 1/np.tan(inclination))
        else:
            # Solve for the height of the Roche Lobe at 0 degree angle (height normal to the orbital plane (beta = 90 deg is the orbital plane))
            H = opt.brentq(rochePotential,L1*1e-3,L1,args=(np.pi/2,0,m1,m2,period,potential_at_L1),maxiter=100)
            return H/a > 1/np.tan(inclination)
        
        
def chandrupatla(f,x0,x1,verbose=False, 
                 eps_m = None, eps_a = None, 
                 maxiter=50, return_iter=False, args=(),):
    # as written in https://www.embeddedrelated.com/showarticle/855.php
    # which in turn is based on Chandrupatla's algorithm as described in Scherer
    # https://books.google.com/books?id=cC-8BAAAQBAJ&pg=PA95
    # This allows vector arguments for x0, x1, and args
    
    # Initialization
    b = x0
    a = x1
    fa = f(a, *args)
    fb = f(b, *args)
    
    # Make sure we know the size of the result
    shape = np.shape(fa)
    assert shape == np.shape(fb)
        
    # In case x0, x1 are scalars, make sure we broadcast them to become the size of the result
    b += np.zeros(shape)
    a += np.zeros(shape)

    fc = fa
    c = a
    
    # Make sure we are bracketing a root in each case
    assert (np.sign(fa) * np.sign(fb) <= 0).all()
    t = 0.5
    # Initialize an array of False,
    # determines whether we should do inverse quadratic interpolation
    iqi = np.zeros(shape, dtype=bool)
    
    # jms: some guesses for default values of the eps_m and eps_a settings
    # based on machine precision... not sure exactly what to do here
    eps = np.finfo(float).eps
    if eps_m is None:
        eps_m = eps
    if eps_a is None:
        eps_a = 2*eps
    
    iterations = 0
    terminate = False
    
    while maxiter > 0:
        maxiter -= 1
        # use t to linearly interpolate between a and b,
        # and evaluate this function as our newest estimate xt
        xt = a + t*(b-a)
        ft = f(xt, *args)
        if verbose:
            output = 'IQI? %s\nt=%s\nxt=%s\nft=%s\na=%s\nb=%s\nc=%s' % (iqi,t,xt,ft,a,b,c)
            if verbose == True:
                print(output)
            else:
                print(output,file=verbose)
        # update our history of the last few points so that
        # - a is the newest estimate (we're going to update it from xt)
        # - c and b get the preceding two estimates
        # - a and b maintain opposite signs for f(a) and f(b)
        samesign = np.sign(ft) == np.sign(fa)
        c  = np.choose(samesign, [b,a])
        b  = np.choose(samesign, [a,b])
        fc = np.choose(samesign, [fb,fa])
        fb = np.choose(samesign, [fa,fb])
        a  = xt
        fa = ft
        
        # set xm so that f(xm) is the minimum magnitude of f(a) and f(b)
        fa_is_smaller = np.abs(fa) < np.abs(fb)
        xm = np.choose(fa_is_smaller, [b,a])
        fm = np.choose(fa_is_smaller, [fb,fa])
        
        """
        the preceding lines are a vectorized version of:

        samesign = np.sign(ft) == np.sign(fa)        
        if samesign
            c = a
            fc = fa
        else:
            c = b
            b = a
            fc = fb
            fb = fa

        a = xt
        fa = ft
        # set xm so that f(xm) is the minimum magnitude of f(a) and f(b)
        if np.abs(fa) < np.abs(fb):
            xm = a
            fm = fa
        else:
            xm = b
            fm = fb
        """
        
        tol = 2*eps_m*np.abs(xm) + eps_a
        tlim = tol/np.abs(b-c)
        terminate = np.logical_or(terminate, np.logical_or(fm==0, tlim > 0.5))
        if verbose:            
            output = "fm=%s\ntlim=%s\nterm=%s" % (fm,tlim,terminate)
            if verbose == True:
                print(output)
            else:
                print(output, file=verbose)

        if np.all(terminate):
            break
        iterations += 1-terminate
        
        # Figure out values xi and phi 
        # to determine which method we should use next
        xi  = (a-b)/(c-b)
        phi = (fa-fb)/(fc-fb)
        iqi = np.logical_and(phi**2 < xi, (1-phi)**2 < 1-xi)
            
        if not shape:
            # scalar case
            if iqi:
                # inverse quadratic interpolation
                t = fa / (fb-fa) * fc / (fb-fc) + (c-a)/(b-a)*fa/(fc-fa)*fb/(fc-fb)
            else:
                # bisection
                t = 0.5
        else:
            # array case
            t = np.full(shape, 0.5)
            a2,b2,c2,fa2,fb2,fc2 = a[iqi],b[iqi],c[iqi],fa[iqi],fb[iqi],fc[iqi]
            t[iqi] = fa2 / (fb2-fa2) * fc2 / (fb2-fc2) + (c2-a2)/(b2-a2)*fa2/(fc2-fa2)*fb2/(fc2-fb2)
        
        # limit to the range (tlim, 1-tlim)
        t = np.minimum(1-tlim, np.maximum(tlim, t))
        
    # done!
    if return_iter:
        return xm, iterations
    else:
        return xm

@jit(nopython = True)
def rochePotential(p2,psi,beta,m1,m2,period,Uconst = 0):   
    a = (constants.G*(m1+m2)*(period/(2*np.pi))**2)**(1/3)
    r2 = m1/(m1+m2)*a

    return -constants.G*m1/np.sqrt(a**2+p2**2-2*a*p2*np.cos(psi)) \
       -constants.G*m2/p2 \
       -0.5*(2*np.pi/period)**2*p2**2*((np.cos(psi)-r2/p2)**2 + (np.sin(psi)*np.sin(beta))**2) \
       -Uconst

@jit(nopython = True)
def polarGradRochePotential(p2,psi,beta,m1,m2,period):
    a = (constants.G*(m1+m2)*(period/(2*np.pi))**2)**(1/3)
    r2 = m1/(m1+m2)*a
    
    dV_dp2 = constants.G*m2/p2**2 +\
         constants.G*m1*(p2-a*np.cos(psi))/(a**2+p2**2-2*a*p2*np.cos(psi))**(1.5) +\
         -(2*np.pi/period)**2*(p2*(1-(np.sin(psi)*np.cos(beta))**2) - r2*np.cos(psi))
             
    dV_dpsi = constants.G*m1*a*p2*np.sin(psi)/(a**2+p2**2-2*a*p2*np.cos(psi))**(1.5) +\
              -(2*np.pi/period)**2*p2*(r2*np.sin(psi) - 0.5*p2*np.sin(2*psi)*np.cos(beta)**2)
    
    return [dV_dp2, p2**(-1)*dV_dpsi]
    # ~ return np.array([dV_dp2, p2**(-1)*dV_dpsi])

def timeDelay_sp(phase, \
                m1_in=1.4, m2_in=0.7, inclination_in=44., period_in=0.787,\
                setting = 'plav'):
    
    m1 = m1_in * constants.M_Sun
    m2 = m2_in * constants.M_Sun
    inclination = inclination_in*np.pi/180
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
        
def timeDelay_3d_full(phase, \
                  m1_in=1.4, m2_in=0.7, inclination_in=44., disk_angle_in = 5, period_in=.787, Q = 25, mode = 'return_delay'):
                      
    # ~ print(phase,m1_in,m2_in,inclination_in,disk_angle_in)
    # Convert to SI units
    m1 = m1_in * constants.M_Sun
    m2 = m2_in * constants.M_Sun
    inclination = inclination_in*np.pi/180
    disk_angle = disk_angle_in*np.pi/180
    
    q = m2_in/m1_in
    
    period = period_in * constants.day
    # Compute the orbital semi-major axis
    a = (constants.G*(m1+m2)*period**2/(4*np.pi**2))**(1/3)
    r2 = m1/(m1+m2)*a
    
    # Compute the location of the L1 Lagrange point
    roots = np.roots([(2*np.pi/period)**2,-r2*(2*np.pi/period)**2-2*a*(2*np.pi/period)**2,2*a*r2*(2*np.pi/period)**2 + a**2*(2*np.pi/period)**2,constants.G*(m1-m2)-r2*(2*np.pi/period)**2*a**2,2*a*constants.G*m2,-a**2*constants.G*m2])
    L1 = np.real(roots[np.argmin(np.abs(np.imag(roots)))]) # There should only be one purely real solution in the 5 complex solutions, use that one
    
    # Compute the value of the effective gravitational potential at the L1 point
    potential_at_L1 = rochePotential(L1,psi=0,beta=0,m1=m1,m2=m2,period=period)
    
    # Make an array of polar coordinates for constructing the Roche Lobe, if not plotting speed up by dropping the bottom half of the Roche lobe
    if mode == 'plot':
        psi_roche = np.linspace(0, np.pi, num=Q, endpoint=True)
    else:
        psi_roche = np.linspace(0, np.pi/2, num=Q, endpoint=True)
        
    r_roche = np.zeros(len(psi_roche))
       
    # Azimuthal angle for the Roche Lobe
    beta_roche = np.linspace(0, 2*np.pi, num=Q, endpoint=True)
    
    # Make a meshgrid of the polar angles
    PSI_roche, BETA_roche = np.meshgrid(psi_roche, beta_roche)
    R_roche = np.tile(r_roche, (len(beta_roche),1))
    # Now we unravel this into 1D arrays
    R_roche, PSI_roche, BETA_roche = R_roche.ravel(), PSI_roche.ravel(), BETA_roche.ravel()
    
    # numerically solve the equation V(p2,psi,beta) = V(L1) for p2 given psi and beta. Then loop over all psi and beta to construct the full lobe
    for i in range(len(R_roche)):
        R_roche[i] = opt.brentq(rochePotential,L1*1e-3,L1,args=(PSI_roche[i],BETA_roche[i],m1,m2,period,potential_at_L1),maxiter=100)
    
    # Distance from points on surface of Roche lobe to compact object
    P1 = np.sqrt(R_roche**2 + a**2 - 2*R_roche*a*np.cos(PSI_roche))
    
    # Angle between p2 (vector from cm of donor to point on surface) and e (vector from cm of binary system to earth)
    p2hat_dot_ehat = -np.cos(PSI_roche)*np.sin(inclination)*np.cos(2*np.pi*phase) +\
                  -np.sin(PSI_roche)*np.sin(BETA_roche)*np.sin(inclination)*np.sin(2*np.pi*phase) +\
                  +np.sin(PSI_roche)*np.cos(BETA_roche)*np.cos(inclination)
                  
    # Angle between psi_hat and e (psi_hat is the polar unit vector in the spherical coordinate system where p2 is the radial coordinate)
    psihat_dot_ehat = np.sin(PSI_roche)*np.sin(inclination)*np.cos(2*np.pi*phase) +\
              -np.cos(PSI_roche)*np.sin(BETA_roche)*np.sin(inclination)*np.sin(2*np.pi*phase) +\
              +np.cos(PSI_roche)*np.cos(BETA_roche)*np.cos(inclination)
    
    # Compute the delay times across the Roche Lobe
    delays = P1 - a*np.cos(2*np.pi*phase)*np.sin(inclination) - R_roche * p2hat_dot_ehat
    
    delays = delays/constants.c
    
    # Compute the radial velocity over the Roche Lobe
    radial_velocity = -1*(2*np.pi/period)*np.sin(inclination)*(R_roche*np.sin(PSI_roche)*np.sin(BETA_roche)*np.cos(2*np.pi*phase) + (r2-R_roche*np.cos(PSI_roche))*np.sin(2*np.pi*phase))
    # Convert the radial velocity to km/s from m/s
    radial_velocity *= 1e-3
    
    #################################
    # Now let's compute the apparent intensity over the Roche Lobe, we have 3 terms A1, A2, and A3
    
    #Let's compute A1: Distance Attenuation
    A1 = P1**(-2)
    
    # We will implement the disk shielding angle here by taking all values of A1A2 corresponding to angles below the shielding angle to zero
    A1[R_roche/P1*np.abs(np.sin(PSI_roche)*np.cos(BETA_roche)) < np.sin(disk_angle)] = 0
    
    #Let's compute A2: Projected area on surface of donor towards accretor
    
    # First we need to compute vectors normal to the Roche Lobe surface via a gradient of the potential function
    vecs_normal_roche = polarGradRochePotential(R_roche,PSI_roche,BETA_roche,m1,m2,period)# returns dV/dp2 vec(p2) + 1/p2*dV/dpsi vec(psi)
    
    # Manage special case of psi = 0 where the gradient function can spuriously evaluate to zero
    psi_zero_indices = PSI_roche == 0
    
    vecs_normal_roche[0][psi_zero_indices] = 1
    vecs_normal_roche[1][psi_zero_indices] = 0
    
    # Normalize the vectors to 1
    vecs_normal_roche = vecs_normal_roche/np.sqrt(vecs_normal_roche[0]**2+vecs_normal_roche[1]**2)

    # Take a normalized dot product of roche normal with the -p1 vector (p1 points from accretor to point on surface of donor)
    A2 = (vecs_normal_roche[0]*(a*np.cos(PSI_roche)-R_roche) - vecs_normal_roche[1]*a*np.sin(PSI_roche)) \
         /(np.sqrt(R_roche**2 + a**2 - 2*R_roche*a*np.cos(PSI_roche)))
    A2[A2<0] = 0 # Make strictly positive or zero
    
    #Let's compute A3: Projected area on surface of donor towards observer
    A3 = vecs_normal_roche[0] * p2hat_dot_ehat + vecs_normal_roche[1] * psihat_dot_ehat
    A3[A3<0] = 0 # Make strictly positive or zero

    T_0 = 10
    apparent_intensities = T_0*A1*A2*A3
    
    if np.sum(R_roche**2*np.sin(PSI_roche)*apparent_intensities) != 0:
        # The Jacobian is sin(psi)*R**2
        centroid_delays = np.average(delays,weights=R_roche**2*np.sin(PSI_roche)*apparent_intensities) # Compute the centroid of the distribution of delay times weighted to the intensities
        centroid_beta = 180/np.pi*np.arcsin(np.sqrt(np.average(np.sin(BETA_roche)**2,weights=R_roche**2*np.sin(PSI_roche)*apparent_intensities)))
        centroid_rv = np.average(radial_velocity,weights=R_roche**2*np.sin(PSI_roche)*apparent_intensities)
    else:
        centroid_delays = 0
        centroid_beta = 0
        centroid_rv = 0
    
    if mode == 'plot':
        pview, aview = 20, 65
        
        x, y, z = R_roche*np.cos(BETA_roche)*np.sin(PSI_roche), R_roche*np.sin(BETA_roche)*np.sin(PSI_roche), R_roche*np.cos(PSI_roche)
        
        mesh_x, mesh_y = (PSI_roche, BETA_roche)
        triangles = mtri.Triangulation(mesh_y, mesh_x).triangles
        
        # Plotting
        fig = plt.figure()
        
        # Make the delays subplot
        cmap = cm.cool
        ax = fig.add_subplot(1,3,1, projection='3d')
        ax.set_axis_off()
        ax.set_xlim3d(-1,1)
        ax.set_ylim3d(-1,1)
        ax.set_zlim3d(-1,1)
        #Calculation of time delay at many points over the surface of the star
        colors = np.mean(delays[triangles], axis=1)
        sm = mpl.cm.ScalarMappable(cmap=cmap)
        sm.set_array([min(colors),max(colors)])
        ticks = np.linspace(min(colors),max(colors),10)   
        plt.colorbar(sm,ax=ax,ticks=ticks,shrink=0.5)
        ax.set_title("Time Delay (s)",y=1.15)
        soa = np.array([[0, 0, 1, 0, 0, 1], [0, 0, 1, np.cos(inclination), -np.sin(inclination)*np.sin(2*np.pi*phase), -np.sin(inclination)*np.cos(2*np.pi*phase)]])
        X, Y, Z, U, V, W = zip(*soa)
        param1 = .75
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        U,V,W = param1*U,param1*V,param1*W
        labels = ['Accretor', 'Observer']
        for i in range(len(labels)):
            ax.text(X[i]+U[i], Y[i]+V[i], Z[i]+W[i], labels[i])
        quiv = ax.quiver(X, Y, Z, U, V, W)
        
        triang = mtri.Triangulation(x/max(R_roche), y/max(R_roche), triangles)
        collec = ax.plot_trisurf(triang, z/max(R_roche), cmap=cmap, shade=False, linewidth=0.)
        collec.set_array(colors)
        collec.autoscale()
        ax.view_init(pview, aview)
        
        # Make the radial velocity subplot
        cmap = cm.jet
        ax = fig.add_subplot(1,3,2, projection='3d')
        ax.set_axis_off()
        ax.set_xlim3d(-1,1)
        ax.set_ylim3d(-1,1)
        ax.set_zlim3d(-1,1)
        #Calculation of time delay at many points over the surface of the star
        colors = np.mean(radial_velocity[triangles], axis=1)
        sm = mpl.cm.ScalarMappable(cmap=cmap)
        sm.set_array([min(colors),max(colors)])
        ticks = np.linspace(min(colors),max(colors),10)   
        plt.colorbar(sm,ax=ax,ticks=ticks,shrink=0.5)
        ax.set_title("Radial Velocity (km/s)",y=1.15)
        soa = np.array([[0, 0, 1, 0, 0, 1], [0, 0, 1, np.cos(inclination), -np.sin(inclination)*np.sin(2*np.pi*phase), -np.sin(inclination)*np.cos(2*np.pi*phase)]])
        X, Y, Z, U, V, W = zip(*soa)
        param1 = .75
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        U,V,W = param1*U,param1*V,param1*W
        labels = ['Accretor', 'Observer']
        for i in range(len(labels)):
            ax.text(X[i]+U[i], Y[i]+V[i], Z[i]+W[i], labels[i])
        quiv = ax.quiver(X, Y, Z, U, V, W)
        
        triang = mtri.Triangulation(x/max(R_roche), y/max(R_roche), triangles)
        collec = ax.plot_trisurf(triang, z/max(R_roche), cmap=cmap, shade=False, linewidth=0.)
        collec.set_array(colors)
        collec.autoscale()
        ax.view_init(pview, aview)
        
        # Make the intensity subplot
        cmap = cm.hot
        ax = fig.add_subplot(1,3,3, projection='3d')
        ax.set_axis_off()
        ax.set_xlim3d(-1,1)
        ax.set_ylim3d(-1,1)
        ax.set_zlim3d(-1,1)
        #Calculation of time delay at many points over the surface of the star
        colors = np.mean(apparent_intensities[triangles], axis=1)
        sm = mpl.cm.ScalarMappable(cmap=cmap)
        sm.set_array([min(colors),max(colors)])
        ticks = np.linspace(min(colors),max(colors),10)   
        plt.colorbar(sm,ax=ax,ticks=ticks,shrink=0.5)
        ax.set_title("Apparent Intensity",y=1.15)
        soa = np.array([[0, 0, 1, 0, 0, 1], [0, 0, 1, np.cos(inclination), -np.sin(inclination)*np.sin(2*np.pi*phase), -np.sin(inclination)*np.cos(2*np.pi*phase)]])
        X, Y, Z, U, V, W = zip(*soa)
        param1 = .75
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        U,V,W = param1*U,param1*V,param1*W
        labels = ['Accretor', 'Observer']
        for i in range(len(labels)):
            ax.text(X[i]+U[i], Y[i]+V[i], Z[i]+W[i], labels[i])
        quiv = ax.quiver(X, Y, Z, U, V, W)
        
        triang = mtri.Triangulation(x/max(R_roche), y/max(R_roche), triangles)
        collec = ax.plot_trisurf(triang, z/max(R_roche), cmap=cmap, shade=False, linewidth=0.)
        collec.set_array(colors)
        collec.autoscale()
        ax.view_init(pview, aview)
        
        q = m2/m1
        egg_radiusDonor = a*(.49*q**(.667))/(.6*q**(.667)+np.log(1+q**(.333)))
        egg_volume = 4/3*np.pi*egg_radiusDonor**3
        fig.suptitle(f'Roche Lobe Volume: {1/3*np.sum(vecs_normal_roche[0]*R_roche**3*np.sin(PSI_roche))*(np.pi/Q)*(2*np.pi/Q)} m^3 \n Eggleton Volume = {egg_volume} m^3 \n Centroid Beta = {centroid_beta} deg \n Centroid Delay = {centroid_delays} s')

        # Make a new plot showing the observed intensity vs. time delay
        fig, axs = plt.subplots(1,2)
        # Here we can scale the apparent intensities by the spherical coordinate system Jacobian
        observed_intensities = (R_roche)**2*np.sin(PSI_roche)*apparent_intensities
        
        args = delays.argsort()
        delays = delays[args]
        observed_intensities = observed_intensities[args]
        
        block_size = 2*Q
        delays_reshaped = delays.reshape(-1, block_size)
        observed_intensities_reshaped = observed_intensities.reshape(-1, block_size)
        # Sum along the rows (axis=1) of the reshaped array
        delays_binned = np.mean(delays_reshaped, axis=1)
        observed_intensities_binned = np.mean(observed_intensities_reshaped, axis=1)
        
        # ~ axs[0].scatter(delays, observed_intensities,s=3)
        axs[0].plot(delays_binned, observed_intensities_binned,'b-',label='binned', zorder = -10)
        axs[0].scatter(delays_binned, observed_intensities_binned,fc = 'white', ec = 'black',label='binned')
        axs[0].axvline(x=centroid_delays,c='black',label=f'centorid = {round(centroid_delays,2)} s',linestyle='--')
        axs[0].set_xlabel('Time Delay (s)')
        axs[0].set_ylabel('Observed Intensity (a.u.)')
        
        observed_intensities = (R_roche)**2*np.sin(PSI_roche)*apparent_intensities
        
        args = radial_velocity.argsort()
        radial_velocity = radial_velocity[args]
        observed_intensities = observed_intensities[args]
        
        block_size = 2*Q
        radial_velocity_reshaped = radial_velocity.reshape(-1, block_size)
        observed_intensities_reshaped = observed_intensities.reshape(-1, block_size)
        # Sum along the rows (axis=1) of the reshaped array
        radial_velocity_binned = np.mean(radial_velocity_reshaped, axis=1)
        observed_intensities_binned = np.mean(observed_intensities_reshaped, axis=1)
        
        # ~ axs[1].scatter(radial_velocity, observed_intensities,s=3)
        axs[1].plot(radial_velocity_binned, observed_intensities_binned,'b-', zorder = -10)
        axs[1].scatter(radial_velocity_binned, observed_intensities_binned,fc = 'white', ec = 'black',label='binned')
        axs[1].axvline(x=centroid_rv,c='black',label=f'centorid = {round(centroid_rv,2)} km/s',linestyle='--')
        axs[1].set_xlabel('Radial Velocity (km/s)')
        axs[1].set_ylabel('Observed Intensity (a.u.)')
        
        axs[0].legend()
        axs[1].legend()
        plt.show()
        
    if mode == 'return_delay':
        return centroid_delays
    if mode == 'return_beta':
        return centroid_beta
    if mode == 'return_rv':
        return centroid_rv
        observed_intensities = R_roche**2*np.sin(PSI_roche)*apparent_intensities
        
        args = radial_velocity.argsort()
        radial_velocity = radial_velocity[args]
        observed_intensities = observed_intensities[args]
        block_size = 2*Q
        radial_velocity_reshaped = radial_velocity.reshape(-1, block_size)
        observed_intensities_reshaped = observed_intensities.reshape(-1, block_size)
        # Sum along the rows (axis=1) of the reshaped array
        radial_velocity_binned = np.mean(radial_velocity_reshaped, axis=1)
        observed_intensities_binned = np.mean(observed_intensities_reshaped, axis=1)
        
        if np.any(observed_intensities_binned > 0):
            vv = np.linspace(min(radial_velocity_binned[observed_intensities_binned>0]),max(radial_velocity_binned[observed_intensities_binned>0]),3*len(radial_velocity_binned[observed_intensities_binned>0]))
            f = scpint.UnivariateSpline(radial_velocity_binned[observed_intensities_binned>0], observed_intensities_binned[observed_intensities_binned>0],k=3)
            ii = f(vv)
        else:
            return 0
        
        if False:
            plt.scatter(radial_velocity_binned,observed_intensities_binned,c='black')
            plt.plot(vv,ii)
            plt.axvline(x=vv[np.argmax(ii)],c='black')
            plt.xlabel('Radial Velocity (km/s)')
            plt.ylabel('Intensity')
            plt.show()
        
        # ~ return vv[np.argmax(ii)]
        # ~ return radial_velocity_binned[np.argmax(observed_intensities_binned)]
        # ~ return tt[np.argmax(yy)]
        return centroid_rv

def timeDelay_3d_asra(phase, \
                  m1_in=1.4, m2_in=0.7, inclination_in=44., disk_angle_in = 5, period_in=.787, Q = 25, mode = 'return_delay', asra_beta = 0):
                          
    # Convert to SI units
    m1 = m1_in * constants.M_Sun
    m2 = m2_in * constants.M_Sun
    inclination = inclination_in*np.pi/180
    disk_angle = disk_angle_in*np.pi/180
    
    q = m2_in/m1_in
    
    period = period_in * constants.day
    omega = 2*np.pi/period
    # Compute the orbital semi-major axis
    G = constants.G
    a = (constants.G*(m1+m2)*period**2/(4*np.pi**2))**(1/3)
    r2 = m1/(m1+m2)*a
    
    # Compute the location of the L1 Lagrange point
    roots = np.roots([(omega)**2,-r2*(omega)**2-2*a*(omega)**2,2*a*r2*(omega)**2 + a**2*(omega)**2,constants.G*(m1-m2)-r2*(omega)**2*a**2,2*a*constants.G*m2,-a**2*constants.G*m2])
    L1 = np.real(roots[np.argmin(np.abs(np.imag(roots)))]) # There should only be one purely real solution in the 5 complex solutions, use that one
    
    # Compute the value of the effective gravitational potential at the L1 point
    potential_at_L1 = rochePotential(L1,psi=0,beta=0,m1=m1,m2=m2,period=period)
    
    asra_beta = np.pi/180*asra_beta
    
    # Make an array of polar coordinates for constructing the Roche Lobe, if not plotting speed up by dropping the bottom half of the Roche lobe
    if mode == 'plot':
        psi_roche = np.linspace(0, np.pi, num=Q, endpoint=True)
    else:
        psi_roche = np.linspace(0, np.pi/2, num=Q, endpoint=True)
        
    r_roche = np.zeros(len(psi_roche))
       
    # Azimuthal angle for the Roche Lobe
    beta_roche = np.linspace(0, 2*np.pi, num=Q, endpoint=True)
    
    # Make a meshgrid of the polar angles
    PSI_roche, BETA_roche = np.meshgrid(psi_roche, beta_roche)

    # numerically solve the equation V(p2,psi,beta) = V(L1) for p2 given psi and beta. Then loop over all psi and beta to construct the full lobe
    for i in range(len(r_roche)):
        r_roche[i] = opt.brentq(rochePotential,L1*1e-3,L1,args=(psi_roche[i],asra_beta,m1,m2,period,potential_at_L1),maxiter=100)
    
    R_roche = np.tile(r_roche, (len(beta_roche),1))
    # Now we unravel this into 1D arrays
    R_roche, PSI_roche, BETA_roche = R_roche.ravel(), PSI_roche.ravel(), BETA_roche.ravel()
    
    # Distance from points on surface of Roche lobe to compact object
    P1 = np.sqrt(R_roche**2 + a**2 - 2*R_roche*a*np.cos(PSI_roche))
    
    # Angle between p2 (vector from cm of donor to point on surface) and e (vector from cm of binary system to earth)
    p2hat_dot_ehat = -np.cos(PSI_roche)*np.sin(inclination)*np.cos(2*np.pi*phase) +\
                  -np.sin(PSI_roche)*np.sin(BETA_roche)*np.sin(inclination)*np.sin(2*np.pi*phase) +\
                  +np.sin(PSI_roche)*np.cos(BETA_roche)*np.cos(inclination)
                  
    # Angle between psi_hat and e (psi_hat is the polar unit vector in the spherical coordinate system where p2 is the radial coordinate)
    psihat_dot_ehat = np.sin(PSI_roche)*np.sin(inclination)*np.cos(2*np.pi*phase) +\
              -np.cos(PSI_roche)*np.sin(BETA_roche)*np.sin(inclination)*np.sin(2*np.pi*phase) +\
              +np.cos(PSI_roche)*np.cos(BETA_roche)*np.cos(inclination)
    
    # Compute the delay times across the Roche Lobe
    delays = P1 - a*np.cos(2*np.pi*phase)*np.sin(inclination) - R_roche * p2hat_dot_ehat
    
    delays = delays/constants.c
    
    # Compute the radial velocity over the Roche Lobe
    radial_velocity = -1*(2*np.pi/period)*np.sin(inclination)*(R_roche*np.sin(PSI_roche)*np.sin(BETA_roche)*np.cos(2*np.pi*phase) + (r2-R_roche*np.cos(PSI_roche))*np.sin(2*np.pi*phase))
    # Convert the radial velocity to km/s from m/s
    radial_velocity *= 1e-3
    
    #################################
    # Now let's compute the apparent intensity over the Roche Lobe, we have 3 terms A1, A2, and A3
    
    #Let's compute A1: Distance Attenuation
    A1 = P1**(-2)
    
    # We will implement the disk shielding angle here by taking all values of A1A2 corresponding to angles below the shielding angle to zero
    A1[R_roche/P1*np.abs(np.sin(PSI_roche)*np.cos(BETA_roche)) < np.sin(disk_angle)] = 0
    
    #Let's compute A2: Projected area on surface of donor towards accretor
    
    # First we need to compute vectors normal to the Roche Lobe surface via a gradient of the potential function
    vecs_normal_roche = polarGradRochePotential(R_roche,PSI_roche,BETA_roche,m1,m2,period)# returns dV/dp2 vec(p2) + 1/p2*dV/dpsi vec(psi)
    
    # Manage special case of psi = 0 where the gradient function can spuriously evaluate to zero
    psi_zero_indices = (PSI_roche == 0)
    vecs_normal_roche[0,psi_zero_indices] = 1
    vecs_normal_roche[1,psi_zero_indices] = 0
    
    # Normalize the vectors to 1
    vecs_normal_roche = vecs_normal_roche/np.sqrt(vecs_normal_roche[0]**2+vecs_normal_roche[1]**2)

    # Take a normalized dot product of roche normal with the -p1 vector (p1 points from accretor to point on surface of donor)
    A2 = (vecs_normal_roche[0]*(a*np.cos(PSI_roche)-R_roche) - vecs_normal_roche[1]*a*np.sin(PSI_roche)) \
         /(np.sqrt(R_roche**2 + a**2 - 2*R_roche*a*np.cos(PSI_roche)))
    A2[A2<0] = 0 # Make strictly positive or zero
    
    #Let's compute A3: Projected area on surface of donor towards observer
    A3 = vecs_normal_roche[0] * p2hat_dot_ehat + vecs_normal_roche[1] * psihat_dot_ehat
    A3[A3<0] = 0 # Make strictly positive or zero

    T_0 = 10
    apparent_intensities = T_0*A1*A2*A3
    
    if np.sum(R_roche**2*np.sin(PSI_roche)*apparent_intensities) != 0:
        # The Jacobian is sin(psi)*R**2
        centroid_delays = np.average(delays,weights=R_roche**2*np.sin(PSI_roche)*apparent_intensities) # Compute the centroid of the distribution of delay times weighted to the intensities
        #centroid_beta = 180/np.pi*np.arcsin(np.sqrt(np.average(np.sin(BETA_roche)**2,weights=R_roche**2*np.sin(PSI_roche)*apparent_intensities)))
        centroid_rv = np.average(radial_velocity,weights=R_roche**2*np.sin(PSI_roche)*apparent_intensities)
    else:
        centroid_delays = 0
        # ~ centroid_beta = 0
        centroid_rv = 0
    
    if mode == 'plot':
        x, y, z = R_roche*np.cos(BETA_roche)*np.sin(PSI_roche), R_roche*np.sin(BETA_roche)*np.sin(PSI_roche), R_roche*np.cos(PSI_roche)
        
        mesh_x, mesh_y = (PSI_roche, BETA_roche)
        triangles = mtri.Triangulation(mesh_y, mesh_x).triangles
        
        # Plotting
        fig = plt.figure()
        
        # Make the delays subplot
        cmap = cm.cool
        ax = fig.add_subplot(1,2,1, projection='3d')
        ax.set_xlim3d(-1,1)
        ax.set_ylim3d(-1,1)
        ax.set_zlim3d(-1,1)
        #Calculation of time delay at many points over the surface of the star
        colors = np.mean(delays[triangles], axis=1)
        sm = mpl.cm.ScalarMappable(cmap=cmap)
        sm.set_array([min(colors),max(colors)])
        ticks = np.linspace(min(colors),max(colors),10)   
        plt.colorbar(sm,ax=ax,ticks=ticks,shrink=0.5)
        ax.set_title("Time Delay (s)",y=1.15)
        soa = np.array([[0, 0, 1, 0, 0, 1], [0, 0, 1, np.cos(inclination), -np.sin(inclination)*np.sin(2*np.pi*phase), -np.sin(inclination)*np.cos(2*np.pi*phase)]])
        X, Y, Z, U, V, W = zip(*soa)
        param1 = .75
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        U,V,W = param1*U,param1*V,param1*W
        labels = ['Accretor', 'Observer']
        for i in range(len(labels)):
            ax.text(X[i]+U[i], Y[i]+V[i], Z[i]+W[i], labels[i])
        quiv = ax.quiver(X, Y, Z, U, V, W)
        
        triang = mtri.Triangulation(x/max(R_roche), y/max(R_roche), triangles)
        collec = ax.plot_trisurf(triang, z/max(R_roche), cmap=cmap, shade=False, linewidth=0.)
        collec.set_array(colors)
        collec.autoscale()
        ax.view_init(38, 0)
        
        # Make the intensity subplot
        cmap = cm.hot
        ax = fig.add_subplot(1,2,2, projection='3d')
        ax.set_xlim3d(-1,1)
        ax.set_ylim3d(-1,1)
        ax.set_zlim3d(-1,1)
        #Calculation of time delay at many points over the surface of the star
        colors = np.mean(apparent_intensities[triangles], axis=1)
        sm = mpl.cm.ScalarMappable(cmap=cmap)
        sm.set_array([min(colors),max(colors)])
        ticks = np.linspace(min(colors),max(colors),10)   
        plt.colorbar(sm,ax=ax,ticks=ticks,shrink=0.5)
        ax.set_title("Apparent Intensity",y=1.15)
        soa = np.array([[0, 0, 1, 0, 0, 1], [0, 0, 1, np.cos(inclination), -np.sin(inclination)*np.sin(2*np.pi*phase), -np.sin(inclination)*np.cos(2*np.pi*phase)]])
        X, Y, Z, U, V, W = zip(*soa)
        param1 = .75
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        U,V,W = param1*U,param1*V,param1*W
        labels = ['Accretor', 'Observer']
        for i in range(len(labels)):
            ax.text(X[i]+U[i], Y[i]+V[i], Z[i]+W[i], labels[i])
        quiv = ax.quiver(X, Y, Z, U, V, W)
        
        triang = mtri.Triangulation(x/max(R_roche), y/max(R_roche), triangles)
        collec = ax.plot_trisurf(triang, z/max(R_roche), cmap=cmap, shade=False, linewidth=0.)
        collec.set_array(colors)
        collec.autoscale()
        ax.view_init(90, 0)

        # Make a new plot showing the observed intensity vs. time delay
        plt.figure()
        # Here we can scale the apparent intensities by the spherical coordinate system Jacobian
        observed_intensities = (R_roche)**2*np.sin(PSI_roche)*apparent_intensities
        
        args = delays.argsort()
        delays = delays[args]
        observed_intensities = observed_intensities[args]
        
        block_size = 2*Q
        delays_reshaped = delays.reshape(-1, block_size)
        observed_intensities_reshaped = observed_intensities.reshape(-1, block_size)
        # Sum along the rows (axis=1) of the reshaped array
        delays_binned = np.mean(delays_reshaped, axis=1)
        observed_intensities_binned = np.mean(observed_intensities_reshaped, axis=1)
        
        plt.scatter(delays, observed_intensities,s=3)
        plt.scatter(delays_binned, observed_intensities_binned,label='binned')
        plt.axvline(x=centroid_delays,c='black',label=f'centorid = {round(centroid_delays,2)} s',linestyle='--')
        plt.xlabel('Time Delay (s)')
        plt.ylabel('Observed Intensity (a.u.)')
        
        plt.legend()
        plt.show()
    if mode == 'return_delay':
        return centroid_delays
    if mode == 'return_rv':
        return centroid_rv
        observed_intensities = R_roche**2*np.sin(PSI_roche)*apparent_intensities
        
        args = radial_velocity.argsort()
        radial_velocity = radial_velocity[args]
        observed_intensities = observed_intensities[args]
        block_size = 2*Q
        radial_velocity_reshaped = radial_velocity.reshape(-1, block_size)
        observed_intensities_reshaped = observed_intensities.reshape(-1, block_size)
        # Sum along the rows (axis=1) of the reshaped array
        radial_velocity_binned = np.mean(radial_velocity_reshaped, axis=1)
        observed_intensities_binned = np.mean(observed_intensities_reshaped, axis=1)
        
        if np.any(observed_intensities_binned > 0):
            vv = np.linspace(min(radial_velocity_binned[observed_intensities_binned>0]),max(radial_velocity_binned[observed_intensities_binned>0]),3*len(radial_velocity_binned[observed_intensities_binned>0]))
            f = scpint.UnivariateSpline(radial_velocity_binned[observed_intensities_binned>0], observed_intensities_binned[observed_intensities_binned>0],k=3)
            ii = f(vv)
        else:
            return 0
        
        if False:
            plt.scatter(radial_velocity_binned,observed_intensities_binned,c='black')
            plt.plot(vv,ii)
            plt.axvline(x=vv[np.argmax(ii)],c='black')
            plt.xlabel('Radial Velocity (km/s)')
            plt.ylabel('Intensity')
            plt.show()
        
        return vv[np.argmax(ii)]
        # ~ return radial_velocity_binned[np.argmax(observed_intensities_binned)]
        # ~ return tt[np.argmax(yy)]
        # ~ return centroid_rv
        
def calibrate_asra(Q = 50, inum = 10, anum = 10, phase = 0.25 ,mode = 'save'):
    """
    Generates the values of the azimuthal angle "beta" for the ASRA to use in approximating the fully 3D model.
    The generated values of beta are computed by taking the full 3d model Roche lobe and averaging beta over the surface
    with weighting to the apparent intensity model. Beta is restricted in this average between 0 and 90 deg because
    the Roche Lobe has C4 symmetry about the axis between CM2 and CM1.
    """
    
    m1_in, m2_in, period_in = 1.4, 0.7, 0.787 # We need these parameters to simulate, but hopefully they are not critical in the correction
    
    # Convert to SI units
    m1 = m1_in * constants.M_Sun
    m2 = m2_in * constants.M_Sun
    period = period_in * constants.day
    
    # Compute the orbital semi-major axis
    a = (constants.G*(m1+m2)*period**2/(4*np.pi**2))**(1/3)
    r2 = m1/(m1+m2)*a
    
    # Compute the location of the L1 Lagrange point
    roots = np.roots([(2*np.pi/period)**2,-r2*(2*np.pi/period)**2-2*a*(2*np.pi/period)**2,2*a*r2*(2*np.pi/period)**2 + a**2*(2*np.pi/period)**2,constants.G*(m1-m2)-r2*(2*np.pi/period)**2*a**2,2*a*constants.G*m2,-a**2*constants.G*m2])
    L1 = np.real(roots[np.argmin(np.abs(np.imag(roots)))])
    
    # Compute the value of the effective gravitational potential at the L1 point
    potential_at_L1 = rochePotential(L1,psi=0,beta=0,m1=m1,m2=m2,period=period)
    
    # Solve for the height of the Roche Lobe at the 45 degree angle (45 deg is a good compromise)
    H = opt.brentq(rochePotential,L1*1e-3,L1,args=(np.pi/2,np.pi/4,m1,m2,period,potential_at_L1),maxiter=100)
    
    print(f'H/L1 = {H/L1}')
    
    i_arr = np.linspace(0,90,inum)
    alpha_arr = np.linspace(0,20,anum)
    
    dhn_arr = np.tan(np.pi/180*alpha_arr)*a/H # Normalized disk height
    beta_arr = np.zeros([len(i_arr),len(alpha_arr)])
    
    print(f'Maximum Normalized Disk Height = {np.max(dhn_arr)}')
    
    for j in range(len(i_arr)):
        print(f'{round(100*j/len(i_arr))} % Complete')
        for k in range(len(alpha_arr)):
            # We always compute from phase 0.5
            beta_arr[j, k] = timeDelay_3d_full(0.5,m1_in=m1_in,m2_in=m2_in,inclination_in=i_arr[j],disk_angle_in=alpha_arr[k],period_in=period_in,Q=Q,mode='return_beta')
    
    beta_interp_func = scpint.RectBivariateSpline(i_arr, alpha_arr,beta_arr, kx=3, ky=3)
    beta_interp_func_dhn = scpint.RectBivariateSpline(i_arr, dhn_arr,beta_arr, kx=3, ky=3)
    
    if mode == 'plot':
        for ai in range(len(alpha_arr)):
            ii = np.linspace(0,90,500)
            bb = beta_interp_func(ii, alpha_arr[ai])
            plt.plot(ii,bb, label = r'$\alpha$ = ' + f'{alpha_arr[ai]} deg', color = cm.jet(ai/len(alpha_arr)))
            plt.scatter(i_arr,beta_arr[:,ai], color = cm.jet(ai/len(alpha_arr)))
        
        plt.xlabel('Inclination (deg)')
        plt.ylabel(r'ASRA $\beta$ (deg)')
        plt.legend()
        plt.show()
        
        fig, ax = plt.subplots()
        im = ax.imshow(beta_arr, extent = [alpha_arr[0],alpha_arr[-1],i_arr[0],i_arr[-1]], aspect= 'auto')
        plt.colorbar(im)
        ax.set_xlabel('Disk Shielding Angle (deg)')
        ax.set_ylabel('Inclination (deg)')
        ax.set_title('Raw Data')
        # ~ ax.set_aspect(len(alpha_arr)/len(i_arr))
        plt.show()
        
        fig, ax = plt.subplots()
        ii = np.linspace(0,90,100)
        hh = np.linspace(0,1.1,100)
        
        II, HH = np.meshgrid(ii, hh, indexing="ij")
        BB = beta_interp_func_dhn.ev(II, HH)
        
        im = ax.imshow(BB, extent = [hh[0],hh[-1],ii[0],ii[-1]], aspect= 'auto')
        plt.colorbar(im)
        ax.set_title('Interpolation')
        ax.set_xlabel('Normalized Disk Height')
        ax.set_ylabel('Inclination (deg)')
        # ~ ax.set_aspect(len(alpha_arr)/len(i_arr))
        plt.show()
    if mode == 'save':
        with open(f'asra_beta.pickle', 'wb') as outp:
            pickle.dump(beta_interp_func_dhn, outp)

def radialVelocity(phase, \
                m1_in=1.4, m2_in=0.7, inclination_in=44., period_in=0.787,\
                setting = 'cm', alpha_in = 0):
                    
    phase = np.mod(phase,1)
    m1 = m1_in * constants.M_Sun
    m2 = m2_in * constants.M_Sun
    inclination = inclination_in*np.pi/180
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
        if alpha_in == 0:
            N0, N1, N2, N3, N4 = 0.886, -1.132, 1.523, -1.892, 0.867
        if alpha_in == 8:
            N0, N1, N2, N3, N4 = 0.975, -1.559, 2.828, -3.576, 1.651

        return 1e-3*np.sin(2*np.pi*phase)*np.sin(inclination)*2*np.pi*a/period*(1/(1+q))*(N0+N1*q+N2*q**2+N3*q**3+N4*q**4)
        
        
    return -1*1e-3*np.sin(2*np.pi*phase)*np.sin(inclination)*2*np.pi*a/period*(1/(1+q))*(1-radiusDonor/a*(1+q))



def timeDelay_radialVelocity_3d(phase, \
                  m1_in=1.4, m2_in=0.7, inclination_in=44., disk_angle_in = 0, period_in=.787, plot = True, Q = 35, beta_approx = 0):
    beta_approx = beta_approx * np.pi/180
    # ~ print(phase,m1_in,m2_in,inclination_in,disk_angle_in)
    # Convert to SI units
    m1 = m1_in * constants.M_Sun
    m2 = m2_in * constants.M_Sun
    inclination = inclination_in*np.pi/180
    disk_angle = disk_angle_in*np.pi/180
    
    q = m2_in/m1_in
    
    period = period_in * constants.day
    # Compute the orbital semi-major axis
    a = (constants.G*(m1+m2)*period**2/(4*np.pi**2))**(1/3)
    r2 = m1/(m1+m2)*a
    
    # Compute the location of the L1 Lagrange point
    roots = np.roots([(2*np.pi/period)**2,-r2*(2*np.pi/period)**2-2*a*(2*np.pi/period)**2,2*a*r2*(2*np.pi/period)**2 + a**2*(2*np.pi/period)**2,constants.G*(m1-m2)-r2*(2*np.pi/period)**2*a**2,2*a*constants.G*m2,-a**2*constants.G*m2])
    L1 = np.real(roots[np.argmin(np.abs(np.imag(roots)))]) # There should only be one purely real solution in the 5 complex solutions, use that one
    
    # ~ print(f'L1 = {L1/constants.AU}, a = {a/constants.AU}, r2 = {r2/constants.AU}')
    
    # Compute the value of the effective gravitational potential at the L1 point
    potential_at_L1 = rochePotential(L1,0,0,m1=m1,m2=m2,period=period)
    # Make an array of polar coordinates for constructing the Roche Lobe, if not plotting speed up by dropping the bottom half of the Roche lobe
    if plot:
        psi_roche = np.linspace(0, np.pi, num=Q, endpoint=True)
    else:
        psi_roche = np.linspace(0, np.pi/2, num=Q, endpoint=True)
        
    r_roche = np.zeros(len(psi_roche))
    for i in range(len(psi_roche)):
        r_roche[i] = opt.brentq(rochePotential,L1*1e-3,L1,args=(psi_roche[i],np.radians(90),m1,m2,period,potential_at_L1),maxiter=100)
       
    # Azimuthal angle for the Roche Lobe
    beta_roche = np.linspace(0, 2*np.pi, num=Q, endpoint=True)
    
    # Make a meshgrid of the polar angles
    PSI_roche, BETA_roche = np.meshgrid(psi_roche, beta_roche)
    R_roche = np.tile(r_roche, (len(beta_roche),1))
    # Now we unravel this into 1D arrays
    R_roche, PSI_roche, BETA_roche = R_roche.ravel(), PSI_roche.ravel(), BETA_roche.ravel()
    
    # Distance from points on surface of Roche lobe to compact object
    p1 = np.sqrt(r_roche**2 + a**2 - 2*r_roche*a*np.cos(psi_roche))
    P1 = np.tile(p1, (len(beta_roche),1)).ravel()
    # Angle between p2 (vector from cm of donor to point on surface) and e (vector from cm of binary system to earth)
    p2hat_dot_ehat = -np.cos(PSI_roche)*np.sin(inclination)*np.cos(2*np.pi*phase) +\
                  -np.sin(PSI_roche)*np.sin(BETA_roche)*np.sin(inclination)*np.sin(2*np.pi*phase) +\
                  +np.sin(PSI_roche)*np.cos(BETA_roche)*np.cos(inclination)
    # Angle between psi_hat and e (psi_hat is the polar unit vector in the spherical coordinate system where p2 is the radial coordinate)
    psihat_dot_ehat = np.sin(PSI_roche)*np.sin(inclination)*np.cos(2*np.pi*phase) +\
              -np.cos(PSI_roche)*np.sin(BETA_roche)*np.sin(inclination)*np.sin(2*np.pi*phase) +\
              +np.cos(PSI_roche)*np.cos(BETA_roche)*np.cos(inclination)
    
    # Compute the delay times across the Roche Lobe
    delays = P1 - a*np.cos(2*np.pi*phase)*np.sin(inclination) - R_roche*p2hat_dot_ehat
    
    delays = delays/constants.c
    
    # Compute the radial velocity over the Roche Lobe
    radial_velocity = r2*(2*np.pi/period)*np.sin(inclination)*np.sin(2*np.pi*phase) +\
                      -1*R_roche*(2*np.pi/period)*np.sin(inclination) *\
                      (np.cos(PSI_roche)*np.sin(2*np.pi*phase) - np.sin(PSI_roche)*np.sin(BETA_roche)*np.cos(2*np.pi*phase))
    # Convert the radial velocity to km/s from m/s
    radial_velocity *= 1e-3
    
    # ~ radial_velocity = 1/(2*np.pi)*np.arctan2(r2-R_roche*np.cos(PSI_roche),R_roche*np.sin(PSI_roche)*np.sin(BETA_roche))
    
    #################################
    # Now let's compute the apparent intensity over the Roche Lobe, we have 3 terms A1, A2, and A3
    
    #Let's compute A1: Distance Attenuation
    
    a1 = p1**(-2) # This is a 1D version of the list
    
    #Let's compute A2: Projected area on surface of donor towards accretor
    
    # First we need to compute vectors normal to the Roche Lobe surface via a gradient of the potential function
    # Since this function only depends on r_roche (aka p2) and psi, we can use the 1D arrays
    vecs_normal_roche = polarGradRochePotential(r_roche,psi_roche,np.radians(0),m1,m2,period)# returns dV/dp2 vec(p2) + 1/p2*dV/dpsi vec(psi)
    
    # Manage special case of psi = 0 where the gradient function can spuriously evaluate to zero
    psi_zero_indices = (psi_roche == 0)
    vecs_normal_roche[0,psi_zero_indices] = 1
    vecs_normal_roche[1,psi_zero_indices] = 0
    
    # Normalize the vectors to 1
    vecs_normal_roche = vecs_normal_roche/np.sqrt(vecs_normal_roche[0]**2+vecs_normal_roche[1]**2)

    # Take a normalized dot product of roche normal with the -p1 vector (p1 points from accretor to point on surface of donor)
    a2 = (vecs_normal_roche[0]*(a*np.cos(psi_roche)-r_roche) - vecs_normal_roche[1]*a*np.sin(psi_roche)) \
         /(np.sqrt(r_roche**2 + a**2 - 2*r_roche*a*np.cos(psi_roche)))
    a2[a2<0] = 0 # Make strictly positive or zero
    
    # Now convert the product A1*A2 to the correct shape accounting for azimuthal angles
    A1A2 = np.tile(a1*a2, (len(beta_roche),1)).ravel()
    
    # We will implement the disk shielding angle here by taking all values of A1A2 corresponding to angles below the shielding angle to zero
    A1A2[R_roche/P1*np.abs(np.sin(PSI_roche)*np.cos(BETA_roche)) < np.sin(disk_angle)] = 0
    
    #Let's compute A3: Projected area on surface of donor towards observer
    
    # We need to convert the normal vectors to a meshgrid because we no longer have azimuthal symmetry
    dV_dp2 = np.tile(vecs_normal_roche[0], (len(beta_roche),1)).ravel()
    dV_dpsi_on_p2 = np.tile(vecs_normal_roche[1], (len(beta_roche),1)).ravel()
    
    A3 = dV_dp2 * p2hat_dot_ehat + dV_dpsi_on_p2 * psihat_dot_ehat
    A3[A3<0] = 0 # Make strictly positive or zero

    T_0 = 10
    apparent_intensities = T_0*A1A2*A3
    
    if np.sum(R_roche**2*np.sin(PSI_roche)*apparent_intensities) != 0:
        centroid_delays = np.average(delays,weights=R_roche**2*np.sin(PSI_roche)*apparent_intensities) # Compute the centroid of the distribution of delay times weighted to the intensities
        centroid_radial_velocity = np.average(radial_velocity,weights=R_roche**2*np.sin(PSI_roche)*apparent_intensities) # Compute the centroid of the distribution of radial velocities weighted to the intensities
    else:
        centroid_delays = 0
        centroid_radial_velocity = 0
    
    if plot == True:
        x, y, z = R_roche*np.cos(BETA_roche)*np.sin(PSI_roche), R_roche*np.sin(BETA_roche)*np.sin(PSI_roche), R_roche*np.cos(PSI_roche)
        
        mesh_x, mesh_y = (PSI_roche, BETA_roche)
        triangles = mtri.Triangulation(mesh_y, mesh_x).triangles
        
        # Plotting
        fig = plt.figure()
        
        # Make the delays subplot
        cmap = cm.cool
        ax = fig.add_subplot(1,3,1, projection='3d')
        ax.set_xlim3d(-1,1)
        ax.set_ylim3d(-1,1)
        ax.set_zlim3d(-1,1)
        #Calculation of time delay at many points over the surface of the star
        colors = np.mean(delays[triangles], axis=1)
        sm = mpl.cm.ScalarMappable(cmap=cmap)
        sm.set_array([min(colors),max(colors)])
        ticks = np.linspace(min(colors),max(colors),10)   
        plt.colorbar(sm,ax=ax,ticks=ticks,shrink=0.5)
        ax.set_title("Time Delay (s)",y=1.15)
        soa = np.array([[0, 0, 1, 0, 0, 1], [0, 0, 1, np.cos(inclination), -np.sin(inclination)*np.sin(2*np.pi*phase), -np.sin(inclination)*np.cos(2*np.pi*phase)]])
        X, Y, Z, U, V, W = zip(*soa)
        param1 = .75
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        U,V,W = param1*U,param1*V,param1*W
        labels = ['Accretor', 'Observer']
        for i in range(len(labels)):
            ax.text(X[i]+U[i], Y[i]+V[i], Z[i]+W[i], labels[i])
        quiv = ax.quiver(X, Y, Z, U, V, W)
        
        triang = mtri.Triangulation(x/max(R_roche), y/max(R_roche), triangles)
        collec = ax.plot_trisurf(triang, z/max(R_roche), cmap=cmap, shade=False, linewidth=0.)
        collec.set_array(colors)
        collec.autoscale()
        ax.view_init(38, 0)
        
        # Make the radial velocity subplot
        cmap = cm.jet
        # ~ cmap = cm.bwr
        ax = fig.add_subplot(1,3,2, projection='3d')
        ax.set_axis_off()
        ax.set_xlim3d(-1,1)
        ax.set_ylim3d(-1,1)
        ax.set_zlim3d(-1,1)
        #Calculation of time delay at many points over the surface of the star
        colors = np.mean(radial_velocity[triangles], axis=1)
        sm = mpl.cm.ScalarMappable(cmap=cmap)
        sm.set_array([min(colors),max(colors)])
        ticks = np.linspace(min(colors),max(colors),10)   
        plt.colorbar(sm,ax=ax,ticks=ticks,shrink=0.5)
        ax.set_title("Radial Velocity",y=1.15)
        soa = np.array([[0, 0, 1, 0, 0, 1], [0, 0, 1, np.cos(inclination), -np.sin(inclination)*np.sin(2*np.pi*phase), -np.sin(inclination)*np.cos(2*np.pi*phase)]])
        X, Y, Z, U, V, W = zip(*soa)
        param1 = .75
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        U,V,W = param1*U,param1*V,param1*W
        labels = ['Accretor', 'Observer']
        # ~ for i in range(len(labels)):
            # ~ ax.text(X[i]+U[i], Y[i]+V[i], Z[i]+W[i], labels[i])
        # ~ quiv = ax.quiver(X, Y, Z, U, V, W)
        
        triang = mtri.Triangulation(x/max(R_roche), y/max(R_roche), triangles)
        collec = ax.plot_trisurf(triang, z/max(R_roche), cmap=cmap, shade=False, linewidth=0.)
        collec.set_array(colors)
        collec.autoscale()
        ax.view_init(38, 0)
        
        # Make the intensity subplot
        cmap = cm.hot
        ax = fig.add_subplot(1,3,3, projection='3d')
        ax.set_axis_off()
        ax.set_xlim3d(-1,1)
        ax.set_ylim3d(-1,1)
        ax.set_zlim3d(-1,1)
        #Calculation of time delay at many points over the surface of the star
        colors = np.mean(apparent_intensities[triangles], axis=1)
        sm = mpl.cm.ScalarMappable(cmap=cmap)
        sm.set_array([min(colors),max(colors)])
        ticks = np.linspace(min(colors),max(colors),10)   
        plt.colorbar(sm,ax=ax,ticks=ticks,shrink=0.5)
        ax.set_title("Apparent Intensity",y=1.15)
        soa = np.array([[0, 0, 1, 0, 0, 1], [0, 0, 1, np.cos(inclination), -np.sin(inclination)*np.sin(2*np.pi*phase), -np.sin(inclination)*np.cos(2*np.pi*phase)]])
        X, Y, Z, U, V, W = zip(*soa)
        param1 = .75
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        U,V,W = param1*U,param1*V,param1*W
        labels = ['Accretor', 'Observer']
        for i in range(len(labels)):
            ax.text(X[i]+U[i], Y[i]+V[i], Z[i]+W[i], labels[i])
        quiv = ax.quiver(X, Y, Z, U, V, W)
        
        triang = mtri.Triangulation(x/max(R_roche), y/max(R_roche), triangles)
        collec = ax.plot_trisurf(triang, z/max(R_roche), cmap=cmap, shade=False, linewidth=0.)
        collec.set_array(colors)
        collec.autoscale()
        ax.view_init(38, 0)

        # Make a new plot showing the observed intensity vs. time delay
        plt.figure()
        # Here we can scale the apparent intensities by the spherical coordinate system Jacobian
        observed_intensities = R_roche**2*np.sin(PSI_roche)*apparent_intensities
        
        args = delays.argsort()
        delays = delays[args]
        observed_intensities = observed_intensities[args]
        
        block_size = Q
        delays_reshaped = delays.reshape(-1, block_size)
        observed_intensities_reshaped = observed_intensities.reshape(-1, block_size)
        # Sum along the rows (axis=1) of the reshaped array
        delays_binned = np.mean(delays_reshaped, axis=1)
        observed_intensities_binned = np.mean(observed_intensities_reshaped, axis=1)
        
        plt.scatter(delays, observed_intensities,s=3)
        plt.scatter(delays_binned, observed_intensities_binned,label='binned')
        # ~ f = scpint.interp1d(delays_binned, observed_intensities_binned,kind='quadratic')
        # ~ tt = np.linspace(min(delays_binned),max(delays_binned),350)
        # ~ yy = f(tt)
        # ~ plt.plot(tt,yy)
        plt.axvline(x=centroid_delays,c='black',label=f'centorid = {round(centroid_delays,2)} s',linestyle='--')
        
        plt.legend()
        plt.show()
    else:
        return centroid_delays, centroid_radial_velocity



# ~ calibrate_asra(mode='plot')
    # ~ print(alphas)
    # ~ print(np.shape(data))
# ~ load_and_interp_munoz_darias()

if __name__ == '__main__':
    model = ASRA_LT_model()
    
    
    



