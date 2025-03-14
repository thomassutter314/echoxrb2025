import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.image as image
from matplotlib.collections import LineCollection
from matplotlib import patches
import matplotlib.tri as mtri
import matplotlib as mpl
import scipy.optimize as opt
import pickle
import corner

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
# ~ plt.rcParams["font.family"] = "Arial"
# ~ plt.rcParams['svg.fonttype'] = 'none'
# ~ plt.rcParams['pdf.use14corefonts'] = True

from mcmc_fitting import MCMC_manager, Priors
import constants
import delay_model

def plot_colorline(x,y,c):
    col = cm.cool((c-np.min(c))/(np.max(c)-np.min(c)))
    ax = plt.gca()
    for i in np.arange(len(x)-1):
        ax.plot([x[i],x[i+1]], [y[i],y[i+1]], c=col[i])
    im = ax.scatter(x, y, c=c, s=0, cmap=cm.jet)
    return im

def multiline(xs, ys, c, ax=None, **kwargs):
    """Plot lines with different colorings

    Parameters
    ----------
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    ax (optional): Axes to plot on.
    kwargs (optional): passed to LineCollection

    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)

    Returns
    -------
    lc : LineCollection instance.
    """

    # find axes
    ax = plt.gca() if ax is None else ax

    # create LineCollection
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)

    # set coloring of line segments
    #    Note: I get an error if I pass c as a list here... not sure why.
    lc.set_array(np.asarray(c))

    # add lines to axes and rescale 
    #    Note: adding a collection doesn't autoscalee xlim/ylim
    ax.add_collection(lc)
    ax.autoscale()
    return lc

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



### Figure 1 ###

def fig1_E(inclination_in = 44, disk_angle_in = 3, m2_in = 0.5*1.4, m1_in = 1.4, period_in = 0.787):
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

def fig1_F(inclination_in = 44, disk_angle_in = 3, m2_in = 0.5*1.4, m1_in = 1.4, period_in = 0.787):
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

def fig1_A(m2_in = 0.7, m1_in = 1.4, period_in = 0.787, inclination_in = 44, annotate = True):
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


    ax.plot([0,a],[0,0],'k--')
    ax.text(x=0.96*a,y=-0.075*a,s='CM1')
    ax.text(x=0,y=-0.075*a,s='CM2')
    ax.text(x=L1,y=0.05*L1,s=r'$L_1$')
    
    if annotate:
        ax.arrow(x=r2, y=0, dx=0.15*a*np.sin(inclination), dy=0.15*a*np.cos(inclination),head_width=0.025*a,fc='black')
        ax.arrow(x=r2, y=0, dx=0, dy=0.15*a,head_width=0.025*a,fc='black')
        ax.arrow(x=r2, y=0, dy=0, dx=0.15*a,head_width=0.025*a,fc='black')
        
        index = 300
        scale = 0.86
        ax.arrow(x=0, y=0, dx=scale*r_roche_0[index]*np.cos(psi_roche[index]), dy=scale*r_roche_0[index]*np.sin(psi_roche[index]),head_width=0.025*a,fc='black')
        
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
    
    ax.set_axis_off()
    
    # ~ plt.legend()
    plt.show()

def fig1_BCD(phase = 0.65, m2_in = 0.7, m1_in = 1.4, period_in = 0.787, inclination_in = 44, disk_angle_in = 3, mode = 'plot', Q = 50):
    
    plot_labels = False
    plot_arrows = True
    
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
    potential_at_L1 = delay_model.rochePotential(L1,psi=0,beta=0,m1=m1,m2=m2,period=period)
    
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
        R_roche[i] = opt.brentq(delay_model.rochePotential,L1*1e-3,L1,args=(PSI_roche[i],BETA_roche[i],m1,m2,period,potential_at_L1),maxiter=100)
    
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
    vecs_normal_roche = delay_model.polarGradRochePotential(R_roche,PSI_roche,BETA_roche,m1,m2,period)# returns dV/dp2 vec(p2) + 1/p2*dV/dpsi vec(psi)
    
    # Manage special case of psi = 0 where the gradient function can spuriously evaluate to zero
    psi_zero_indices = PSI_roche == 0
    # ~ vecs_normal_roche[0,psi_zero_indices] = 1
    # ~ vecs_normal_roche[1,psi_zero_indices] = 0
    
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
        pview, aview = 10, -16
        
        # ~ pview, aview = 0, 90
        
        x, y, z = R_roche*np.cos(BETA_roche)*np.sin(PSI_roche), R_roche*np.sin(BETA_roche)*np.sin(PSI_roche), R_roche*np.cos(PSI_roche)
        
        triangles = mtri.Triangulation(PSI_roche, BETA_roche).triangles
        
        # Plotting
        fig = plt.figure()
        
        # Make the delays subplot
        cmap = cm.Blues
        ax = fig.add_subplot(1,1,1, projection='3d')
        ax.set_axis_off()
        ax.set_xlim3d(-1,1)
        ax.set_ylim3d(-1,1)
        ax.set_zlim3d(-1,1)
        #Calculation of time delay at many points over the surface of the star
        colors = np.mean(delays[triangles], axis=1)
        sm = mpl.cm.ScalarMappable(cmap=cmap)
        
        vmin, vmax = min(delays), max(delays)
        sm.set_array([vmin,vmax])
        ticks = np.linspace(vmin,vmax,3)   
        
        cbar = plt.colorbar(sm,ax=ax,ticks=ticks,shrink=0.5, label = "Time Delay (s)")
        # ~ ax.set_title("Time Delay (s)",y=1.15)
        soa = np.array([[0, 0, 1, 0, 0, 1], [0, 0, 1, np.cos(inclination), -np.sin(inclination)*np.sin(2*np.pi*phase), -np.sin(inclination)*np.cos(2*np.pi*phase)]])
        X, Y, Z, U, V, W = zip(*soa)
        param1 = .75
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        U,V,W = param1*U,param1*V,param1*W
        labels = ['Accretor', 'Observer']
        if plot_labels:
            for i in range(len(labels)):
                ax.text(X[i]+U[i], Y[i]+V[i], Z[i]+W[i], labels[i])
        if plot_arrows:
            quiv = ax.quiver(X, Y, Z, U, V, W)
        
        
        # ~ # create x,y
        # ~ xx_plane, yy_plane = np.meshgrid(np.arange(-1,1,0.1), np.arange(-1,1,0.1))
        # ~ # calculate corresponding z
        # ~ normal = [1,1e-3,1e-3]
        # ~ zz_plane = (-normal[0] * xx_plane - normal[1] * yy_plane)/normal[2]
        # ~ ax.plot_surface(xx_plane, yy_plane, zz_plane, alpha=0.2)
        
        triang = mtri.Triangulation(x/max(R_roche), y/max(R_roche), triangles)
        collec = ax.plot_trisurf(triang, z/max(R_roche), cmap=cmap, shade=False, linewidth=0.2, antialiased = True, edgecolor = 'black', vmin = np.min(colors), vmax = 14)
        #fig.colorbar(collec)
        collec.set_array(colors)
        # ~ collec.autoscale()
        
        # ~ indices = np.logical_and(delays > 10,delays < 12)
        # ~ scale_up_scatter = 1.01
        # ~ x_c = scale_up_scatter*x[indices]/max(R_roche)
        # ~ y_c = scale_up_scatter*y[indices]/max(R_roche)
        # ~ z_c = scale_up_scatter*z[indices]/max(R_roche)
        
        # ~ ax.scatter(x_c, y_c, z_c, color = 'black')

        ax.view_init(pview, aview)
        
        # Make the radial velocity subplot
        cmap = cm.Reds
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1, projection='3d')
        ax.set_axis_off()
        ax.set_xlim3d(-1,1)
        ax.set_ylim3d(-1,1)
        ax.set_zlim3d(-1,1)
        #Calculation of time delay at many points over the surface of the star
        colors = np.mean(radial_velocity[triangles], axis=1)
        sm = mpl.cm.ScalarMappable(cmap=cmap)
        sm.set_array([np.min(colors),np.max(colors)])
        ticks = np.linspace(np.min(colors),np.max(colors),3)   
        cbar = plt.colorbar(sm,ax=ax,ticks=ticks,shrink=0.5,label = 'Radial Velocity (km/s)')
        # ~ ax.set_title("Radial Velocity (km/s)",y=1.15)
        soa = np.array([[0, 0, 1, 0, 0, 1], [0, 0, 1, np.cos(inclination), -np.sin(inclination)*np.sin(2*np.pi*phase), -np.sin(inclination)*np.cos(2*np.pi*phase)]])
        X, Y, Z, U, V, W = zip(*soa)
        param1 = .75
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        U,V,W = param1*U,param1*V,param1*W
        labels = ['Accretor', 'Observer']
        if plot_labels:
            for i in range(len(labels)):
                ax.text(X[i]+U[i], Y[i]+V[i], Z[i]+W[i], labels[i])
        if plot_arrows:
            quiv = ax.quiver(X, Y, Z, U, V, W)
        
        triang = mtri.Triangulation(x/max(R_roche), y/max(R_roche), triangles)
        collec = ax.plot_trisurf(triang, z/max(R_roche), cmap=cmap, shade=False, linewidth=0.2, antialiased = True, edgecolor = 'black', vmin = np.min(colors), vmax = 120)
        #fig.colorbar(collec)
        collec.set_array(colors)
        # ~ collec.autoscale()
        ax.view_init(pview, aview)
        
        # Make the intensity subplot
        cmap = cm.hot
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1, projection='3d')
        ax.set_axis_off()
        ax.set_xlim3d(-1,1)
        ax.set_ylim3d(-1,1)
        ax.set_zlim3d(-1,1)
        #Calculation of time delay at many points over the surface of the star
        colors = np.mean(apparent_intensities[triangles], axis=1)
        colors = 100*colors/np.max(colors)
        sm = mpl.cm.ScalarMappable(cmap=cmap)
        sm.set_array([min(colors),max(colors)])
        ticks = np.linspace(min(colors),max(colors),3)   
        cbar = plt.colorbar(sm,ax=ax,ticks=ticks,shrink=0.5, label = 'Apparent Intensity')
        
        # ~ ax.set_title("Apparent Intensity",y=1.15)
        soa = np.array([[0, 0, 1, 0, 0, 1], [0, 0, 1, np.cos(inclination), -np.sin(inclination)*np.sin(2*np.pi*phase), -np.sin(inclination)*np.cos(2*np.pi*phase)]])
        X, Y, Z, U, V, W = zip(*soa)
        param1 = .75
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        U,V,W = param1*U,param1*V,param1*W
        labels = ['Accretor', 'Observer']
        if plot_labels:
            for i in range(len(labels)):
                ax.text(X[i]+U[i], Y[i]+V[i], Z[i]+W[i], labels[i])
        if plot_arrows:
            quiv = ax.quiver(X, Y, Z, U, V, W)
        
        triang = mtri.Triangulation(x/max(R_roche), y/max(R_roche), triangles)
        # ~ collec = ax.plot_trisurf(triang, z/max(R_roche), cmap=cmap, shade=False, linewidth=0.)
        
        collec = ax.plot_trisurf(triang, z/max(R_roche), cmap=cmap, shade=False, linewidth=0.2, antialiased = True, edgecolor = 'black', vmin = np.min(colors), vmax = np.max(colors))
        #fig.colorbar(collec)
        
        collec.set_array(colors)
        collec.autoscale()
        ax.view_init(pview, aview)
        
        q = m2/m1
        egg_radiusDonor = a*(.49*q**(.667))/(.6*q**(.667)+np.log(1+q**(.333)))
        egg_volume = 4/3*np.pi*egg_radiusDonor**3
        # ~ fig.suptitle(f'Roche Lobe Volume: {1/3*np.sum(vecs_normal_roche[0]*R_roche**3*np.sin(PSI_roche))*(np.pi/Q)*(2*np.pi/Q)} m^3 \n Eggleton Volume = {egg_volume} m^3 \n Centroid Beta = {centroid_beta} deg \n Centroid Delay = {centroid_delays} s')

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
        
        axs[0].scatter(delays, observed_intensities,s=3)
        axs[0].scatter(delays_binned, observed_intensities_binned,label='binned')
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
        
        axs[1].scatter(radial_velocity, observed_intensities,s=3)
        axs[1].scatter(radial_velocity_binned, observed_intensities_binned,label='binned')
        axs[1].axvline(x=centroid_rv,c='black',label=f'centorid = {round(centroid_rv,2)} km/s',linestyle='--')
        axs[1].set_xlabel('Radial Velocity (km/s)')
        axs[1].set_ylabel('Observed Intensity (a.u.)')
        
        plt.legend()
        plt.show()
        

### Figure 2 ###

def fig2():
    m1_in = 1.4
    period_in = 0.787
    
    inclination_in = 40
    alphas = np.arange(0,18,5)
    # ~ alphas = [0,10]
    
    fig, axs = plt.subplots(2,1)

    for alpha_index in range(len(alphas)):
        Q = np.linspace(0.1,1,10)
        print('alpha',alphas[alpha_index])
        k_corr = np.zeros(len(Q))
        for i in range(len(Q)):
            k_corr[i] = delay_model.timeDelay_3d_full(0.25, m1_in = m1_in, m2_in = m1_in*Q[i], period_in = period_in, disk_angle_in = alphas[alpha_index], inclination_in = inclination_in, Q = 25, mode='return_rv')/delay_model.radialVelocity(0.25, m1_in = m1_in, m2_in = m1_in*Q[i], period_in = period_in, inclination_in = inclination_in)
       
        axs[0].plot(Q[k_corr != 0], k_corr[k_corr != 0], label = r'$\alpha$' + f' = {alphas[alpha_index]}',color=cm.plasma(alpha_index/len(alphas)))
        
        
    q = np.linspace(0.1,1,30)
    k_corr_egg = delay_model.radialVelocity(0.25, m1_in = m1_in, m2_in = m1_in*q, period_in = period_in, inclination_in = inclination_in,setting='egg')/delay_model.radialVelocity(0.25, m1_in = m1_in, m2_in = m1_in*q, period_in = period_in, inclination_in = inclination_in)
    axs[0].plot(q, k_corr_egg, 'r--', label =  'SP Eggleton')
    
    k_corr_egg = delay_model.radialVelocity(0.25, m1_in = m1_in, m2_in = m1_in*q, period_in = period_in, inclination_in = inclination_in,setting='plav')/delay_model.radialVelocity(0.25, m1_in = m1_in, m2_in = m1_in*q, period_in = period_in, inclination_in = inclination_in)
    axs[0].plot(q, k_corr_egg, 'k-.', label =  'SP Plavec')
        
    axs[0].set_xlabel(r'$q=m_2/m_1$')
    axs[0].set_ylabel(r'$K_{\text{corr}}$')
    # ~ axs[0].legend()
    # ~ axs[0].set_title(f'inclination = {inclination_in} deg')
    # ~ axs[0].set_ylim([-.1,1.25])
    
    # ~ phases = np.linspace(0.2,0.8,20)
    phases = np.linspace(0.25,0.3,30)
    tt, vv = np.zeros(len(phases)), np.zeros(len(phases))
    
    model = delay_model.ASRA_LT_model(period_in = 0.787)
    
    # ~ inclinations = [30,50,70,90]
    inclinations = np.linspace(1,89,10)
    for alpha_index in range(len(alphas)):
        print(alpha_index)
        max_phases = []
        for inclination in inclinations:
            tt, vv = model.evaluate_array(phases, m1_in = m1_in, m2_in = 0.5*m1_in, period_in = period_in, disk_angle_in = alphas[alpha_index], inclination_in = inclination, Q = 50)
            # ~ print(alphas[alpha_index], inclination, max(vv))
            # ~ axs[1].plot(phases, vv)
            # ~ axs[1].scatter([phases[np.argmax(vv)]], [np.max(vv)])
            max_phases.append(phases[np.argmax(-vv)])
        
        popt = np.polyfit(inclinations, max_phases, deg = 2)
        ii = np.linspace(0,90,100)
        vv_poly = popt[0]*ii**2 + popt[1]*ii + 0.25
        axs[1].plot(ii, vv_poly, color=cm.plasma(alpha_index/len(alphas)))
        # ~ axs[1].scatter(inclinations, max_phases, color=cm.plasma(alpha_index/len(alphas)))
    
    axs[1].set_ylabel('Maximizing Phase')
    axs[1].set_xlabel(r'Inclination $(^{\circ})$')
        
    # ~ plt.xlabel('Inclination')
    # ~ plt.ylabel('Max Phase')
    plt.show()

def fig2_phase_map(phase = 0.65, m2_in = 0.7, m1_in = 1.4, period_in = 0.787, inclination_in = 44, disk_angle_in = 3, mode = 'plot', Q = 22):
    
    print('Each panel on the Roche lobe contributes its own radial velocity signal.')
    print('This plot shows the phase for which that signal is a maximum as a function of position on the Roche lobe.')

    
    plot_labels = False
    plot_arrows = True
    
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
    potential_at_L1 = delay_model.rochePotential(L1,psi=0,beta=0,m1=m1,m2=m2,period=period)
    
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
        R_roche[i] = opt.brentq(delay_model.rochePotential,L1*1e-3,L1,args=(PSI_roche[i],BETA_roche[i],m1,m2,period,potential_at_L1),maxiter=100)
    
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
    vecs_normal_roche = delay_model.polarGradRochePotential(R_roche,PSI_roche,BETA_roche,m1,m2,period)# returns dV/dp2 vec(p2) + 1/p2*dV/dpsi vec(psi)
    
    # Manage special case of psi = 0 where the gradient function can spuriously evaluate to zero
    psi_zero_indices = (PSI_roche == 0)
    #vecs_normal_roche[0,psi_zero_indices] = 1
    #vecs_normal_roche[1,psi_zero_indices] = 0
    
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
    
    
    maximizing_phases = np.arctan2((r2 - R_roche * np.cos(PSI_roche)),(R_roche * np.sin(PSI_roche) * np.sin(BETA_roche)))/(2*np.pi)
    
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
        pview, aview = 10, -16
        
        # ~ pview, aview = 0, 90
        
        x, y, z = R_roche*np.cos(BETA_roche)*np.sin(PSI_roche), R_roche*np.sin(BETA_roche)*np.sin(PSI_roche), R_roche*np.cos(PSI_roche)
        
        triangles = mtri.Triangulation(PSI_roche, BETA_roche).triangles
        
        # Plotting
        fig = plt.figure()
        
        # Make the maximizing phase plot
        cmap = cm.bwr
        ax = fig.add_subplot(1,1,1, projection='3d')
        ax.set_axis_off()
        ax.set_xlim3d(-1,1)
        ax.set_ylim3d(-1,1)
        ax.set_zlim3d(-1,1)
        #Calculation of time delay at many points over the surface of the star
        
        
        colors = np.mean(maximizing_phases[triangles], axis=1)
        
        
        sm = mpl.cm.ScalarMappable(cmap=cmap)
        
        vmin, vmax = min(delays), max(delays)
        sm.set_array([vmin,vmax])
        ticks = np.linspace(vmin,vmax,3)   
        
        # ~ cbar = plt.colorbar(sm,ax=ax,ticks=ticks,shrink=0.5, label = "Time Delay (s)")
        # ~ ax.set_title("Time Delay (s)",y=1.15)
        soa = np.array([[0, 0, 1, 0, 0, 1], [0, 0, 1, np.cos(inclination), -np.sin(inclination)*np.sin(2*np.pi*phase), -np.sin(inclination)*np.cos(2*np.pi*phase)]])
        X, Y, Z, U, V, W = zip(*soa)
        param1 = .75
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        U,V,W = param1*U,param1*V,param1*W
        labels = ['Accretor', 'Observer']
        if plot_labels:
            for i in range(len(labels)):
                ax.text(X[i]+U[i], Y[i]+V[i], Z[i]+W[i], labels[i])
        if plot_arrows:
            quiv = ax.quiver(X, Y, Z, U, V, W)
        
        triang = mtri.Triangulation(x/max(R_roche), y/max(R_roche), triangles)
        collec = ax.plot_trisurf(triang, z/max(R_roche), cmap=cmap, shade=False, linewidth=0.2, antialiased = True, edgecolor = 'black', vmin = np.min(colors), vmax = np.max(colors))
        fig.colorbar(collec)
        collec.set_array(colors)

        ax.set_axis_off()
        ax.set_xlim3d(-1,1)
        ax.set_ylim3d(-1,1)
        ax.set_zlim3d(-1,1)

        
        plt.legend()
        plt.show()
      

### Figure 3 ###      

def fig3():

    echo = '1737768183'
    
    rv = '1737762632'
    
    both = '1737764995'
    
    alpha = 1
    s = 1
    
    no_fill_contours = False
    fill_contours = False
    plot_datapoints = False
    plot_density = True
    smooth = 1
    levels = [0.68, 0.95]
    
    contour_kwargs = {'linestyles': '-'}
    
    c_red = 'red'
    c_blue = 'blue'
    c_purple = '#580066'
    
    
    with open(f'mcmc_results//{rv}//sampler.pickle', 'rb') as inp:
        sampler = pickle.load(inp)
    samples1 = sampler.get_chain(discard=0, thin=1, flat=True)
    # ~ plt.scatter(samples[:,2],samples[:,1]/samples[:,0], s = s, alpha = alpha, label = 'radial velocity only', c = 'red')

    with open(f'mcmc_results//{echo}//sampler.pickle', 'rb') as inp:
        sampler = pickle.load(inp)
    samples2 = sampler.get_chain(discard=0, thin=1, flat=True)
    # ~ plt.scatter(samples[:,2],samples[:,1]/samples[:,0], s = s, alpha = alpha, label = 'echo delay only', c = 'blue')

    with open(f'mcmc_results//{both}//sampler.pickle', 'rb') as inp:
        sampler = pickle.load(inp)
    samples3 = sampler.get_chain(discard=0, thin=1, flat=True)
    
    # ~ {'mpl2005', 'mpl2014', 'serial', 'threaded'}
    
    corner.hist2d(samples1[:,2],samples1[:,1]/samples1[:,0], levels=levels, smooth = smooth,\
                  plot_datapoints = plot_datapoints, color = c_red, alpha = alpha, label = 'radial velocity only', no_fill_contours = no_fill_contours, plot_density = plot_density, fill_contours = fill_contours, contour_kwargs = contour_kwargs)
    
    corner.hist2d(samples2[:,2],samples2[:,1]/samples2[:,0], levels=levels, smooth = smooth,\
                  plot_datapoints = plot_datapoints, color = c_blue, alpha = alpha, label = 'echo delay only', no_fill_contours = no_fill_contours, plot_density = plot_density, fill_contours = fill_contours)
    
    corner.hist2d(samples3[:,2], samples3[:,1]/samples3[:,0], levels=levels, smooth = smooth,\
                  plot_datapoints = plot_datapoints, color = c_purple, alpha = alpha, label = 'both', no_fill_contours = no_fill_contours, plot_density = plot_density, fill_contours = fill_contours)
    
    plt.xlabel(r'Inclination ($^{\circ}$)')
    plt.ylabel(r'q $(m_2/m_1)$')
    print('mass ratio_3',np.percentile(samples3[:,1]/samples3[:,0], 16),np.percentile(samples3[:,1]/samples3[:,0], 50),np.percentile(samples3[:,1]/samples3[:,0], 84))
    print('inclination_3',np.percentile(samples3[:,2], 16),np.percentile(samples3[:,2], 50),np.percentile(samples3[:,2], 84))
    
    plt.ylim([0,1])
    plt.xlim([0,89])
    
    plt.axvline(x=44,c='black')
    plt.axhline(y=0.5,c='black')
    plt.scatter([44],[0.5],marker='s',color='black')
    
    # ~ plt.legend()
    # ~ fig = plt.gcf()
    # ~ fig.set_size_inches(4, 4)
    plt.plot([],[], c = 'red', label = 'Radial Velocity Only')
    plt.plot([],[], c = 'blue', label = 'Echo Only')
    plt.plot([],[], c = 'purple', label = 'Both')
    plt.legend()
    
    fig = plt.gcf()
    fig.set_size_inches(4, 4)
    plt.show()
    
    plt.figure()
    plt.hist(samples1[:,0]+samples1[:,1], color = 'red', density = True, bins = 25)
    plt.hist(samples2[:,0]+samples2[:,1], color = 'blue', density = True, bins = 25, alpha = 0.75)
    plt.hist(samples3[:,0]+samples3[:,1], color = 'purple', density = True, bins = 25, alpha = 0.75)
    print('total mass_3',np.percentile(samples3[:,0]+samples3[:,1], 16),np.percentile(samples3[:,0]+samples3[:,1], 50),np.percentile(samples3[:,0]+samples3[:,1], 84))
    plt.axvline(x = 0.7+1.4, c = 'black')
    plt.xlabel('Total Mass')

    plt.figure()
    plt.hist(samples1[:,2], color = 'red', density = True, bins = 25)
    plt.hist(samples2[:,2], color = 'blue', density = True, bins = 25, alpha = 0.75)
    plt.hist(samples3[:,2], color = 'purple', density = True, bins = 25, alpha = 0.75)
    plt.axvline(x = 0.7+1.4, c = 'black')
    plt.xlabel('Inclination')
    
    plt.figure()
    plt.hist(samples1[:,1], color = 'red', density = True, bins = 25)
    plt.hist(samples2[:,1], color = 'blue', density = True, bins = 25, alpha = 0.75)
    plt.hist(samples3[:,1], color = 'purple', density = True, bins = 25, alpha = 0.75)
    plt.axvline(x = 0.7+1.4, c = 'black')
    plt.xlabel('M_2')
    
    plt.figure()
    plt.hist(samples1[:,0], color = 'red', density = True, bins = 25)
    plt.hist(samples2[:,0], color = 'blue', density = True, bins = 25, alpha = 0.75)
    plt.hist(samples3[:,0], color = 'purple', density = True, bins = 25, alpha = 0.75)
    plt.axvline(x = 0.7+1.4, c = 'black')
    plt.xlabel('M_1')
    
    fig = plt.gcf()
    fig.set_size_inches(4, 4)
    plt.show()
    
    
    ################
    corner.hist2d(samples1[:,2],samples1[:,0]+samples1[:,1], levels=[0.68,0.95], smooth = True,\
                  plot_datapoints = plot_datapoints, color = 'red', alpha = alpha, label = 'both', no_fill_contours = no_fill_contours, plot_density = True)
    
    corner.hist2d(samples2[:,2],samples2[:,0]+samples2[:,1], levels=[0.68,0.95], smooth = True,\
                  plot_datapoints = plot_datapoints, color = c_blue, alpha = alpha, label = 'echo delay only', no_fill_contours = no_fill_contours, plot_density = True)
    
    corner.hist2d(samples3[:,2],samples3[:,0]+samples3[:,1], levels=[0.68,0.95], smooth = True,\
                  plot_datapoints = plot_datapoints, color = c_purple, alpha = alpha, label = 'both', no_fill_contours = no_fill_contours, plot_density = True)
    
    
    plt.xlabel(r'Inclination ($^{\circ}$)')
    plt.ylabel(r'$(M_\odot)$')
    
    plt.ylim([0,5])
    plt.xlim([0,89])
    # ~ plt.xlim([0,18])
    
    plt.axvline(x=44,c='black')
    # ~ plt.axvline(x=3,c='black')
    plt.axhline(y=2.1,c='black')
    plt.scatter([44],[2.1],marker='s',color='black')
    
    # ~ plt.legend()
    fig = plt.gcf()
    fig.set_size_inches(4, 4)
    plt.show()
    
    
    
    ################
    
    # load echo data and radial velocity data
    echoDelay_data = np.loadtxt(f'mcmc_results//{both}//echo_delay_data.csv',delimiter=',',skiprows=1)
    
    fig, axs = plt.subplots(1,2)
    asra_lt = delay_model.ASRA_LT_model()
    
    pp = np.linspace(0,1,100)
    VV = []
    N = 2000
    print(len(samples3)/N)
    offset = 10
    for i in range(len(samples3)//N - offset):
        guess = samples3[i*N + offset]
        tt, vv = asra_lt.evaluate_array(pp,*guess)
        print(f'{i}/{len(samples3)//N}')
        axs[0].plot(pp, tt, c = 'purple', alpha = 0.08)
        # ~ axs[2,1].plot(pp,vv,'g-', alpha = 0.15)
    
    N = 5
    for i in range(len(samples3)//N):
        if i%100 == 0:
            print(f'{i}/{len(samples3)//N}')
        guess = samples3[i*N]
        tt, vv = asra_lt.evaluate_array([0.65,0.70,0.75],*guess)
        VV.append(max(vv))
        
    
    axs[0].set_xlim([0,1])
    axs[0].set_xlabel('Orbital Phase')
    axs[0].set_ylabel('Echo Delay (s)')
    
    axs[0].errorbar(echoDelay_data[:,0],echoDelay_data[:,1],echoDelay_data[:,2], fmt='o', capsize=3, color = 'black', markersize = 4)
    
    axs[1].hist(VV,density=True,color='purple',bins = 40,alpha = 1,histtype='step') #axs[1,1].hist(samples1[:,0]+samples1[:,1], color = c_red, density = True, bins = 30, alpha = 1, histtype='step')
    axs[1].hist(VV,density=True,color='purple',bins = 40,alpha = 0.5,histtype='stepfilled')
    
    # ~ plt.hist(samples3[:,1], color = c_purple, density = True, bins = 30, alpha = 0.5, histtype='stepfilled', hatch = '')
    # ~ plt.hist(samples3[:,1], color = c_purple, density = True, bins = 30, alpha = 1, histtype='step')
    
    axs[1].set_ylabel('PDF')
    
    rv = 88.49
    axs[1].axvline(x=rv,linestyle='--',c='black')
    axs[1].axvspan(rv-5,rv+5,alpha=0.3,color='black')
    
        # ~ plt.axvspan(44-6,44+6,color='black',alpha=0.3, zorder = 10)
    # ~ plt.axvline(x=44-6,c='black')
    # ~ plt.axvline(x=44+6,c='black')
    
    axs[1].set_xlabel(r'$K_{em}$ (km/s)')
    
    fig.set_size_inches(8, 3)
    
    plt.show()

### Figure 4 ###

def fig4():
    echo = '1718317261'

    rv = '1718317701'

    both = '1718318015'
    both_conn = '1718320552'

    
    alpha = 1
    s = 1
    
    no_fill_contours = True
    fill_contours = False
    plot_datapoints = False
    plot_density = True
    smooth = 1
    levels = [0.68, 0.95]
    
    contour_kwargs = {'linestyles': '-'}
    
    c_red = 'red'
    c_blue = 'blue'
    c_purple = '#580066'
    
    
    with open(f'mcmc_results//{rv}//sampler.pickle', 'rb') as inp:
        sampler = pickle.load(inp)
    samples1 = sampler.get_chain(discard=150, thin=1, flat=True)
    # ~ plt.scatter(samples[:,2],samples[:,1]/samples[:,0], s = s, alpha = alpha, label = 'radial velocity only', c = 'red')
    
    with open(f'mcmc_results//{echo}//sampler.pickle', 'rb') as inp:
        sampler = pickle.load(inp)
    samples2 = sampler.get_chain(discard=150, thin=1, flat=True)
    # ~ plt.scatter(samples[:,2],samples[:,1]/samples[:,0], s = s, alpha = alpha, label = 'echo delay only', c = 'blue')

    with open(f'mcmc_results//{both}//sampler.pickle', 'rb') as inp:
        sampler = pickle.load(inp)
    samples3 = sampler.get_chain(discard=150, thin=1, flat=True)
    
    with open(f'mcmc_results//{both_conn}//sampler.pickle', 'rb') as inp:
        sampler = pickle.load(inp)
    samples3_conn = sampler.get_chain(discard=150, thin=1, flat=True)
    
    # ~ {'mpl2005', 'mpl2014', 'serial', 'threaded'}
    
    corner.hist2d(samples1[:,2],samples1[:,1]/samples1[:,0], levels=levels, smooth = smooth,\
                  plot_datapoints = plot_datapoints, color = c_red, alpha = alpha, label = 'radial velocity only', no_fill_contours = no_fill_contours, plot_density = plot_density, fill_contours = fill_contours, contour_kwargs = contour_kwargs)
    
    corner.hist2d(samples2[:,2],samples2[:,1]/samples2[:,0], levels=levels, smooth = smooth,\
                  plot_datapoints = plot_datapoints, color = c_blue, alpha = alpha, label = 'echo delay only', no_fill_contours = no_fill_contours, plot_density = plot_density, fill_contours = fill_contours)
    

    # ~ corner.hist2d(samples3_conn[:,2], samples3_conn[:,1]/samples3_conn[:,0], levels=levels, smooth = smooth,\
                # ~ plot_datapoints = plot_datapoints, color = 'green', alpha = alpha, label = 'both', no_fill_contours = no_fill_contours, plot_density = plot_density, fill_contours = fill_contours)
    
    
    plt.xlabel(r'Inclination ($^{\circ}$)')
    plt.ylabel(r'q $(m_2/m_1)$')
    
    plt.ylim([0,1])
    plt.xlim([0,89])
    fig = plt.gcf()
    fig.set_size_inches(3, 3)
    
    plt.figure()
    corner.hist2d(samples3[:,2], samples3[:,1]/samples3[:,0], levels=levels, smooth = smooth,\
        plot_datapoints = plot_datapoints, color = c_purple, alpha = alpha, label = 'both', no_fill_contours = no_fill_contours, plot_density = plot_density, fill_contours = fill_contours)
    
    plt.axvspan(44-6,44+6,color='black',alpha=0.3, zorder = 10)
    plt.axvline(x=44-6,c='black')
    plt.axvline(x=44+6,c='black')
    
    # ~ plt.legend()
    plt.xlabel(r'Inclination ($^{\circ}$)')
    plt.ylabel(r'q $(m_2/m_1)$')
    fig = plt.gcf()
    fig.set_size_inches(3, 3)
    
    
    plt.show()
    
    
    
    plt.figure()
    plt.hist(samples1[:,0]+samples1[:,1], color = c_red, density = True, bins = 30, alpha = 1, histtype='step')
    plt.hist(samples2[:,0]+samples2[:,1], color = c_blue, density = True, bins = 30, alpha = 1, histtype='step')
    plt.hist(samples3[:,0]+samples3[:,1], color = c_purple, density = True, bins = 30, alpha = 1, histtype='stepfilled')
    plt.hist(samples3[:,0]+samples3[:,1], color = c_purple, density = True, bins = 30, alpha = 1, histtype='step')
    plt.hist(samples3_conn[:,0]+samples3_conn[:,1], color = 'black', density = True, bins = 25, alpha = 1, histtype='step', hatch = '//', zorder = -5)
    plt.show()
    
    
    # ~ corner.hist2d(samples3[:,1] + samples3[:,0], samples3[:,1], levels=levels, smooth = smooth,\
                # ~ plot_datapoints = plot_datapoints, color = c_purple, alpha = alpha, label = 'both', no_fill_contours = no_fill_contours, plot_density = plot_density, fill_contours = fill_contours)

    # ~ plt.figure()
    # ~ plt.hist(samples1[:,2], color = 'red', density = True, bins = 25)
    # ~ plt.hist(samples2[:,2], color = 'blue', density = True, bins = 25, alpha = 0.75)
    # ~ plt.hist(samples3[:,2], color = c_purple, density = True, bins = 25, alpha = 0.75)

    # ~ plt.xlabel('Inclination')
    
    plt.figure()
    # ~ plt.hist(samples1[:,1], color = c_red, density = True, bins = 30, alpha = 0.5, histtype='stepfilled', hatch = '')
    plt.hist(samples1[:,1], color = c_red, density = True, bins = 30, alpha = 1, histtype='step')
    
    # ~ plt.hist(samples2[:,1], color = c_blue, density = True, bins = 30, alpha = 0.5, histtype='stepfilled', hatch = '')
    plt.hist(samples2[:,1], color = c_blue, density = True, bins = 30, alpha = 1, histtype='step')
    
    plt.hist(samples3[:,1], color = c_purple, density = True, bins = 30, alpha = 0.5, histtype='stepfilled', hatch = '')
    plt.hist(samples3[:,1], color = c_purple, density = True, bins = 30, alpha = 1, histtype='step')
    
    plt.hist(samples3_conn[:,1], color = 'black', density = True, bins = 25, alpha = 1, histtype='step', hatch = '//', zorder = -5)
    plt.xlabel(r'$m_2$')
    plt.ylabel(r'PDF')
    fig = plt.gcf()
    fig.set_size_inches(3, 3)
    
    # ~ plt.figure()
    # ~ plt.hist(samples3[:,0], color = c_purple, density = True, bins = 30, alpha = 0.75, histtype='stepfilled', hatch = '')
    # ~ plt.hist(samples3_conn[:,0], color = 'black', density = True, bins = 25, alpha = 1, histtype='step', hatch = '//')
    # ~ plt.xlabel('M1')
    
    plt.figure()
    # ~ plt.hist(samples1[:,1], color = c_red, density = True, bins = 30, alpha = 0.5, histtype='stepfilled', hatch = '')
    plt.hist(samples1[:,1]/samples1[:,0], color = c_red, density = True, bins = 30, alpha = 1, histtype='step')
    
    # ~ plt.hist(samples2[:,1], color = c_blue, density = True, bins = 30, alpha = 0.5, histtype='stepfilled', hatch = '')
    plt.hist(samples2[:,1]/samples2[:,0], color = c_blue, density = True, bins = 30, alpha = 1, histtype='step')
    
    
    plt.hist(samples3[:,1]/samples3[:,0], color = c_purple, density = True, bins = 30, alpha = 0.5, histtype='stepfilled', hatch = '')
    plt.hist(samples3[:,1]/samples3[:,0], color = c_purple, density = True, bins = 30, alpha = 1, histtype='step')
    
    plt.hist(samples3_conn[:,1]/samples3_conn[:,0], color = 'black', density = True, bins = 25, alpha = 1, histtype='step', hatch = '//', zorder = -5)
    plt.xlabel(r'$q (m2/m1)$')
    plt.ylabel(r'PDF')
    fig = plt.gcf()
    fig.set_size_inches(3, 3)
    
    # ~ plt.figure()
    # ~ plt.hist(samples3[:,0]+samples3[:,1], color = c_purple, density = True, bins = 30, alpha = 0.75, histtype='stepfilled', hatch = '')
    # ~ plt.hist(samples3_conn[:,0]+samples3_conn[:,1], color = 'black', density = True, bins = 25, alpha = 1, histtype='step', hatch = '//')
    # ~ plt.xlabel('Total Mass')
    
    plt.show()
    
    ################
    corner.hist2d(samples1[:,2],samples1[:,0]+samples1[:,1], levels=[0.68,0.95], smooth = True,\
                  plot_datapoints = plot_datapoints, color = 'red', alpha = alpha, label = 'both', no_fill_contours = no_fill_contours, plot_density = True)
    
    corner.hist2d(samples2[:,2],samples2[:,0]+samples2[:,1], levels=[0.68,0.95], smooth = True,\
                  plot_datapoints = plot_datapoints, color = c_blue, alpha = alpha, label = 'echo delay only', no_fill_contours = no_fill_contours, plot_density = True)
    
    corner.hist2d(samples3[:,2],samples3[:,0]+samples3[:,1], levels=[0.68,0.95], smooth = True,\
                  plot_datapoints = plot_datapoints, color = c_purple, alpha = alpha, label = 'both', no_fill_contours = no_fill_contours, plot_density = True)
    
    
    plt.xlabel(r'Inclination ($^{\circ}$)')
    plt.ylabel(r'$(M_\odot)$')
    
    plt.ylim([0,5])
    plt.xlim([0,89])
    # ~ plt.xlim([0,18])
    
    plt.axvline(x=44,c='black')
    # ~ plt.axvline(x=3,c='black')
    plt.axhline(y=2.1,c='black')
    plt.scatter([44],[2.1],marker='s',color='black')
    
    # ~ plt.legend()
    fig = plt.gcf()
    fig.set_size_inches(4, 4)
    plt.show()
    
    
    
    ################
    
    # load echo data and radial velocity data
    echoDelay_data = np.loadtxt(f'mcmc_results//{both}//echo_delay_data.csv',delimiter=',',skiprows=1)
    
    fig, axs = plt.subplots(1,2)
    asra_lt = delay_model.ASRA_LT_model()
    
    pp = np.linspace(0,1,100)
    VV = []
    N = 1000
    print(len(samples3)/N)
    for i in range(len(samples3)//N):
        guess = samples3[i*N]
        tt, vv = asra_lt.evaluate_array(pp,*guess)
        print(f'{i}/{len(samples3)//N}')
        axs[0].plot(pp, tt, c = 'purple', alpha = 0.1)
        # ~ axs[2,1].plot(pp,vv,'g-', alpha = 0.15)
    
    N = 5
    for i in range(len(samples3)//N):
        if i%100 == 0:
            print(f'{i}/{len(samples3)//N}')
        guess = samples3[i*N]
        tt, vv = asra_lt.evaluate_array([0.65,0.70,0.75],*guess)
        VV.append(max(vv))
        
    
    axs[0].set_xlim([0,1])
    axs[0].set_xlabel('Orbital Phase')
    axs[0].set_ylabel('Echo Delay (s)')
    
    axs[0].errorbar(echoDelay_data[:,0],echoDelay_data[:,1],echoDelay_data[:,2], fmt='o', capsize=3, color = 'black')     
    
    axs[1].hist(VV,density=True,color='purple',bins = 40,alpha = 1,histtype='step')
    axs[1].hist(VV,density=True,color='purple',bins = 40,alpha = 0.5,histtype='stepfilled')
    
    axs[1].set_ylabel('PDF')
    axs[1].axvline(x=75.0,linestyle='--',c='black')
    axs[1].axvspan(75.0-0.8,75.0+0.8,alpha=0.4,color='black')
    axs[1].set_xlabel(r'$K_{em}$ (km/s)')
    
    fig.set_size_inches(8, 5)
    
    plt.show()

### Figure 5 ###

def fig5():
    echo = '1725003719'# with K1 constraint and no eclipse constraint
    rv = '1725002583' # with K1 constraint and no eclipse constraint
    both = '1725004053'# with K1 constraint and no eclipse constraint
    
    alpha = 1
    no_fill_contours = True
    fill_contours = False
    plot_datapoints = False
    plot_density = True
    smooth = 1
    levels = [0.68, 0.95]
    contour_kwargs = {'linestyles': '-'}
    c_red = 'red'
    c_blue = 'blue'
    c_purple = '#580066'
    
    
    with open(f'mcmc_results//{rv}//sampler.pickle', 'rb') as inp:
        sampler = pickle.load(inp)
    samples1 = sampler.get_chain(discard=150, thin=1, flat=True)
    
    with open(f'mcmc_results//{echo}//sampler.pickle', 'rb') as inp:
        sampler = pickle.load(inp)
    samples2 = sampler.get_chain(discard=150, thin=1, flat=True)

    with open(f'mcmc_results//{both}//sampler.pickle', 'rb') as inp:
        sampler = pickle.load(inp)
    samples3 = sampler.get_chain(discard=150, thin=1, flat=True)
    
    # Print some measurements
    print(f'q = {np.mean(samples3[:, 1]/samples3[:, 0])} pm {np.std(samples3[:, 1]/samples3[:, 0])}') #m1F, m2F, iF, alphaF = samples[:,0], samples[:,1], samples[:,2], samples[:,3]
    print(f'M = {np.mean(samples3[:, 1] + samples3[:, 0])} pm {np.std(samples3[:, 1] + samples3[:, 0])}')
    
    m1F, m2F, iF, alphaF = samples3[:,0], samples3[:,1], samples3[:,2], samples3[:,3]
    period_in = 0.15804693
    a = (constants.G*(m1F+m2F)*constants.M_Sun*(period_in * constants.day)**2/(4*np.pi**2))**(1/3)
    K1 = np.sin(iF*np.pi/180)*2*np.pi*a/(period_in * constants.day) * (m2F/(m1F + m2F))
    K2 = 1e-3*np.sin(iF*np.pi/180)*2*np.pi*a/(period_in * constants.day) * (m1F/(m1F + m2F))
    
    print(f'K1 = {np.mean(K1)} pm {np.std(K1)}')
    print(f'K2 = {np.mean(K2)} pm {np.std(K2)}')
    # ~ print(f'{alphaF}')
    
    plt.plot(alphaF)
    plt.show()
    
    # Inclination vs. mass ratio 2D histogram plot
    corner.hist2d(samples1[:,2],samples1[:,1]/samples1[:,0], levels=levels, smooth = smooth,\
                  plot_datapoints = plot_datapoints, color = c_red, alpha = alpha, label = 'radial velocity only', no_fill_contours = no_fill_contours, plot_density = plot_density, fill_contours = fill_contours, contour_kwargs = contour_kwargs)
    
    corner.hist2d(samples2[:,2],samples2[:,1]/samples2[:,0], levels=levels, smooth = smooth,\
                  plot_datapoints = plot_datapoints, color = c_blue, alpha = alpha, label = 'echo delay only', no_fill_contours = no_fill_contours, plot_density = plot_density, fill_contours = fill_contours)
    
    corner.hist2d(samples3[:,2], samples3[:,1]/samples3[:,0], levels=levels, smooth = smooth,\
                  plot_datapoints = plot_datapoints, color = c_purple, alpha = alpha, label = 'both', no_fill_contours = no_fill_contours, plot_density = plot_density, fill_contours = fill_contours)
    
    # ~ corner.hist2d(samples3_conn[:,2], samples3_conn[:,1]/samples3_conn[:,0], levels=levels, smooth = smooth,\
                # ~ plot_datapoints = plot_datapoints, color = 'green', alpha = alpha, label = 'both', no_fill_contours = no_fill_contours, plot_density = plot_density, fill_contours = fill_contours)
    
    eclipse_conn = np.loadtxt('variables_used_in_paper//eclipse_conn.csv')
    popt = np.polyfit(eclipse_conn[:, 0], eclipse_conn[:, 1], deg = 5)
    tt = np.linspace(min(eclipse_conn[:, 0]) - 1, max(eclipse_conn[:, 0]), 500)
    yy = popt[0]*tt**5 + popt[1]*tt**4 + popt[2]*tt**3 + popt[3]*tt**2 + popt[4]*tt + popt[5]
    # ~ plt.scatter(eclipse_conn[:, 0], eclipse_conn[:, 1])
    plt.plot(tt, yy, 'k-')
    
    # ~ plt.fill_between(tt, 1, y2 = yy, hatch='//', zorder=2, fc='none')
    plt.fill_between(tt, 1, y2 = yy, zorder=2, fc='black', alpha = 0.3)
    
    plt.xlabel(r'Inclination ($^{\circ}$)')
    plt.ylabel(r'q $(m_2/m_1)$')
    
    # ~ plt.legend()
    fig = plt.gcf()
    fig.set_size_inches(5, 5)
    
    
    plt.ylim([0.15,0.52])
    plt.xlim([40,89])
    
    # ~ plt.axhspan(0.21,0.34,color='black',alpha=0.3, zorder = 10)
    # ~ plt.axhline(y=0.21,c='black')
    # ~ plt.axhline(y=0.34,c='black')
    # ~ plt.axvline(x=74,c='black')
    
    fig, axs = plt.subplots(2, 3, figsize = (8,5))
    axs[0,0].set_ylabel('PDF (a.u.)')
    axs[1,0].set_ylabel('PDF (a.u.)')
    
    
    axs[0,1].set_yticks([])
    axs[0,2].set_yticks([])
    axs[1,1].set_yticks([])
    axs[1,2].set_yticks([])
    
    axs[1,1].hist(samples1[:,0]+samples1[:,1], color = c_red, density = True, bins = 30, alpha = 1, histtype='step')
    
    axs[1,1].hist(samples2[:,0]+samples2[:,1], color = c_blue, density = True, bins = 30, alpha = 1, histtype='step')
    
    axs[1,1].hist(samples3[:,0]+samples3[:,1], color = c_purple, density = True, bins = 30, alpha = 0.5, histtype='stepfilled', hatch = '')
    axs[1,1].hist(samples3[:,0]+samples3[:,1], color = c_purple, density = True, bins = 30, alpha = 1, histtype='step')
    axs[1,1].set_xlabel(r'Total Mass (M$_{\odot}$)')

    axs[1,0].hist(samples1[:,1]/samples1[:,0], color = c_red, density = True, bins = 30, alpha = 1, histtype='step')
    
    axs[1,0].hist(samples2[:,1]/samples2[:,0], color = c_blue, density = True, bins = 30, alpha = 1, histtype='step')
    
    axs[1,0].hist(samples3[:,1]/samples3[:,0], color = c_purple, density = True, bins = 30, alpha = 0.5, histtype='stepfilled', hatch = '')
    axs[1,0].hist(samples3[:,1]/samples3[:,0], color = c_purple, density = True, bins = 30, alpha = 1, histtype='step')
    
    axs[1,0].set_xlabel(r'q $(m_2/m_2)$')

    
    axs[0,0].hist(samples1[:,2], color = c_red, density = True, bins = 30, alpha = 1, histtype='step')
    
    axs[0,0].hist(samples2[:,2], color = c_blue, density = True, bins = 30, alpha = 1, histtype='step')
    
    axs[0,0].hist(samples3[:,2], color = c_purple, density = True, bins = 30, alpha = 0.5, histtype='stepfilled', hatch = '')
    axs[0,0].hist(samples3[:,2], color = c_purple, density = True, bins = 30, alpha = 1, histtype='step')
    
    axs[0,0].set_xlabel(r'Inclination ($^{\circ}$)')

    axs[0,1].hist(samples1[:,3], color = c_red, density = True, bins = 30, alpha = 1, histtype='step')
    
    axs[0,1].hist(samples2[:,3], color = c_blue, density = True, bins = 30, alpha = 1, histtype='step')
    
    axs[0,1].hist(samples3[:,3], color = c_purple, density = True, bins = 30, alpha = 0.5, histtype='stepfilled', hatch = '')
    axs[0,1].hist(samples3[:,3], color = c_purple, density = True, bins = 30, alpha = 1, histtype='step')
    
    axs[0,1].set_xlabel(r'$\alpha$ ($^{\circ}$)')
    
    # Plot K1
    period_in = 0.15804693
    
    samples = samples1
    m1F, m2F, iF, alphaF = samples[:,0], samples[:,1], samples[:,2], samples[:,3]
    a = (constants.G*(m1F+m2F)*constants.M_Sun*(period_in * constants.day)**2/(4*np.pi**2))**(1/3)
    K1 = 1e-3*np.sin(iF*np.pi/180)*2*np.pi*a/(period_in * constants.day) * (m2F/(m1F + m2F))
    axs[1,2].hist(K1, color = c_red, density = True, bins = 30, alpha = 1, histtype='step')
        
    samples = samples2
    m1F, m2F, iF, alphaF = samples[:,0], samples[:,1], samples[:,2], samples[:,3]
    a = (constants.G*(m1F+m2F)*constants.M_Sun*(period_in * constants.day)**2/(4*np.pi**2))**(1/3)
    K1 = 1e-3*np.sin(iF*np.pi/180)*2*np.pi*a/(period_in * constants.day) * (m2F/(m1F + m2F))
    axs[1,2].hist(K1, color = c_blue, density = True, bins = 30, alpha = 1, histtype='step')
        
    samples = samples3
    m1F, m2F, iF, alphaF = samples[:,0], samples[:,1], samples[:,2], samples[:,3]
    a = (constants.G*(m1F+m2F)*constants.M_Sun*(period_in * constants.day)**2/(4*np.pi**2))**(1/3)
    K1 = 1e-3*np.sin(iF*np.pi/180)*2*np.pi*a/(period_in * constants.day) * (m2F/(m1F + m2F))
    axs[1,2].hist(K1, color = c_purple, density = True, bins = 30, alpha = 1, histtype='step')
    axs[1,2].hist(K1, color = c_purple, density = True, bins = 30, alpha = 0.5, histtype='stepfilled')
    
    axs[1,2].axvline(x = 101.5 - 11.5, linestyle = '--', color = 'k')
    axs[1,2].axvline(x = 101.5 + 11.5, linestyle = '--', color = 'k')
    
    axs[1,2].set_xlabel(r'$K_1$ (km/s)')
    
    # Plot K2
    period_in = 0.15804693
    
    samples = samples1
    m1F, m2F, iF, alphaF = samples[:,0], samples[:,1], samples[:,2], samples[:,3]
    a = (constants.G*(m1F+m2F)*constants.M_Sun*(period_in * constants.day)**2/(4*np.pi**2))**(1/3)
    K2 = 1e-3*np.sin(iF*np.pi/180)*2*np.pi*a/(period_in * constants.day) * (m1F/(m1F + m2F))
    axs[0,2].hist(K2, color = c_red, density = True, bins = 30, alpha = 1, histtype='step')
    
        
    samples = samples2
    m1F, m2F, iF, alphaF = samples[:,0], samples[:,1], samples[:,2], samples[:,3]
    a = (constants.G*(m1F+m2F)*constants.M_Sun*(period_in * constants.day)**2/(4*np.pi**2))**(1/3)
    K2 = 1e-3*np.sin(iF*np.pi/180)*2*np.pi*a/(period_in * constants.day) * (m1F/(m1F + m2F))
    axs[0,2].hist(K2, color = c_blue, density = True, bins = 30, alpha = 1, histtype='step')
    
        
    samples = samples3
    m1F, m2F, iF, alphaF = samples[:,0], samples[:,1], samples[:,2], samples[:,3]
    a = (constants.G*(m1F+m2F)*constants.M_Sun*(period_in * constants.day)**2/(4*np.pi**2))**(1/3)
    K2 = 1e-3*np.sin(iF*np.pi/180)*2*np.pi*a/(period_in * constants.day) * (m1F/(m1F + m2F))
    axs[0,2].hist(K2, color = c_purple, density = True, bins = 30, alpha = 1, histtype='step')
    axs[0,2].hist(K2, color = c_purple, density = True, bins = 30, alpha = 0.5, histtype='stepfilled')
    
    axs[0,2].set_xlabel(r'$K_2$ (km/s)')
    
    
    plt.show()
    
    # Plot posterior fits against data
    fig, axs = plt.subplots(1,2)
    asra_lt = delay_model.ASRA_LT_model()
    
    echoDelay_data = np.loadtxt(f'mcmc_results//{both}//echo_delay_data.csv',delimiter=',',skiprows=1)
    
    pp = np.linspace(0,1,100)
    VV = []
    N = 1000
    print(len(samples3)/N)
    for i in range(len(samples3)//N):
        guess = samples3[i*N]
        print(guess)
        tt, vv = asra_lt.evaluate_array(pp, guess[0], guess[1], guess[2], guess[3], period_in)
        print(f'{i}/{len(samples3)//N}')
        axs[0].plot(pp, tt, c = 'purple', alpha = 0.1)
        # ~ axs[2,1].plot(pp,vv,'g-', alpha = 0.15)
    
    N = 5
    for i in range(len(samples3)//N):
        if i%100 == 0:
            print(f'{i}/{len(samples3)//N}')
        guess = samples3[i*N]
        tt, vv = asra_lt.evaluate_array([0.65,0.70,0.75], guess[0], guess[1], guess[2], guess[3], period_in)
        VV.append(max(vv))
        
    
    axs[0].set_xlim([0,1])
    axs[0].set_xlabel('Orbital Phase')
    axs[0].set_ylabel('Echo Delay (s)')
    
    axs[0].errorbar(echoDelay_data[:,0],echoDelay_data[:,1],echoDelay_data[:,2], fmt='o', capsize=3, color = 'black')     
    
    axs[1].hist(VV,density=True,color='purple',bins = 40,alpha = 1,histtype='step')
    axs[1].hist(VV,density=True,color='purple',bins = 40,alpha = 0.5,histtype='stepfilled')
    
    axs[1].set_ylabel('PDF')
    axs[1].axvline(x=277,linestyle='--',c='black')
    axs[1].axvspan(277 - 22, 277 + 22, alpha=0.4, color='black')
    axs[1].set_xlabel(r'$K_{em}$ (km/s)')
    
    fig.set_size_inches(8, 5)
    
    plt.show()

def fig5_extra():
    echo = '1725003719'# with K1 constraint and no eclipse constraint
    rv = '1725002583' # with K1 constraint and no eclipse constraint
    both = '1725004053'# with K1 constraint and no eclipse constraint
    
    alpha = 1
    s = 1
    
    no_fill_contours = True
    fill_contours = False
    plot_datapoints = True
    plot_density = True
    smooth = 1
    levels = [0.68, 0.95]
    
    contour_kwargs = {'linestyles': '-'}
    
    c_red = 'red'
    c_blue = 'blue'
    c_purple = '#580066'
    
    
    with open(f'mcmc_results//{rv}//sampler.pickle', 'rb') as inp:
        sampler = pickle.load(inp)
    samples1 = sampler.get_chain(discard=150, thin=1, flat=True)
    # ~ plt.scatter(samples[:,2],samples[:,1]/samples[:,0], s = s, alpha = alpha, label = 'radial velocity only', c = 'red')
    
    with open(f'mcmc_results//{echo}//sampler.pickle', 'rb') as inp:
        sampler = pickle.load(inp)
    samples2 = sampler.get_chain(discard=150, thin=1, flat=True)
    # ~ plt.scatter(samples[:,2],samples[:,1]/samples[:,0], s = s, alpha = alpha, label = 'echo delay only', c = 'blue')

    with open(f'mcmc_results//{both}//sampler.pickle', 'rb') as inp:
        sampler = pickle.load(inp)
    samples3 = sampler.get_chain(discard=150, thin=1, flat=True)
    
    # ~ with open(f'mcmc_results//{both_conn}//sampler.pickle', 'rb') as inp:
        # ~ sampler = pickle.load(inp)
    # ~ samples3_conn = sampler.get_chain(discard=150, thin=1, flat=True)
    
    # ~ {'mpl2005', 'mpl2014', 'serial', 'threaded'}
    
    corner.hist2d(samples1[:,2],samples1[:,1]/samples1[:,0], levels=levels, smooth = smooth,\
                  plot_datapoints = plot_datapoints, color = c_red, alpha = alpha, label = 'radial velocity only', no_fill_contours = no_fill_contours, plot_density = plot_density, fill_contours = fill_contours, contour_kwargs = contour_kwargs)
    
    corner.hist2d(samples2[:,2],samples2[:,1]/samples2[:,0], levels=levels, smooth = smooth,\
                  plot_datapoints = plot_datapoints, color = c_blue, alpha = alpha, label = 'echo delay only', no_fill_contours = no_fill_contours, plot_density = plot_density, fill_contours = fill_contours)
    
    corner.hist2d(samples3[:,2], samples3[:,1]/samples3[:,0], levels=levels, smooth = smooth,\
                  plot_datapoints = plot_datapoints, color = c_purple, alpha = alpha, label = 'both', no_fill_contours = no_fill_contours, plot_density = plot_density, fill_contours = fill_contours)
    
    # ~ corner.hist2d(samples3_conn[:,2], samples3_conn[:,1]/samples3_conn[:,0], levels=levels, smooth = smooth,\
                # ~ plot_datapoints = plot_datapoints, color = 'green', alpha = alpha, label = 'both', no_fill_contours = no_fill_contours, plot_density = plot_density, fill_contours = fill_contours)
    
    eclipse_conn = np.loadtxt('variables_used_in_paper//eclipse_conn.csv')
    popt = np.polyfit(eclipse_conn[:, 0], eclipse_conn[:, 1], deg = 5)
    tt = np.linspace(min(eclipse_conn[:, 0]) - 1, max(eclipse_conn[:, 0]), 500)
    yy = popt[0]*tt**5 + popt[1]*tt**4 + popt[2]*tt**3 + popt[3]*tt**2 + popt[4]*tt + popt[5]
    # ~ plt.scatter(eclipse_conn[:, 0], eclipse_conn[:, 1])
    plt.plot(tt, yy, 'k-')
    
    
    k1_conn = np.loadtxt('variables_used_in_paper//K1_conn_ara.csv', skiprows = 1, delimiter = ',')
    plt.scatter(k1_conn[:, 0], k1_conn[:, 1])
    plt.scatter(k1_conn[:, 2], k1_conn[:, 3])
    
    plt.xlabel(r'Inclination ($^{\circ}$)')
    plt.ylabel(r'q $(m_2/m_1)$')
    
    plt.ylim([0,1])
    plt.xlim([0,89])
    
    # ~ plt.axhspan(0.21,0.34,color='black',alpha=0.3, zorder = 10)
    # ~ plt.axhline(y=0.21,c='black')
    # ~ plt.axhline(y=0.34,c='black')
    # ~ plt.axvline(x=74,c='black')
    
    # ~ plt.legend()
    fig = plt.gcf()
    fig.set_size_inches(6, 5)
    
    plt.figure()
    period_in = 0.15804693
    

    # rv only
    samples = samples1
    a = (constants.G*(samples[:,0]+samples[:,1])*constants.M_Sun*(period_in * constants.day)**2/(4*np.pi**2))**(1/3)
    K1 = 1e-3*np.sin(samples[:,2]*np.pi/180)*2*np.pi*a/(period_in * constants.day) * (samples[:,1]/(samples[:,0] + samples[:,1]))
    K2 = 1e-3*np.sin(samples[:,2]*np.pi/180)*2*np.pi*a/(period_in * constants.day) * (samples[:,0]/(samples[:,0] + samples[:,1]))
    corner.hist2d(K1, K2, levels=levels, smooth = smooth,\
                  plot_datapoints = plot_datapoints, color = c_red, alpha = alpha, label = 'both', no_fill_contours = no_fill_contours, plot_density = plot_density, fill_contours = fill_contours)
    
    # echo only
    samples = samples2
    a = (constants.G*(samples[:,0]+samples[:,1])*constants.M_Sun*(period_in * constants.day)**2/(4*np.pi**2))**(1/3)
    K1 = 1e-3*np.sin(samples[:,2]*np.pi/180)*2*np.pi*a/(period_in * constants.day) * (samples[:,1]/(samples[:,0] + samples[:,1]))
    K2 = 1e-3*np.sin(samples[:,2]*np.pi/180)*2*np.pi*a/(period_in * constants.day) * (samples[:,0]/(samples[:,0] + samples[:,1]))
    corner.hist2d(K1, K2, levels=levels, smooth = smooth,\
                  plot_datapoints = plot_datapoints, color = c_blue, alpha = alpha, label = 'both', no_fill_contours = no_fill_contours, plot_density = plot_density, fill_contours = fill_contours)
    
    # both
    samples = samples3
    a = (constants.G*(samples[:,0]+samples[:,1])*constants.M_Sun*(period_in * constants.day)**2/(4*np.pi**2))**(1/3)
    K1 = 1e-3*np.sin(samples[:,2]*np.pi/180)*2*np.pi*a/(period_in * constants.day) * (samples[:,1]/(samples[:,0] + samples[:,1]))
    K2 = 1e-3*np.sin(samples[:,2]*np.pi/180)*2*np.pi*a/(period_in * constants.day) * (samples[:,0]/(samples[:,0] + samples[:,1]))
    corner.hist2d(K1, K2, levels=levels, smooth = smooth,\
                  plot_datapoints = plot_datapoints, color = c_purple, alpha = alpha, label = 'both', no_fill_contours = no_fill_contours, plot_density = plot_density, fill_contours = fill_contours)
    
    
    
    plt.axvline(x = 101.5 - 2*11.5, linestyle = '--', color = 'k')
    plt.axvline(x = 101.5 + 2*11.5, linestyle = '--', color = 'k')
    plt.xlabel(r'$K_1$')
    plt.ylabel(r'$K_2$')
    
    
    plt.figure()

    ss1 = np.copy(samples1)
    ss2 = np.copy(samples2)
    ss3 = np.copy(samples3)
    
    
    # ~ plt.scatter(ss1[:,2],ss1[:,1]/ss1[:,0], color = c_red, s = 1, alpha = 0.01)
    # ~ plt.scatter(ss2[:,2],ss2[:,1]/ss2[:,0], color = c_blue, s = 1, alpha = 0.01)
    # ~ plt.scatter(ss3[:,2],ss3[:,1]/ss3[:,0], color = c_purple, s = 1, alpha = 0.01)
    
    corner.hist2d(ss1[:,2],ss1[:,1]/ss1[:,0], levels=levels, smooth = smooth,\
              plot_datapoints = plot_datapoints, color = c_red, alpha = alpha, label = 'radial velocity only', no_fill_contours = no_fill_contours, plot_density = plot_density, fill_contours = fill_contours)
    
    corner.hist2d(ss2[:,2],ss2[:,1]/ss2[:,0], levels=levels, smooth = smooth,\
                  plot_datapoints = plot_datapoints, color = c_blue, alpha = alpha, label = 'echo only', no_fill_contours = no_fill_contours, plot_density = plot_density, fill_contours = fill_contours)
    
    
    plt.xlabel(r'inclination')
    plt.ylabel(r'mass ratio')
    
    plt.axvline(x = 55,c = 'black')
    plt.axvline(x = 89,c = 'black')
    plt.axhline(y = 0.35,c = 'black')
    plt.axhline(y = 0.22,c = 'black')
    
    plt.figure()
    
    # ~ print(np.logical_and(samples1[:,2] > 55, samples1[:,2] < 75, samples1[:,1]/samples1[:,0] > 0.4))
    
    ss1 = samples1[np.logical_and(samples1[:,0] + samples1[:,1] < 3.15, np.logical_and(np.logical_and(samples1[:,0] + samples1[:,1] > 2.18, samples1[:,2] > 50), samples1[:,2] < 65))]
    ss2 = samples2[np.logical_and(samples2[:,0] + samples2[:,1] < 3.15, np.logical_and(np.logical_and(samples2[:,0] + samples2[:,1] > 2.18, samples2[:,2] > 50), samples2[:,2] < 65))]
    ss3 = samples3[np.logical_and(samples3[:,0] + samples3[:,1] < 3.15, np.logical_and(np.logical_and(samples3[:,0] + samples3[:,1] > 2.18, samples3[:,2] > 50), samples3[:,2] < 65))]
    
    
    # ~ ss1 = samples1
    # ~ ss2 = samples2
    # ~ ss3 = samples3
    corner.hist2d(ss1[:,3],ss1[:,1]/ss1[:,0], levels=levels, smooth = smooth,\
              plot_datapoints = plot_datapoints, color = c_red, alpha = alpha, label = 'radial velocity only', no_fill_contours = no_fill_contours, plot_density = plot_density, fill_contours = fill_contours)
    
    corner.hist2d(ss2[:,3],ss2[:,1]/ss2[:,0], levels=levels, smooth = smooth,\
              plot_datapoints = plot_datapoints, color = c_blue, alpha = alpha, label = 'radial velocity only', no_fill_contours = no_fill_contours, plot_density = plot_density, fill_contours = fill_contours)
    
    
    # ~ plt.scatter(ss1[:,3],ss1[:,1]/ss1[:,0], color = c_red, s = 1, alpha = 0.03)
    # ~ plt.scatter(ss2[:,3],ss2[:,1]/ss2[:,0], color = c_blue, s = 1, alpha = 0.03)
    # ~ plt.scatter(ss3[:,3],ss3[:,1]/ss3[:,0], color = c_purple, s = 1, alpha = 0.03)
    
    plt.xlabel(r'Alpha')
    plt.ylabel(r'Mass Ratio')
    
    # ~ plt.scatter(ss1[:,2],ss1[:,1]/ss1[:,0], color = c_red, s = 1, alpha = 0.01)
    # ~ plt.scatter(ss2[:,2],ss2[:,1]/ss2[:,0], color = c_blue, s = 1, alpha = 0.01)
    # ~ plt.scatter(ss3[:,2],ss3[:,1]/ss3[:,0], color = c_purple, s = 1, alpha = 0.01)
    # ~ plt.xlabel(r'inclination')
    # ~ plt.ylabel(r'mass ratio')

    # ~ plt.hist(ss1[:,0], color = 'red', histtype='step')
    # ~ plt.hist(ss2[:,0], color = 'blue',  histtype='step')
    # ~ plt.hist(ss3[:,0], color = 'purple',  histtype='step')
    

    
    fig = plt.gcf()
    fig.set_size_inches(6, 5)
    plt.show()
    
    
    
    plt.figure()
    
    corner.hist2d(samples1[:,0],samples1[:,1]/samples1[:,0], levels=levels, smooth = smooth,\
                  plot_datapoints = plot_datapoints, color = c_red, alpha = alpha, label = 'radial velocity only', no_fill_contours = no_fill_contours, plot_density = plot_density, fill_contours = fill_contours, contour_kwargs = contour_kwargs)
    
    corner.hist2d(samples2[:,0],samples2[:,1]/samples2[:,0], levels=levels, smooth = smooth,\
                  plot_datapoints = plot_datapoints, color = c_blue, alpha = alpha, label = 'echo delay only', no_fill_contours = no_fill_contours, plot_density = plot_density, fill_contours = fill_contours)
    
    corner.hist2d(samples3[:,0], samples3[:,1]/samples3[:,0], levels=levels, smooth = smooth,\
                  plot_datapoints = plot_datapoints, color = c_purple, alpha = alpha, label = 'both', no_fill_contours = no_fill_contours, plot_density = plot_density, fill_contours = fill_contours)
    
    plt.xlabel('m1')
    plt.ylabel('Mass Ratio')
    
    plt.show()
    
    # ~ corner.hist2d(samples3[:,1] + samples3[:,0], samples3[:,1], levels=levels, smooth = smooth,\
                # ~ plot_datapoints = plot_datapoints, color = c_purple, alpha = alpha, label = 'both', no_fill_contours = no_fill_contours, plot_density = plot_density, fill_contours = fill_contours)

    # ~ plt.figure()
    # ~ plt.hist(samples1[:,2], color = 'red', density = True, bins = 25)
    # ~ plt.hist(samples2[:,2], color = 'blue', density = True, bins = 25, alpha = 0.75)
    # ~ plt.hist(samples3[:,2], color = c_purple, density = True, bins = 25, alpha = 0.75)

    # ~ plt.xlabel('Inclination')
    
    plt.figure()
    # ~ plt.hist(samples1[:,1], color = c_red, density = True, bins = 30, alpha = 0.5, histtype='stepfilled', hatch = '')
    plt.hist(samples1[:,1], color = c_red, density = True, bins = 30, alpha = 1, histtype='step')
    
    # ~ plt.hist(samples2[:,1], color = c_blue, density = True, bins = 30, alpha = 0.5, histtype='stepfilled', hatch = '')
    plt.hist(samples2[:,1], color = c_blue, density = True, bins = 30, alpha = 1, histtype='step')
    
    plt.hist(samples3[:,1], color = c_purple, density = True, bins = 30, alpha = 0.5, histtype='stepfilled', hatch = '')
    plt.hist(samples3[:,1], color = c_purple, density = True, bins = 30, alpha = 1, histtype='step')
    
    # ~ plt.hist(samples3_conn[:,1], color = 'black', density = True, bins = 25, alpha = 1, histtype='step', hatch = '//', zorder = -5)
    plt.xlabel(r'$m_2$')
    plt.ylabel(r'PDF')
    fig = plt.gcf()
    fig.set_size_inches(3, 4)
    
    plt.figure()
    # ~ plt.hist(samples1[:,1], color = c_red, density = True, bins = 30, alpha = 0.5, histtype='stepfilled', hatch = '')
    plt.hist(samples1[:,3], color = c_red, density = True, bins = 30, alpha = 1, histtype='step')
    
    # ~ plt.hist(samples2[:,1], color = c_blue, density = True, bins = 30, alpha = 0.5, histtype='stepfilled', hatch = '')
    plt.hist(samples2[:,3], color = c_blue, density = True, bins = 30, alpha = 1, histtype='step')
    
    plt.hist(samples3[:,3], color = c_purple, density = True, bins = 30, alpha = 0.5, histtype='stepfilled', hatch = '')
    plt.hist(samples3[:,3], color = c_purple, density = True, bins = 30, alpha = 1, histtype='step')
    
    # ~ plt.hist(samples3_conn[:,1], color = 'black', density = True, bins = 25, alpha = 1, histtype='step', hatch = '//', zorder = -5)
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'PDF')
    fig = plt.gcf()
    fig.set_size_inches(3, 4)
    
    # ~ plt.figure()
    # ~ plt.hist(samples3[:,0], color = c_purple, density = True, bins = 30, alpha = 0.75, histtype='stepfilled', hatch = '')
    # ~ plt.hist(samples3_conn[:,0], color = 'black', density = True, bins = 25, alpha = 1, histtype='step', hatch = '//')
    # ~ plt.xlabel('M1')
    
    plt.figure()
    # ~ plt.hist(samples1[:,1], color = c_red, density = True, bins = 30, alpha = 0.5, histtype='stepfilled', hatch = '')
    plt.hist(samples1[:,2], color = c_red, density = True, bins = 30, alpha = 1, histtype='step')
    
    # ~ plt.hist(samples2[:,1], color = c_blue, density = True, bins = 30, alpha = 0.5, histtype='stepfilled', hatch = '')
    plt.hist(samples2[:,2], color = c_blue, density = True, bins = 30, alpha = 1, histtype='step')
    
    
    plt.hist(samples3[:,2], color = c_purple, density = True, bins = 30, alpha = 0.5, histtype='stepfilled', hatch = '')
    plt.hist(samples3[:,2], color = c_purple, density = True, bins = 30, alpha = 1, histtype='step')
    
    # ~ plt.hist(samples3_conn[:,2], color = 'black', density = True, bins = 25, alpha = 1, histtype='step', hatch = '//', zorder = -5)
    plt.xlabel(r'Inclination')
    plt.ylabel(r'PDF')
    fig = plt.gcf()
    fig.set_size_inches(3, 4)
    
    
    
    # Plot K1
    plt.figure()
    period_in = 0.15804693
    
    samples = samples1
    m1F, m2F, iF, alphaF = samples[:,0], samples[:,1], samples[:,2], samples[:,3]
    a = (constants.G*(m1F+m2F)*constants.M_Sun*(period_in * constants.day)**2/(4*np.pi**2))**(1/3)
    K1 = 1e-3*np.sin(iF*np.pi/180)*2*np.pi*a/(period_in * constants.day) * (m2F/(m1F + m2F))
    plt.hist(K1, color = c_red, density = True, bins = 30, alpha = 1, histtype='step')
    
        
    samples = samples2
    m1F, m2F, iF, alphaF = samples[:,0], samples[:,1], samples[:,2], samples[:,3]
    a = (constants.G*(m1F+m2F)*constants.M_Sun*(period_in * constants.day)**2/(4*np.pi**2))**(1/3)
    K1 = 1e-3*np.sin(iF*np.pi/180)*2*np.pi*a/(period_in * constants.day) * (m2F/(m1F + m2F))
    plt.hist(K1, color = c_blue, density = True, bins = 30, alpha = 1, histtype='step')
    
        
    samples = samples3
    m1F, m2F, iF, alphaF = samples[:,0], samples[:,1], samples[:,2], samples[:,3]
    a = (constants.G*(m1F+m2F)*constants.M_Sun*(period_in * constants.day)**2/(4*np.pi**2))**(1/3)
    K1 = 1e-3*np.sin(iF*np.pi/180)*2*np.pi*a/(period_in * constants.day) * (m2F/(m1F + m2F))
    plt.hist(K1, color = c_purple, density = True, bins = 30, alpha = 1, histtype='step')
    
    plt.xlabel(r'K1')
    plt.ylabel(r'PDF')
    fig = plt.gcf()
    fig.set_size_inches(3, 4)
    
    
    # Plot K2
    plt.figure()
    period_in = 0.15804693
    
    samples = samples1
    m1F, m2F, iF, alphaF = samples[:,0], samples[:,1], samples[:,2], samples[:,3]
    a = (constants.G*(m1F+m2F)*constants.M_Sun*(period_in * constants.day)**2/(4*np.pi**2))**(1/3)
    K2 = 1e-3*np.sin(iF*np.pi/180)*2*np.pi*a/(period_in * constants.day) * (m1F/(m1F + m2F))
    plt.hist(K2, color = c_red, density = True, bins = 30, alpha = 1, histtype='step')
    
        
    samples = samples2
    m1F, m2F, iF, alphaF = samples[:,0], samples[:,1], samples[:,2], samples[:,3]
    a = (constants.G*(m1F+m2F)*constants.M_Sun*(period_in * constants.day)**2/(4*np.pi**2))**(1/3)
    K2 = 1e-3*np.sin(iF*np.pi/180)*2*np.pi*a/(period_in * constants.day) * (m1F/(m1F + m2F))
    plt.hist(K2, color = c_blue, density = True, bins = 30, alpha = 1, histtype='step')
    
        
    samples = samples3
    m1F, m2F, iF, alphaF = samples[:,0], samples[:,1], samples[:,2], samples[:,3]
    a = (constants.G*(m1F+m2F)*constants.M_Sun*(period_in * constants.day)**2/(4*np.pi**2))**(1/3)
    K2 = 1e-3*np.sin(iF*np.pi/180)*2*np.pi*a/(period_in * constants.day) * (m1F/(m1F + m2F))
    plt.hist(K2, color = c_purple, density = True, bins = 30, alpha = 1, histtype='step')
    plt.hist(K2, color = c_purple, density = True, bins = 30, alpha = 0.5, histtype='stepfilled')
    
    plt.xlabel(r'K2')
    plt.ylabel(r'PDF')
    fig = plt.gcf()
    fig.set_size_inches(3, 4)
    
    # ~ plt.figure()
    # ~ plt.hist(samples3[:,0]+samples3[:,1], color = c_purple, density = True, bins = 30, alpha = 0.75, histtype='stepfilled', hatch = '')
    # ~ plt.hist(samples3_conn[:,0]+samples3_conn[:,1], color = 'black', density = True, bins = 25, alpha = 1, histtype='step', hatch = '//')
    # ~ plt.xlabel('Total Mass')
    
    
    plt.figure()
    # ~ plt.hist(samples1[:,1], color = c_red, density = True, bins = 30, alpha = 0.5, histtype='stepfilled', hatch = '')
    plt.hist(samples1[:,0], color = c_red, density = True, bins = 30, alpha = 1, histtype='step')
    
    # ~ plt.hist(samples2[:,1], color = c_blue, density = True, bins = 30, alpha = 0.5, histtype='stepfilled', hatch = '')
    plt.hist(samples2[:,0], color = c_blue, density = True, bins = 30, alpha = 1, histtype='step')
    
    
    plt.hist(samples3[:,0], color = c_purple, density = True, bins = 30, alpha = 0.5, histtype='stepfilled', hatch = '')
    plt.hist(samples3[:,0], color = c_purple, density = True, bins = 30, alpha = 1, histtype='step')
    
    # ~ plt.hist(samples3_conn[:,0], color = 'black', density = True, bins = 25, alpha = 1, histtype='step', hatch = '//', zorder = -5)
    plt.xlabel(r'$m_1$')
    plt.ylabel(r'PDF')
    fig = plt.gcf()
    fig.set_size_inches(3, 4)
    
    plt.show()
    
    ################
    
    fig, ax = plt.subplots()

    corner.hist2d(samples1[:,2],samples1[:,0]+samples1[:,1], levels=[0.68,0.95], smooth = True,\
                  plot_datapoints = plot_datapoints, color = 'red', alpha = alpha, label = 'both', no_fill_contours = no_fill_contours, plot_density = True)
    
    corner.hist2d(samples2[:,2],samples2[:,0]+samples2[:,1], levels=[0.68,0.95], smooth = True,\
                  plot_datapoints = plot_datapoints, color = c_blue, alpha = alpha, label = 'echo delay only', no_fill_contours = no_fill_contours, plot_density = True)
    
    corner.hist2d(samples3[:,2],samples3[:,0]+samples3[:,1], levels=[0.68,0.95], smooth = True,\
                  plot_datapoints = plot_datapoints, color = c_purple, alpha = alpha, label = 'both', no_fill_contours = no_fill_contours, plot_density = True)
    
    
    square = patches.Rectangle((50, 2.18), 15, 3.15-2.18, edgecolor='black', facecolor='none')
    ax.add_patch(square)
    
    plt.xlabel(r'Inclination ($^{\circ}$)')
    plt.ylabel(r'Total Mass $(M_\odot)$')
    
    plt.ylim([0,5])
    plt.xlim([0,89])
    # ~ plt.xlim([0,18])
    
    # ~ plt.axvline(x=44,c='black')
    # ~ plt.axvline(x=3,c='black')
    # ~ plt.axhline(y=2.1,c='black')
    # ~ plt.scatter([44],[2.1],marker='s',color='black')
    
    fig.set_size_inches(4, 4)
    plt.show()
    
    
    
    ################
    
    # load echo data and radial velocity data
    print('both', both)
    echoDelay_data = np.loadtxt(f'mcmc_results//{both}//echo_delay_data.csv',delimiter=',',skiprows=1)
    
    fig, axs = plt.subplots(1,2)
    asra_lt = delay_model.ASRA_LT_model()
    
    pp = np.linspace(0,1,100)
    VV = []
    N = 5000
    print(len(samples3)/N)
    for i in range(len(samples3)//N):
        guess = samples3[i*N]
        tt, vv = asra_lt.evaluate_array(pp,*guess)
        print(f'{i}/{len(samples3)//N}')
        axs[0].plot(pp, tt, c = 'purple', alpha = 0.1)
        # ~ axs[2,1].plot(pp,vv,'g-', alpha = 0.15)
    
    N = 500
    for i in range(len(samples3)//N):
        if i%100 == 0:
            print(f'{i}/{len(samples3)//N}')
        guess = samples3[i*N]
        tt, vv = asra_lt.evaluate_array([0.65,0.70,0.75],*guess)
        VV.append(max(vv))
        
    
    axs[0].set_xlim([0,1])
    axs[0].set_xlabel('Orbital Phase')
    axs[0].set_ylabel('Echo Delay (s)')
    
    axs[0].errorbar(echoDelay_data[:,0],echoDelay_data[:,1],echoDelay_data[:,2], fmt='o', capsize=3, color = 'black')     
    
    axs[1].hist(VV,density=True,color='purple',bins = 40)
    axs[1].set_ylabel('PDF')
    axs[1].axvline(x=79.8,linestyle='--',c='black')
    axs[1].axvspan(79.8-5,79.8+5,alpha=0.4,color='black')
    axs[1].set_xlabel(r'$K_{em}$ (km/s)')
    
    fig.set_size_inches(8, 5)
    
    plt.show()

def fig5_gen_constraints():
    M = 2
    P = 0.15804693
    model = delay_model.ASRA_LT_model()
    
    if True:
    
        ii = np.linspace(1,90,300)
        qq = np.linspace(0.0000001,1,100)
        mm = np.linspace(0.1,2.5,1)
        eclipse_map = np.empty([len(mm), len(qq), len(ii)])
        
        for j1 in range(len(ii)):
            print(j1,len(ii))
            for j2 in range(len(qq)):
                for j3 in range(len(mm)):
                    eclipse_map[j3, j2, j1] = model.check_eclipse(m1_in=mm[j3]/(1+qq[j2]), m2_in=mm[j3]/(1+1/qq[j2]), inclination_in=ii[j1], period_in=P, fancy = True)
        
        
        eclipse_map_avg = np.mean(eclipse_map, axis = 0)
        
        i_solved = np.empty(len(qq))
        for j2 in range(len(qq)):
            i_solved[j2] = ii[np.where(eclipse_map_avg[j2, :] >= 1)[0][0]]
            
        data_to_save = np.array([i_solved, qq]).transpose()
        np.savetxt('eclipse_conn.csv', data_to_save)
        
        plt.imshow(eclipse_map_avg, extent = [min(ii), max(ii), min(qq), max(qq)], aspect = 'auto', cmap = 'Greys_r', origin='lower')
        plt.scatter(i_solved, qq)
        plt.xlabel('Inclination')
        plt.ylabel(r'Mass Ratio $(m_2/m_1)$')
        plt.title('eclipse map')
        plt.show()
    
    
    ii = np.linspace(1,90,300)
    qq = np.linspace(0.0001,1,300)
    mm = np.linspace(1.62,1.64,10)
    K1_map = np.empty([len(mm), len(qq), len(ii)])
    
    for j1 in range(len(ii)):
        for j2 in range(len(qq)):
            for j3 in range(len(mm)):
                a = (constants.G*mm[j3]*constants.M_Sun*(P * constants.day)**2/(4*np.pi**2))**(1/3)
                K1 = 1e-3*np.sin(ii[j1]*np.pi/180)*2*np.pi*a/(P * constants.day)*qq[j2]/(1 + qq[j2])
                # ~ K1_map[j3, j2, j1] = K1 > 101.5 - 2*11.5 and K1 < 101.5 + 2*11.5
                K1_map[j3, j2, j1] = np.exp(-0.5*((K1 - 101.5)/(11.5))**2)
    
    K1_map_avg = np.mean(K1_map, axis = 0)
    K1_map_avg = K1_map_avg/np.sum(K1_map_avg)
    K1_map_avg[K1_map_avg < 1.07e-5] = 0 # 95% confidence interval
    K1_map_avg[K1_map_avg > 0] = 1
    
    plt.imshow(K1_map_avg, extent = [min(ii), max(ii), min(qq), max(qq)], aspect = 'auto', cmap = 'Greys_r', origin='lower')
    plt.xlabel('Inclination')
    plt.ylabel(r'Mass Ratio $(m_2/m_1)$')
    plt.title(r'K1_map, $101.5 \pm 11.5$ km/s 1-$\sigma$')
    plt.show()
    
    both_map = K1_map_avg*(1-eclipse_map_avg)
    plt.imshow(both_map, extent = [min(ii), max(ii), min(qq), max(qq)], aspect = 'auto', cmap = 'Greys_r', origin='lower')
    plt.xlabel('Inclination')
    plt.ylabel(r'Mass Ratio $(m_2/m_1)$')
    plt.title(r'Both')
    plt.show()
  
  
### appendix plots ###

# Delay curve collections for single orbital parameter sets
def appendix_1(m2_in = 0.7, m1_in = 1.4, period_in = 0.787, inclination_in = 44, disk_angle_in = 5):
    asra_lt = delay_model.ASRA_LT_model()
    
    # Trends in m1, m2, inclination, disk angle, (perhaps also orbital period)
    
    pp = np.linspace(0,1,100)
    
    cmap = cm.bwr
    fig, axs = plt.subplots(2,4)
    
    m1m1 = np.linspace(1,2,5, endpoint = True)
    TT = np.empty([len(m1m1),len(pp)])
    VV = np.empty([len(m1m1),len(pp)])
    PP = np.ones([len(m1m1),len(pp)]) * pp
    for i in range(len(m1m1)):
        print(f'{i+1}/{len(m1m1)}')
        TT[i,:], VV[i,:] = asra_lt.evaluate_array(pp,m1_in=m1m1[i],m2_in = m2_in, period_in = period_in, inclination_in = inclination_in, disk_angle_in = disk_angle_in)
    lc1 = multiline(PP, TT, c = m1m1, ax = axs[0,0], cmap = cmap)
    lc2 = multiline(PP, VV, c = m1m1, ax = axs[1,0], cmap = cmap)
    fig.colorbar(lc1)
    fig.colorbar(lc2)
    
    m2m2 = np.linspace(0.1,1,5, endpoint = True)
    TT = np.empty([len(m2m2),len(pp)])
    VV = np.empty([len(m2m2),len(pp)])
    PP = np.ones([len(m2m2),len(pp)]) * pp
    for i in range(len(m2m2)):
        print(f'{i+1}/{len(m2m2)}')
        TT[i,:], VV[i,:] = asra_lt.evaluate_array(pp,m1_in=m1_in,m2_in = m2m2[i], period_in = period_in, inclination_in = inclination_in, disk_angle_in = disk_angle_in)
    lc1 = multiline(PP, TT, c = m2m2, ax = axs[0,1], cmap = cmap)
    lc2 = multiline(PP, VV, c = m2m2, ax = axs[1,1], cmap = cmap)
    fig.colorbar(lc1)
    fig.colorbar(lc2)
    
    ii = np.linspace(0,90,5, endpoint = True)
    TT = np.empty([len(ii),len(pp)])
    VV = np.empty([len(ii),len(pp)])
    PP = np.ones([len(ii),len(pp)]) * pp
    for i in range(len(ii)):
        print(f'{i+1}/{len(ii)}')
        TT[i,:], VV[i,:] = asra_lt.evaluate_array(pp,m1_in=m1_in,m2_in = m2_in, period_in = period_in, inclination_in = ii[i], disk_angle_in = disk_angle_in)
    lc1 = multiline(PP, TT, c = ii, ax = axs[0,2], cmap = cmap)
    lc2 = multiline(PP, VV, c = ii, ax = axs[1,2], cmap = cmap)
    fig.colorbar(lc1)
    fig.colorbar(lc2)
    
    aa = np.linspace(0,15,5, endpoint = True)
    TT = np.empty([len(aa),len(pp)])
    VV = np.empty([len(aa),len(pp)])
    PP = np.ones([len(aa),len(pp)]) * pp
    for i in range(len(aa)):
        print(f'{i+1}/{len(aa)}')
        TT[i,:], VV[i,:] = asra_lt.evaluate_array(pp,m1_in=m1_in,m2_in = m2_in, period_in = period_in, inclination_in = inclination_in, disk_angle_in = aa[i])
    lc1 = multiline(PP, TT, c = aa, ax = axs[0,3], cmap = cmap)
    lc2 = multiline(PP, VV, c = aa, ax = axs[1,3], cmap = cmap)
    fig.colorbar(lc1)
    fig.colorbar(lc2)
    
    titles = [r'$m_1$' + r' $(M_{\odot})$',r'$m_2$' + r' $(M_{\odot})$',r'$i$' + r' $(^{\circ})$',r'$\alpha$' + r' $(^{\circ})$']
    for i in range(4):
        axs[1,i].set_xlabel('Orbital Phase')
        axs[0,i].set_title(titles[i])

    
    axs[0,0].set_ylabel('Time Delay (s)')
    axs[1,0].set_ylabel('Radial Velocity (km/s)')
    
    plt.show()

# 2D cuts of Roche Lobe along 0 and 90 degree planes
def appendix_2(m2_in = 0.7, m1_in = 1.4, period_in = 0.787):
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
        r_roche_90[i] = opt.brentq(delay_model.rochePotential,L1*1e-3,L1,args=(psi_roche[i],np.radians(90),m1,m2,period,potential_at_L1),maxiter=100)/constants.AU
        r_roche_0[i] = opt.brentq(delay_model.rochePotential,L1*1e-3,L1,args=(psi_roche[i],np.radians(0),m1,m2,period,potential_at_L1),maxiter=100)/constants.AU
        r_roche_45[i] = opt.brentq(delay_model.rochePotential,L1*1e-3,L1,args=(psi_roche[i],np.radians(45),m1,m2,period,potential_at_L1),maxiter=100)/constants.AU
        
    fig, ax = plt.subplots()
    
    plt.axhline(y=0,c='k',linestyle='--', zorder= -5)
    plt.axvline(x=0,c='k',linestyle='--', zorder = -5)
    
    ax.plot(r_roche_90*np.cos(psi_roche), r_roche_90*np.sin(psi_roche), linewidth = 1, color = 'green', label = r'$\beta$ = 90 deg')
    ax.plot(r_roche_0*np.cos(psi_roche), r_roche_0*np.sin(psi_roche), linewidth = 1, linestyle = '-', color = 'blue', label = r'$\beta$ = 0 deg')
    # ~ ax.plot(r_roche_45*np.cos(psi_roche), r_roche_45*np.sin(psi_roche), linewidth = 1, linestyle = '-', color = 'k', label = r'$\beta$ = 45 deg')
    
    
    egg_radiusDonor = a*(.49*q**(.667))/(.6*q**(.667)+np.log(1+q**(.333)))/constants.AU
    Egg_radiusDonor = egg_radiusDonor*np.ones(np.shape(psi_roche))
    
    ax.plot(Egg_radiusDonor*np.cos(psi_roche), Egg_radiusDonor*np.sin(psi_roche), linewidth = 1, linestyle = '--', color = 'red', label = r'Eggleton')
    
    res = 10*(r_roche_90 - r_roche_0)
    ax.plot(res*np.cos(psi_roche), res*np.sin(psi_roche), linewidth = 1, color = 'purple', label = r'10X Residual')
    
    ax.scatter([a/constants.AU],[0],s=150,c='black')
    
    ax.text(x=0.9*a/constants.AU,y=0.05*a,s='Compact\nObject')
    ax.text(x=L1/constants.AU,y=0.05*L1,s=r'$L_1$')
    
    print(f'Eggleton Volume: {4/3*np.pi*egg_radiusDonor**3}')
    # ~ print(f'beta 0 Volume: {2*np.pi*np.sum(r_roche_0**2*np.sin(psi_roche))*np.pi/len(psi_roche)}')
    

    # ~ ax.set_rmax(2)
    # ~ ax.set_rticks([0.5, 1, 1.5, 2])  # Less radial ticks
    L = 1.1*max(r_roche_0)
    plt.xlim([-L,1.2*a/constants.AU])
    plt.ylim([-L,L])
    ax.set_aspect('equal')
    ax.set_xlabel('X coord (A.U.)')
    ax.set_ylabel('Y coord (A.U.)')
    plt.legend()
    plt.show()
         

### Additional plots ###

def plot_hist_3d():
    echo = '1719128205' # without K1 constraint
    rv = '1719125933' # with K1 constraint
    both = '1719126447' # with K1 constraint
    
    with open(f'mcmc_results//{rv}//sampler.pickle', 'rb') as inp:
        sampler = pickle.load(inp)
    samples1 = sampler.get_chain(discard=150, thin=5, flat=True)
    # ~ plt.scatter(samples[:,2],samples[:,1]/samples[:,0], s = s, alpha = alpha, label = 'radial velocity only', c = 'red')
    
    with open(f'mcmc_results//{echo}//sampler.pickle', 'rb') as inp:
        sampler = pickle.load(inp)
    samples2 = sampler.get_chain(discard=150, thin=5, flat=True)
    # ~ plt.scatter(samples[:,2],samples[:,1]/samples[:,0], s = s, alpha = alpha, label = 'echo delay only', c = 'blue')

    with open(f'mcmc_results//{both}//sampler.pickle', 'rb') as inp:
        sampler = pickle.load(inp)
    samples3 = sampler.get_chain(discard=150, thin=5, flat=True)
    
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    
    ss1 = samples1
    ss2 = samples2
    
    ax.scatter(ss1[:,2], ss1[:,1] + ss1[:,0], ss1[:,1]/ss1[:,0], color = 'red', s = 1, alpha = 0.25)
    ax.scatter(ss2[:,2], ss2[:,1] + ss2[:,0], ss2[:,1]/ss2[:,0], color = 'blue', s = 1, alpha = 0.25)
    
    ax.set_xlabel('Inclination (deg)')
    ax.set_ylabel(r'Total Mass (M$_\odot$)')
    ax.set_zlabel('Mass Ratio ($m_2$/$m_1$)')
    
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
  
def rv_for_different_disk_angles(m1_in = 1, inclination_in = 90):
    data_dir = r'C:\Users\thoma\Documents\GitHub\echoxrb\data\mz_data'
    m2_in = 0.6*m1_in
    period_in = 1.5
    phases = np.linspace(0,1,20)
    yy_3d = np.zeros(len(phases))
    vv_3d = np.zeros(len(phases))
    scale = 1
    
    for i in range(len(phases)):
        vv_3d[i] = delay_model.timeDelay_3d_full(phases[i], mode = 'rv', Q = 25, m1_in = m1_in, m2_in = m2_in, inclination_in = inclination_in, disk_angle_in = 0, period_in = period_in)
    plt.plot(phases,vv_3d*scale)
    
    data = np.loadtxt(f"{data_dir}//alpha0.txt")
    plt.scatter(data[:,0],data[:,1],label=r'$\alpha$ = 0')
    
    for i in range(len(phases)):
        vv_3d[i] = delay_model.timeDelay_3d_full(phases[i], mode = 'rv', Q = 25, m1_in = m1_in, m2_in = m2_in, inclination_in = inclination_in, disk_angle_in = 8, period_in = period_in)
    plt.plot(phases,vv_3d*scale)
    
    data = np.loadtxt(f"{data_dir}//alpha8.txt")
    plt.scatter(data[:,0],data[:,1],label=r'$\alpha$ = 8')
    
    for i in range(len(phases)):
        vv_3d[i] = delay_model.timeDelay_3d_full(phases[i], mode = 'rv', Q = 25, m1_in = m1_in, m2_in = m2_in, inclination_in = inclination_in, disk_angle_in = 14, period_in = period_in)
    plt.plot(phases,vv_3d*scale)
    
    data = np.loadtxt(f"{data_dir}//alpha14.txt")
    plt.scatter(data[:,0],data[:,1],label=r'$\alpha$ = 14')
    
    for i in range(len(phases)):
        vv_3d[i] = delay_model.timeDelay_3d_full(phases[i], mode = 'rv', Q = 25, m1_in = m1_in, m2_in = m2_in, inclination_in = inclination_in, disk_angle_in = 18, period_in = period_in)
    plt.plot(phases,vv_3d*scale)
    
    data = np.loadtxt(f"{data_dir}//alpha18.txt")
    plt.scatter(data[:,0],data[:,1],label=r'$\alpha$ = 18')
    
    plt.xlabel('Orbital Phase')
    plt.ylabel('Radial Velocity (km/s)')
    plt.legend()
    plt.show()

def rv_for_different_inclinations(m1_in = 1.5):
    data_dir = r'C:\Users\thoma\Documents\GitHub\echoxrb\data\mz_data'
    m2_in = 0.6*m1_in
    period_in = 0.60
    phases = np.linspace(0,1,20)
    yy_3d = np.zeros(len(phases))
    vv_3d = np.zeros(len(phases))
    scale = 0.9
    
    for i in range(len(phases)):
        print(i)
        vv_3d[i] = delay_model.timeDelay_3d_full(phases[i], mode = 'rv', Q = 25, m1_in = m1_in, m2_in = m2_in, inclination_in = 90, disk_angle_in = 0, period_in = period_in)
    plt.plot(phases,vv_3d*scale)
    
    data = np.loadtxt(f"{data_dir}//i90.txt")
    plt.scatter(data[:,0],data[:,1],label=r'i = 90')
    
    for i in range(len(phases)):
        vv_3d[i] = delay_model.timeDelay_3d_full(phases[i], mode = 'rv', Q = 25, m1_in = m1_in, m2_in = m2_in, inclination_in = 40, disk_angle_in = 0, period_in = period_in)
    plt.plot(phases,vv_3d*scale)
    
    data = np.loadtxt(f"{data_dir}//i40.txt")
    plt.scatter(data[:,0],100*data[:,1],label=r'i = 40')
    
    for i in range(len(phases)):
        vv_3d[i] = delay_model.timeDelay_3d_full(phases[i], mode = 'rv', Q = 25, m1_in = m1_in, m2_in = m2_in, inclination_in = 20, disk_angle_in = 0, period_in = period_in)
    plt.plot(phases,vv_3d*scale)
    
    data = np.loadtxt(f"{data_dir}//i20.txt")
    plt.scatter(data[:,0],data[:,1],label=r'i = 20')
    plt.axvline(x = 0.25)
    plt.xlabel('Orbital Phase')
    plt.ylabel('Radial Velocity (km/s)')
    plt.legend()
    plt.show()
 
def plot_ratio_Tem_T2():
    asra_lt = delay_model.ASRA_LT_model()
    
    Q = np.linspace(0.1,1,15)**3
    kcorr = np.empty(len(Q))
    tcorr_max = np.empty(len(Q))
    tcorr_min = np.empty(len(Q))
    
    disk_angle_in = 0
    M_in = 2
    
    for qi in range(len(Q)):
        m1_in = M_in/(1+Q[qi])
        m2_in = Q[qi]*m1_in
        _, Kem = asra_lt.evaluate(0.75, m1_in = m1_in, m2_in = m2_in, disk_angle_in = disk_angle_in)
        Tem_max, _ = asra_lt.evaluate(0.5, m1_in = m1_in, m2_in = m2_in, disk_angle_in = disk_angle_in)
        Tem_min, _ = asra_lt.evaluate(0, m1_in = m1_in, m2_in = m2_in, disk_angle_in = disk_angle_in)
        K2 = delay_model.radialVelocity(0.75, m1_in = m1_in, m2_in = m2_in)
        T2_max = delay_model.timeDelay_sp(0.5, m1_in = m1_in, m2_in = m2_in, setting = 'cm')
        T2_min = delay_model.timeDelay_sp(0, m1_in = m1_in, m2_in = m2_in, setting = 'cm')
        
        kcorr[qi] = Kem/K2
        tcorr_max[qi] = Tem_max/T2_max
        tcorr_min[qi] = Tem_min/T2_min

    plt.scatter(Q,kcorr,label = r'$\kappa$')
    
    plt.scatter(Q,tcorr_max,label = r'$\tau_{max}$')
    plt.scatter(Q,tcorr_min,label = r'$\tau_{min}$')
    plt.xlabel('Mass Ratio (q)')
    plt.ylabel('T_em/T2')
    
    
    
    xx = np.linspace(0,1,100)
    yy = 0.5/(0.5+xx)
    
    plt.plot(xx,yy)
    

    plt.legend()
    plt.show()
