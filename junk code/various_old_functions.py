# The chopping block
def timeDelay_radialVelocity_3d(phase, \
                  m1_in=1.4, m2_in=0.7, inclination_in=44., disk_angle_in = 5, period_in=.787, plot = True, Q = 35, beta_approx = 0):
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
        cmap = cm.PiYG
        ax = fig.add_subplot(1,3,2, projection='3d')
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
        ax = fig.add_subplot(1,3,3, projection='3d')
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
        
        
    return 1e-3*np.sin(2*np.pi*phase)*np.sin(inclination)*2*np.pi*a/period*(1/(1+q))*(1-radiusDonor/a*(1+q))

def go():
    # ~ inclinations = np.array([0,45,90])
    # ~ disk_angles = np.linspace(0,20,20)
    
    # ~ disk_angles = np.linspace(0,20,20)
    inclinations = np.linspace(0,90,90)
    disk_angles = np.array([0,5,10,15,20])
    
    # ~ beta_corrs = np.zeros([len(inclinations),len(disk_angles)])
    beta_corrs2 = np.zeros([len(inclinations),len(disk_angles)])
    
    for j in range(len(inclinations)):
        print(f"j = {j}")
        for k in range(len(disk_angles)):
            # ~ print(f"    k = {k}")
            beta_corrs2[j, k] = timeDelay_3d_asra_limited(45,inclination_in = inclinations[j], disk_angle_in = disk_angles[k], m2_in = 0.1, t0=0, betaReturn = True, Q = 50)
    
    # ~ fig, ax = plt.subplots()
    # ~ im = ax.imshow(beta_corrs2)
    # ~ fig.colorbar(im, cax=ax)
    # ~ ax.set_aspect(1/5)
    # ~ plt.show()
    
    
    
    # ~ plt.plot(disk_angles, beta_corrs[0,:], label = inclinations[0])
    # ~ plt.scatter(disk_angles, beta_corrs2[0,:], label = inclinations[0])
    # ~ plt.plot(disk_angles, beta_corrs[1,:], label = inclinations[1])
    # ~ plt.scatter(disk_angles, beta_corrs2[1,:], label = inclinations[1])
    # ~ plt.plot(disk_angles, beta_corrs[2,:], label = inclinations[2])
    # ~ plt.scatter(disk_angles, beta_corrs2[2,:], label = inclinations[2])
    
    # ~ plt.plot(inclinations, beta_corrs[:,0], label = disk_angles[0])
    plt.scatter(inclinations, beta_corrs2[:,0], label = disk_angles[0])
    # ~ plt.plot(inclinations, beta_corrs[:,1], label = disk_angles[1])
    plt.scatter(inclinations, beta_corrs2[:,1], label = disk_angles[1])
    # ~ plt.plot(inclinations, beta_corrs[:,2], label = disk_angles[2])
    plt.scatter(inclinations, beta_corrs2[:,2], label = disk_angles[2])
    # ~ plt.plot(inclinations, beta_corrs[:,3], label = disk_angles[3])
    plt.scatter(inclinations, beta_corrs2[:,3], label = disk_angles[3])
    # ~ plt.plot(inclinations, beta_corrs[:,4], label = disk_angles[4])
    plt.scatter(inclinations, beta_corrs2[:,4], label = disk_angles[4])
    
    
    plt.legend()
    plt.show()

def timeDelay_3d_asra_limited(beta_approx, inclination_in, disk_angle_in, t0, phase = 0.5, m1_in=1.4, m2_in=0.7, period_in=.787, plot = False, Q = 75, betaReturn = False):
    
    beta_approx = beta_approx*np.pi/180
                      
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
    if plot:
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
        r_roche[i] = opt.brentq(rochePotential,L1*1e-3,L1,args=(psi_roche[i],beta_approx,m1,m2,period,potential_at_L1),maxiter=100)
    
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
        if betaReturn:
            return 180/np.pi*np.arcsin(np.sqrt(np.average(np.sin(BETA_roche)**2,weights=R_roche**2*np.sin(PSI_roche)*apparent_intensities)))
        centroid_delays = np.average(delays,weights=R_roche**2*np.sin(PSI_roche)*apparent_intensities) # Compute the centroid of the distribution of delay times weighted to the intensities
    else:
        centroid_delays = 0
    
    if plot == True:
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
    else:
        return centroid_delays - t0

def fig_mz():
    # ~ plt.scatter(alphas,N0,label=r'$N_0$')
    # ~ plt.scatter(alphas,N1,label=r'$N_1$')
    # ~ plt.scatter(alphas,N2,label=r'$N_2$')
    # ~ plt.scatter(alphas,N3,label=r'$N_3$')
    # ~ plt.scatter(alphas,N4,label=r'$N_4$')
    
    # ~ plt.legend()
    # ~ plt.show()
    data = np.loadtxt(r'data/mz_coeffs.txt',delimiter=',')
    alphas = data[:,0]
    # ~ alphas = np.array([0])
    
    
    m1_in = 1.14
    period_in = 0.232
    
    inclinations = [40,90]
    fig, axs = plt.subplots(1,len(inclinations))
    for i_index in range(len(inclinations)):
        inclination_in = inclinations[i_index]
    
        if inclination_in > 60:
            N0 = data[:,1]
            N1 = data[:,2]
            N2 = data[:,3]
            N3 = data[:,4]
            N4 = data[:,5]
        if inclination_in <= 60:
            N0 = data[:,6]
            N1 = data[:,7]
            N2 = data[:,8]
            N3 = data[:,9]
            N4 = data[:,10]
    
        q = np.linspace(0,1.5,100)
        
        for alpha_index in range(len(alphas)):
            print('alpha',alphas[alpha_index])
            
            k_corr_mz = N0[alpha_index]+N1[alpha_index]*q+N2[alpha_index]*q**2+N3[alpha_index]*q**3+N4[alpha_index]*q**4
            
            Q = np.linspace(0.1,1.5,5)
            k_corr = np.zeros(len(Q))
            
            for i in range(len(Q)):
                k_corr[i] = timeDelay_3d_full(0.25, m1_in = m1_in, m2_in = m1_in*Q[i], period_in = period_in, disk_angle_in = alphas[alpha_index], inclination_in = inclination_in, Q = 25, mode='return_rv')/radialVelocity(0.25, m1_in = m1_in, m2_in = m1_in*Q[i], period_in = period_in, inclination_in = inclination_in)
            
            axs[i_index].plot(q, k_corr_mz, color = cm.jet(alpha_index/len(alphas)))
            axs[i_index].scatter(Q, k_corr, label = r'$\alpha$' + f' = {alphas[alpha_index]}',color=cm.jet(alpha_index/len(alphas)))
        
        
        k_corr_egg = radialVelocity(0.27, m1_in = m1_in, m2_in = m1_in*q, period_in = period_in, inclination_in = inclination_in,setting='egg')/radialVelocity(0.25, m1_in = m1_in, m2_in = m1_in*q, period_in = period_in, inclination_in = inclination_in)
        axs[i_index].plot(q, k_corr_egg, 'r--', label =  'SP Eggleton')
        
        k_corr_egg = radialVelocity(0.27, m1_in = m1_in, m2_in = m1_in*q, period_in = period_in, inclination_in = inclination_in,setting='plav')/radialVelocity(0.25, m1_in = m1_in, m2_in = m1_in*q, period_in = period_in, inclination_in = inclination_in)
        axs[i_index].plot(q, k_corr_egg, 'k-.', label =  'SP Plavec')
        
        axs[i_index].set_xlabel(r'$q=m_2/m_1$')
        axs[i_index].set_ylabel(r'$K_corr$')
        axs[i_index].legend()
        axs[i_index].set_title(f'inclination = {inclination_in} deg')
        axs[i_index].set_ylim([-.1,1.25])
        

    plt.show()
    

def timeDelay_3d_asra_partial(phase, \
                  m1_in=1.4, m2_in=0.7, inclination_in=44., disk_angle_in = 5, period_in=.787, plot = False, Q = 25, beta_approx = 45):
    # Constant setting for azimuthal angle during construction of Roche Lobe (best value is usually 45 deg)
    beta_approx = beta_approx * np.pi/180
    
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
    if plot:
        psi_roche = np.linspace(0, np.pi, num=Q, endpoint=True)
    else:
        psi_roche = np.linspace(0, np.pi/2, num=Q, endpoint=True)
        
    r_roche = np.zeros(len(psi_roche))
    # Solve for r_roche as a function of psi_roche at fixed beta (this is the azimuthally symmetric approximation, we fix beta at a value - usually 45 deg)
    for i in range(len(psi_roche)):
        r_roche[i] = opt.brentq(rochePotential,L1*1e-3,L1,args=(psi_roche[i],beta_approx,m1,m2,period,potential_at_L1),maxiter=100)
       
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
    
    #################################
    # Now let's compute the apparent intensity over the Roche Lobe, we have 3 terms A1, A2, and A3
    
    #Let's compute A1: Distance Attenuation
    
    a1 = p1**(-2) # This is a 1D version of the list because using ASRA p1 has no beta dependence (in the full model it does)
    
    #Let's compute A2: Projected area on surface of donor towards accretor
    
    # First we need to compute vectors normal to the Roche Lobe surface via a gradient of the potential function
    # Since this function only depends on r_roche (aka p2) and psi, we can use the 1D arrays (again this is only true in ASRA, in the full model there is beta dependence)
    vecs_normal_roche = polarGradRochePotential(r_roche,psi_roche,beta_approx,m1,m2,period)# returns dV/dp2 vec(p2) + 1/p2*dV/dpsi vec(psi)
    
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
        # The Jacobian is sin(psi)*R**2
        centroid_delays = np.average(delays,weights=R_roche**2*np.sin(PSI_roche)*apparent_intensities) # Compute the centroid of the distribution of delay times weighted to the intensities
    else:
        centroid_delays = 0
    
    if plot == True:
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
    else:
        return centroid_delays
