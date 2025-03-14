import numpy as np
import matplotlib.pyplot as plt
import constants
import math

#from scipy import decorate
#import scipy.ndimage.filters._ni_support
#from scipy._lib.decorator import decorate
from scipy.ndimage.filters import gaussian_filter
#from scipy.interpolate import interpolate

plt.ion()
plt.rc('text', usetex=True)

def go():
	
	###To do multiple time delay curves###
	#m1=0.2
	#while(m1<0.4):
	#	m2=1.4	
	#	R1=m1**(0.8)
	#	R2=m2**(0.8)
	#	eccentricity=0.7
	#	inclination=0
	#	semimajorAxes(m1,m2,R1,R2, eccentricity,inclination)
	#	m1+=0.005

	m1=0.4
	m2=1.4
	R1=m1**(0.8)
	R2=m2**(0.8)
	eccentricity=0.9
	inclination=45*(3.14159/180)
	P=16.6*constants.day
	phi=-45*(3.14159/180)
	#print(phi)

	xarray1,yarray1,xarray2,yarray2,rarray,tarray,thetaarray = orbit(m1,m2,R1,R2, eccentricity,P)
	xarray1A,yarray1A,xarray2A,yarray2A,rarrayA,tarrayA,thetaarrayA = xarray1,yarray1,xarray2,yarray2,rarray,tarray,thetaarray

	#plotMotion(xarray1,yarray1,xarray2,yarray2)

	xarray1,yarray1,xarray2,yarray2,rarray,tarray,inclination,phi = orient(xarray1,yarray1,xarray2,yarray2,rarray,tarray,inclination,phi)

	#plotMotion(xarray1,yarray1,xarray2,yarray2)
	
	timeDelayCurve(rarray,tarray,P)
	obs_timeDelayCurve(rarray,thetaarray,phi,inclination,tarray,P)
	
	#multipleTimeDelayCurve(rarray,tarray,P)
	
	#gaussianCurve(rarray[500])
	#gaussianCurve(rarray[0])
def orient(xarray1,yarray1,xarray2,yarray2,rarray,tarray,inclination,phi): #orients the results of orbit according to the angles of inclination and longitude of periastron
        i = inclination
        x_1a = np.cos(phi)*xarray1
        x_1b = (-1)*np.sin(phi)*yarray1
        x_1 = np.add(x_1a,x_1b) # rotated x1
        y_1a = np.sin(phi)*xarray1
        y_1b = np.cos(phi)*yarray1
        y_1 = np.add(y_1a,y_1b) # rotated y1
        x_2a = np.cos(phi)*xarray2
        x_2b = (-1)*np.sin(phi)*yarray2
        x_2 = np.add(x_2a,x_2b) #rotated x2
        y_2a = np.sin(phi)*xarray2
        y_2b = np.cos(phi)*yarray2
        y_2 = np.add(y_2a,y_2b)#rotated y2
        #The y component is along the orbital plane and perpandicular to the line of sight.
        #The x component is along the orbital plane and perpandicular to the y component.
        x_1 = np.cos(i)*x_1 #projection onto plane perpandicular to the line of sight.
        x_2 = np.cos(i)*x_2
        return(x_1,y_1,x_2,y_2,rarray,tarray,inclination,phi)
        


def orbit(m1,m2,R1,R2,eccentricity,period):
	"""Do the math to calculate the oribit of a binary system
	Stores values for position in arrays (x,y and r), as well as the values for the time steps
	(2-body problem solver)
	"""
	m1=m1*constants.M_Sun#mass 1
	m2=m2*constants.M_Sun#mass 2
	M=m1+m2
	R1=R1*constants.R_Sun#radius 1
	R2=R2*constants.R_Sun#radius 2
	P=period#period 233280 seconds
	e=float(eccentricity)
	#i=71.88*constants.degrees_to_radians#orbital inclination
	#phi=0 #orientation periastron
	vcmxp=0#center of mass velocity vetors
	vcmyp=0
	vcmzp=0


	xarray1=np.array([])
	yarray1=np.array([])
	xarray2=np.array([])
	yarray2=np.array([])

	rarray=np.array([])
	tarray=np.array([])
	thetaarray=np.array([])


	mu=m1*m2/M
	a = (constants.G * M * P**2 / (2.0*np.pi)**2)**(1.0/3.0) #Kepler III solving for semi-major axis of equiv mass orbit
	a1=(mu/m1)*a #translate 1-body equiv mass back into 2-body real semi-major axes. Semi-major axis of 1st mass.
	a2=(mu/m2)*a #Semi-major axis of 2nd mass.


	N=10001
	
	t=0.
	dt=P/np.float(N)
	theta=0.
	L_ang=mu*np.sqrt(constants.G*M*a*(1-e*e))
	dAdt=L_ang/(2*mu)

	
	


	while(t < P + dt/2):# and theta <2*np.pi):
                #theta is a dummy variable used to evaluate the polar function
		r=np.float(a*(1.-e*e)/(1.+e*np.cos(theta))) #Polar function for distance from focus (equiv body problem).
		x=r*np.cos(theta) #Decomposed r into x and y components
		y=r*np.sin(theta)
		x1=(mu/m1)*x # translate equiv body problem into real positions
		y1=(mu/m1)*y
		x2= -(mu/m2)*x
		y2= -(mu/m2)*y


		
		xarray1=np.concatenate((xarray1,[x1])) #stores locations as arrays
		yarray1=np.concatenate((yarray1,[y1]))
		xarray2=np.concatenate((xarray2,[x2]))
		yarray2=np.concatenate((yarray2,[y2]))

		rarray=np.concatenate((rarray,[r]))
		tarray=np.concatenate((tarray,[t]))
		thetaarray=np.concatenate((thetaarray,[theta]))

		
		dtheta = (2.*dAdt/(r*r))*dt
		theta += dtheta
		t += dt

	return(xarray1,yarray1,xarray2,yarray2,rarray,tarray,thetaarray)

		

	
	
	

def plotMotion(xarray1,yarray1,xarray2,yarray2):
	"""Plots the position of the orbits of the two masses in an XRB
	"""
	fig = plt.figure(1)
	try:
		fig.close()
	except:
		dumdum=1
	fig.clf()
	ax = fig.add_subplot(111)
	#yarray1I = interpolate.interp1d(xarray1, yarray1)
	#xarray1I = interpolate.interp1d(yarray1, xarray1)
        #tFine = np.linspace(-15, 70, 10000)
        #yFine = yarrayI(tFine);
	#tFine = np.linspace(-15,70,10000)
	#yFine = yarray1I(tFine)
	#interpolate the data before plotting scatter.
	ax.scatter(xarray1,yarray1,s=1,color = 'blue',label='Donor')
	#ax.scatter(tFine,yFine,s=1,label='Donor') #Interpolated
	ax.scatter(xarray2,yarray2,s=1,color='red',label='Accretor')  #plotting without interpolation
	#ax.scatter(xarray1[0],yarray1[0],s=50,color = 'green',label = 'origin')
	#ax.scatter(xarray1[500],yarray1[500],s=50)
	#ax.plot(xarray1[0],yarray1[0],color='orange',label='initial position')
	ax.plot(0,0,'gx',label='Center of Mass')
	ax.scatter(xarray1[0],yarray1[0],s=50,color='pink',label='pos1')
	ax.scatter(xarray2[0],yarray2[0],s=50,color='pink',label='pos2')
	#ax.scatter(0,0,s=50,color='green',label='Center of Mass')
	#xmin = -3.5*10**(10)
	#xmax = 1.5*10**(10)
	#ymin = -3.5*10**(10)
	#ymax = 1.5*10**(10)
	#ax.set_xlim([xmin,xmax])
	#ax.set_ylim([ymin,ymax])
	ax.set_aspect('equal', 'box')
	dum = ax.legend(loc='upper right')
def plotMotion_compare(xarray1,yarray1,xarray2,yarray2,xarray1A,yarray1A,xarray2A,yarray2A):
	"""Plots the position of the orbits of the two masses in an XRB
	"""
	fig = plt.figure(1)
	try:
		fig.close()
	except:
		dumdum=1
	fig.clf()
	ax = fig.add_subplot(111)
	#yarray1I = interpolate.interp1d(xarray1, yarray1)
	#xarray1I = interpolate.interp1d(yarray1, xarray1)
        #tFine = np.linspace(-15, 70, 10000)
        #yFine = yarrayI(tFine);
	#tFine = np.linspace(-15,70,10000)
	#yFine = yarray1I(tFine)
	#interpolate the data before plotting scatter.
	ax.scatter(xarray1,yarray1,s=1,color = 'red',label='Donor')
	ax.scatter(xarray1A,yarray1A,s=1,color = 'blue',label='Donor1')
	#ax.scatter(tFine,yFine,s=1,label='Donor') #Interpolated
	ax.scatter(xarray2,yarray2,s=1,color='red',label='Accretor')  #plotting without interpolation
	ax.scatter(xarray2A,yarray2A,s=1,color='blue',label='Accretor1')  #plotting without interpolation
	#ax.scatter(xarray1[0],yarray1[0],s=50,color = 'green',label = 'initial position')
	#ax.scatter(xarray1A[0],yarray1A[0],s=50,color = 'green',label = 'initial position2')
	#ax.scatter(xarray1[500],yarray1[500],s=50)
	#ax.plot(xarray1[0],yarray1[0],color='orange',label='initial position')
	ax.plot(0,0,'gx',label='Center of Mass')
	#ax.scatter(xarray1[0],yarray1[0],s=50,color='pink',label='pos1')
	#ax.scatter(xarray2[0],yarray2[0],s=50,color='pink',label='pos2')
	#ax.scatter(0,0,s=50,color='green',label='Center of Mass')
	xmin = -3.5*10**(10)
	xmax = 1.5*10**(10)
	ymin = -3.5*10**(10)
	ymax = 1.5*10**(10)
	#ax.set_xlim([xmin,xmax])
	#ax.set_ylim([ymin,ymax])
	ax.set_aspect('equal', 'box')
	dum = ax.legend(loc='upper right')


def timeDelayCurve(a,time,Period):
	"""calculates a single phase vs time delay curve for given parameters
	"""
	#Time delay calculates light travel time between bodies as a function of orbital phase
	delta_t=np.divide(a,constants.c)
	t_over_P=np.divide(time,Period)

	fig = plt.figure(2)
	try:
		fig.close()
	except:
		dumdum=1
	fig.clf()
	ax = fig.add_subplot(111)
	ax.scatter(delta_t,t_over_P,s=1,color = 'k',alpha = 0.1)
	plt.xlabel(r"$\displaystyle \Delta t$")
	plt.ylabel(r"$\displaystyle \frac{t}{P}$")
def obs_timeDelayCurve(r,theta,phi,i,time,Period):
        #let mass #2 be the donar star and mass #1 be the neutron star.
        A1 = r
        A2 = 1 + np.sin(i)*np.sin(theta-phi)
        A = np.multiply(A1,A2)
        #B1 = r
        #B2 = np.sin(theta-phi)
        #B = np.multiply(B1,B2)
        #delta_tauc = np.add(A,B)
        delta_tauc = A
        delta_tau = np.divide(delta_tauc,constants.c)
        t_over_P = np.divide(time,Period)
        fig = plt.figure(3)
        try:
                fig.close()
        except:
                dumdum=1
        fig.clf()
        ax = fig.add_subplot(111)
        ax.scatter(delta_tau,t_over_P,s=1,color = 'k',alpha = 0.1)
        plt.xlabel(r"$\displaystyle \Delta t$")
        plt.ylabel(r"$\displaystyle \frac{t}{P}$")
def multipleTimeDelayCurve(a,time,Period):
	"""Use when plotting several phase vs time delay curves to get all curves on the same plot
	"""
	delta_t=np.divide(a,constants.c)
	#uncertainty=np.random.normal(0,0.01*delta_t,len(delta_t))
	#dt_u=delta_t+uncertainty

	t_over_P=np.divide(time,Period)
	plt.plot(delta_t,t_over_P,color='k',alpha = 0.05)
	#plt.plot(dt_u,t_over_P,color='k',alpha=0.05)
	plt.xlabel(r"$\displaystyle \Delta t$")
	plt.ylabel(r"$\displaystyle \frac{t}{P}$")
	plt.show()

def gaussianCurve(Radius):
	"""calculate and plot a 'fake' time delay transfer function
	"""
	sigma = np.divide(Radius,constants.c)
	bin=10*sigma
	mu=bin/2
	xrange=np.linspace(0,bin,num=bin)
	noiseGauss = np.random.normal(0,sigma/1000000,np.size(xrange))



	
	fig = plt.figure(3)
	try:
		fig.close()
	except:
		dumdum=1
	fig.clf()
	ax = fig.add_subplot(211)
	

	uncty = np.ones(np.size(xrange))*0.1 # for 0.1 units error

	#gaussian function for xray and covolved optical
	func=np.array(1/(sigma * np.sqrt(2 * np.pi)) *np.exp(-(xrange - mu)**2 / (2 * sigma**2))+noiseGauss*uncty)
	filter = gaussian_filter(func,0.5*sigma)

	#plot
	ax.plot(xrange,func,label='function',color='r')
	plt.ylabel(r"$\displaystyle I_x$")
	dum = ax.legend(loc='upper right')

	ax2 = fig.add_subplot(212)
	ax2.plot(xrange,filter,label='convovled function',color='g')
	plt.ylabel(r"$\displaystyle I_o$")	
	dum2 = ax2.legend(loc='upper right')
go()

