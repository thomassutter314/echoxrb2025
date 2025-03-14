#
# echoPriors.py
#

# 2018-05-04 WIC - priors object split off from the main simulator

import numpy as np
import os

import matplotlib.pylab as plt

class Priors(object):

    """Class implementing priors for simulator_2.py . Mostly contains
    a selection of methods for priors, all of which take the form
    method(theta), with the hyper-parameters specified at the class
    level. Arguments when initializing:

    namePrior='uninformative' -- name for the prior function

    figPrior = ''  -- filename for the prior figure.
    plotPrior = True -- plot the prior to png?"""

    def __init__(self, namePrior='uninformative', \
                     figPrior='fig_priors.png', \
                     plotPrior=True, \
                     filParams=''):

        ### Which function are we using?
        self.namePrior = namePrior[:] ### Name of prior method


        ### Set up physical boundaries and default hyperparameters
        self.setupPhysicalBounds()
        self.setupDefaultPars() ### set up default hyperpars

        ### Set up defaults for mixed prior
        self.setupMixedPrior()

        ### filename for input parameters and function, if given
        self.filParams=filParams[:]
        self.loadParams()

        self.selectMethod() ### ensure prior method is set
        
        ### Figure name for the prior
        self.figPrior = figPrior[:]
        self.plotPrior = plotPrior ## control variable

    def evaluate(self, theta=np.array([]) ):

        """Evaluates the prior for a particular set of theta"""
        ### 2018-05-06 - check whether the parameter-set is within
        ### physical bounds. If not, return -np.inf here.
        if not self.parsAreWithinBounds(theta):
            return -np.inf

        return self.methPrior(theta)

    def parsAreWithinBounds(self, theta=np.array([]), nMin=8):

        """Returns True if bounds are within physical bounds, False
        otherwise"""

        ### (Method named so that the call reads logically)

        ### 2018-05-06 WIC - Might be better to return 0, -np.inf?q
        goodval = True
        badval = False

        # Do some input parsing. We might decide later to set the
        # minimum number of parameters as a class-level variable
        # (e.g. if we have a set of seven parameters but only care
        # about the first six).
        if np.size(theta) < nMin:
            return badval

        # count the number of parameters that are outside the physical
        # bounds. Return "badval" if any are nonzero.
        bBad = (theta < self.boundsPhysLo) \
            | (theta > self.boundsPhysHi)

        if np.sum(bBad) > 0:
            return badval

        # If we got here, then the parameters were within the bounds.
        return goodval

    def loadParams(self):

        """Loads parameters from file if given"""

        if len(self.filParams) < 3:
            return

        if not os.access(self.filParams, os.R_OK):
            return

        print("Priors.loadParams INFO: loading priors from %s" \
                  % (self.filParams))

        # This is a little bit painful without just using something
        # more mature like astropy.table or pandas:
        hypers = np.genfromtxt(self.filParams, usecols=(1,2))

        # Convert the angular arguments to radians
        hypers[4] = np.radians(hypers[4])
        hypers[5] = np.radians(hypers[5])
        hypers[7] = np.radians(hypers[7])

        # transpose into hyperparams
        self.hyper = np.transpose(hypers)

        # now we need to read in the function names. This only really
        # has meaning for the mixed prior...
        strNames = np.genfromtxt(self.filParams, usecols=(0), dtype='str')
        self.mixedNames = list(strNames)

        # Finally, read in the name of the function
        with open(self.filParams, 'r') as rObj:
            for sLine in rObj:
                if sLine.find('#') < 0:
                    continue
                if sLine.find('NAME') < 0:
                    continue

                vLine = sLine.strip().split()
                self.namePrior = vLine[-1]

    def selectMethod(self):

        """Selects the method for the prior"""

        try:
            self.methPrior = getattr(self, self.namePrior)
        except:
            self.methPrior = self.uninformative

        ### Consider reorganizing this!
        if self.namePrior.find('ixed') > -1:
            self.findMixedMethods()
        
    def setupPhysicalBounds(self):

        """Sets up physical bounds. These are NOT the same as the
        prior parameters."""
    
        ### 2018-05-06 WIC - **do not** enforce +/- pi limits on the
        ### angles here.
        self.boundsPhysLo = np.array(\
            [0.00,  0.00,    0., 0.,    -np.inf, -np.inf,-np.inf,0 ] )
        self.boundsPhysHi = np.array(\
            [np.inf, np.inf, 1., np.inf, np.inf, np.inf,np.inf, np.inf ] )

    def setupDefaultPars(self):

        """Sets default hyperparameters"""

        ### Written out here to make easier to debug in the future

        boundsLo = np.array([1.35, 0., .001, 0.,      0.0,    -np.pi, -np.inf,0])
        boundsHi = np.array([1.45, 8., .999, 30.45, np.pi/2.,  np.pi, np.inf,np.inf])

        self.hyper = np.vstack(( boundsLo, boundsHi ))

    def setupMixedPrior(self):

        """Sets up the defaults for a mixed prior."""

        if self.namePrior.find('mixed') < 0:
            return

        # we set up the default parameters for bounded flat prior,
        # then update them with non-flat examples
        if np.size(self.hyper) < 7:
            self.setupDefaultPars()

        # Adjust the hyperparameters for defaults.
        self.hyper[0][2] = 0.45
        self.hyper[1][2] = 0.05
        self.hyper[0][3] = 16.3
        self.hyper[1][3] = 0.1

        nMeths = np.shape(self.hyper)[-1]
        self.mixedNames = ['binaryBoundedOne' for i in range(nMeths)]

        ### Let's try some gaussians. Eccentricity and period
        self.mixedNames[2] = 'gaussianOne'
        self.mixedNames[3] = 'gaussianOne'

        self.findMixedMethods()

    def findMixedMethods(self):

        """Resolves the mixed methods"""
        
        ### Now resolve the methods. Don't bother defensive
        ### programming here, if the priors are inconsistent then this
        ### shouldn't work at all.
        self.mixedMeths = []
        for iMeth in range(len(self.mixedNames)):
            thisMeth = getattr(self, self.mixedNames[iMeth])
            self.mixedMeths.append(thisMeth)

    def binaryBoundedOne(self, x=np.array([]), low=0., hi=+1.):

        """Flat bounded prior on a single parameter. 

        Returns the natural log of the prior"""

        yScal = 1.0/(hi - low)
        lnPrior = x * 0. ### + np.log(yScal)
        bOut = (x <= low) | (x > hi)
        lnPrior[bOut] = -np.inf
        
        return lnPrior

    def unifLogOne(self, x=np.array([]), low=0., hi=100.):

        """Uniform prior in log(x)"""
        
        const = 1.0/low**2. - 1.0/hi**2.
        lnPrior = np.log(const) - np.log(x)
        bOut = (x <= low) | (x > hi)
        lnPrior[bOut] = -np.inf

        return lnPrior

    def gaussianOne(self, x=np.array([]), mu=0., sig=1.):

        """Gaussian prior on a single parameter.

        Returns the natural log of the prior"""

        lnConst = -0.5 * np.log(2.0 * np.pi)
        lnVar = -np.log(sig)
        lnBrack = -0.5 * ((x - mu) / sig)**2

        ### safety valve
        ###bBad = np.abs(lnBrack) > 7.
        ###lnBrack[bBad] = -np.inf

        return lnConst + lnVar + lnBrack

    def sampleMixedPrior(self, size=1, scalars=True):

        """Draws samples from the mixed prior"""

        sampleTheta = np.array([])
        for iMeth in range(len(self.mixedNames)):
            thisName = self.mixedNames[iMeth]
            thisMeth = getattr(self, '%sSample' % (thisName))
            
            par1 = self.hyper[0][iMeth]
            par2 = self.hyper[1][iMeth]

            # draw the sample
            thisSample = thisMeth(par1, par2, size)

            # If this is the first parameter, we construct the new
            # vector whatever happens.
            if np.size(sampleTheta) < 1:
                sampleTheta = np.copy(thisSample)
                continue
            
            # Otherwise, what we do depends on whether we're asking
            # for vectors or scalars in the sample.
            if size < 2 and scalars:
                sampleTheta = np.hstack(( sampleTheta, \
                                              np.asscalar(thisSample) ))
            else:
                sampleTheta = np.vstack(( sampleTheta, thisSample ))

        return sampleTheta

    def binaryBoundedOneSample(self, low=0., hi=+1., size=1):

        """Samples from the binary bounded prior"""

        thisSample = np.random.uniform(low=low, high=hi, size=size)
        
        return thisSample

    def unifLogOneSample(self, low=0.01, hi=100., size=1):

        """Samples from the uniform log prior"""

        logSample = np.random.uniform(low=np.log(low), \
                                          high=np.log(hi), \
                                          size=size)

        thisSample = np.exp(logSample)
        return thisSample

    def gaussianOneSample(self, mu=0., sig=1., size=1):

        """Samples from the gaussian prior"""

        thisSample = np.random.normal(mu, sig, size)

        return thisSample

    def setupPlotVariables(self):

        """Sets up a few useful characteristics for plotting."""

        ### Borrowed from Thomas' plot routines
        self.plotLabels = [r'$m_1$', r'$m_2$', r'eccentricity', \
                               r'period (days)', \
                               r'inclination (rad)',r'$\omega$ (rad)',r'$t_0$',r'$\alpha$ (rad)']

        ### Change these to update the plot ranges for each
        ### parameter. 
        angOut = np.pi+0.3
        self.plotLimsLo = [1.0, -1.0, -0.2, -1.0, -angOut, -angOut, -10,0]
        self.plotLimsHi = [2.2, 10.0,  1.2, 35.0,  angOut,  angOut, 10,1.2]

        ### We specify the method for the uniformly-spaced grid. If we
        ### want to make one of these logspace (say) we just change
        ### the method identified in the appropriate place in the
        ### list.
        nMeth = len(self.plotLimsLo)
        self.plotSpacerMethods = [np.linspace for i in range(nMeth)]

        self.plotNfine = 1000 ### number of fine points to use
        self.plotNcols = 3 ### number of columns in the plot

        self.plotNrows = int(np.ceil(nMeth/float(self.plotNcols)) )
        
    def wrapPlotPriors(self):

        """Wrapper that builds and plots the priors, one axis per
        parameter."""
        
        ### Don't plot anything if we don't want to plot!
        if not self.plotPrior:
            return

        ### Set up the plot information here
        self.setupPlotVariables()
        self.adjustPlotLimsMixed()

        self.buildPlotX()
        self.buildPlotY()
        self.showPriors()

    def showPriors(self, figNum=1):

        """Actually builds the figures"""

        if not self.plotPrior: ### Do nothing if we don't want to plot
            return

        if len(self.plotFineX) < 1:
            return

        fig=plt.figure(figNum)
        fig.clf()
        fig.subplots_adjust(wspace=0.3, hspace=0.4)

        ### Set up the number of rows from the number of columns and
        ### the number of parameter-sets we've evaluated

        nSets = len(self.plotFineX)
        self.plotNrows = int(np.ceil(nSets/float(self.plotNcols)) )

        # OK now we loop through and plot
        for iPlot in range(nSets):
            thisAx = fig.add_subplot(self.plotNcols, self.plotNrows, iPlot+1)

            thisX = self.plotFineX[iPlot]
            thisY = self.plotFineY[iPlot]
            thisL = self.plotLabels[iPlot]

            ### Strip off the units to create the Y-axis
            sLabelY = thisL.split('(')[0].strip()

            dum = thisAx.plot(thisX, thisY, 'b-', lw=2)
            thisAx.set_xlabel(thisL, fontsize=10)
            thisAx.set_ylabel(r'Prior(%s)' % (sLabelY), fontsize=10)
            thisAx.tick_params(axis='both', labelsize=10)

            ### give a little more room in the axis range. Clumsy hack
            ### for now...
            yLo = np.min(thisY)
            yHi = np.max(thisY)
            yOff = (yHi-yLo)*0.1
            thisAx.set_ylim(yLo - yOff, yHi+yOff)

            if self.namePrior.find('ixed') > 0:
                if self.mixedNames[iPlot].find('Log') > -1:
                    xLo = self.hyper[0][iPlot] * 0.1
                    thisAx.set_xlim(xLo, np.max(thisX))
                    thisAx.set_xscale('log')

            ### show the grid
            dum = thisAx.grid(which='both')
            
        # save the figure
        if len(self.figPrior) > 3:
            fig.savefig(self.figPrior)

    def buildPlotX(self):

        """Builds the uniformly-spaced parameter values for plotting
        to show the prior"""

        self.plotFineX = []
        for iPlot in range(len(self.plotSpacerMethods)):
            thisMeth = self.plotSpacerMethods[iPlot]
            limLo = self.plotLimsLo[iPlot]
            limHi = self.plotLimsHi[iPlot]
        
            # generate the fine grid for this parameter
            thisUnif = thisMeth(limLo, limHi, num=self.plotNfine, \
                                    endpoint=True)

            # and append onto the list
            self.plotFineX.append(thisUnif)
    
    def buildPlotY(self):

        """Evaluates the prior for each individual parameter, for
        plotting purposes"""
    
        ### When used for real, the calling routine only cares about
        ### the prior applied to all the parameters at once. However
        ### here we want the effect of the prior on each individual
        ### parameter. That means we're going to have to use different
        ### methods for evaluating each prior than we will on the
        ### actual routines. 

        ### Find the plot-version of the prior method we chose
        nameMeth = '%sPlot' % (self.namePrior)
        try:
            methPlot = getattr(self, nameMeth)
        except:
            methPlot = self.uninformativePlot

        # Now do it
        methPlot()

    def initPlotY(self):

        """Initializes the fine-Y plotting"""

        self.plotFineY = [np.array([]) for i in range(len(self.plotFineX))]

    def uninformativePlot(self):

        """As the uninformative prior, but for plotting purposes"""

        self.initPlotY()
        for iPlot in range(len(self.plotFineX)):
            thisX = self.plotFineX[iPlot]

            self.plotFineY[iPlot] = thisX*0. + 1.

    def binaryBoundedPlot(self):

        """As the flat bounded prior, but for plotting purposes"""

        self.initPlotY()
        for iPlot in range(len(self.plotFineX)):
            thisX = self.plotFineX[iPlot]

            ### call binaryBoundedOne instead
            thisLnY = self.binaryBoundedOne(thisX, \
                                              self.hyper[0][iPlot], \
                                              self.hyper[1][iPlot])
            thisY = np.exp(thisLnY)

            #thisY = thisX*0. + 1. ### Good unless otherwise stated.
            #bOut = (thisX <= self.hyper[0][iPlot]) | \
            #    (thisX > self.hyper[1][iPlot])
            #thisY[bOut] = 0.

            self.plotFineY[iPlot] = thisY

    def mixedPlot(self, theta=np.array([])):

        """Mixed prior for plotting. Returns ln(prior)"""

        self.initPlotY()
        for iPlot in range(len(self.plotFineX)):
            thisX = self.plotFineX[iPlot]
        
            # now call the particular method we want
            thisMeth = self.mixedMeths[iPlot]
            thisY = thisMeth(thisX, self.hyper[0][iPlot], \
                                 self.hyper[1][iPlot])

            self.plotFineY[iPlot] = np.exp(thisY)

    def adjustPlotLimsMixed(self, nSig=7., lAvoid=[2]):

        """If mixed prior methods are used, adjust the plot limits for
        the gaussians. Does not adjust any indices in list lAvoid"""

        ### Call this right before plotting.

        ### Do nothing if we're not actually using a mixed
        ### method. (Doing things this way allows us to call this
        ### method every time through without having to put
        ### conditionals in the main program flow).
        if self.namePrior.find('ixed') < 0:
            return
            
        for iName in range(len(self.mixedNames)):
            if self.mixedNames[iName].find('auss') < 0:
                continue
            
            ### Don't do anything if we've asked not to adjust this
            ### parameter
            if iName in lAvoid:
                continue

            thisMu = self.hyper[0][iName]
            thisSi = self.hyper[1][iName]
            
            self.plotLimsLo[iName] = thisMu - nSig*thisSi
            self.plotLimsHi[iName] = thisMu + nSig*thisSi

    ### Methods for the prior evaluation come below.

    def uninformative(self, theta=np.array([])):

        """Uniformative prior. Returns ln(prior)."""

        return 0.

    def binaryBounded(self, theta=np.array([])):

        """Flat prior within bounds. Returns ln(prior)"""
        
        ### Since this is a simple yes/no test, we can forget about
        ### normalizing by area and simply apply the test to all the
        ### variables at once.
        thisPrior = 0.
        bOut = (theta <= self.hyper[0]) | (theta > self.hyper[1])
        if np.sum(bOut) > 0:
            thisPrior = -np.inf
    
        return thisPrior

    def mixed(self, theta=np.array([])):

        """Mixed prior, with a separate (class-level specified) method
        for each parameter"""

        ### This is where the OO-approach helps me here, since we can
        ### draw on other instance-level parameters that the other
        ### methods do not need.
        
        ### Loop through the methods to evaluate

        lnPriors = theta*0.
        for iPar in range(np.size(theta)):
            thisMeth = self.mixedMeths[iPar]
            thisHyp0 = self.hyper[0][iPar]
            thisHyp1 = self.hyper[1][iPar]
            
            ### Some of the ___One methods expect arrays. Rather than
            ### trying to put in conditionals on what kind of
            ### variables we're actually dealing with, instead convert
            ### the parameter into a one-element array, and convert
            ### the output into a float.
            thisPar = np.array([theta[iPar]])
            lnPriors[iPar] = np.float(thisMeth(thisPar, thisHyp0, thisHyp1))

        # if ANY of the priors are -np.inf, return -np.inf
        lnPrior = np.sum(lnPriors)
        
        if np.any(lnPriors < -1e5):
            lnPrior = -np.inf
        
        return lnPrior
