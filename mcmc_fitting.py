import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import scipy.optimize as opt
import pickle
import time
import os

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

def genPseudoData(echo_phases, echo_noise, rv_noise, \
                    m1_in=1.4, m2_in=0.7, inclination_in=44., disk_angle_in=5, period_in=.787):
                        
    echo_noises = echo_noise + np.zeros(len(echo_phases))

    period = period_in * constants.day
    delays = np.zeros(len(echo_phases))
    for i in range(len(echo_phases)):
        delays[i] = delay_model.timeDelay_3d_full(echo_phases[i],m1_in=m1_in,m2_in=m2_in,inclination_in=inclination_in,disk_angle_in=disk_angle_in,period_in=period_in,mode = 'return_delay') +\
                    np.random.normal(scale=echo_noises[i])
    
    np.savetxt('echo_pseudo_data.csv',np.array([echo_phases, delays, echo_noises]).transpose(),delimiter=',',header='phase, delay (s), error (s)')

    k_em = delay_model.timeDelay_3d_full(0.75,m1_in=m1_in,m2_in=m2_in,inclination_in=inclination_in,disk_angle_in=disk_angle_in,period_in=period_in,mode = 'return_rv') +\
                    np.random.normal(scale=rv_noise)
                    
    np.savetxt('radial_pseudo_data.csv',np.array([0.75, k_em, rv_noise]),delimiter=',',header='phase, radial velocity (km/s), error (km/s)')
    
class Priors():
    def __init__(self, dist_params):
        self.dist_type = dist_params[0]
        self.param1 = dist_params[1] # g: loc, u: min
        self.param2 = dist_params[2] # g: scale, u: max
        
    def evaluate(self, x, log = True):
        if self.dist_type == 'g':
            ln_y = -0.5*np.log(2*np.pi*self.param2**2) - 0.5*((x - self.param1)/self.param2)**2
            
        if self.dist_type == 'u':
            ln_y = np.piecewise(x, [(x >= self.param1) & (x <= self.param2)], [-1*np.log(self.param2-self.param1), -np.inf])
        
        if log:
            return ln_y
        else:
            return np.exp(ln_y)

class MCMC_manager():
    def __init__(self, save = True):
        # modes are "both", "echo", and "rv"
        self.asra_lt = delay_model.ASRA_LT_model()
        
        self.run_id = str(int(time.time()))
        self.save_dir = f'mcmc_results//{self.run_id}'
        self.save = save
        
        if self.save:
            # Create the directory for the saved information
            os.makedirs(self.save_dir)
        
        self.settingsDict = {}
        with open('mcmc_settings.txt','r') as ssf:
            for line in ssf:
                line = line.replace(" ", "") # remove all spaces from string
                line = line[:line.find("#")]
                if len(line) > 0:
                    self.settingsDict[line[:line.find(":")]] = line[line.find(":")+1:].replace('\n','')
                
        self.Q = int(self.settingsDict['Q'])
        self.walker_number = int(self.settingsDict['walker_number'])
        self.walker_steps = int(self.settingsDict['walker_steps'])
        self.period_in = float(self.settingsDict['period_in'])
        echoDelay_dataLoc = self.settingsDict['echoDelay_dataLoc']
        radialVelocity_dataLoc = self.settingsDict['radialVelocity_dataLoc']
        self.mode = self.settingsDict['mode']
        
        print(f'MCMC run ID = {self.run_id}')
        print(f'mode = {self.mode}')
        print(f'Saving Results = {self.save} \n')
        print('MCMC run parameters:')
        print('______________________')
        print(f'Roche Triangle Number = {self.Q**2}')
        print(f'Walker Number = {self.walker_number}')
        print(f'Step Number = {self.walker_steps}')
        print(f'Orbital Period = {self.period_in} days')
        print(f'echoDelay_dataLoc = {echoDelay_dataLoc}')
        print(f'radialVelocity_dataLoc = {radialVelocity_dataLoc} \n')
        
        # Labels for the fit parameters
        self.var_labels = [r'$m_1$ ($M_\odot$)',r'$m_2$ ($M_\odot$)',r'$i$ ($^{\circ}$)',r'$\alpha$ ($^{\circ}$)']
        
        # echoDelay_dataLoc and radialVelocity_dataLoc contain the paths for this data
        # if either path is given as None, then we aslsume that data is not present on this run
        # The organization of the data is column 0 gives observation times, column 1 gives observed value, and column 2 gives error
        # Comma delimiter and 1st row is headers
        
        if echoDelay_dataLoc == 'none':
            self.echoDelay_data = None
        else:
            self.echoDelay_data = np.loadtxt(echoDelay_dataLoc,delimiter=',',skiprows=1)
            
        if radialVelocity_dataLoc == 'none':
            self.radialVelocity_data = None
        else:
            self.radialVelocity_data = np.loadtxt(radialVelocity_dataLoc,delimiter=',',skiprows=1)
        
        # Load the initial guess distributions
        self.m1_guess_params = self.interpret_dist_string(self.settingsDict['m1_guess'])
        self.m2_guess_params = self.interpret_dist_string(self.settingsDict['m2_guess'])
        self.i_guess_params = self.interpret_dist_string(self.settingsDict['i_guess'])
        self.alpha_guess_params = self.interpret_dist_string(self.settingsDict['alpha_guess'])
        
        # Generate samples from guess distributions
        m1_guess_sample = self.sample_dist(sample_N = self.walker_number, dist_params = self.m1_guess_params)
        m2_guess_sample = self.sample_dist(sample_N = self.walker_number, dist_params = self.m2_guess_params)
        i_guess_sample = self.sample_dist(sample_N = self.walker_number, dist_params = self.i_guess_params)
        alpha_guess_sample = self.sample_dist(sample_N = self.walker_number, dist_params = self.alpha_guess_params)
        
        # Combine in a single higher dimensional guess distribution
        self.guess_sample = np.array([m1_guess_sample,m2_guess_sample,i_guess_sample,alpha_guess_sample]).transpose()
        
        # Save guess sample
        if self.save:
            np.save(f'{self.save_dir}//guess_sample',self.guess_sample)
        
        # Load the prior parameters
        self.m1_prior_params = self.interpret_dist_string(self.settingsDict['m1_prior'])
        self.m2_prior_params = self.interpret_dist_string(self.settingsDict['m2_prior'])
        self.i_prior_params = self.interpret_dist_string(self.settingsDict['i_prior'])
        self.alpha_prior_params = self.interpret_dist_string(self.settingsDict['alpha_prior'])
        
        # Generate the priors
        self.m1_prior = Priors(self.m1_prior_params)
        self.m2_prior = Priors(self.m2_prior_params)
        self.i_prior = Priors(self.i_prior_params)
        self.alpha_prior = Priors(self.alpha_prior_params)
        
        self.plot_priors_and_sample(self.guess_sample, fname = 'initial_state.png')
        plt.show()
        
        self.execute_mcmc()
        
    def execute_mcmc(self, save = True):
        self.metadataDict = {} # A dictionary for storing meta data on this MCMC run
        start_time = time.time()
        self.metadataDict['start_time'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))
        self.metadataDict['mode'] = self.mode
        print('______________________')
        print('Starting MCMC Run')
        
        sampler = emcee.EnsembleSampler(self.walker_number, 4, self.lnprob, args=(self.echoDelay_data[:,0], self.echoDelay_data[:,1], self.echoDelay_data[:,2], self.radialVelocity_data[0], self.radialVelocity_data[1], self.radialVelocity_data[2]))
        sampler.run_mcmc(self.guess_sample, self.walker_steps, progress=True)
        
        taus = sampler.get_autocorr_time(tol = 0)
        print(f'Autocorrelation Time: {taus} \n (All these values should be at least 100 times smaller than step number)')
        
        finish_time = time.time()
        self.metadataDict['finish_time'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(finish_time))
        self.metadataDict['run_duration'] = f'{round(finish_time - start_time,1)} s'
        self.metadataDict['mean_acceptance_fraction'] = f'{np.mean(sampler.acceptance_fraction)}'
        self.metadataDict['auto_corr_time'] = f'{taus}'
        
        # Save the sampler, some metadata, and the input data
        if self.save:
            # sampler
            with open(f'{self.save_dir}//sampler.pickle', 'wb') as outp:
                pickle.dump(sampler, outp)
                
            # metadata           
            with open(f'{self.save_dir}//metadata.txt','w') as ssf:
                # Write the input settings
                ssf.write('Input Settings \n \n')
                for key, value in self.settingsDict.items():
                    ssf.write('%s:%s\n' % (key, value))
                    
                ssf.write('\nMCMC Run Metadata \n \n')
                
                for key, value in self.metadataDict.items():
                    ssf.write('%s:%s\n' % (key, value))
                    
            # input echo and radial velocity data
            if type(self.echoDelay_data) != type(None):
                np.savetxt(f'{self.save_dir}//echo_delay_data.csv',self.echoDelay_data,delimiter=',',header='phase, delay (s), error (s)')
            if type(self.radialVelocity_data) != type(None):
                np.savetxt(f'{self.save_dir}//radial_velocity_data.csv',self.radialVelocity_data,delimiter=',',header='phase, velocity (km/s), error (s)')
                

        samples = sampler.get_chain()
        flat_samples = sampler.get_chain(discard=int(5*max(taus)), thin=int(0.5*max(taus)), flat=True)

        fig = corner.corner(flat_samples, labels=self.var_labels, quantiles=[0.16, 0.5, 0.84],\
                            show_titles=True, title_kwargs={"fontsize": 12}, smooth = True)
        if self.save:
            plt.savefig(f'{self.save_dir}//posterior_corner.png')
            
        fig, axs = plt.subplots(4)
        for i in range(4):
            axs[i].plot(samples[:,:,i], alpha = 0.3)
            axs[i].set_ylabel(self.var_labels[i])
            axs[0].yaxis.set_label_coords(-0.1, 0.5)
        axs[-1].set_xlabel('Step Number')
        if self.save:
            plt.savefig(f'{self.save_dir}//walker_paths.png')
        
        inds = np.random.randint(len(flat_samples), size = min([100,len(flat_samples)]))
        random_sample = flat_samples[inds]
        self.plot_priors_and_sample(random_sample,'final_state.png')

        plt.show()
                    
    # Make a fit function with only the fit variables as inputs
    def model(self, x, m1_in, m2_in, inclination_in, disk_angle_in):
        tt = np.zeros(len(x)) # delay times
        vv = np.zeros(len(x)) # radial velocities
        tt, vv = self.asra_lt.evaluate_array(x, m1_in=m1_in,m2_in=m2_in,inclination_in=inclination_in,disk_angle_in=disk_angle_in,period_in=self.period_in,Q=self.Q)
        return tt, vv
    
    # Log probability function for MCMC
    def lnprob(self, theta, x_echo, y_echo, y_echo_err, x_rv, y_rv, y_rv_err):
        m1F, m2F, iF, alphaF  = theta #model parameters, N = 4
        
        # Make sure the parameters are within the bounds of reason, if not, then return -infinity (This is separate from the prior)
        if not self.check_physical_param_boundry(m1F, m2F, iF, alphaF):
            return -np.inf
        
        echo_delay, _ = self.model(x_echo,m1_in=m1F,m2_in=m2F,inclination_in=iF,disk_angle_in=alphaF)
        _, rv_em = self.model([0.68,0.715,0.75],m1_in=m1F,m2_in=m2F,inclination_in=iF,disk_angle_in=alphaF)
        
        rv_em = np.max(rv_em)
        
        # ~ print(rv_em)
        
        if self.mode == 'both' or self.mode == 'echo':
            ln_like_echo = np.sum(-0.5*((y_echo-echo_delay)/y_echo_err)**2 - np.log(y_echo_err) - 0.5*np.log(2*np.pi))
            
        if self.mode == 'both' or self.mode == 'rv':
            ln_like_rv = -0.5*((y_rv-rv_em)/y_rv_err)**2 - np.log(y_rv_err) - 0.5*np.log(2*np.pi)
        
        ln_prior = self.m1_prior.evaluate(m1F) + self.m2_prior.evaluate(m2F) + self.i_prior.evaluate(iF) + self.alpha_prior.evaluate(alphaF)
        
        if self.mode == 'both':
            return ln_like_echo + ln_like_rv + ln_prior
        if self.mode == 'echo':
            return ln_like_echo + ln_prior
        if self.mode == 'rv':
            return ln_like_rv + ln_prior
            
        
    # Enterprets the shorthand for distributions of the form (g,loc,scale) and (u,min,max)
    def interpret_dist_string(self, string):
        i1 = string.find('(')
        i2 = string[i1+1:].find(',') + i1 + 1
        i3 = string[i2+1:].find(',') + i2 + 1
        i4 = string[i3+1:].find(')') + i3 + 1
        
        dist_type = string[i1+1:i2]
        p1 = float(string[i2+1:i3])
        p2 = float(string[i3+1:i4])
        
        return (dist_type,p1,p2)
    
    # Generates a sample of sample_N elements from the distribution described by dist_params    
    def sample_dist(self, sample_N, dist_params):
        if dist_params[0] == 'g':
            return np.random.normal(loc = dist_params[1], scale = dist_params[2], size = sample_N)
        if dist_params[0] == 'u':
            return np.random.uniform(low = dist_params[1], high = dist_params[2], size = sample_N)
    
    # Plot the priors
    def plot_priors_and_sample(self, sample, fname = 'sample.png'):
        f = 1
        fig, axs = plt.subplots(3,2, figsize=(10, 10))
        
        m1, m2 = min(sample[:,0]), max(sample[:,0])
        w = m2 - m1
        tt = np.linspace(m1-f*w,m2+f*w,500)
        yy = self.m1_prior.evaluate(tt, log = False)
        axs[0,0].plot(tt,yy,c='k')
        axs[0,0].hist(sample[:,0],density=True,color='green')
        axs[0,0].set_xlabel(self.var_labels[0])
        
        m1, m2 = min(sample[:,1]), max(sample[:,1])
        w = m2 - m1
        tt = np.linspace(m1-f*w,m2+f*w,500)
        yy = self.m2_prior.evaluate(tt, log = False)
        axs[1,0].plot(tt,yy,c='k')
        axs[1,0].hist(sample[:,1],density=True,color='green')
        axs[1,0].set_xlabel(self.var_labels[1])
        
        m1, m2 = min(sample[:,2]), max(sample[:,2])
        w = m2 - m1
        tt = np.linspace(m1-f*w,m2+f*w,500)
        yy = self.i_prior.evaluate(tt, log = False)
        axs[0,1].plot(tt,yy,c='k')
        axs[0,1].hist(sample[:,2],density=True,color='green')
        axs[0,1].set_xlabel(self.var_labels[2])
        
        m1, m2 = min(sample[:,3]), max(sample[:,3])
        w = m2 - m1
        tt = np.linspace(m1-f*w,m2+f*w,500)
        yy = self.alpha_prior.evaluate(tt, log = False)
        axs[1,1].plot(tt,yy,c='k')
        axs[1,1].hist(sample[:,3],density=True,color='green')
        axs[1,1].set_xlabel(self.var_labels[3])
        
        # Error bar scatter plot of data
        axs[2,0].errorbar(self.echoDelay_data[:,0],self.echoDelay_data[:,1],self.echoDelay_data[:,2], fmt='o', capsize=3, color = 'black')        
        # Plot the initial guess curves
        pp = np.linspace(0,1,50)
        VV = []
        for guess in sample:
            tt, vv = self.model(pp,*guess)
            VV.append(max(vv))
            axs[2,0].plot(pp,tt,'g-', alpha = 0.15)
            # ~ axs[2,1].plot(pp,vv,'g-', alpha = 0.15)
            
        axs[2,0].set_xlabel('Orbital Phase')
        axs[2,0].set_ylabel('Echo Delay (s)')
        
        axs[2,1].hist(VV,density=True,color='green')
        axs[2,1].axvline(x=self.radialVelocity_data[1],linestyle='--',c='black')
        axs[2,1].axvspan(self.radialVelocity_data[1]-self.radialVelocity_data[2],self.radialVelocity_data[1]+self.radialVelocity_data[2],alpha=0.5,color='black')
        axs[2,1].set_xlabel(r'$K_{em}$ (km/s)')
        
        
        
        if self.save:
            plt.savefig(f'{self.save_dir}//{fname}')
        
    # Function checks if the inputs are within what is physically allowed
    def check_physical_param_boundry(self, m1, m2, i, alpha):
        if (m1 < 0) or (m2 < 0) or (i < 0) or (i > 90) or (alpha < 0) or (alpha > 90):
            return False
        else:
            return True

def load_sampler():
    
    echo = '1712124156'
    rv = '1712125465'
    both = '1712128213'
    
    alpha = 0.5
    s = 3
    
    with open(f'mcmc_results//{rv}//sampler.pickle', 'rb') as inp:
        sampler = pickle.load(inp)
    samples = sampler.get_chain(discard=10, thin=10, flat=True)
    plt.scatter(samples[:,2],samples[:,1]/samples[:,0], s = s, alpha = alpha)
    
    with open(f'mcmc_results//{echo}//sampler.pickle', 'rb') as inp:
        sampler = pickle.load(inp)
    samples = sampler.get_chain(discard=10, thin=10, flat=True)
    plt.scatter(samples[:,2],samples[:,1]/samples[:,0], s = s, alpha = alpha)
    
    with open(f'mcmc_results//{both}//sampler.pickle', 'rb') as inp:
        sampler = pickle.load(inp)
    samples = sampler.get_chain(discard=10, thin=10, flat=True)
    plt.scatter(samples[:,2],samples[:,1]/samples[:,0], s = s, alpha = alpha)
    
    
    plt.show()
    
    
if __name__ == '__main__':
    #load_sampler()
    # ~ genPseudoData(np.array([0.2,0.4,0.5,0.6,0.8]),0.4,5)
    mcmc_obj = MCMC_manager()



