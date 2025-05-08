Simulation of X-ray/optical light echos in X-ray binary systems with implementation of MCMC recovery of orbital parameters.
The MCMC algorithm takes measurements of (orbital phase, echo delay time, echo delay time 1-sigma error) and outputs a posterior distribution.
Radial velocity measurements can be included as an additional constraint. The radial velocity data can be entered as a single value of (K_em, K_em 1-sigma error) or as a full curve: (phase, velocity, velocity 1-sigma error).
Along with echo delay and radial velocity measurements, further constraints can be included with prior distributions. For instance, in neutron star systems, we apply a uniform prior on the neutron star mass between 1 and 2.5 solar masses (based on physical constraints and the distribution of known neutron star masses).

