<u>Introduction</u>

Simulation of X-ray/optical light echos in X-ray binary systems with implementation of MCMC recovery of orbital parameters.
The MCMC algorithm takes measurements of (orbital phase, echo delay time, echo delay time 1-sigma error) and outputs a posterior distribution.
Radial velocity measurements can be included as an additional constraint. The radial velocity data can be entered as a single value of (K_em, K_em 1-sigma error) or as a full curve: (phase, velocity, velocity 1-sigma error).
Along with echo delay and radial velocity measurements, further constraints can be included with prior distributions. For instance, in neutron star systems, we apply a uniform prior on the neutron star mass between 1 and 2.5 solar masses (based on physical constraints and the distribution of known neutron star masses).

To get started, open ``delay_model.py`` and run the function: ``timeDelay_3d_full(phase = 0.75, mode = 'plot', Q = 100)``
This will construct a fully 3D model of the Roche lobe of the donor star and color plot the surface with echo time delay, radial velocity, and apparent intensity.
By integrating the apparent intensity along contours of equal time delay, a function of observed intensity against time delay is constructed. The same procedure is used to generate a function of observed intensity against radial velocity. These are plotted in the 2nd plot window.

This fully 3D model is slow. To speed things up, we introduce an approximate model called the "Azimuthally Symmetric Roche Approximation" ASRA-model. A Roche lobe is not azimuthally symmetric; however, the deviation from this symmetry is very slight. We can compute a 2D cross-section of the Roche lobe at some specific azimuthal angle (beta) and rapidly generate a 3D figure by rotating about the central axis. Using a look-up table approach, we build a map of beta against the orbital parameters so that the ASRA-model is generated for a beta that minimizes the deviation from the full 3D model. The ASRA model is accurate to within less than 1% for a very large range of orbital parameters.

To visualize the Roche lobe deviation from azimuthal symmetry, open ``generate_figures.py`` and run ``appendix_2()``. This will plot 2D cross sections at beta = 0 and 90 deg.

