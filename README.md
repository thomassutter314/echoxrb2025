**Introduction**

Simulation of X-ray/optical light echoes in X-ray binary systems with implementation of MCMC recovery of orbital parameters.
The MCMC algorithm takes measurements of (orbital phase, echo delay time, echo delay time 1-sigma error) and outputs a posterior distribution.
Radial velocity measurements can be included as an additional constraint. The radial velocity data can be entered as a single value of (K_em, K_em 1-sigma error) or as a full curve: (phase, velocity, velocity 1-sigma error).
Along with echo delay and radial velocity measurements, further constraints can be included with prior distributions. For instance, in neutron star systems, we apply a uniform prior on the neutron star mass between 1 and 2.5 solar masses (based on physical constraints and the distribution of known neutron star masses).

To get started, open ``delay_model.py`` and run the function: ``timeDelay_3d_full(phase = 0.75, mode = 'plot', Q = 100)``
This will construct a fully 3D model of the Roche lobe of the donor star and color plot the surface with echo time delay, radial velocity, and apparent intensity.
By integrating the apparent intensity along contours of equal time delay, a function of observed intensity against time delay is constructed. The same procedure is used to generate a function of observed intensity against radial velocity. These are plotted in the 2nd plot window.

This fully 3D model is slow. To speed things up, we introduce an approximate model called the "Azimuthally Symmetric Roche Approximation" ASRA-model. A Roche lobe is not azimuthally symmetric; however, the deviation from this symmetry is very slight. We can compute a 2D cross-section of the Roche lobe at some specific azimuthal angle (beta) and rapidly generate a 3D figure by rotating about the central axis. Using a look-up table approach, we build a map of beta against the orbital parameters so that the ASRA-model is generated for a beta that minimizes the deviation from the full 3D model.

To visualize the Roche lobe deviation from azimuthal symmetry, open ``generate_figures.py`` and run ``appendix_2()``. This will plot 2D cross sections at beta = 0 and 90 deg. Several figures from "Orbital parameter estimation from X-ray/optical echo mapping in X-ray binaries" can be reproduced from the functions in this file. See the functions: ``fig1_A``, ``fig1_BCD``, ``fig1_E``, ``fig1_F``, ``fig2``, ``fig3``, ``fig4``, and ``fig5``.

**Parameter names**

m1: mass of compact object (in solar masses)

m2: mass of donor star (in solar masses)

i: inclination (deg)

alpha: disk shielding angle (deg)

**MCMC instructions**

To perform MCMC parameter recovery on data, we need to adjust the ``mcmc_settings.txt`` file. The initial entries can be set to ``walker_number:50``, ``walker_steps:1000``, ``mode:both, Q:25``. This specifies that the fit will consider both radial velocity and echo delay data. The number of panels on the simulated Roche lobe is $\sim Q^2$. We then specify the orbital period for the system under consideration with the line ``period_in:0.787``. This is in units of days.

For an initial run, we can use pseudo data that was generated using the model (with added gaussian noise). We specify the location of the files containing the echo and radial velocity data:

```
echoDelay_dataLoc:echo_pseudo_data.csv
radialVelocity_dataLoc:radial_pseudo_data.csv
```

We then need to specify the guess distributions for the walkers. This will set the initial positions of the walkers.

```
m1_guess:(g,1.45,0.1)
m2_guess:(g,0.7,0.1)
i_guess:(g,44,5)
alpha_guess:(u,1,10)
```

The format here specifies either a Gaussian or uniform distribution. ``(g,1.45,0.1)`` means a Gaussian centered at 1.45 with $\sigma = 0.1$. ``(u,1,10)`` means a uniform distribution between 1 and 10.

The same format is used to set the priors on the orbital parameters:

```
m1_prior:(u,1.2,2.5)
m2_prior:(u,0.01,100)
i_prior:(u,0,90)
alpha_prior:(u,0,18)
```

Finally, we can specify whether the donor is known to eclipse the compact object for an additional constraint. We will ignore this for now: ``eclipse: None``. If we set this variable as either True or False, then a hard constraint associated with the presence or absence of an eclipse will be added.

Save the ``mcmc_settings.txt`` file and open ``mcmc_fitting.py``. Run ``mcmc_obj = MCMC_manager()``. This will automatically begin running a parameter recovery. Results are saved in the ``mcmc_results`` directory. Specifically, the files will be in a subdirectory named as the UNIX time when the run began. The state of the sampler is saved as ``sampler.pickle``.

To see an example of how to pull results from this file and plot, refer to the ``fig3`` function of ``generate_figures.py``.



