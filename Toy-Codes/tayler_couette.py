#
# Axisymmetric Taylor-Couette flow in Dedalus (Cylindrical coords)
#
# TC is characterized by 3 non-dimensional numbers:
#   eta = R1/R2    ratio of inner cylinder radius (R1) to outer radius (R2)
#   mu = Om2/Om1   ratio of outer cylinder rotation rate to inner rate (Om1)
#   Re = Om1*R1*delta/nu   Reynolds number, delta = R2-R1 = gap width
#
# non dimensionalize with:
#   [L] = delta = R2-R1
#   [V] = R1*Om1
#   [M] = rho*delta**3
#
# choose:
#   delta = 1
#   R1*Om1 = 1
#   rhow = 1
#
# R. Orvedahl 4-8-2015

import sys
import getopt
import numpy
import pylab
import time
import h5py
import logging

import dedalus.public as de
from dedalus.core.field import Field
from dedalus.extras import flow_tools


def run(subs):

    # set lowest level of all loggers to INFO, i.e. dont follow DEBUG stuff
    root = logging.root
    for h in root.handlers:
        h.setLevel("INFO")

    # name logger using the module name
    logger = logging.getLogger(__name__)

    # Input params from Barenghi (1991) J. Comp. Phys.
    eta = 1./1.444  # R1/R2
    alpha = 3.13    # vertical wavenumber
    Re = 80.        # Reynolds number in units of R1.*Om1*delta/nu
    mu = 0.         # Om2/Om1

    # Computed quantities
    Omega_in = 1.
    Omega_out = mu*Omega_in
    R_in = eta/(1.-eta)
    R_out = 1./(1.-eta)
    height = 2.*numpy.pi/alpha
    V_l = 1.
    V_r = Omega_out*R_out

    # Problem Domain
    r_basis = de.Chebyshev('r', 32, interval=(R_in, R_out), dealias=3/2)
    z_basis = de.Fourier('z', 32, interval=(0., height), dealias=3/2)
    domain = de.Domain([z_basis, r_basis], grid_dtype=numpy.float64)

    # Equations
    #    incompressible, axisymmetric, Navier-Stokes in Cylindrical
    #    expand the non-constant-coefficients (NCC) to precision of 1.e-8
    TC = de.IVP(domain, variables=['p', 'u', 'v', 'w', 'ur', 'vr', 'wr'],
                ncc_cutoff=1.e-8)
    TC.parameters['nu'] = 1./Re
    TC.parameters['V_l'] = V_l
    TC.parameters['V_r'] = V_r
    mu = TC.parameters['V_r']/TC.parameters['V_l'] * eta

    # multiply equations through by r**2 to avoid 1/r and 1/r**2 terms
    if (subs):
        # setup substitutions
        logger.info("not implemented yet")
        sys.exit(2)

        # define substitutions (can be used in equations & analysis tasks)
        TC.substitutions['KE'] = 'rho*(u*u+w**2)/2'
        TC.substitutions['Lap(u, uz)'] = 'dx(dx(u)) + dz(uz)'

        # equations using substituions
        TC.add_equation("r*ur + u + r*dz(w) = 0")
        TC.add_equation("r*r*dt(u) - r*r*nu*dr(ur) - r*nu*ur - " +
               "r*r*nu*dz(dz(u)) + nu*u + r*r*dr(p) = -r*r*u*ur - " +
               "r*r*w*dz(u) + r*v*v")
        TC.add_equation("r*r*dt(v) - r*r*nu*dr(vr) - r*nu*vr - " +
               "r*r*nu*dz(dz(v)) + nu*v  = -r*r*u*vr - r*r*w*dz(v) - r*u*v")
        TC.add_equation("r*dt(w) - r*nu*dr(wr) - nu*wr - r*nu*dz(dz(w)) + " +
               "r*dz(p) = -r*u*wr - r*w*dz(w)")
        TC.add_equation("ur - dr(u) = 0")
        TC.add_equation("vr - dr(v) = 0")
        TC.add_equation("wr - dr(w) = 0")

    else:
        # equations with no substitutions
        TC.add_equation("r*ur + u + r*dz(w) = 0")
        TC.add_equation("r*r*dt(u) - r*r*nu*dr(ur) - r*nu*ur - " +
               "r*r*nu*dz(dz(u)) + nu*u + r*r*dr(p) = -r*r*u*ur - " +
               "r*r*w*dz(u) + r*v*v")
        TC.add_equation("r*r*dt(v) - r*r*nu*dr(vr) - r*nu*vr - " +
               "r*r*nu*dz(dz(v)) + nu*v  = -r*r*u*vr - r*r*w*dz(v) - r*u*v")
        TC.add_equation("r*dt(w) - r*nu*dr(wr) - nu*wr - r*nu*dz(dz(w)) + " +
               "r*dz(p) = -r*u*wr - r*w*dz(w)")
        TC.add_equation("ur - dr(u) = 0")
        TC.add_equation("vr - dr(v) = 0")
        TC.add_equation("wr - dr(w) = 0")

    # Boundary Conditions
    r = domain.grid(1, scales=domain.dealias)
    z = domain.grid(0, scales=domain.dealias)

    p_analytic = (eta/(1-eta**2))**2 * (-1./(2*r**2*(1-eta)**2) - \
                  2*numpy.log(r) +0.5*r**2 * (1.-eta)**2)
    v_analytic = eta/(1-eta**2) * ((1. - mu)/(r*(1-eta)) - \
                  r * (1.-eta) * (1 - mu/eta**2)) 

    TC.add_bc("left(u) = 0")
    TC.add_bc("left(v) = V_l")
    TC.add_bc("left(w) = 0")
    TC.add_bc("right(u) = 0", condition="nz != 0")
    TC.add_bc("right(v) = V_r")
    TC.add_bc("right(w) = 0")
    TC.add_bc("integ(p,'r') = 0", condition="nz == 0")

    # Timestepper
    dt = max_dt = 1.
    Omega_1 = TC.parameters['V_l']/R_in
    period = 2.*numpy.pi/Omega_1
    logger.info('Period: %f' %(period))

    ts = de.timesteppers.RK443
    IVP = TC.build_solver(ts)
    IVP.stop_sim_time = 15.*period
    IVP.stop_wall_time = numpy.inf
    IVP.stop_iteration = 10000000

    # alias the state variables
    p = IVP.state['p']
    u = IVP.state['u']
    v = IVP.state['v']
    w = IVP.state['w']
    ur = IVP.state['ur']
    vr = IVP.state['vr']
    wr = IVP.state['wr']

    # new field
    phi = Field(domain, name='phi')

    for f in [phi, p, u, v, w, ur, vr, wr]:
        f.set_scales(domain.dealias, keep_data=False)

    v['g'] = v_analytic
    #p['g'] = p_analytic

    v.differentiate(1,vr)

    # incompressible perturbation, arbitrary vorticity
    phi['g'] = 1.e-3*numpy.random.randn(*v['g'].shape)
    phi.differentiate(1,u)
    u['g'] *= -1.*numpy.sin(numpy.pi*(r - R_in))
    phi.differentiate(0,w)
    w['g'] *= numpy.sin(numpy.pi*(r - R_in))
    u.differentiate(1,ur)
    w.differentiate(1,wr)

    # Time step size
    CFL = flow_tools.CFL(IVP, initial_dt=1.e-3, cadence=5, safety=0.3,
                         max_change=1.5, min_change=0.5)
    CFL.add_velocities(('u', 'w'))

    # Analysis
    # integrated energy every 10 steps
    analysis1 = IVP.evaluator.add_file_handler("scalar_data", iter=10)
    analysis1.add_task("integ(0.5 * (u*u + v*v + w*w))", name="total KE")
    analysis1.add_task("integ(0.5 * (u*u + w*w))", name="meridional KE")
    analysis1.add_task("integ((u*u)**0.5)", name="u_rms")
    analysis1.add_task("integ((w*w)**0.5)", name="w_rms")

    # Snapshots every half inner rotation period
    analysis2 = IVP.evaluator.add_file_handler('snapshots', sim_dt=0.5*period,
                                               max_size=2**30)
    analysis2.add_system(IVP.state, layout='g')

    # Radial profiles every 100 steps
    analysis3 = IVP.evaluator.add_file_handler("radial_profiles", iter=100)
    analysis3.add_task("integ(r*v, 'z')", name="Angular Momentum")

    # MAIN LOOP
    dt = CFL.compute_dt()
    start_time = time.time()

    while IVP.ok:
        IVP.step(dt)
        if (IVP.iteration % 10 == 0):
            logger.info('Iteration: %i, Time: %e, dt: %e' % \
                                          (IVP.iteration, IVP.sim_time, dt))
        dt = CFL.compute_dt()

    end_time = time.time()

    logger.info('Total time: %f' % (end_time-start_time))
    logger.info('Iterations: %i' % (IVP.iteration))
    logger.info('Average timestep: %f' %(IVP.sim_time/IVP.iteration))
    logger.info('Period: %f' %(period))

    # plot the last snapshot of the background v\theta-hat with arrows 
    # representing the meridional flow
    #pylab.figsize(12,8)
    pylab.pcolormesh((r[0]*numpy.ones_like(z)).T, (z*numpy.ones_like(r)).T, 
                     v['g'].T, cmap='PuOr')
    pylab.quiver((r[0]*numpy.ones_like(z)).T, (z*numpy.ones_like(r)).T, 
                 u['g'].T, w['g'].T, width=0.005)
    pylab.axis('image')
    pylab.xlabel('r', fontsize=18)
    pylab.ylabel('z', fontsize=18)
    pylab.show()

    logger.info('\n\tSimulation Complete\n')


def analyze():

    # hard coded --- pretty ugly...
    period = 14.151318
    Re = 80.

    # extract time series data
    def get_timeseries(data, field):
        data_1d = []
        time = data['scales/sim_time'][:]
        data_out = data['tasks/%s'%field][:,0,0]
        return time, data_out

    # read data from HDF5 file
    data = h5py.File("scalar_data/scalar_data_s1/scalar_data_s1_p0.h5", "r")
    t, ke   = get_timeseries(data, 'total KE')
    t, kem  = get_timeseries(data, 'meridional KE')
    t, urms = get_timeseries(data, 'u_rms')
    t, wrms = get_timeseries(data, 'w_rms')

    # to compare to Barenghi (1991), scale results because the 
    # non-dimensionalizations are different
    t_window = (t/period > 2) & (t/period < 14)

    gamma_w, log_w0 = numpy.polyfit(t[t_window], numpy.log(wrms[t_window]),1)

    gamma_w_scaled = gamma_w*Re
    gamma_barenghi = 0.430108693
    rel_error_barenghi = (gamma_barenghi - gamma_w_scaled)/gamma_barenghi

    print ("gamma_w = %10.8f" % (gamma_w_scaled)) # expect ~ 0.37201041
    print ("relative error = %10.8f" % (rel_error_barenghi)) # ~1.35078136e-1

    # plot RMS w and compare to fitted exponential
    fig = pylab.figure()
    ax = fig.add_subplot(111)
    ax.semilogy(t/period, wrms, 'ko', label=r'$w_{rms}$', ms=10)
    ax.semilogy(t/period, numpy.exp(log_w0)*numpy.exp(gamma_w*t), 'k-.',
                label=r'$\gamma_w = %f$' % gamma_w)
    ax.legend(loc='upper right', fontsize=18).draw_frame(False)
    ax.set_xlabel(r"$t/t_{ic}$",fontsize=18)
    ax.set_ylabel(r"$w_{rms}$",fontsize=18)

    pylab.show()


if __name__ == "__main__":

    try:
        opts, args = getopt.getopt(sys.argv[1:], "rs", ["subs", "run"])
    except getopt.GetoptError:
        print ("\n\n\tBad Command Line Args\n\n")
        sys.exit(2)

    simulate = False
    subs = False

    for opt, arg in opts:

        if opt in ("-r", "--run"):
           simulate = True

        elif opt in ("-s", "--subs"):
           subs = True

    # run the simulation
    if (simulate):
        run(subs)

    # do the analysis
    else:
        analyze()


