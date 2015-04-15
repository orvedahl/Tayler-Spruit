#
# Pure Hydro, linearize and find dispersion relation
#
# background: rho=const, P=const, u=0
#
# linearized eqns:
#      P1/P0 = rho1/rho0 + T1/T0
#
#      dP1/dt = -gamma*P0*div(u1) + (gamma-1)*kappa*Lap(T1)
#
#      drho1/dt = -rho0*div(u1)
#
#      rho0*du1/dt = -grad(P1)
#
# R. Orvedahl 4-13-2015

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

    print ("\n\nBegin Linearized Pure Hydro, Const/Static Background\n")
    print ("Using substitutions: ",subs)
    print()

    # set lowest level of all loggers to INFO, i.e. dont follow DEBUG stuff
    root = logging.root
    for h in root.handlers:
        h.setLevel("INFO")

    # name logger using the module name
    logger = logging.getLogger(__name__)

    # Input parameters
    gamma = 1.4
    kappa = 1.e-4
    c_v = 1.
    k_z = 3.13    # vertical wavenumber
    R_in  = 1.
    R_out = 3.

    # background is const
    rho0 = 5.
    T0 = 10.
    p0 = (gamma-1)*c_v*rho0*T0

    height = 2.*numpy.pi/k_z

    # Problem Domain
    r_basis = de.Chebyshev('r', 32, interval=(R_in, R_out), dealias=3/2)
    z_basis = de.Fourier(  'z', 32, interval=(0., height), dealias=3/2)
    th_basis = de.Fourier('th', 32, interval=(0., 2.*numpy.pi), dealias=3/2)
    domain = de.Domain([r_basis, th_basis, z_basis], grid_dtype=numpy.float64)

    # alias the grids
    z = domain.grid( 2, scales=domain.dealias)
    th = domain.grid(1, scales=domain.dealias)
    r = domain.grid( 0, scales=domain.dealias)

    # Equations
    #    incompressible, axisymmetric, Navier-Stokes in Cylindrical
    #    expand the non-constant-coefficients (NCC) to precision of 1.e-8
    TC = de.EVP(domain, variables=['rho', 'p', 'T', 'u', 'v', 'w', 'Tr'],
                ncc_cutoff=1.e-8, eigenvalue='omega')
    TC.parameters['gamma'] = gamma
    TC.parameters['kappa'] = kappa
    TC.parameters['p0'] = p0
    TC.parameters['T0'] = T0
    TC.parameters['rho0'] = rho0

    # multiply equations through by r**2 or r to avoid 1/r**2 & 1/r terms
    if (subs):

        # substitute for dt()
        TC.substitutions['dt(A)'] = '-1.j*omega*A'

        # r*div(U)
        TC.substitutions['r_div(A,B,C)'] = 'A + r*dr(A) + dth(B) + r*dz(C)'

        # r-component of gradient
        TC.substitutions['grad_1(A)'] = 'dr(A)'

        # th-component of gradient
        TC.substitutions['r_grad_2(A)'] = 'r*dth(A)'

        # z-component of gradient
        TC.substitutions['grad_3(A)'] = 'dz(A)'

        # r*r*Laplacian(scalar)
        TC.substitutions['r2_Lap(f, fr)'] = \
                             'r*r*dr(fr) + r*dr(f) + dth(f) + r*r*dz(dz(f))'

        # equations using substituions
        TC.add_equation("p/p0 - rho/rho0 - T/T0 = 0")

        TC.add_equation("r*r*dt(p) + gamma*p0*r*r_div(u,v,w) - " + \
               "(gamma-1)*kappa*r2_Lap(T, Tr) = 0")

        TC.add_equation("r*dt(rho) + rho0*r_div(u,v,w) = 0")

        TC.add_equation("rho0*dt(u) + grad_1(p) = 0")
        TC.add_equation("r*rho0*dt(v) + r_grad_2(p) = 0")
        TC.add_equation("rho0*dt(w) + grad_3(p) = 0")

        TC.add_equation("Tr - dr(T) = 0")

    else:
        print ("\nNon-Substitutions Version is not coded up\n")
        sys.exit(2)

    # Boundary Conditions
    TC.add_bc("left(u) = 0")
    TC.add_bc("left(v) = 0")
    TC.add_bc("left(w) = 0")
    TC.add_bc("right(u) = 0", condition="nz != 0")
    TC.add_bc("right(v) = 0")
    TC.add_bc("right(w) = 0")
    TC.add_bc("integ(p,'r') = 0", condition="nz == 0")

    ###############################################################
    # Force break since the following has not been edited yet...
    ###############################################################
    print ("\nThe rest of the script needs to be changed still\n")
    sys.exit(2)

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
    logger.info('\n\tSimulation Complete\n')

    return

def analyze(subs):

    print ("\nUsing Subs: ",subs)
    print ()

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
    if (subs):
        dir = "./"
    else:
        dir = "Subs-False/"
    data = h5py.File(dir+"scalar_data/scalar_data_s1/scalar_data_s1_p0.h5", "r")
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
        opts, args = getopt.getopt(sys.argv[1:], "rs",
                                   ["subs", "run"])
    except getopt.GetoptError:
        print ("\n\n\tBad Command Line Args\n\n")
        sys.exit(2)

    simulate = False
    subs = True

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
        analyze(subs)


