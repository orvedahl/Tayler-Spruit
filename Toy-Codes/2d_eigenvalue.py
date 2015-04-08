#
# Dedalus code to solve:
#
#               u_xx + u_yy = -lambda*u
#
#               u(x,0) = u(x,b) = u_x(0,y) = u_x(a,y) = 0
#
#     eigenvector solution:
#
#               Phi_nm(x,y) = X_n(x) * Y_m(y)
#               X_n(x) = cos(n*pi*x/a)        n=0,1,2,3,4,...
#               Y_m(x) = sin(m*pi*x/b)        m=1,2,3,4,...
#
#     eigenvalue solution:
#
#               lambda_nm = (n*pi/a)**2 + (m*pi/b)**2
#

import numpy as np
import time
import os
import sys
import getopt
import equations

import logging
logger = logging.getLogger(__name__)

from dedalus.public import *
from dedalus.tools  import post
from dedalus.extras import flow_tools
#from dedalus2.extras.checkpointing import Checkpoint

def usage():
    print ("\nTayler-Spruit Dynamo with Dedalus\n")
    print ("Runtime parameters:")
    print ("\t--nu             velocity diffusion\n")
    print ("\t--kappa          thermal diffusion\n")
    print ("\t--g              gravity\n")
    print ("\t--D              species diffusion\n")
    print ("\t--Tbot           bottom temperature\n")
    print ("\t--Ttop           top temperature\n")
    print ("\t--T0             ignition temperature\n")
    print ("\t--Lx             domain size in x\n")
    print ("\t--Lz             domain size in z\n")
    print ("\t--tstop          simulation stop time\n")
    print ("\t--twall-stop     wall-time limit\n")
    print ("\t--nx             resolution in x\n")
    print ("\t--nz             resolution in z\n")
    print ("\t--dir-str        add string to output directory\n")
#####################################################################
initial_time = time.time()

logger.info("Starting Dedalus script {:s}".format(sys.argv[0]))

# what directory to put output data
data_dir_prefix = "/charybdis/toomre/ryor5023/Projects/Tayler-Spruit/"

# parse command line arguments
Lx = 4.          # set domain
Lz = 4.
nx_tmp = 128     # resolution
nz_tmp = 128
tstop = 20       # simulation stop time
tstop_wall = 100 # max walltime limit in hours
dir_str = ""     # extra string on directory name

# parse command args
try:
    opts, args = getopt.getopt(sys.argv[1:], "", ["Lx=", "Lz=",
                                "tstop=", "twall-stop=", 
                               "nx=", "nz=", "dir-str=", "help"])
except getopt.GetoptError:
    print ("\n\n\tBad Command Line Args\n\n")
    sys.exit(2)

for opt, arg in opts:
    if opt in ("--help"):
        usage()
        sys.exit(2)
    elif opt in ("--Lx"):
        Lx = float(arg)
    elif opt in ("--Lz"):
        Lz = float(arg)
    elif opt in ("--tstop"):
        tstop = float(arg)
    elif opt in ("--twall-stop"):
        tstop_wall = float(arg)
    elif opt in ("--nx"):
        nx_tmp = int(arg)
    elif opt in ("--nz"):
        nz_tmp = int(arg)
    elif opt in ("--dir-str"):
        dir_str = arg

# if dir_str is empty, set to default
if (dir_str == ""):
    dir_str = ""

nx = np.int(nx_tmp)
nz = np.int(nz_tmp)

x_basis = Fourier(nx,   interval=[0., Lx], dealias=2/3)
z_basis = Chebyshev(nz, interval=[0., Lz], dealias=2/3)
domain = Domain([x_basis, z_basis], grid_dtype=np.float64)

# save data in directory named after script
script_name = sys.argv[0].split('.py')[0] # this is path of python executable
script_name = script_name.split("/")[-1]  # this removes any "/" in filename
if (dir_str != ""):
    if (dir_str[0] != "_"):
        dir_str = "_" + dir_str
data_dir = data_dir_prefix + script_name + "_" + str(nx_tmp) + "x" + \
           str(nz_tmp) + dir_str + "/"

if domain.distributor.rank == 0:
  if not os.path.exists('{:s}/'.format(data_dir)):
    os.mkdir('{:s}/'.format(data_dir))

# setup equations & BCs
e_val_prob = equations.eigenvalue_2d(domain)
pde = e_val_prob.set_problem()

logger.info("Nx = {:g}, Nz = {:g}".format(nx, nz))

# set time steppers and solvers
#ts = timesteppers.MCNAB2
#cfl_safety_factor = 0.3
ts = timesteppers.RK443
cfl_safety_factor = 0.1*4

# Build solver
solver = solvers.IVP(pde, domain, ts)

# extract grid info
x = domain.grid(0)
z = domain.grid(1)

# initial conditions
u = solver.state['u']
w = solver.state['w']
T = solver.state['T']
T_z = solver.state['T_z']
c = solver.state['c']
c_z = solver.state['c_z']

solver.evaluator.vars['Lx'] = Lx
solver.evaluator.vars['Lz'] = Lz

z0 = 0.6875*Lz   # old was: 2.75
width  = 0.0125*Lz   # old was: 0.05

T['g'] = z
c['g'] = 0.5*(1.-np.tanh( (z-z0)/width ))

T['g'] += Tbot

T.differentiate(1,T_z)
c.differentiate(1,c_z)

A0 = 1.e-2
T['g'] += A0*np.sin(np.pi*z/Lz)*np.random.randn(*T['g'].shape)

#u['g'] = 0.0
#w['g'] = 0.0

# to print min/max, must first do a global reduce:
global_reduce = flow_tools.GlobalArrayReducer(domain.distributor.comm_world)
min_c = global_reduce.global_min(c['g'])
max_c = global_reduce.global_max(c['g'])
min_T = global_reduce.global_min(T['g'])
max_T = global_reduce.global_max(T['g'])

logger.info("c = {:g} --> {:g}".format(min_c, max_c))
logger.info("T = {:g} --> {:g}".format(min_T, max_T))

# integrate parameters
max_dt = 1.e-2
cfl_cadence = 1
report_cadence = 10

# setup cfl safety stuff
cfl = flow_tools.CFL(solver, initial_dt=max_dt, cadence=cfl_cadence,
                     safety=cfl_safety_factor, max_change=1.5, 
                     min_change=0.5, max_dt=max_dt)
cfl.add_velocities(('u', 'w'))

# stopping criteria
output_time_cadence = 0.05
solver.iteration = 0
solver.stop_iteration= np.inf
solver.stop_sim_time = tstop
solver.stop_wall_time = tstop_wall*3600

logger.info("output cadence = {:g}".format(output_time_cadence))

# things to calculate/report during simulation
flow = flow_tools.GlobalFlowProperty(solver, cadence=1)
flow.add_property("sqrt(u*u+w*w)*Lz/nu", name="Re")
flow.add_property("sqrt(u*u+w*w)*Lz/kappa", name="Pe")
flow.add_property("0.5*(u*u+w*w)", name="KE")

# analysis output
analysis_slice = solver.evaluator.add_file_handler(data_dir+"slices", 
                sim_dt=output_time_cadence, max_writes=20, parallel=False)

analysis_slice.add_task("T", name="T")
analysis_slice.add_task("c", name="c")
#analysis_slice.add_task("T - Integrate(T, dx)/Lx", name="T'")
analysis_slice.add_task("u", name="u")
analysis_slice.add_task("w", name="w")
#analysis_slice.add_task("(dx(w) - dz(u))**2", name="enstrophy")

#-----
# not quite sure what this does (most likely deals with restarts):
do_checkpointing=False
if do_checkpointing:
    checkpoint = Checkpoint(data_dir)
    checkpoint.set_checkpoint(solver, wall_dt=1800)
#-----

# MAIN EVOLUTION LOOP
start_time = time.time()
while solver.ok:

    # get timestep
    dt = cfl.compute_dt()

    # advance solution through dt
    solver.step(dt)
    
    # update lists
    if solver.iteration % report_cadence == 0:

        peak_Re = flow.max("Re")
        avg_Re  = flow.grid_average("Re")
        peak_Pe = flow.max("Pe")
        avg_Pe  = flow.grid_average("Pe")
        KE      = flow.grid_average("KE")

        log_str = 'Iteration: {:5d}, Time: {:8.3e}, dt: {:8.3e},'
        log_string = log_str.format(solver.iteration, solver.sim_time, dt)
        log_str = " Re: {:8.3g}/{:8.3g}, Pe: {:8.3g}/{:8.3g}"
        log_string += log_str.format(peak_Re, avg_Re, peak_Pe, avg_Pe)
        log_string += " Avg KE: {:8.3e}".format(KE)

        logger.info(log_string)
        
end_time = time.time()

# Print statistics
elapsed_time = end_time - start_time
elapsed_sim_time = solver.sim_time
N_iterations = solver.iteration 
logger.info('main loop time: {:e}'.format(elapsed_time))
logger.info('Iterations: {:d}'.format(N_iterations))
logger.info('iter/sec: {:g}'.format(N_iterations/(elapsed_time)))
logger.info('Average timestep: {:e}'.format(elapsed_sim_time / N_iterations))

logger.info('beginning join operation')
if do_checkpointing:
    logger.info(data_dir+'/checkpoint/')
    post.merge_analysis(data_dir+'/checkpoint/')
logger.info(analysis_slice.base_path)
post.merge_analysis(analysis_slice.base_path)

if (domain.distributor.rank==0):

    N_TOTAL_CPU = domain.distributor.comm_world.size
    
    # Print statistics
    print('-' * 40)
    total_time = end_time-initial_time
    main_loop_time = end_time - start_time
    startup_time = start_time-initial_time
    print('  startup time:', startup_time)
    print('main loop time:', main_loop_time)
    print('    total time:', total_time)
    print('Iterations:', solver.iteration)
    print('Average timestep:', solver.sim_time / solver.iteration)
    print('scaling:',
          ' {:d} {:d} {:d} {:d} {:d} {:d}'.format(\
                         N_TOTAL_CPU, 0, N_TOTAL_CPU,nx, 0, nz),
          ' {:8.3g} {:8.3g} {:8.3g} {:8.3g} {:8.3g}'.format(startup_time,
                         main_loop_time, main_loop_time/solver.iteration, 
                         main_loop_time/solver.iteration/(nx*nz), 
                         N_TOTAL_CPU*main_loop_time/solver.iteration/(nx*nz)))
    print('-' * 40)


