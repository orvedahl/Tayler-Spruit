import numpy as np
from mpi4py import MPI

from dedalus import public as de

#import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__.split('.')[-1])

class problem_domain:
    def __init__(self, n1=256, n2=256, n3=256, L1=30, L2=30, L3=10,
                  geometry="cylinder", magnetic=False, 
                  const_thermo=True, Cv=1.5, gamma=1.4):
        
        logger.info("set domain...")

        self.geometry = geometry

        if (self.geometry == "cylinder"):
            1_basis = de.Fourier(  'r', nx, interval=[0., L1], dealias=3/2)
            2_basis = de.Fourier(  't', nx, interval=[0., L2], dealias=3/2)
            3_basis = de.Chebyshev('z', nz, interval=[0., L3], dealias=3/2)
            self.domain = de.Domain([1_basis, 3_basis], grid_dtype=np.float64)
        elif (self.geometry == "cartesian"):
            1_basis = de.Fourier(  'x', nx, interval=[0., L1], dealias=3/2)
            2_basis = de.Fourier(  'y', nx, interval=[0., L2], dealias=3/2)
            3_basis = de.Chebyshev('z', nz, interval=[0., L3], dealias=3/2)
            self.domain = de.Domain([1_basis, 3_basis], grid_dtype=np.float64)
        elif (self.geometry == "sphere"):
            1_basis = de.Fourier(  'r', nx, interval=[0., L1], dealias=3/2)
            2_basis = de.Fourier(  't', nx, interval=[0., L2], dealias=3/2)
            3_basis = de.Chebyshev('p', nz, interval=[0., L3], dealias=3/2)
            self.domain = de.Domain([1_basis, 3_basis], grid_dtype=np.float64)
        else:
            logger.info("ERROR: Unrecognized Geometry: {0}".format(geometry)

        self.magnetic = magnetic

        self.const_thermo = const_thermo
        self._set_thermodynamics(Cv, gamma)


    def _set_thermodynamics(self, Cv, gamma):

        logger.info("set thermodynamics...")

        if (self.const_thermo):
            self.gamma = gamma
            self.Cv = Cv
        else:
            # get global size of Lx
            self.x = self.domain.grid(0)
            self.Lx = self.domain.bases[0].interval[1] - \
                      self.domain.bases[0].interval[0]
            self.nx = self.domain.bases[0].coeff_size

            # set up gamma and Cv profiles
            self.gamma = gamma
            self.Cv = Cv


    def _set_background(self):
        
        logger.info("set background state...")

        # set gravity
        self.g = 9.8

        rho0 = 1.
        T0 = 1.
        P0 = 1.
        u0 = 1.
        B0 = 1.

        if (self.magnetic):
            return rho0, T0, P0, u0, B0
        else:
            return rho0, T0, P0, u0


    def _set_parameters(self):
        self.problem.parameters['gamma'] = self.gamma
        self.problem.parameters['Cv'] = self.Cv

        self.problem.parameters['g']  = self.g

        self.problem.parameters['L1'] = self.L1
        self.problem.parameters['L2'] = self.L2
        self.problem.parameters['L3'] = self.L3

        
    def get_problem(self):
        return self.problem


class FC_Static(problem_domain):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def set_IVP_problem(self, kappa, gamma):

        self.problem = de.IVP(self.domain,
                      variables=['u','u_z','w','w_z','s', 'Q_z', 'pomega'],
                      cutoff=1e-10)

        #self._set_thermodynamics(self.Cv, self.gamma) # already handled???
        self._set_parameters()
        self._set_subs()

        # setup the equations
        self.problem.add_equation("P1 - P0/rho0*rho1 - P0/T0*T1 = 0")
        self.problem.add_equation("dt(P1) + gamma*P0*(dx(u) + dy(v)) - (gamma-1)*kappa*(dx(dx(u)) + dy(dy(v))) = 0")
        self.problem.add_equation("dt(rho1) + rho0*(dx(u) + dy(v)) = 0")
        self.problem.add_equation("dt(u) + dx(P1)/rho0")
        self.problem.add_equation("dt(v) + dy(P1)/rho0")

    def set_eigenvalue_problem(self, ):

        self.problem = de.IVP(self.domain, variables=['u','u_z','w','w_z','s', 'Q_z', 'pomega'], cutoff=1e-6)

        self._set_diffusivity(Rayleigh, Prandtl)
        self._set_parameters()

        self.problem.add_equation("dz(w) - w_z = 0")
        self.problem.add_equation("dz(u) - u_z = 0")
        self.problem.add_equation("T0*dz(s) + Q_z = 0")


    def set_BC(self, fixed_flux=False):

        self.problem.add_bc( "left(s) = 0")
        self.problem.add_bc("right(s) = 0")
        self.problem.add_bc( "left(u) = 0")
        self.problem.add_bc("right(u) = 0")
        self.problem.add_bc( "left(w) = 0", condition="nx != 0")
        self.problem.add_bc( "left(pomega) = 0", condition="nx == 0")
        self.problem.add_bc("right(w) = 0")


    def _set_subs(self):
        self.problem.substitutions['rho_full'] = 'rho0*exp(ln_rho1)'

        self.problem.substitutions['KE'] = 'rho_full*(u**2+w**2)/2'
        self.problem.substitutions['PE'] = 'rho_full*phi'
        self.problem.substitutions['PE_fluc'] = 'rho_fluc*phi'
        self.problem.substitutions['IE'] = 'rho_full*Cv*(T1+T0)'
        self.problem.substitutions['IE_fluc'] = 'rho_full*Cv*T1+rho_fluc*T0'
        self.problem.substitutions['P'] = 'rho_full*(T1+T0)'
        self.problem.substitutions['P_fluc'] = 'rho_full*T1+rho_fluc*T0'
        self.problem.substitutions['h'] = 'IE + P'
        self.problem.substitutions['h_fluc'] = 'IE_fluc + P_fluc'
        self.problem.substitutions['u_rms'] = 'sqrt(u*u)'
        self.problem.substitutions['w_rms'] = 'sqrt(w*w)'
        self.problem.substitutions['Re_rms'] = 'sqrt(u**2+w**2)*Lz/nu'
        self.problem.substitutions['Pe_rms'] = 'sqrt(u**2+w**2)*Lz/chi'

        # analysis operators
        self.problem.substitutions['plane_avg(A)'] = 'integ(A, "x")/Lx'
        self.problem.substitutions['vol_avg(A)']   = 'integ(A)/Lx/Lz'

    def initialize_output(self, solver, data_dir, **kwargs):
        analysis_tasks = []
        self.analysis_tasks = analysis_tasks
        
        analysis_slice = solver.evaluator.add_file_handler(data_dir+"slices", max_writes=20, parallel=False, **kwargs)
        analysis_slice.add_task("s", name="s")
        analysis_slice.add_task("s - plane_avg(s)", name="s'")
        analysis_slice.add_task("u", name="u")
        analysis_slice.add_task("w", name="w")
        analysis_slice.add_task("(dx(w) - dz(u))**2", name="enstrophy")
        analysis_tasks.append(analysis_slice)

        analysis_profile = solver.evaluator.add_file_handler(data_dir+"profiles", max_writes=20, parallel=False, **kwargs)
        analysis_profile.add_task("plane_avg(KE)", name="KE")
        analysis_profile.add_task("plane_avg(PE)", name="PE")
        analysis_profile.add_task("plane_avg(IE)", name="IE")
        analysis_profile.add_task("plane_avg(PE_fluc)", name="PE_fluc")
        analysis_profile.add_task("plane_avg(IE_fluc)", name="IE_fluc")
        analysis_profile.add_task("plane_avg(KE + PE + IE)", name="TE")
        analysis_profile.add_task("plane_avg(KE + PE_fluc + IE_fluc)", name="TE_fluc")
        analysis_profile.add_task("plane_avg(w*(KE))", name="KE_flux_z")
        analysis_profile.add_task("plane_avg(w*(PE))", name="PE_flux_z")
        analysis_profile.add_task("plane_avg(w*(IE))", name="IE_flux_z")
        analysis_profile.add_task("plane_avg(w*(P))",  name="P_flux_z")
        analysis_profile.add_task("plane_avg(w*(h))",  name="enthalpy_flux_z")
        analysis_profile.add_task("plane_avg(u_rms)", name="u_rms")
        analysis_profile.add_task("plane_avg(w_rms)", name="w_rms")
        analysis_profile.add_task("plane_avg(Re_rms)", name="Re_rms")
        analysis_profile.add_task("plane_avg(Pe_rms)", name="Pe_rms")
        analysis_tasks.append(analysis_profile)

        analysis_scalar = solver.evaluator.add_file_handler(data_dir+"scalar", max_writes=20, parallel=False, **kwargs)
        analysis_scalar.add_task("vol_avg(KE)", name="KE")
        analysis_scalar.add_task("vol_avg(PE)", name="PE")
        analysis_scalar.add_task("vol_avg(IE)", name="IE")
        analysis_scalar.add_task("vol_avg(PE_fluc)", name="PE_fluc")
        analysis_scalar.add_task("vol_avg(IE_fluc)", name="IE_fluc")
        analysis_scalar.add_task("vol_avg(KE + PE + IE)", name="TE")
        analysis_scalar.add_task("vol_avg(KE + PE_fluc + IE_fluc)", name="TE_fluc")
        analysis_scalar.add_task("vol_avg(u_rms)", name="u_rms")
        analysis_scalar.add_task("vol_avg(w_rms)", name="w_rms")
        analysis_scalar.add_task("vol_avg(Re_rms)", name="Re_rms")
        analysis_scalar.add_task("vol_avg(Pe_rms)", name="Pe_rms")
        analysis_tasks.append(analysis_scalar)

        return self.analysis_tasks

