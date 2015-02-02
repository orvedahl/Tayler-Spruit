import numpy as np
from dedalus2.public import *

import logging
logger = logging.getLogger(__name__.split('.')[-1])

class Flame:

    def __init__(self, domain):
        self.domain = domain

    def set_problem(self, nu, kappa, D, g, T0, Tbot, Ttop):

        # write parameter values
        output = "nu = {:g}, kappa = {:g}, D = {:g}"
        logger.info(output.format(nu, kappa, D))
        output = "g = {:g}, T0 = {:g}, Tbot = {:g}, Ttop = {:g}"
        logger.info(output.format(g, T0, Tbot, Ttop))

        # define problem fields, axes, parameters, ...
        self.problem = ParsedProblem(axis_names=['x', 'z'],
                        field_names=['u','u_z','w','w_z','T','T_z','c', 'c_z', 'p'],
                        param_names=['nu','kappa','D','g','T0','Tbot','Ttop'])

        # define the equation set
        self.problem.add_equation("dt(w) + dz(p) - g*T - nu*dx(dx(w)) - nu*dz(w_z) = -u*dx(w) - w*w_z")
        self.problem.add_equation("dt(u) + dx(p)       - nu*dx(dx(u)) - nu*dz(u_z) = -u*dx(u) - w*u_z")
        self.problem.add_equation("dt(T) - kappa*dx(dx(T)) - kappa*dz(T_z) = -u*dx(T) - w*T_z + c**2*T**6/T0**6")
        self.problem.add_equation("dt(c) - D*dx(dx(c)) - D*dz(c_z) = -u*dx(c) - w*c_z - c**2*T**6/T0**6")
        self.problem.add_equation("dz(u) - u_z = 0")
        self.problem.add_equation("dz(w) - w_z = 0")
        self.problem.add_equation("dz(T) - T_z = 0")
        self.problem.add_equation("dz(c) - c_z = 0")
        self.problem.add_equation("dx(u) + w_z = 0")  # div(U) = 0

        # specify boundary conditions: left=bottm, right=top
        self.problem.add_left_bc( "T = Tbot")
        self.problem.add_right_bc("T = Ttop")
        self.problem.add_left_bc( "c_z = 0")
        self.problem.add_right_bc("c_z = 0")
        self.problem.add_left_bc( "u = 0")
        self.problem.add_right_bc("u = 0")
        self.problem.add_left_bc( "w = 0", condition="dx != 0")
        self.problem.add_left_bc( "p = 0", condition="dx == 0")
        self.problem.add_right_bc("w = 0")

        # store the parameters
        self.problem.parameters['nu'] = nu         # viscosity
        self.problem.parameters['kappa'] = kappa   # thermal diffusion
        self.problem.parameters['D'] = D           # C diffusion
        self.problem.parameters['g'] = g           # gravity
        self.problem.parameters['T0'] = T0         # reference Temp
        self.problem.parameters['Tbot'] = Tbot     # T at bottom
        self.problem.parameters['Ttop'] = Ttop     # T at top

        self.problem.expand(self.domain, order=1)

        return self.problem

