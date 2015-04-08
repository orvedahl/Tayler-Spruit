import numpy as np
from dedalus.public import *

import logging
logger = logging.getLogger(__name__.split('.')[-1])

class eigenvalue_2d:
    def __init__(self, domain):
        self.domain = domain
        
    def set_problem(self):
        self.problem = de.EVP(self.domain, variables=['u','u_x','u_y',],
                              eigenvalue='l')

        self.problem.add_equation("dz(w) - w_z = 0")
        self.problem.add_equation("dz(u) - u_z = 0")
        self.problem.add_equation("dz(T) - T_z = 0")

        logger.info("Imposing fixed boundaries")
        self.problem.add_left_bc( "T = 1") # bottom = hotter/lighter
        self.problem.add_right_bc("T = 0") # top = colder/heavier
        self.problem.add_left_bc( "u = 0") # \vec{u} = 0 BC
        self.problem.add_right_bc("u = 0")
        self.problem.add_left_bc( "w = 0", condition="dx != 0") # for dx=0, w=const, so need to
        self.problem.add_left_bc( "P = 0", condition="dx == 0") # "manually" account for this
        self.problem.add_right_bc("w = 0")                      # by only setting w=0 for dx!=0

        self.problem.expand(self.domain, order=1)

        return self.problem

