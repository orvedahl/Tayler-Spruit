#!/usr/bin/env python
#
# Script to solve for eigenvalues of fluid system:
#
#	-no gravity
#	-no rotation
#	-no viscosity
#	-thermal diffusion
#	-background is constant in space & time
#
# 2015-3-26 R. Orvedahl
#

import sys
import os
import getopt
from scipy import special
import numpy
import pylab

import secant

#####################################################################
#  Main program
#####################################################################
def main():

    derivs = False
    show_plot = True

    # solve for the zeros of:
    #   F(kr) = J'_m(kr R1) * Y'_m(kr R2) - J'_m(kr R2) * Y'_m(kr R1) = 0

    nk = 1000
    k1 = 1.e-16     # close but no cigar to zero
    k2 = 10.
    R1 = 1.         # inner radius
    R2 = 3.         # outer radius

    kr = numpy.linspace(k1, k2, nk)

    # plot data
    if (show_plot):
        pylab.clf()
        pylab.plot(kr, exact(kr, R1, R2, derivs), color='r')
        pylab.plot(kr, numpy.zeros((len(kr))), color='b')
        pylab.title("$F(k_r) = J^\prime_1(k_r R_1) Y^\prime_1(k_r R_2) - " \
                    +"J^\prime_1(k_r R_2) Y^\prime_1(k_r R_1)$")
        pylab.xlabel("$k_r$")
        pylab.ylabel("$F(k_r)$")
        pylab.xlim(0.,10.)
        pylab.ylim(-.5,.5)
        pylab.show()

    # estimated values of wavenumbers that solve F(k r)=0
    # chosen based on plotting the function
    k1 = 0.51
    k2 = 1.76
    k3 = 3.24
    k4 = 4.77
    k5 = 6.33
    k6 = 7.89

    # do root finds to get exact wavenumbers
    k1 = secant.secant(exact, k1, args=(R1, R2, derivs))
    k2 = secant.secant(exact, k2, args=(R1, R2, derivs))
    k3 = secant.secant(exact, k3, args=(R1, R2, derivs))
    k4 = secant.secant(exact, k4, args=(R1, R2, derivs))
    k5 = secant.secant(exact, k5, args=(R1, R2, derivs))
    k6 = secant.secant(exact, k6, args=(R1, R2, derivs))

    # print values of the function given a specific k
    print "\nEigenvalues and F(k) --> expect F(k)=0:\n"
    print "\tk1 = ",k1,"\tF(k1) = ", exact(k1, R1, R2, derivs)
    print "\tk2 = ",k2,"\tF(k2) = ", exact(k2, R1, R2, derivs)
    print "\tk3 = ",k3,"\tF(k3) = ", exact(k3, R1, R2, derivs)
    print "\tk4 = ",k4,"\tF(k4) = ", exact(k4, R1, R2, derivs)
    print "\tk5 = ",k5,"\tF(k5) = ", exact(k5, R1, R2, derivs)
    print "\tk6 = ",k6,"\tF(k6) = ", exact(k6, R1, R2, derivs)
    print "\nFor each eigenvalue, solve for A, B:\n"
    print "\tk1 = ",k1,"\tA, B = ", solve_for_AB(k1, R1, R2)
    print "\tk2 = ",k2,"\tA, B = ", solve_for_AB(k2, R1, R2)
    print "\tk3 = ",k3,"\tA, B = ", solve_for_AB(k3, R1, R2)
    print "\tk4 = ",k4,"\tA, B = ", solve_for_AB(k4, R1, R2)
    print "\tk5 = ",k5,"\tA, B = ", solve_for_AB(k5, R1, R2)
    print "\tk6 = ",k6,"\tA, B = ", solve_for_AB(k6, R1, R2)

    print "\n\n---Complete---\n"

#####################################################################
# dispersion relation
#####################################################################
def dispersion(k):

    # need to solve the cubic equation
    w = 1.0

    return w

#####################################################################
# solve for A, B given a specific wavenumber k
#####################################################################
def solve_for_AB(k, R1, R2):

    # solve the system for A,B
    #
    #  / J'_m(k R1)   Y'_m(k R1) \ /A\  --  /0\
    #  \ J'_m(k R2)   Y'_m(k R2) / \B/  --  \0/

    M = numpy.zeros((2,2))
    M[0][0] = special.jvp(1,k*R1,1) # Jprime_1 at R1
    M[0][1] = special.yvp(1,k*R1,1) # Yprime_1 at R1
    M[1][0] = special.jvp(1,k*R2,1) # Jprime_1 at R2
    M[1][1] = special.yvp(1,k*R2,1) # Yprime_1 at R2

    # close but no cigar to zero, otherwise it returns x=(0,0)
    b = numpy.array([1.e-16,1.e-16])

    x = numpy.linalg.solve(M, b)

    return x[0], x[1]

#####################################################################
# exact result given k
#####################################################################
def exact(k, R1, R2, derivs):

    if (derivs):
        # Bessel evaluated at R1
        J0_1 = special.j0(k*R1)
        Y0_1 = special.y0(k*R1)
        J1_1 = special.j1(k*R1)
        Y1_1 = special.y1(k*R1)

        # Bessel evaluated at R2
        J0_2 = special.j0(k*R2)
        Y0_2 = special.y0(k*R2)
        J1_2 = special.j1(k*R2)
        Y1_2 = special.y1(k*R2)

        #
        # d Z1(k r)/dr = k*Z0(k r) - Z1(k r) / r
        #
        Jprime_1 = k*J0_1 - J1_1/R1
        Jprime_2 = k*J0_2 - J1_2/R2
        Yprime_1 = k*Y0_1 - Y1_1/R1
        Yprime_2 = k*Y0_2 - Y1_2/R2

    else:
        Jprime_1 = special.jvp(1,k*R1,1)
        Jprime_2 = special.jvp(1,k*R2,1)
        Yprime_1 = special.yvp(1,k*R1,1)
        Yprime_2 = special.yvp(1,k*R2,1)

    # F(k r) = J'_m(kr R1) * Y'_m(kr R2) - J'_m(kr R2) * Y'_m(kr R1) = 0
    return Jprime_1 * Yprime_2 - Jprime_2 * Yprime_1
    
#####################################################################
#  If called as a command line program, this serves as the main prog
#####################################################################
if __name__ == "__main__":

    main()

