#
# module to do secant root finding in a general way
#
# if the function is defined by discrete data:
#
#	from scipy.interpolate import interp1d
#	func = interp1d(xdata, ydata, bounds_error=False, kind='cubic')
#	...
#	x_zero = secant(func, x0)
#
# Orvedahl R. 17 Feb 2014

def secant(func, x0, **kwargs): 

    # set defaults
    xm1 = None
    tol = 1.e-8
    maxiter = 1000
    args = ()

    # override defaults if named parameters were passed
    if (kwargs):

        for key, val in kwargs.items():

            # xm1 was passed
            if (key == "xm1"):
                xm1 = float(val)

            # tolerance was passed
            elif (key == "tol"):
                tol = float(val)

            # maxiter was passed
            elif (key == "maxiter"):
                maxiter = int(maxiter)

            # function arguments were passed
            elif (key == "args"):
                args = val

            else:
                print "\nWARNING: Unrecognized argument sent to secant"
                print "\targname = "+str(key)+" argvalue = "+str(val)

    # if xm1 is not given use x0 to get xm1
    if (xm1==None):
        xm1 = 0.99*x0

    f0  = func(x0, *args)
    fm1 = func(xm1, *args)
    # initial change
    dx = -f0 * (x0 - xm1) / (f0 - fm1)

    xm1 = x0
    x = x0 + dx

    cnt = 1
    # loop to find root, previous estimate for root will always be xm1
    while(abs(dx) > tol):

        # new function value and old function value
        fx = func(x, *args)
        fm1 = func(xm1, *args)

        # new estimate for change
        dx = -fx * (x - xm1) / (fx - fm1)

        # store old value and update x
        xm1 = x

        if (cnt == maxiter):
            print "\n\tWARNING: maxiter reached in secant\n"
            break

        x += dx

    return x


