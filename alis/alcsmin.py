"""
Perform Levenberg-Marquardt least-squares minimization, based on MINPACK-1.

RJC: This is a modified version of MPFIT, which allows CPU multiprocessing
     and has output designed for ALIS, in addition to multiple bug fixes and extensions.

                                   AUTHORS
  The original version of this software, called LMFIT, was written in FORTRAN
  as part of the MINPACK-1 package by XXX.

  Craig Markwardt converted the FORTRAN code to IDL.  The information for the
  IDL version is:

     Craig B. Markwardt, NASA/GSFC Code 662, Greenbelt, MD 20770
     craigm@lheamail.gsfc.nasa.gov
     UPDATED VERSIONs can be found on this WEB PAGE:
        http://cow.physics.wisc.edu/~craigm/idl/idl.html

 Mark Rivers created this Python version from Craig's IDL version.
    Mark Rivers, University of Chicago
    Building 434A, Argonne National Laboratory
    9700 South Cass Avenue, Argonne, IL 60439
    rivers@cars.uchicago.edu
    Updated versions can be found at http://cars.uchicago.edu/software
 
 Sergey Koposov converted Mark's Python version from Numeric to numpy
    Sergey Koposov, University of Cambridge, Institute of Astronomy,
    Madingley road, CB3 0HA, Cambridge, UK
    koposov@ast.cam.ac.uk
    Updated versions can be found at http://code.google.com/p/astrolibpy/source/browse/trunk/
"""

import numpy
import types
import signal
#import scipy.linalg
from alis import almsgs
from alis.alsave import print_model
from multiprocessing import Pool as mpPool
from multiprocessing.pool import ApplyResult
msgs=almsgs.msgs()


try:
    from copyreg import pickle  # Python 3
except:
    from copy_reg import pickle

from types import MethodType

def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

#def _pickle_fortran(fortran):
#	return _unpickle_fortran, ()

#def _unpickle_fortran():
#	return

class alfit(object):

#	blas_enorm32, = scipy.linalg.get_blas_funcs(['nrm2'],numpy.array([0],dtype=numpy.float32))
#	blas_enorm64, = scipy.linalg.get_blas_funcs(['nrm2'],numpy.array([0],dtype=numpy.float64))


    def __init__(self, fcn, xall=None, functkw={}, funcarray=[None, None, None], parinfo=None,
                 ftol=1.e-10, xtol=1.e-10, gtol=1.e-10, atol=1.e-10,
                 damp=0., miniter=0, maxiter=200, factor=100., nprint=1,
                 iterfunct='default', iterkw={}, nocovar=0, limpar=False,
                 rescale=0, autoderivative=1, verbose=2, modpass=None,
                 diag=None, epsfcn=None, ncpus=None, fstep=1.0, debug=0, convtest=False):
        """
  Inputs:
    fcn:
       The function to be minimized.  The function should return the weighted
       deviations between the model and the data, as described above.

    xall:
       An array of starting values for each of the parameters of the model.
       The number of parameters should be fewer than the number of measurements.

       This parameter is optional if the parinfo keyword is used (but see
       parinfo).  The parinfo keyword provides a mechanism to fix or constrain
       individual parameters.

  Keywords:

     autoderivative:
        If this is set, derivatives of the function will be computed
        automatically via a finite differencing procedure.  If not set, then
        fcn must provide the (analytical) derivatives.
           Default: set (=1)
           NOTE: to supply your own analytical derivatives,
                 explicitly pass autoderivative=0

     ftol:
        A nonnegative input variable. Termination occurs when both the actual
        and predicted relative reductions in the sum of squares are at most
        ftol (and status is accordingly set to 1 or 3).  Therefore, ftol
        measures the relative error desired in the sum of squares.
           Default: 1E-10

     functkw:
        A dictionary which contains the parameters to be passed to the
        user-supplied function specified by fcn via the standard Python
        keyword dictionary mechanism.  This is the way you can pass additional
        data to your user-supplied function without using global variables.

        Consider the following example:
           if functkw = {'xval':[1.,2.,3.], 'yval':[1.,4.,9.],
                         'errval':[1.,1.,1.] }
        then the user supplied function should be declared like this:
           def myfunct(p, fjac=None, xval=None, yval=None, errval=None):

        Default: {}   No extra parameters are passed to the user-supplied
                      function.

     gtol:
        A nonnegative input variable. Termination occurs when the cosine of
        the angle between fvec and any column of the jacobian is at most gtol
        in absolute value (and status is accordingly set to 4). Therefore,
        gtol measures the orthogonality desired between the function vector
        and the columns of the jacobian.
           Default: 1e-10

     iterkw:
        The keyword arguments to be passed to iterfunct via the dictionary
        keyword mechanism.  This should be a dictionary and is similar in
        operation to FUNCTKW.
           Default: {}  No arguments are passed.

     iterfunct:
        The name of a function to be called upon each NPRINT iteration of the
        ALFIT routine.  It should be declared in the following way:
           def iterfunct(myfunct, p, iter, fnorm, functkw=None,
                         parinfo=None, quiet=0, dof=None, [iterkw keywords here])
           # perform custom iteration update

        iterfunct must accept all three keyword parameters (FUNCTKW, PARINFO
        and QUIET).

        myfunct:  The user-supplied function to be minimized,
        p:		The current set of model parameters
        iter:	 The iteration number
        functkw:  The arguments to be passed to myfunct.
        fnorm:	The chi-squared value.
        quiet:	Set when no textual output should be printed.
        dof:	  The number of degrees of freedom, normally the number of points
                  less the number of free parameters.
        See below for documentation of parinfo.

        In implementation, iterfunct can perform updates to the terminal or
        graphical user interface, to provide feedback while the fit proceeds.
        If the fit is to be stopped for any reason, then iterfunct should return a
        a status value between -15 and -1.  Otherwise it should return None
        (e.g. no return statement) or 0.
        In principle, iterfunct should probably not modify the parameter values,
        because it may interfere with the algorithm's stability.  In practice it
        is allowed.

        Default: an internal routine is used to print the parameter values.

        Set iterfunct=None if there is no user-defined routine and you don't
        want the internal default routine be called.

     maxiter:
        The maximum number of iterations to perform.  If the number is exceeded,
        then the status value is set to 5 and ALFIT returns.
        Default: 200 iterations

     nocovar:
        Set this keyword to prevent the calculation of the covariance matrix
        before returning (see COVAR)
        Default: clear (=0)  The covariance matrix is returned

     nprint:
        The frequency with which iterfunct is called.  A value of 1 indicates
        that iterfunct is called with every iteration, while 2 indicates every
        other iteration, etc.  Note that several Levenberg-Marquardt attempts
        can be made in a single iteration.
        Default value: 1

     ncpus:
        Number of CPUs to use during parallel processing
        Default value: None  (This means use all CPUs)

     parinfo
        Provides a mechanism for more sophisticated constraints to be placed on
        parameter values.  When parinfo is not passed, then it is assumed that
        all parameters are free and unconstrained.  Values in parinfo are never
        modified during a call to ALFIT.

        See description above for the structure of PARINFO.

        Default value: None  All parameters are free and unconstrained.

     quiet:
        Set this keyword when no textual output should be printed by ALFIT

     damp:
        A scalar number, indicating the cut-off value of residuals where
        "damping" will occur.  Residuals with magnitudes greater than this
        number will be replaced by their hyperbolic tangent.  This partially
        mitigates the so-called large residual problem inherent in
        least-squares solvers (as for the test problem CURVI,
        http://www.maxthis.com/curviex.htm).
        A value of 0 indicates no damping.
           Default: 0

        Note: DAMP doesn't work with autoderivative=0

     xtol:
        A nonnegative input variable. Termination occurs when the relative error
        between two consecutive iterates is at most xtol (and status is
        accordingly set to 2 or 3).  Therefore, xtol measures the relative error
        desired in the approximate solution.
        Default: 1E-10

   Outputs:

     Returns an object of type alfit.  The results are attributes of this class,
     e.g. alfit.status, alfit.errmsg, alfit.params, npfit.niter, alfit.covar.

     .status
        An integer status code is returned.  All values greater than zero can
        represent success (however .status == 5 may indicate failure to
        converge). It can have one of the following values:

        -16
           A parameter or function value has become infinite or an undefined
           number.  This is usually a consequence of numerical overflow in the
           user's model function, which must be avoided.

        -15 to -1
           These are error codes that either MYFUNCT or iterfunct may return to
           terminate the fitting process.  Values from -15 to -1 are reserved
           for the user functions and will not clash with ALIS.

        0  Improper input parameters.

        1  Both actual and predicted relative reductions in the sum of squares
           are at most ftol.

        2  Relative error between two consecutive iterates is at most xtol

        3  Conditions for status = 1 and status = 2 both hold.

        4  The cosine of the angle between fvec and any column of the jacobian
           is at most gtol in absolute value.

        5  The maximum number of iterations has been reached.

        6  ftol is too small. No further reduction in the sum of squares is
           possible.

        7  xtol is too small. No further improvement in the approximate solution
           x is possible.

        8  gtol is too small. fvec is orthogonal to the columns of the jacobian
           to machine precision.

        9  The absolute difference in the chi-squared between successive iterations is less than atol

     .fnorm
        The value of the summed squared residuals for the returned parameter
        values.

     .covar
        The covariance matrix for the set of parameters returned by ALFIT.
        The matrix is NxN where N is the number of  parameters.  The square root
        of the diagonal elements gives the formal 1-sigma statistical errors on
        the parameters if errors were treated "properly" in fcn.
        Parameter errors are also returned in .perror.

        To compute the correlation matrix, pcor, use this example:
           cov = alfit.covar
           pcor = cov * 0.
           for i in range(n):
              for j in range(n):
                 pcor[i,j] = cov[i,j]/sqrt(cov[i,i]*cov[j,j])

        If nocovar is set or ALFIT terminated abnormally, then .covar is set to
        a scalar with value None.

     .errmsg
        A string error or warning message is returned.

     .nfev
        The number of calls to MYFUNCT performed.

     .niter
        The number of iterations completed.

     .perror
        The formal 1-sigma errors in each parameter, computed from the
        covariance matrix.  If a parameter is held fixed, or if it touches a
        boundary, then the error is reported as zero.

        If the fit is unweighted (i.e. no errors were given, or the weights
        were uniformly set to unity), then .perror will probably not represent
        the true parameter uncertainties.

        *If* you can assume that the true reduced chi-squared value is unity --
        meaning that the fit is implicitly assumed to be of good quality --
        then the estimated parameter uncertainties can be computed by scaling
        .perror by the measured chi-squared value.

           dof = len(x) - len(alfit.params) # deg of freedom
           # scaled uncertainties
           pcerror = alfit.perror * sqrt(alfit.fnorm / dof)

        """
        self.niter = 0
        self.params = None
        self.covar = None
        self.perror = None
        self.status = 0  # Invalid input flag set while we check inputs
        self.debug = debug
        self.errmsg = ''
        self.nfev = 0
        self.damp = damp
        self.dof=0
        self.ncpus = ncpus
        self.fstep = fstep

        # Include a function to deal with signal interruptions
        self.handler=True
        signal.signal(signal.SIGQUIT, self.signal_handler)

        if fcn==None:
            self.errmsg = "Usage: parms = alfit('myfunct', ... )"
            return

        if iterfunct == 'default':
            iterfunct = self.defiter

        # Parameter damping doesn't work when user is providing their own
        # gradients.
        if (self.damp != 0) and (autoderivative == 0):
            self.errmsg =  'keywords DAMP and AUTODERIVATIVE are mutually exclusive'
            return

        # Parameters can either be stored in parinfo, or x. x takes precedence if it exists
        if (xall is None) and (parinfo is None):
            self.errmsg = 'must pass parameters in P or PARINFO'
            return

        # Be sure that PARINFO is of the right type
        if parinfo is not None:
            #if type(parinfo) != types.ListType:
            if not isinstance(parinfo, list):
                self.errmsg = 'PARINFO must be a list of dictionaries.'
                return
            else:
                if not isinstance(parinfo[0], dict): #type(parinfo[0]) != types.DictionaryType:
                    self.errmsg = 'PARINFO must be a list of dictionaries.'
                    return
            if ((xall is not None) and (len(xall) != len(parinfo))):
                self.errmsg = 'number of elements in PARINFO and P must agree'
                return

        # If the parameters were not specified at the command line, then
        # extract them from PARINFO
        if xall is None:
            xall = self.parinfo(parinfo, 'value')
            if xall is None:
                self.errmsg = 'either P or PARINFO(*)["value"] must be supplied.'
                return

        # Make sure parameters are numpy arrays
        xall = numpy.asarray(xall)
        # In the case if the xall is not float or if is float but has less
        # than 64 bits we do convert it into double
        if xall.dtype.kind != 'f' or xall.dtype.itemsize<=4:
            xall = xall.astype(numpy.float)

        npar = len(xall)
        self.fnorm  = -1.
        fnorm1 = -1.

        # TIED parameters?
        ptied = self.parinfo(parinfo, 'tied', default='', n=npar)
        self.qanytied = 0
        for i in range(npar):
            ptied[i] = ptied[i].strip()
            if ptied[i] != '':
                self.qanytied = 1
        self.ptied = ptied

        # FIXED parameters ?
        pfixed = self.parinfo(parinfo, 'fixed', default=0, n=npar)
        pfixed = (pfixed == 1)
        for i in range(npar):
            pfixed[i] = pfixed[i] or (ptied[i] != '') # Tied parameters are also effectively fixed

        # Finite differencing step, absolute and relative, and sidedness of deriv.
        step = self.parinfo(parinfo, 'step', default=0., n=npar)
        dstep = self.parinfo(parinfo, 'relstep', default=0., n=npar)
        dside = self.parinfo(parinfo, 'mpside',  default=0, n=npar)

        # Maximum and minimum steps allowed to be taken in one iteration
        maxstep = self.parinfo(parinfo, 'mpmaxstep', default=0., n=npar)
        minstep = self.parinfo(parinfo, 'mpminstep', default=0., n=npar)
        qmin = minstep != 0
        qmin[:] = False # Remove minstep for now!!
        qmax = maxstep != 0
        if numpy.any(qmin & qmax & (maxstep<minstep)):
            self.errmsg = 'MPMINSTEP is greater than MPMAXSTEP'
            return
        wh = (numpy.nonzero((qmin!=0.) | (qmax!=0.)))[0]
        qminmax = len(wh > 0)

        # Finish up the free parameters
        ifree = (numpy.nonzero(pfixed != 1))[0]
        nfree = len(ifree)
        if nfree == 0:
            self.errmsg = 'No free parameters'
            return

        # Compose only VARYING parameters
        self.params = xall.copy()	  # self.params is the set of parameters to be returned
        x = self.params[ifree]  # x is the set of free parameters

        # LIMITED parameters ?
        limited = self.parinfo(parinfo, 'limited', default=[0,0], n=npar)
        limits = self.parinfo(parinfo, 'limits', default=[0.,0.], n=npar)
        if (limited is not None) and (limits is not None):
            # Error checking on limits in parinfo
            if numpy.any((limited[:,0] & limited[:,1]) &
                                 (limits[:,0] >= limits[:,1]) &
                                 (pfixed == 0)):
                self.errmsg = 'Parameter limits are not consistent'
                return
            if numpy.any( ((limited[:,0]==1) & (xall < limits[:,0])) |
                                 ((limited[:,1] == 1) & (xall > limits[:,1])) ):
                # Find the parameter that is not within the limits
                outlim = numpy.where( ((limited[:,0] == 1) & (xall < limits[:,0])) | ((limited[:,1] == 1) & (xall > limits[:,1])) )[0]
                if limpar: # Push parameters to the model limits
                    for ol in range(len(outlim)):
                        if ((limited[outlim[ol],0] == 1) & (xall[outlim[ol]] < limits[outlim[ol],0])):
                            newval = limits[outlim[ol],0]
                        else:
                            newval = limits[outlim[ol],1]
                        msgs.warn("A parameter that = {0:s} is not within specified limits on line -".format(self.params[outlim][ol])+msgs.newline()+modpass['line'][outlim[ol]],verbose=verbose)
                        msgs.info("Setting this parameter to the limiting value of the model: {0:f}".format(newval))
                        xall[outlim][ol], self._params[outlim][ol] = newval, newval
                else:
                    self.errmsg = [outlim,str(self.params[outlim][0])]
                    self.status = -21
                    return

            # Transfer structure values to local variables
            qulim = (limited[:,1])[ifree]
            ulim  = (limits [:,1])[ifree]
            qllim = (limited[:,0])[ifree]
            llim  = (limits [:,0])[ifree]

            if numpy.any((qulim!=0.) | (qllim!=0.)):
                qanylim = 1
            else:
                qanylim = 0
        else:
            # Fill in local variables with dummy values
            qulim = numpy.zeros(nfree)
            ulim  = x * 0.
            qllim = qulim
            llim  = x * 0.
            qanylim = 0

        n = len(x)
        # Check input parameters for errors
        if (n < 0) or (ftol < 0) or (xtol < 0) or (gtol < 0) \
                    or (maxiter < 0) or (factor <= 0):
            self.errmsg = 'input keywords are inconsistent'
            return

        if rescale != 0:
            self.errmsg = 'DIAG parameter scales are inconsistent'
            if len(diag) < n:
                return
            if numpy.any(diag <= 0):
                return
            self.errmsg = ''

        [self.status, fvec, emab] = self.call(fcn, self.params, functkw, getemab=True)

        if self.status < 0:
            self.errmsg = 'first call to "'+str(fcn)+'" failed'
            return
        # If the returned fvec has more than four bits I assume that we have
        # double precision
        # It is important that the machar is determined by the precision of
        # the returned value, not by the precision of the input array
        if numpy.array([fvec]).dtype.itemsize>4:
            self.machar = machar(double=1)
#			self.blas_enorm = alfit.blas_enorm64
        else:
            self.machar = machar(double=0)
#			self.blas_enorm = alfit.blas_enorm32
        machep = self.machar.machep

        m = len(fvec)
        if m < n:
            self.errmsg = 'number of parameters must not exceed data'
            return
        self.dof = m-nfree
        self.fnorm = self.enorm(fvec)

        # Allow multiprocessing to call funcderiv in this class
        pickle(MethodType, _pickle_method, _unpickle_method)
#		pickle(type(self.blas_enorm), _pickle_fortran, _unpickle_fortran)

        # Initialize Levelberg-Marquardt parameter and iteration counter

        par = 0.
        self.niter = 1
        qtf = x * 0.
        self.status = 0

        # Beginning of the outer loop

        while(1):

            # If requested, call fcn to enable printing of iterates
            self.params[ifree] = x
            if self.qanytied:
                self.params = self.tie(self.params, ptied)

            if (nprint > 0) and (iterfunct is not None):
                if ((self.niter-1) % nprint) == 0:
                    mperr = 0
                    xnew0 = self.params.copy()

                    dof = numpy.max([len(fvec) - len(x), 0])
                    status = iterfunct(fcn, self.params, self.niter, self.fnorm**2,
                      functkw=functkw, parinfo=parinfo, verbose=verbose,
                      modpass=modpass, convtest=convtest, dof=dof, funcarray=funcarray, **iterkw)
                    if status is not None:
                        self.status = status

                    # Check for user termination
                    if self.status < 0:
                        self.errmsg = 'WARNING: premature termination by ' + str(iterfunct)
                        return

                    # If parameters were changed (grrr..) then re-tie
                    if numpy.max(numpy.abs(xnew0-self.params)) > 0:
                        if self.qanytied:
                            self.params = self.tie(self.params, ptied)
                        x = self.params[ifree]


            # Calculate the jacobian matrix
            self.status = 2
            catch_msg = 'calling ALFIT_FDJAC2'
            fjac = self.fdjac2(fcn, x, fvec, step, qulim, ulim, dside,
                          epsfcn=epsfcn, emab=emab,
                          autoderivative=autoderivative, dstep=dstep,
                          functkw=functkw, ifree=ifree, xall=self.params)
            if fjac is None:
                self.errmsg = 'WARNING: premature termination by FDJAC2'
                return

            # Determine if any of the parameters are pegged at the limits
            if qanylim:
                catch_msg = 'zeroing derivatives of pegged parameters'
                whlpeg = (numpy.nonzero(qllim & (x == llim)))[0]
                nlpeg = len(whlpeg)
                whupeg = (numpy.nonzero(qulim & (x == ulim)))[0]
                nupeg = len(whupeg)
                # See if any "pegged" values should keep their derivatives
                if nlpeg > 0:
                    # Total derivative of sum wrt lower pegged parameters
                    for i in range(nlpeg):
                        sum0 = numpy.sum(fvec * fjac[:,whlpeg[i]])
                        if sum0 > 0:
                            fjac[:,whlpeg[i]] = 0
                if nupeg > 0:
                    # Total derivative of sum wrt upper pegged parameters
                    for i in range(nupeg):
                        sum0 = numpy.sum(fvec * fjac[:,whupeg[i]])
                        if sum0 < 0:
                            fjac[:,whupeg[i]] = 0

            # Compute the QR factorization of the jacobian
            [fjac, ipvt, wa1, wa2] = self.qrfac(fjac, pivot=1)

            # On the first iteration if "diag" is unspecified, scale
            # according to the norms of the columns of the initial jacobian
            catch_msg = 'rescaling diagonal elements'
            if self.niter == 1:
                if (rescale==0) or (len(diag) < n):
                    diag = wa2.copy()
                    diag[diag == 0.] = 1.

                # On the first iteration, calculate the norm of the scaled x
                # and initialize the step bound delta
                wa3 = diag * x
                xnorm = self.enorm(wa3)
                delta = factor*xnorm
                if delta == 0.:
                    delta = factor

            # Form (q transpose)*fvec and store the first n components in qtf
            catch_msg = 'forming (q transpose)*fvec'
            wa4 = fvec.copy()
            for j in range(n):
                lj = ipvt[j]
                temp3 = fjac[j,lj]
                if temp3 != 0:
                    fj = fjac[j:,lj]
                    wj = wa4[j:]
                    # *** optimization wa4(j:*)
                    wa4[j:] = wj - fj * numpy.sum(fj*wj) / temp3
                fjac[j,lj] = wa1[j]
                qtf[j] = wa4[j]
            # From this point on, only the square matrix, consisting of the
            # triangle of R, is needed.
            fjac = fjac[0:n, 0:n]
            fjac.shape = [n, n]
            temp = fjac.copy()
            for i in range(n):
                temp[:,i] = fjac[:, ipvt[i]]
            fjac = temp.copy()

            # Check for overflow.  This should be a cheap test here since FJAC
            # has been reduced to a (small) square matrix, and the test is
            # O(N^2).
            #wh = where(finite(fjac) EQ 0, ct)
            #if ct GT 0 then goto, FAIL_OVERFLOW

            # Compute the norm of the scaled gradient
            catch_msg = 'computing the scaled gradient'
            gnorm = 0.
            if self.fnorm != 0:
                for j in range(n):
                    l = ipvt[j]
                    if wa2[l] != 0:
                        sum0 = numpy.sum(fjac[0:j+1,j]*qtf[0:j+1])/self.fnorm
                        gnorm = numpy.max([gnorm,numpy.abs(sum0/wa2[l])])

            # Test for convergence of the gradient norm
            if gtol != 0.0:
                if gnorm <= gtol:
                    self.status = 4
                    break
            if maxiter == 0:
                self.status = 5
                break

            # Rescale if necessary
            if rescale == 0:
                diag = numpy.choose(diag>wa2, (wa2, diag))

            # Beginning of the inner loop
            while(1):

                # Determine the levenberg-marquardt parameter
                catch_msg = 'calculating LM parameter (ALIS_)'
                [fjac, par, wa1, wa2] = self.lmpar(fjac, ipvt, diag, qtf,
                                                     delta, wa1, wa2, par=par)
                # Store the direction p and x+p. Calculate the norm of p
                wa1 = -wa1

                if (qanylim == 0) and (qminmax == 0):
                    # No parameter limits, so just move to new position WA2
                    alpha = 1.
                    wa2 = x + wa1

                else:

                    # Respect the limits.  If a step were to go out of bounds, then
                    # we should take a step in the same direction but shorter distance.
                    # The step should take us right to the limit in that case.
                    alpha = 1.

                    if qanylim:
                        # Do not allow any steps out of bounds
                        catch_msg = 'checking for a step out of bounds'
                        if nlpeg > 0:
                            wa1[whlpeg] = numpy.clip( wa1[whlpeg], 0., numpy.max(wa1))
                        if nupeg > 0:
                            wa1[whupeg] = numpy.clip(wa1[whupeg], numpy.min(wa1), 0.)

                        dwa1 = numpy.abs(wa1) > machep
                        whl = (numpy.nonzero(((dwa1!=0.) & qllim) & ((x + wa1) < llim)))[0]
                        if len(whl) > 0:
                            t = ((llim[whl] - x[whl]) /
                                  wa1[whl])
                            alpha = numpy.min([alpha, numpy.min(t)])
                        whu = (numpy.nonzero(((dwa1!=0.) & qulim) & ((x + wa1) > ulim)))[0]
                        if len(whu) > 0:
                            t = ((ulim[whu] - x[whu]) /
                                  wa1[whu])
                            alpha = numpy.min([alpha, numpy.min(t)])

                    # Obey any max step values.
                    if qminmax:
                        nwa1 = wa1 * alpha
                        whmax = (numpy.nonzero((qmax != 0.) & (maxstep > 0)))[0]
                        if len(whmax) > 0:
                            mrat = numpy.max(numpy.abs(nwa1[whmax]) /
                                       numpy.abs(maxstep[ifree[whmax]]))
                            if mrat > 1:
                                alpha = alpha / mrat

                    # The minimization will fail if the model contains a pegged parameter, and alpha is forced to the machine precision. If this happens, reset alpha to be some small number 100 times the machine precision.
                    if numpy.abs(alpha) < 1.0E6*machep:
                        msgs.warn("A parameter step was out of bounds, and resulted in a scalar close"+msgs.newline()+"to the machine precision")
                        msgs.info("Adopting a small scale factor -- check that the subsequent chi-squared is lower")
                        alpha = 0.1

                    # Scale the resulting vector
                    wa1 = wa1 * alpha
                    wa2 = x + wa1

                    # Adjust the final output values.  If the step put us exactly
                    # on a boundary, make sure it is exact.
                    sgnu = (ulim >= 0) * 2. - 1.
                    sgnl = (llim >= 0) * 2. - 1.
                    # Handles case of
                    #        ... nonzero *LIM ... ...zero * LIM
                    ulim1 = ulim * (1 - sgnu * machep) - (ulim == 0) * machep
                    llim1 = llim * (1 + sgnl * machep) + (llim == 0) * machep
                    wh = (numpy.nonzero((qulim!=0) & (wa2 >= ulim1)))[0]
                    if len(wh) > 0:
                        wa2[wh] = ulim[wh]
                    wh = (numpy.nonzero((qllim!=0.) & (wa2 <= llim1)))[0]
                    if len(wh) > 0:
                        wa2[wh] = llim[wh]

                    # Make smaller steps if any tied parameters go out of limits.
                    if self.qanytied:
                        arrom = numpy.append(0.0,10.0**numpy.arange(-16.0,1.0)[::-1])
                        xcopy = self.params.copy()
                        xcopy[ifree] = wa2.copy()
                        watemp = numpy.zeros(npar)
                        watemp[ifree] = wa1.copy()
                        for pqt in range(npar):
                            if self.ptied[pqt] == '': continue
                            cmd = "parval = "+parinfo[pqt]['tied'].replace("p[","xcopy[")
                            namespace = dict({'xcopy':xcopy, 'numpy':numpy})
                            exec(cmd, namespace)
                            parval = namespace['parval']
                            # Check if this parameter is lower than the enforced limit
                            if parinfo[pqt]['limited'][0] == 1:
                                if parval < parinfo[pqt]['limits'][0]:
                                    madetlim = False
                                    for nts in range(1,arrom.size):
                                        xcopyB = self.params.copy()
                                        xcopyB[ifree] = x + arrom[nts]*wa1
                                        cmd = "tmpval = "+parinfo[pqt]['tied'].replace("p[","xcopyB[")
                                        namespace = dict({'xcopyB': xcopyB, 'numpy':numpy})
                                        exec(cmd, namespace)
                                        tmpval = namespace['tmpval']
                                        if tmpval > parinfo[pqt]['limits'][0]: # Then we shouldn't scale the parameters by more than arrom[nts]
                                            arromB = numpy.linspace(arrom[nts],arrom[nts-1],91)[::-1]
                                            xcopyB[ifree] -= arrom[nts]*wa1
                                            for ntsB in range(1,arromB.size):
                                                xcopyB[ifree] = x + arromB[ntsB]*wa1
                                                cmd = "tmpval = "+parinfo[pqt]['tied'].replace("p[","xcopyB[")
                                                namespace = dict({'xcopyB': xcopyB, 'numpy':numpy})
                                                exec(cmd, namespace)
                                                tmpval = namespace['tmpval']
                                                if tmpval > parinfo[pqt]['limits'][0]:
                                                    # Find the parameters used in this linking, and scale there wa1 values appropriately
                                                    strspl = (" "+parinfo[pqt]['tied']).split("p[")
                                                    for ssp in range(1,len(strspl)):
                                                        watemp[int(strspl[ssp].split("]")[0])] *= arromB[ntsB]
                                                    madetlim = True
                                                if madetlim: break
                                                xcopyB[ifree] -= arromB[ntsB]*wa1
                                        if madetlim: break
                                    if not madetlim:
                                        strspl = (" "+parinfo[pqt]['tied']).split("p[")
                                        for ssp in range(1,len(strspl)):
                                            watemp[int(strspl[ssp].split("]")[0])] *= 0.0
                            # Check if this parameter is higher than the enforced limit
                            elif parinfo[pqt]['limited'][1] == 1:
                                if parval > parinfo[pqt]['limits'][1]:
                                    madetlim = False
                                    for nts in range(1,arrom.size):
                                        xcopyB = self.params.copy()
                                        xcopyB[ifree] = x + arrom[nts]*wa1*alpha
                                        cmd = "tmpval = "+parinfo[pqt]['tied'].replace("p[","xcopyB[")
                                        namespace = dict({'xcopyB': xcopyB, 'numpy':numpy})
                                        exec(cmd, namespace)
                                        tmpval = namespace['tmpval']
                                        if tmpval < parinfo[pqt]['limits'][1]: # Then we shouldn't scale the parameters by more than arrom[nts]
                                            arromB = numpy.linspace(arrom[nts],arrom[nts-1],91)[::-1]
                                            xcopyB[ifree] -= arrom[nts]*wa1*alpha
                                            for ntsB in range(1,arromB.size):
                                                xcopyB[ifree] = x + arromB[ntsB]*wa1*alpha
                                                cmd = "tmpval = "+parinfo[pqt]['tied'].replace("p[","xcopyB[")
                                                namespace = dict({'xcopyB': xcopyB, 'numpy':numpy})
                                                exec(cmd, namespace)
                                                tmpval = namespace['tmpval']
                                                if tmpval < parinfo[pqt]['limits'][1]:
                                                    # Find the parameters used in this linking, and scale there wa1 values appropriately
                                                    strspl = (" "+parinfo[pqt]['tied']).split("p[")
                                                    for ssp in range(1,len(strspl)):
                                                        watemp[int(strspl[ssp].split("]")[0])] *= arromB[ntsB]
                                                    madetlim = True
                                                if madetlim: break
                                        if madetlim: break
                                    if not madetlim:
                                        strspl = (" "+parinfo[pqt]['tied']).split("p[")
                                        for ssp in range(1,len(strspl)):
                                            watemp[int(strspl[ssp].split("]")[0])] *= 0.0
                        wa2 = wa2 + watemp[ifree] - wa1
                        del xcopy, watemp, arrom

                # endelse
                wa3 = diag * wa1
                pnorm = self.enorm(wa3)

                # On the first iteration, adjust the initial step bound
                if self.niter == 1:
                    delta = numpy.min([delta,pnorm])

                self.params[ifree] = wa2

                # Evaluate the function at x+p and calculate its norm
                mperr = 0
                catch_msg = 'calling '+str(fcn)
                [self.status, wa4, emab] = self.call(fcn, self.params, functkw, getemab=True)
                #[self.status, wa4] = self.call(fcn, self.params, functkw)
                if self.status < 0:
                    self.errmsg = 'WARNING: premature termination by "'+fcn+'"'
                    return
                fnorm1 = self.enorm(wa4)

                # Compute the scaled actual reduction
                catch_msg = 'computing convergence criteria'
                actred = -1.
                if (0.1 * fnorm1) < self.fnorm:
                    actred = 1.0 - (fnorm1/self.fnorm)**2

                # Compute the scaled predicted reduction and the scaled directional
                # derivative
                for j in range(n):
                    wa3[j] = 0
                    wa3[0:j+1] = wa3[0:j+1] + fjac[0:j+1,j]*wa1[ipvt[j]]

                # Remember, alpha is the fraction of the full LM step actually
                # taken
                temp1 = self.enorm(alpha*wa3)/self.fnorm
                temp2 = (numpy.sqrt(alpha*par)*pnorm)/self.fnorm
                prered = temp1*temp1 + (temp2*temp2)/0.5
                dirder = -(temp1*temp1 + temp2*temp2)

                # Compute the ratio of the actual to the predicted reduction.
                ratio = 0.0
                if prered != 0.0:
                    ratio = actred/prered
#				print ratio, actred, prered

                # Update the step bound
                if ratio <= 0.25:
                    if actred >= 0.0:
                        temp = .5
                    else:
                        temp = .5*dirder/(dirder + .5*actred)
                    if ((0.1*fnorm1) >= self.fnorm) or (temp < 0.1):
                        temp = 0.1
                    delta = temp*numpy.min([delta,pnorm/0.1])
                    par = par/temp
                else:
                    if (par == 0) or (ratio >= 0.75):
                        delta = pnorm/0.5
                        par = 0.5*par

                # Get the absolute reduction
                absred = self.fnorm**2 - fnorm1**2

                # Test for successful iteration
                if ratio >= 0.0001:
                    # Successful iteration.  Update x, fvec, and their norms
                    x = wa2
                    wa2 = diag * x
                    fvec = wa4
                    xnorm = self.enorm(wa2)
                    self.fnorm = fnorm1
                    self.niter = self.niter + 1

                # Tests for convergence
                if ftol != 0.0:
                    if (numpy.abs(actred) <= ftol) and (prered <= ftol) \
                         and (0.5 * ratio <= 1):
                         self.status = 1
                if xtol != 0.0:
                    if delta <= xtol*xnorm:
                        self.status = 2
                if ftol != 0.0:
                    if (numpy.abs(actred) <= ftol) and (prered <= ftol) \
                         and (0.5 * ratio <= 1) and (self.status == 2):
                         self.status = 3
                if atol != 0.0 and atol/fnorm1**2 > machep and ratio >= 0.0001:
                    if absred < atol:
                        self.status = 9

                # If we haven't undertaken the minimum number of interations, then keep going.
                if self.niter < miniter and (self.status in [1,2,3]):
                    self.status = 0
                # End if conditions are satisfied
                if self.status != 0:
                    break

                # Tests for termination and stringent tolerances
                if self.niter >= maxiter:
                    self.status = 5
                if (numpy.abs(actred) <= machep) and (prered <= machep) \
                    and (0.5*ratio <= 1.0):
                    self.status = 6
                if delta <= machep*xnorm and xtol != 0.0:
                    self.status = 7
                if gnorm <= machep and gtol != 0.0:
                    self.status = 8
                if self.status != 0:
                    break

                # End of inner loop. Repeat if iteration unsuccessful
                if ratio >= 0.0001:
                    break

                # Check for over/underflow
                if ~numpy.all(numpy.isfinite(wa1) & numpy.isfinite(wa2) & \
                            numpy.isfinite(x)) or ~numpy.isfinite(ratio):
                    errmsg = ('''parameter or function value(s) have become
                        'infinite; check model function for over- 'and underflow''')
                    self.status = -16
                    break
                #wh = where(finite(wa1) EQ 0 OR finite(wa2) EQ 0 OR finite(x) EQ 0, ct)
                #if ct GT 0 OR finite(ratio) EQ 0 then begin

            if self.status != 0:
                break;

            if self.handler == False:
                break;

        # End of outer loop.

        catch_msg = 'in the termination phase'
        # Termination, either normal or user imposed.
        if len(self.params) == 0:
            return
        if nfree == 0:
            self.params = xall.copy()
        else:
            self.params[ifree] = x
        if (nprint > 0) and (self.status > 0):
            catch_msg = 'calling ' + str(fcn)
            [status, fvec] = self.call(fcn, self.params, functkw)
            catch_msg = 'in the termination phase'
            self.fnorm = self.enorm(fvec)

        if (self.fnorm is not None) and (fnorm1 is not None):
            self.fnorm = numpy.max([self.fnorm, fnorm1])
            self.fnorm = self.fnorm**2.

        self.covar = None
        self.perror = None
        # (very carefully) set the covariance matrix COVAR
        if (self.status > 0) and (nocovar==0) and (n is not None) \
                       and (fjac is not None) and (ipvt is not None):
            sz = fjac.shape
            if (n > 0) and (sz[0] >= n) and (sz[1] >= n) \
                and (len(ipvt) >= n):

                catch_msg = 'computing the covariance matrix'
                cv = self.calc_covar(fjac[0:n,0:n], ipvt[0:n])
                cv.shape = [n, n]
                nn = len(xall)

                # Fill in actual covariance matrix, accounting for fixed
                # parameters.
                self.covar = numpy.zeros([nn, nn], dtype=float)
                for i in range(n):
                    self.covar[ifree,ifree[i]] = cv[:,i]

                # Compute errors in parameters
                catch_msg = 'computing parameter errors'
                self.perror = numpy.zeros(nn, dtype=float)
                d = numpy.diagonal(self.covar)
                wh = (numpy.nonzero(d >= 0))[0]
                if len(wh) > 0:
                    self.perror[wh] = numpy.sqrt(d[wh])
        elif self.handler == False: self.status = -20
        return


    def __str__(self):
        return {'params': self.params,
               'niter': self.niter,
               'params': self.params,
               'covar': self.covar,
               'perror': self.perror,
               'status': self.status,
               'debug': self.debug,
               'errmsg': self.errmsg,
               'nfev': self.nfev,
               'damp': self.damp
               #,'machar':self.machar
               }.__str__()

    # The signal handler
    def signal_handler(self, signum, handler):
        if self.handler:
            self.handler=False

    # Default procedure to be called every iteration.  It simply prints
    # the parameter values.
    def defiter(self, fcn, x, iter, fnorm=None, functkw=None,
                       verbose=2, iterstop=None, parinfo=None,
                       format=None, pformat='%.10g', dof=1,
                       modpass=None, convtest=False, funcarray=[None,None,None]):

        if self.debug:
            print('Entering defiter...')
        if verbose == 0:
            return
        if fnorm is None:
            [status, fvec] = self.call(fcn, x, functkw)
            fnorm = self.enorm(fvec)**2

        # Determine which parameters to print
        nprint = len(x)
        if convtest: msgs.test("CONVERGENCE",verbose=verbose)
        if verbose <= 0: return
        print("ITERATION ", ('%6i' % iter),"   CHI-SQUARED = ",('%.10g' % fnorm)," DOF = ", ('%i' % dof)," (REDUCED = {0:f})".format(fnorm/float(dof)))
        if verbose == 1 or modpass == None:
            return
        else:
            prstr, cvstr = print_model(x, modpass, verbose=verbose, funcarray=funcarray)
            print(prstr+cvstr[0]+cvstr[2])
            return 0



    # Procedure to parse the parameter values in PARINFO, which is a list of dictionaries
    def parinfo(self, parinfo=None, key='a', default=None, n=0):
        if self.debug:
            print('Entering parinfo...')
        if (n == 0) and (parinfo is not None):
            n = len(parinfo)
        if n == 0:
            values = default

            return values
        values = []
        for i in range(n):
            if (parinfo is not None) and (key in parinfo[i].keys()):
                values.append(parinfo[i][key])
            else:
                values.append(default)

        # Convert to numeric arrays if possible
        test = default
        if isinstance(default, list): #type(default) == types.ListType:
            test=default[0]
        if isinstance(test, int): #types.IntType):
            values = numpy.asarray(values, int)
        elif isinstance(test, float): #types.FloatType):
            values = numpy.asarray(values, float)
        return values

    # Call user function or procedure, with _EXTRA or not, with
    # derivatives or not.
    def call(self, fcn, x, functkw, fjac=None, ddpid=None, pp=None, emab=None, getemab=False):
        if self.debug:
            print('Entering call...')
        if self.qanytied:
            x = self.tie(x, self.ptied)
        self.nfev = self.nfev + 1
        if fjac is None:
            if self.damp > 0:
                # Apply the damping if requested.  This replaces the residuals
                # with their hyperbolic tangent.  Thus residuals larger than
                # DAMP are essentially clipped.
                [status, f] = fcn(x, fjac=fjac, ddpid=ddpid, pp=pp, emab=emab, getemab=getemab, **functkw)
                f = numpy.tanh(f/self.damp)
                return [status, f]
            return fcn(x, fjac=fjac, ddpid=ddpid, pp=pp, emab=emab, getemab=getemab, **functkw)
        else:
            return fcn(x, fjac=fjac, ddpid=ddpid, pp=pp, emab=emab, getemab=getemab, **functkw)


    def enorm(self, vec):
        #ans = self.blas_enorm(vec)
        ans = numpy.sqrt(numpy.dot(vec.T, vec))
        return ans

    def funcderiv(self, fcn, fvec, functkw, j, xp, ifree, hj, emab, oneside):
        pp = xp.copy()
        pp[ifree] += hj
        [status, fp] = self.call(fcn, xp, functkw, ddpid=j, pp=pp, emab=emab)
        if status < 0:
            return None
        if oneside:
            # COMPUTE THE ONE-SIDED DERIVATIVE
            fjac = (fp-fvec)/hj
        else:
            # COMPUTE THE TWO-SIDED DERIVATIVE
            pp[ifree] -= 2.0*hj # There's a 2.0 here because hj was recently added to pp (see second line of funcderiv)
            [status, fm] = self.call(fcn, xp, functkw, ddpid=j, pp=pp, emab=emab)
            if status < 0:
                return None
            fjac = (fp-fm)/(2.0*hj)
        return [j,fjac]

    def fdjac2(self, fcn, x, fvec, step=None, ulimited=None, ulimit=None, dside=None,
               epsfcn=None, emab=None, autoderivative=1,
               functkw=None, xall=None, ifree=None, dstep=None):

        if self.debug:
            print('Entering fdjac2...')
        machep = self.machar.machep
        if epsfcn is None:
            epsfcn = machep
        if xall is None:
            xall = x
        if ifree is None:
            ifree = numpy.arange(len(xall))
        if step is None:
            step = x * 0.
        nall = len(xall)

        eps = numpy.sqrt(numpy.max([epsfcn, machep]))
        m = len(fvec)
        n = len(x)

        # Compute analytical derivative if requested
        if autoderivative == 0:
            mperr = 0
            fjac = numpy.zeros(nall, dtype=float)
            fjac[ifree] = 1.0  # Specify which parameters need derivatives
            [status, fp, fjac] = self.call(fcn, xall, functkw, fjac=fjac)

            if fjac.size != m*nall:
                print('Derivative matrix was not computed properly.')
                return None

            # This definition is consistent with CURVEFIT
            # Sign error found (thanks Jesus Fernandez <fernande@irm.chu-caen.fr>)
            fjac.shape = [m,nall]
            fjac = -fjac

            # Select only the free parameters
            if len(ifree) < nall:
                fjac = fjac[:,ifree]
                fjac.shape = [m, n]
                return fjac

        fjac = numpy.zeros([m, n], dtype=numpy.float64)

        h = eps * numpy.abs(x) * self.fstep

        # if STEP is given, use that
        # STEP includes the fixed parameters
        if step is not None:
            stepi = step[ifree]
            wh = (numpy.nonzero(stepi > 0))[0]
            if len(wh) > 0:
                h[wh] = stepi[wh]

        # if relative step is given, use that
        # DSTEP includes the fixed parameters
        if len(dstep) > 0:
            dstepi = dstep[ifree]
            wh = (numpy.nonzero(dstepi > 0))[0]
            if len(wh) > 0:
                h[wh] = numpy.abs(dstepi[wh]*x[wh])

        # In case any of the step values are zero
        h[h == 0.0] = eps * self.fstep

        # In case any of the step values are very small
        h[h < 1.0E-10] = 1.0E-10

        # Reverse the sign of the step if we are up against the parameter
        # limit, or if the user requested it.
        # DSIDE includes the fixed parameters (ULIMITED/ULIMIT have only
        # varying ones)
        mask = dside[ifree] == -1
        if len(ulimited) > 0 and len(ulimit) > 0:
            mask = (mask | ((ulimited!=0) & (x > ulimit-h)))
            wh = (numpy.nonzero(mask))[0]
            if len(wh) > 0:
                h[wh] = - h[wh]

        # Loop through parameters, computing the derivative for each
        pool = mpPool(processes=self.ncpus)
        async_results = []
        for j in range(n):
            if numpy.abs(dside[ifree[j]]) <= 1:
                # COMPUTE THE ONE-SIDED DERIVATIVE
                async_results.append(pool.apply_async(self.funcderiv, (fcn,fvec,functkw,j,xall,ifree[j],h[j],emab,True)))
            else:
                # COMPUTE THE TWO-SIDED DERIVATIVE
                async_results.append(pool.apply_async(self.funcderiv, (fcn,fvec,functkw,j,xall,ifree[j],h[j],emab,False)))
        pool.close()
        pool.join()
        map(ApplyResult.wait, async_results)
        for j in range(n):
            getVal = async_results[j].get()
            if getVal == None: return None
            fjac[0:,getVal[0]] = getVal[1]
        return fjac

#
#       The following code is for the not multi-processing
#
#		# Loop through parameters, computing the derivative for each
#		async_results = []
#		for j in range(n):
#			if numpy.abs(dside[ifree[j]]) <= 1:
#				# COMPUTE THE ONE-SIDED DERIVATIVE
#				async_results.append(self.funcderiv(fcn,fvec,functkw,j,xall,ifree[j],h[j],emab,True))
#			else:
#				# COMPUTE THE TWO-SIDED DERIVATIVE
#				async_results.append(self.funcderiv(fcn,fvec,functkw,j,xall,ifree[j],h[j],emab,False))
#		for j in range(n):
#			getVal = async_results[j]
#			if getVal == None: return None
#			# Note optimization fjac(0:*,j)
#			fjac[0:,getVal[0]] = getVal[1]
#		return fjac




    def qrfac(self, a, pivot=0):

        if self.debug: print('Entering qrfac...')
        machep = self.machar.machep
        sz = a.shape
        m = sz[0]
        n = sz[1]

        # Compute the initial column norms and initialize arrays
        acnorm = numpy.zeros(n, dtype=float)
        for j in range(n):
            acnorm[j] = self.enorm(a[:,j])
        rdiag = acnorm.copy()
        wa = rdiag.copy()
        ipvt = numpy.arange(n)

        # Reduce a to r with householder transformations
        minmn = numpy.min([m,n])
        for j in range(minmn):
            if pivot != 0:
                # Bring the column of largest norm into the pivot position
                rmax = numpy.max(rdiag[j:])
                kmax = (numpy.nonzero(rdiag[j:] == rmax))[0]
                ct = len(kmax)
                kmax = kmax + j
                if ct > 0:
                    kmax = kmax[0]

                    # Exchange rows via the pivot only.  Avoid actually exchanging
                    # the rows, in case there is lots of memory transfer.  The
                    # exchange occurs later, within the body of ALFIT, after the
                    # extraneous columns of the matrix have been shed.
                    if kmax != j:
                        temp = ipvt[j] ; ipvt[j] = ipvt[kmax] ; ipvt[kmax] = temp
                        rdiag[kmax] = rdiag[j]
                        wa[kmax] = wa[j]

            # Compute the householder transformation to reduce the jth
            # column of A to a multiple of the jth unit vector
            lj = ipvt[j]
            ajj = a[j:,lj]
            ajnorm = self.enorm(ajj)
            if ajnorm == 0:
                break
            if a[j,lj] < 0:
                ajnorm = -ajnorm

            ajj = ajj / ajnorm
            ajj[0] = ajj[0] + 1
            # *** Note optimization a(j:*,j)
            a[j:,lj] = ajj

            # Apply the transformation to the remaining columns
            # and update the norms

            # NOTE to SELF: tried to optimize this by removing the loop,
            # but it actually got slower.  Reverted to "for" loop to keep
            # it simple.
            if j+1 < n:
                for k in range(j+1, n):
                    lk = ipvt[k]
                    ajk = a[j:,lk]
                    # *** Note optimization a(j:*,lk)
                    # (corrected 20 Jul 2000)
                    if a[j,lj] != 0:
                        a[j:,lk] = ajk - ajj * numpy.sum(ajk*ajj)/a[j,lj]
                        if (pivot != 0) and (rdiag[k] != 0):
                            temp = a[j,lk]/rdiag[k]
                            rdiag[k] = rdiag[k] * numpy.sqrt(numpy.max([(1.-temp**2), 0.]))
                            temp = rdiag[k]/wa[k]
                            if (0.05*temp*temp) <= machep:
                                rdiag[k] = self.enorm(a[j+1:,lk])
                                wa[k] = rdiag[k]
            rdiag[j] = -ajnorm
        return [a, ipvt, rdiag, acnorm]



    def qrsolv(self, r, ipvt, diag, qtb, sdiag):
        if self.debug:
            print('Entering qrsolv...')
        sz = r.shape
        m = sz[0]
        n = sz[1]

        # copy r and (q transpose)*b to preserve input and initialize s.
        # in particular, save the diagonal elements of r in x.

        for j in range(n):
            r[j:n,j] = r[j,j:n]
        x = numpy.diagonal(r).copy()
        wa = qtb.copy()

        # Eliminate the diagonal matrix d using a givens rotation
        for j in range(n):
            l = ipvt[j]
            if diag[l] == 0:
                break
            sdiag[j:] = 0
            sdiag[j] = diag[l]

            # The transformations to eliminate the row of d modify only a
            # single element of (q transpose)*b beyond the first n, which
            # is initially zero.

            qtbpj = 0.
            for k in range(j,n):
                if sdiag[k] == 0:
                    break
                if numpy.abs(r[k,k]) < numpy.abs(sdiag[k]):
                    cotan  = r[k,k]/sdiag[k]
                    sine   = 0.5/numpy.sqrt(.25 + .25*cotan*cotan)
                    cosine = sine*cotan
                else:
                    tang   = sdiag[k]/r[k,k]
                    cosine = 0.5/numpy.sqrt(.25 + .25*tang*tang)
                    sine   = cosine*tang

                # Compute the modified diagonal element of r and the
                # modified element of ((q transpose)*b,0).
                r[k,k] = cosine*r[k,k] + sine*sdiag[k]
                temp = cosine*wa[k] + sine*qtbpj
                qtbpj = -sine*wa[k] + cosine*qtbpj
                wa[k] = temp

                # Accumulate the transformation in the row of s
                if n > k+1:
                    temp = cosine*r[k+1:n,k] + sine*sdiag[k+1:n]
                    sdiag[k+1:n] = -sine*r[k+1:n,k] + cosine*sdiag[k+1:n]
                    r[k+1:n,k] = temp
            sdiag[j] = r[j,j]
            r[j,j] = x[j]

        # Solve the triangular system for z.  If the system is singular
        # then obtain a least squares solution
        nsing = n
        wh = (numpy.nonzero(sdiag == 0))[0]
        if len(wh) > 0:
            nsing = wh[0]
            wa[nsing:] = 0

        if nsing >= 1:
            wa[nsing-1] = wa[nsing-1]/sdiag[nsing-1] # Degenerate case
            # *** Reverse loop ***
            for j in range(nsing-2,-1,-1):
                sum0 = numpy.sum(r[j+1:nsing,j]*wa[j+1:nsing])
                wa[j] = (wa[j]-sum0)/sdiag[j]

        # Permute the components of z back to components of x
        x[ipvt] = wa
        return (r, x, sdiag)




    def lmpar(self, r, ipvt, diag, qtb, delta, x, sdiag, par=None):

        if self.debug:
            print('Entering lmpar...')
        dwarf = self.machar.minnum
        machep = self.machar.machep
        sz = r.shape
        m = sz[0]
        n = sz[1]

        # Compute and store in x the gauss-newton direction.  If the
        # jacobian is rank-deficient, obtain a least-squares solution
        nsing = n
        wa1 = qtb.copy()
        rthresh = numpy.max(numpy.abs(numpy.diagonal(r))) * machep
        wh = (numpy.nonzero(numpy.abs(numpy.diagonal(r)) < rthresh))[0]
        if len(wh) > 0:
            nsing = wh[0]
            wa1[wh[0]:] = 0.0
        if nsing >= 1:
            # *** Reverse loop ***
            for j in range(nsing-1,-1,-1):
                wa1[j] = wa1[j]/r[j,j]
                if j-1 >= 0:
                    wa1[0:j] = wa1[0:j] - r[0:j,j]*wa1[j]

        # Note: ipvt here is a permutation array
        x[ipvt] = wa1

        # Initialize the iteration counter.  Evaluate the function at the
        # origin, and test for acceptance of the gauss-newton direction
        iter = 0
        wa2 = diag * x
        dxnorm = self.enorm(wa2)
        fp = dxnorm - delta
        if fp <= 0.1*delta:
            return [r, 0., x, sdiag]

        # If the jacobian is not rank deficient, the newton step provides a
        # lower bound, parl, for the zero of the function.  Otherwise set
        # this bound to zero.

        parl = 0.
        if nsing >= n:
            wa1 = diag[ipvt] * wa2[ipvt] / dxnorm
            wa1[0] = wa1[0] / r[0,0] # Degenerate case
            for j in range(1,n):   # Note "1" here, not zero
                sum0 = numpy.sum(r[0:j,j]*wa1[0:j])
                wa1[j] = (wa1[j] - sum0)/r[j,j]

            temp = self.enorm(wa1)
            parl = ((fp/delta)/temp)/temp

        # Calculate an upper bound, paru, for the zero of the function
        for j in range(n):
            sum0 = numpy.sum(r[0:j+1,j]*qtb[0:j+1])
            wa1[j] = sum0/diag[ipvt[j]]
        gnorm = self.enorm(wa1)
        paru = gnorm/delta
        if paru == 0:
            paru = dwarf/numpy.min([delta,0.1])

        # If the input par lies outside of the interval (parl,paru), set
        # par to the closer endpoint

        par = numpy.max([par,parl])
        par = numpy.min([par,paru])
        if par == 0:
            par = gnorm/dxnorm

        # Beginning of an interation
        while(1):
            iter = iter + 1

            # Evaluate the function at the current value of par
            if par == 0:
                par = numpy.max([dwarf, paru*0.001])
            temp = numpy.sqrt(par)
            wa1 = temp * diag
            [r, x, sdiag] = self.qrsolv(r, ipvt, wa1, qtb, sdiag)
            wa2 = diag*x
            dxnorm = self.enorm(wa2)
            temp = fp
            fp = dxnorm - delta

            if (numpy.abs(fp) <= 0.1*delta) or \
               ((parl == 0) and (fp <= temp) and (temp < 0)) or \
               (iter == 10):
               break;

            # Compute the newton correction
            wa1 = diag[ipvt] * wa2[ipvt] / dxnorm

            for j in range(n-1):
                wa1[j] = wa1[j]/sdiag[j]
                wa1[j+1:n] = wa1[j+1:n] - r[j+1:n,j]*wa1[j]
            wa1[n-1] = wa1[n-1]/sdiag[n-1] # Degenerate case

            temp = self.enorm(wa1)
            parc = ((fp/delta)/temp)/temp

            # Depending on the sign of the function, update parl or paru
            if fp > 0:
                parl = numpy.max([parl,par])
            if fp < 0:
                paru = numpy.min([paru,par])

            # Compute an improved estimate for par
            par = numpy.max([parl, par+parc])

            # End of an iteration
        # Termination
        return [r, par, x, sdiag]


    # Procedure to tie one parameter to another.
    def tie(self, p, ptied=None):
        if self.debug:
            print('Entering tie...')
        if ptied is None:
            return
        for i in range(len(ptied)):
            if ptied[i] == '':
                continue
            cmd = 'p[' + str(i) + '] = ' + ptied[i]
            namespace = dict({'p': p, 'numpy':numpy})
            exec(cmd, namespace)
            p = namespace['p']
        return p

    def calc_covar(self, rr, ipvt=None, tol=1.e-14):

        if self.debug:
            print('Entering calc_covar...')
        if numpy.ndim(rr) != 2:
            print('r must be a two-dimensional matrix')
            return -1
        s = rr.shape
        n = s[0]
        if s[0] != s[1]:
            print('r must be a square matrix')
            return -1

        if ipvt is None:
            ipvt = numpy.arange(n)
        r = rr.copy()
        r.shape = [n,n]

        # For the inverse of r in the full upper triangle of r
        l = -1
        tolr = tol * numpy.abs(r[0,0])
        for k in range(n):
            if numpy.abs(r[k,k]) <= tolr:
                break
            r[k,k] = 1./r[k,k]
            for j in range(k):
                temp = r[k,k] * r[j,k]
                r[j,k] = 0.
                r[0:j+1,k] = r[0:j+1,k] - temp*r[0:j+1,j]
            l = k

        # Form the full upper triangle of the inverse of (r transpose)*r
        # in the full upper triangle of r
        if l >= 0:
            for k in range(l+1):
                for j in range(k):
                    temp = r[j,k]
                    r[0:j+1,j] = r[0:j+1,j] + temp*r[0:j+1,k]
                temp = r[k,k]
                r[0:k+1,k] = temp * r[0:k+1,k]

        # For the full lower triangle of the covariance matrix
        # in the strict lower triangle or and in wa
        wa = numpy.repeat([r[0,0]], n)
        for j in range(n):
            jj = ipvt[j]
            sing = j > l
            for i in range(j+1):
                if sing:
                    r[i,j] = 0.
                ii = ipvt[i]
                if ii > jj:
                    r[ii,jj] = r[i,j]
                if ii < jj:
                    r[jj,ii] = r[i,j]
            wa[jj] = r[j,j]

        # Symmetrize the covariance matrix in r
        for j in range(n):
            r[0:j+1,j] = r[j,0:j+1]
            r[j,j] = wa[j]

        return r

class machar:
    def __init__(self, double=1):
        if double == 0:
            info = numpy.finfo(numpy.float32)
        else:
            info = numpy.finfo(numpy.float64)

        self.machep = info.eps
        self.maxnum = info.max
        self.minnum = info.tiny

        self.maxlog = numpy.log(self.maxnum)
        self.minlog = numpy.log(self.minnum)
        self.rdwarf = numpy.sqrt(self.minnum*1.5) * 10
        self.rgiant = numpy.sqrt(self.maxnum) * 0.1

