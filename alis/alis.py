""" Absorption LIne Software
"""
from __future__ import absolute_import, division, print_function

# Import standard libraries
import os
import pdb
import sys
import time
import copy
import signal
import warnings
import traceback
import numpy as np
# Import a Chi-Squared minimisation package
from alis.alcsmin import alfit
# Import useful ALIS definitions:
from alis import alconv
from alis import alload
from alis import alplot
from alis import alsave
from alis import alutils
from alis import alfunc_base
from alis import almsgs
msgs = almsgs.msgs()

try: input = raw_input
except NameError: pass

#from multiprocessing import Pool as mpPool
#import pycuda.driver as cuda
#import pycuda.autoinit
#from pycuda.compiler import SourceModule

### TO BE IMPLEMENTED ###
# Supermongo
# Make sure there is some data in the fitted region (when you load a model)
# Make sure each snip contains an absorption feature (or give a warning).
# generate starting parameter files

# def myfunct_wrap(p, fjac=None, x=None, y=None, err=None, fdict=None, ddpid=None, pp=None, getemab=False, emab=None, output=0):
#     instance=ClassMain(None, getinst=True)
#     if fdict is None: fdict = dict({})
#     instance.__dict__.update(fdict)
#     return ClassMain.myfunct(instance, p, fjac=fjac, x=x, y=y, err=err, ddpid=ddpid, pp=pp, getemab=getemab, emab=emab, output=output)

class ClassMain:

    def __init__(self, argflag, getinst=False, modelfile=None, parlines=[], datlines=[], modlines=[], lnklines=[], data=None, fitonly=False, verbose=None):
        if getinst: return # Just get an instance

        # Set parameters
        self._argflag = argflag
        if verbose is not None: self._argflag['out']['verbose'] = verbose
        self._retself = False
        self._fitonly = fitonly
        self._isonefits = False

        # First send all signals to messages to be dealt
        # with (i.e. someone hits ctrl+c)
        signal.signal(signal.SIGINT, msgs.signal_handler)

        # Ignore all warnings given by python
        warnings.resetwarnings()
        warnings.simplefilter("ignore")

        # Record the starting time
        self._tstart=time.time()

        # Load the Input file
        if modelfile is not None:
            self._argflag['run']['modname'] = modelfile
            self._parlines, self._datlines, self._modlines, self._lnklines = alload.load_input(self)
            self._retself = True
        elif parlines != [] or modlines != [] or datlines != [] or lnklines != []:
            self._parlines, self._datlines, self._modlines, self._lnklines = parlines, datlines, modlines, lnklines
            self._argflag = alload.set_params(self._parlines, copy.deepcopy(self._argflag), setstr="Model ")
            alload.check_argflag(self._argflag)
            self._retself = True
        else:
            self._parlines, self._datlines, self._modlines, self._lnklines = alload.load_input(self)
        # Load the atomic data
        self._atomic = alload.load_atomic(self)

        # Get the calls to each of the functions
        function=alfunc_base.call(getfuncs=True, verbose=self._argflag['out']['verbose'])
        funccall=alfunc_base.call(verbose=self._argflag['out']['verbose'])
        funcinst=alfunc_base.call(prgname=self._argflag['run']['prognm'], getinst=True, verbose=self._argflag['out']['verbose'],atomic=self._atomic)
        self._funcarray=[function,funccall,funcinst]


        # Update the verbosity of messages for functions
        for i in self._funcarray[2].keys():
            self._funcarray[2][i]._verbose = self._argflag['out']['verbose']

        # Load the Data
        alload.load_data(self, self._datlines, data=data)

        # Load the Model
        self._modpass = alload.load_model(self, self._modlines)

        # Load the Links
        self._links = alload.load_links(self, self._lnklines)

        if self._argflag['out']['modelname'] in ['', "\"\""]:
            self._argflag['out']['modelname'] = self._argflag['run']['modname'] + '.out'

        #		print self._modpass
#		print self._snipid
#		print self._specid
#		sys.exit()

        # Fit the data!
        self.main()

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# The fitting code

    # Now for the fitting code
    def main(self):
        init_fit = None
        msgs.info("Preparing model parameters",verbose=self._argflag['out']['verbose'])
        # Prepare the parameters and their limits
        wavf, fluf, errf = np.array([]), np.array([]), np.array([])
        for sp in range(len(self._posnfull)):
            wavf = np.append(wavf, self._wavefit[sp])
            fluf = np.append(fluf, self._fluxfit[sp])
            errf = np.append(errf, self._fluefit[sp])
        # Load the parameter information array
        parinfo, self._levadd = alload.load_parinfo(self)
        # Load parameter influence
        self._pinfl = alload.load_par_influence(self, self._modpass['p0'])
        # Sub-pixellate the spectrum
        self._wavespx, self._contspx, self._zerospx, self._posnspx, self._nexbins = alload.load_subpixels(self, self._modpass['p0'])
        # If there are some linked parameters, apply the links here
        npar = len(self._modpass['p0'])
        self._ptied = []
        lnkcnt=0
        for i in range(npar):
            if 'tied' in parinfo[i].keys():
                self._ptied.append(parinfo[i]['tied'].replace("numpy","np"))
                self._modpass['mlnk'].append([-2-lnkcnt,parinfo[i]['tied'].replace("numpy","np")])
                lnkcnt += 1
            else: self._ptied.append('')
        self._qanytied = 0
        for i in range(npar):
            self._ptied[i] = self._ptied[i].strip()
            if self._ptied[i] != '':
                self._qanytied = 1
        if self._qanytied: self._modpass['p0'] = alload.load_tied(self._modpass['p0'], self._ptied)
        # NOW PREPARE TO FIT THE DATA!
        fdict = self.__dict__#.copy()
        fa = {'x':wavf, 'y':fluf, 'err':errf, 'fdict':fdict}
        if self._argflag['run']['ngpus'] is not None and self._argflag['run']['ngpus'] != 0:
            msgs.info("Performing a GPU accelerated analysis")
        # Calculate the initial Chi-Squared
        msgs.info("Calculating the starting chi-squared",verbose=self._argflag['out']['verbose'])
        # Calculate the starting function
        init_fit = alfit(self, self._modpass['p0'], dofit=False, parinfo=parinfo,
                         ncpus=self._argflag['run']['ncpus'], ngpus=self._argflag['run']['ngpus'],
                         fstep=self._argflag['chisq']['fstep'])
        start_func = init_fit.myfunct(self._modpass['p0'], output=2)
        if not self._argflag['generate']['data'] and self._argflag['sim']['beginfrom'] == "":
            self._chisq_init = np.sum(((fluf-start_func)/errf)**2)
            numfreepars = len(parinfo) - [parinfo[i]['fixed'] for i in range(len(parinfo))].count(1)
            if np.isnan(self._chisq_init): msgs.error("Initial chi-squared is not a number")
            if self._chisq_init == np.Inf: msgs.error("Input chi-squared is Infinite"+msgs.newline()+"Perhaps the error spectrum is zero?")
            msgs.info("Chi-squared using input parameters = " + str(self._chisq_init),verbose=self._argflag['out']['verbose'])
            msgs.info("Number of free parameters: {0:d}".format(numfreepars),verbose=self._argflag['out']['verbose'])
        if self._argflag['plot']['only']:
            # Go straight to plotting
            self._fitparams = self._modpass['p0']
            alplot.make_plots_all(self)
            plotCasePDF = ((self._argflag['out']['plots'].lower() == 'true') or ((self._argflag['out']['plots'].lower() != 'false') and (self._argflag['out']['plots'] != '')))
            if plotCasePDF:
                alplot.plot_pdf(self)
            fileend=input(msgs.input()+"Press enter to view the fits -")
            alplot.plot_showall()
        elif self._argflag['sim']['random'] is not None and self._argflag['sim']['beginfrom'] != "":
            from alis import alsims
            # If doing systematics you can skip the initial fit
            # Get the covariance matrix from the best-fitting model
            covar = np.loadtxt(self._argflag['sim']['beginfrom']+".covar")
            # Load the best-fitting parameters
            parlines, datlines, modlines, lnklines = alload.load_input(self, filename=self._argflag['sim']['beginfrom'], updateself=False)
            modpass = alload.load_model(self, modlines, updateself=False)
            # Calculate the best-fitting model
            model = np.copy(start_func)  #self.myfunct(modpass['p0'], output=1)
            msgs.info("Starting simulations",verbose=self._argflag['out']['verbose'])
            alsims.sim_random(self, covar, modpass['p0'], parinfo)
        elif self._argflag['sim']['perturb'] is not None and self._argflag['sim']['beginfrom'] != "":
            from alis import alsims
            # If doing systematics you can skip the initial fit
            # Get the covariance matrix from the best-fitting model
            covar = np.loadtxt(self._argflag['sim']['beginfrom']+".covar")
            # Load the best-fitting parameters
            parlines, datlines, modlines, lnklines = alload.load_input(self, filename=self._argflag['sim']['beginfrom'], updateself=False)
            modpass = alload.load_model(self, modlines, updateself=False)
            # Calculate the best-fitting model
            model = np.copy(start_func)  #self.myfunct(modpass['p0'], output=1)
            msgs.info("Starting simulations",verbose=self._argflag['out']['verbose'])
            alsims.perturb(self, covar, modpass['p0'], parinfo)
        elif self._argflag['iterate']['model'] is not None:
            # The user wants to iterate over the model
            # Get the identifier text
            if len(self._argflag['iterate']['model'])==2:
                filename, idtxt = self._argflag['iterate']['model']
            elif len(self._argflag['iterate']['model'])==1:
                filename, idtxt = self._argflag['iterate']['model'][0], 'itermodule'
            else:
                msgs.error("You can only pass two arguments (separated by a comma) to iterate+model"+msgs.newline()+"The first argument is a python module, the second is a text string")
            path, file = os.path.split(filename)
            name, ext = os.path.splitext(file)
            # Set the import loader
            try:
                from ihooks import BasicModuleLoader as srcloader
            except ImportError:
                msgs.error("Cannot iterate model without 'ihooks' module installed." + msgs.newline() +
                           "Install ihooks to continue (note: ihooks is currently only supported in python 2.*)")
            impload = srcloader()
            modu = impload.find_module_in_dir(name, path)
            if not modu: msgs.error("Could not import {0:s}".format(name))
            itermod = impload.load_module(name, modu)
            # Now iterate
            complete = False
            iternum = 1
            while not complete:
                msgs.info("Iterating the model, Iteration {0:d}".format(iternum),verbose=self._argflag['out']['verbose'])
                # Fit this iteration
                # functkw={}, funcarray=[None, None, None], ftol=1.e-10, xtol=1.e-10, gtol=1.e-10, atol=1.e-10,
                # miniter=0, maxiter=200, factor=100., nprint=1, iterkw={}, nocovar=0, limpar=False, rescale=0,
                # verbose=2, modpass=None, diag=None, epsfcn=None, convtest=False.
                init_fit.minimise(self._modpass['p0'], parinfo=parinfo, functkw=fa, funcarray=self._funcarray,
                        verbose=self._argflag['out']['verbose'], modpass=self._modpass, miniter=self._argflag['chisq']['miniter'], maxiter=self._argflag['chisq']['maxiter'],
                        atol=self._argflag['chisq']['atol'], ftol=self._argflag['chisq']['ftol'], gtol=self._argflag['chisq']['gtol'], xtol=self._argflag['chisq']['xtol'],
                        limpar=self._argflag['run']['limpar'])
                model = init_fit.myfunct(init_fit.params, output=2)
                # Pass the fitted information to the user module and obtain and updated model and completion status
                newmodln = alsave.modlines(self, init_fit.params, self._modpass, verbose=self._argflag['out']['verbose'])
                newmerln = alsave.modlines(self, init_fit.perror, self._modpass, verbose=self._argflag['out']['verbose'])
                modlines, complete = itermod.loader(self, idtxt, newmodln, newmerln, init_fit, model=model)
                if not complete: # then update the appropriate variables
                    self._modlines = modlines
                    # Reload the new model
                    self._modpass = alload.load_model(self, self._modlines)
                    # Reload the parameter information array
                    parinfo, self._levadd = alload.load_parinfo(self)
                    # Reload parameter influence
                    self._pinfl = alload.load_par_influence(self, self._modpass['p0'])
                    # Re-sub-pixellate the spectrum
                    self._wavespx, self._contspx, self._zerospx, self._posnspx, self._nexbins = alload.load_subpixels(self, self._modpass['p0'])
                    # Update the parameter dictionary
                    fa['fdict'] = self.__dict__
                    iternum += 1
            self._fitresults = init_fit
            self._fitparams = init_fit.params
            self._tend=time.time()
        elif self._argflag['generate']['data']:
            # Save the generated data
            #if self._argflag['out']['fits'] or self._argflag['out']['onefits']:
            # Go to modelfits to generate the fake data. It won't be written out unless self._argflag['out']['fits'] is True
            self, fnames = alsave.save_modelfits(self)
            self._fitparams = self._modpass['p0']
            # Plot the results
            plotCasePDF = ((self._argflag['out']['plots'].lower() == 'true') or ((self._argflag['out']['plots'].lower() != 'false') and (self._argflag['out']['plots'] != '')))
            if self._argflag['plot']['fits'] or self._argflag['plot']['residuals'] or (plotCasePDF):
                alplot.make_plots_all(self)
                if plotCasePDF:
                    alplot.plot_pdf(self)
                if self._argflag['plot']['fits'] or self._argflag['plot']['residuals']:
                    null=input(msgs.input()+"Press enter to view the fits -")
                    alplot.plot_showall()
        else:
            msgs.info("Commencing chi-squared minimisation",verbose=self._argflag['out']['verbose'])
            init_fit.minimise(self._modpass['p0'], parinfo=parinfo, functkw=fa, funcarray=self._funcarray,
                      verbose=self._argflag['out']['verbose'], modpass=self._modpass, miniter=self._argflag['chisq']['miniter'], maxiter=self._argflag['chisq']['maxiter'],
                      atol=self._argflag['chisq']['atol'], ftol=self._argflag['chisq']['ftol'], gtol=self._argflag['chisq']['gtol'], xtol=self._argflag['chisq']['xtol'],
                      limpar=self._argflag['run']['limpar'])
            self._tend=time.time()
            niter=init_fit.niter
            if (init_fit.status <= 0):
                if init_fit.status == -20: # Interrupted fit
                    msgs.info("Fitting routine was interrupted",verbose=self._argflag['out']['verbose'])
                    msgs.warn("Setting ERRORS = PARAMETERS",verbose=self._argflag['out']['verbose'])
                    init_fit.perror = init_fit.params
                elif init_fit.status == -21: # Parameters are not within the specified limits
                    if type(self._modpass['line'][init_fit.errmsg[0][0]]) is int: msgs.error("A parameter that = {0:s} is not within the specified limits on line -".format(init_fit.errmsg[1])+msgs.newline()+self._modlines[self._modpass['line'][init_fit.errmsg[0][0]]])
                    else: msgs.error("A parameter that = {0:s} is not within specified limits on line -".format(init_fit.errmsg[1])+msgs.newline()+self._modpass['line'][init_fit.errmsg[0][0]])

                elif init_fit.status == -16:
                    msgs.error("There was an error in the chi-squared minimization - "+msgs.newline()+"A parameter or function value has become infinite or an undefined"+msgs.newline()+"number. This is usually a consequence of numerical overflow in the"+msgs.newline()+"user's model function, which must be avoided.")
                else:
                    if init_fit.errmsg == "": msgs.error("There was an error in the chi-squared minimization - "+msgs.newline()+"please contact the author")
                    msgs.error(init_fit.errmsg)
            else:
                msgs.info("Reason for convergence:"+msgs.newline()+alutils.getreason(init_fit.status,verbose=self._argflag['out']['verbose']),verbose=self._argflag['out']['verbose'])
            # Create the best-fitting model, this generates some arrays that are now required
            msgs.info("Best-fitting model parameters found",verbose=self._argflag['out']['verbose'])
            # If the user just wants to fit the data and return, do so.
            if self._fitonly:
                self._fitresults = init_fit
                self._fitparams = init_fit.params
                model = init_fit.myfunct(init_fit.params, output=1)
                return self
            # Otherwise, do everything else
            if self._argflag['run']['convergence']:
                if init_fit.status == -20:
                    msgs.warn("Cannot check convergence for an interrupted fit",verbose=self._argflag['out']['verbose'])
                else:
                    msgs.info("Beginning test for convergence",verbose=self._argflag['out']['verbose'])
                    mpars = copy.deepcopy(init_fit)
                    lowby = 0
                    mt = copy.deepcopy(init_fit) # The testing set of parameters
                    # Initialise the fit - TODO : Is this needed? Maybe could reuse init_fit?
                    mc = alfit(self, mpars.params, parinfo=parinfo, ncpus=self._argflag['run']['ncpus'],
                               ngpus=self._argflag['run']['ngpus'],
                               fstep=self._argflag['chisq']['fstep'])
                    while True:
                        if mpars.status != 5: # If the maximum number of iterations was reached, don't lower the tolerance
                            self._argflag['chisq']['atol'] /= 10.0
                            self._argflag['chisq']['ftol'] /= 10.0
                            self._argflag['chisq']['gtol'] /= 10.0
                            self._argflag['chisq']['xtol'] /= 10.0
                            lowby -= 1
                        # Now minimise
                        mc.minimise(mpars.params, parinfo=parinfo, functkw=fa, funcarray=self._funcarray,
                                    verbose=self._argflag['out']['verbose'], modpass=self._modpass, miniter=self._argflag['chisq']['miniter'], maxiter=self._argflag['chisq']['maxiter'],
                                    atol=self._argflag['chisq']['atol'], ftol=self._argflag['chisq']['ftol'], gtol=self._argflag['chisq']['gtol'], xtol=self._argflag['chisq']['xtol'],
                                    limpar=self._argflag['run']['limpar'], convtest=True)
                        # Update parameters
                        mpars = copy.deepcopy(mc)
                        # Keep going until this run has reached the tolerances
                        if mpars.status == 5 and self._argflag['run']['convnostop']:
                            continue
                        # Check that the chi-squared didn't fail
                        if mpars.status <= 0:
                            if mpars.status == -20:
                                msgs.info("Convergence check was interrupted",verbose=self._argflag['out']['verbose'])
                                # Force non-convergence
                                mpars.perror = 10.0*(mpars.params-mt.params)/self._argflag['run']['convcriteria']
                            else: msgs.error(mpars.errmsg)
                        # Keep going if the tolerances haven't been lowered
                        if lowby == 0: continue
                        # Check the solution
                        # Find out which parameters have converged
                        whrconv = np.where( (np.abs(mpars.params-mt.params)/mt.perror) >= self._argflag['run']['convcriteria'])[0]
                        outputconvfile = True
                        if np.size(whrconv) == 0 and mpars.niter > 1 and mpars.status != 5:
                            msgs.info("Solution has converged",verbose=self._argflag['out']['verbose'])
                            if outputconvfile:
                                fwrite = open(self._argflag['run']['modname'].rstrip("mod")+"convY", "w")
                                fwrite.close()
                            # Use the best-fit results from the convergence test
                            init_fit = mpars
                            msgs.info("Reason for convergence:"+msgs.newline()+alutils.getreason(mpars.status,verbose=self._argflag['out']['verbose']),verbose=self._argflag['out']['verbose'])
                            break
                        elif mc.niter == 1:
                            if outputconvfile:
                                fwrite = open(self._argflag['run']['modname'].rstrip("mod")+"convN", "w")
                                fwrite.close()
                            msgs.warn("Solution has probably not converged yet (after 1 iteration)",verbose=self._argflag['out']['verbose'])
                            if self._argflag['run']['convnostop']:
                                msgs.info("User has forced to continue convergence with 'noconvstop'",verbose=self._argflag['out']['verbose'])
                                continue
                            else: break
                        elif mc.status == 5:
                            if outputconvfile:
                                fwrite = open(self._argflag['run']['modname'].rstrip("mod")+"convN", "w")
                                fwrite.close()
                            msgs.warn("Solution has probably not converged yet."+msgs.newline()+"Maximum number of iterations reached",verbose=self._argflag['out']['verbose'])
                            if self._argflag['run']['convnostop']:
                                msgs.info("User has forced to continue convergence with 'noconvstop'",verbose=self._argflag['out']['verbose'])
                                continue
                            else: break
                        else:
                            if outputconvfile:
                                fwrite = open(self._argflag['run']['modname'].rstrip("mod")+"convN", "w")
                                fwrite.close()
                            msgs.warn("Solution has not converged for {0:d}/{1:d} free parameters".format(np.size(whrconv), numfreepars),verbose=self._argflag['out']['verbose'])
                            msgs.info("Maximum parameter difference was {0:f} sigma".format( np.max((np.abs(mpars.params-mt.params)/mt.perror)[whrconv]) ),verbose=self._argflag['out']['verbose'])
                            if not self._argflag['run']['convnostop']: break # Stop if convergence isn't forced
                            mt = copy.deepcopy(mpars)
                            continue
                    msgs.info("Convergence test complete",verbose=self._argflag['out']['verbose'])
            msgs.info("Generating best-fit model",verbose=self._argflag['out']['verbose'])
            model = init_fit.myfunct(init_fit.params, output=1)
            # Prepare some arrays for eventual plotting
#			if self._posnLya != 0:
#				elnames, elwaves, rdshft, comparr = alplot.prep_arrs(self._snip_ions, self._snip_detl, self._posnfit)
            # Write out the results of the convergence test:
            if self._argflag['run']['convergence']:
                if init_fit.status != -20 and mc.status != -20 and mpars.status != -20:
                    if self._argflag['out']['convtest'] != "":
                        fit_info=[(self._tend - self._tstart)/3600.0, init_fit.fnorm, init_fit.dof, init_fit.niter, init_fit.status]
                        diff = np.abs(mpars.params-mt.params)/mt.perror
                        diff[np.where((mpars.params==mt.params) & (mt.perror==0.0))[0]] = 0.0
                        alconv.save_convtest(self, diff, self._argflag['run']['convcriteria'],fit_info)
            # Store the fitting results
            self._fitresults = init_fit
            self._fitparams = init_fit.params
            # Write out the data and model fits
            if self._argflag['out']['fits'] or self._argflag['out']['onefits']:
                fnames = alsave.save_modelfits(self)
            # Write a supermongo file for the plots.
            if self._argflag['out']['sm']:
                msgs.error("Sorry, supermongo generated files are not implemented yet")
                msgs.info("Generating Supermongo files to plot the output",verbose=self._argflag['out']['verbose'])
                #alsave.save_smfiles(self._modname_dla, fnames, elnames, elwaves, comparr, rdshft)
            # Write an output of the parameters for the best-fitting model
            if self._argflag['run']['blind'] and init_fit.status != -20:
                if self._argflag['run']['convergence']:
                    if mc.status != -20 and mpars.status != -20:
                        msgs.info("Printing out the parameter errors:",verbose=self._argflag['out']['verbose'])
                        print(alsave.print_model(init_fit.perror, self._modpass, blind=True, verbose=self._argflag['out']['verbose'],funcarray=self._funcarray))
                else:
                    msgs.info("Printing out the parameter errors:",verbose=self._argflag['out']['verbose'])
                    print(alsave.print_model(init_fit.perror, self._modpass, blind=True, verbose=self._argflag['out']['verbose'],funcarray=self._funcarray))
            if self._argflag['out']['model']:
                fit_info=[(self._tend - self._tstart)/3600.0, init_fit.fnorm, init_fit.dof, init_fit.niter, init_fit.status]
                alsave.save_model(self, init_fit.params, init_fit.perror, fit_info)
            if self._argflag['out']['covar'] != "":
                alsave.save_covar(self, init_fit.covar)
            # Plot the results
            plotCasePDF = ((self._argflag['out']['plots'].lower() == 'true') or ((self._argflag['out']['plots'].lower() != 'false') and (self._argflag['out']['plots'] != '')))
            if self._argflag['plot']['fits'] or self._argflag['plot']['residuals'] or plotCasePDF:
                alplot.make_plots_all(self)
                if plotCasePDF:
                    alplot.plot_pdf(self)
                if self._argflag['plot']['fits'] or self._argflag['plot']['residuals']:
                    null=input(msgs.input()+"Press enter to view the fits -")
                    alplot.plot_showall()
#			if self._argflag['plot']['fits']:
#				alplot.make_plots_all(self)
#				fileend=input(msgs.input()+"Press enter to view the fits -")
#				alplot.plot_showall()
            self.print_runtime()
            # If simulations were requested, do them now
            if self._argflag['sim']['random'] != None:
                from alis import alsims
                if init_fit.perror is None or init_fit.perror is init_fit.params: msgs.warn("Fitting routine interrupted. Cannot perform simulations",verbose=self._argflag['out']['verbose'])
                else:
                    msgs.info("Starting simulations",verbose=self._argflag['out']['verbose'])
                    alsims.sim_random(self, init_fit.covar, init_fit.params, parinfo)
            elif self._argflag['sim']['perturb'] != None:
                from alis import alsims
                if init_fit.perror is None or init_fit.perror is init_fit.params: msgs.warn("Fitting routine interrupted. Cannot perform simulations",verbose=self._argflag['out']['verbose'])
                else:
                    msgs.info("Starting simulations",verbose=self._argflag['out']['verbose'])
                    alsims.perturb(self, init_fit.covar, init_fit.params, parinfo)
        if self._retself == True:
            return self

    def print_runtime(self):
        hours = (self._tend - self._tstart) / 3600.0
        if hours >= 1.0:
            msgs.info("Total fitting time in hours: %s" % (hours),
                      verbose=self._argflag['out']['verbose'])
        else:
            mins = (self._tend - self._tstart) / 60.0
            if mins >= 1.0:
                msgs.info("Total fitting time in minutes: %s" % (mins),
                          verbose=self._argflag['out']['verbose'])
            else:
                msgs.info("Total fitting time in seconds: %s" % (self._tend - self._tstart),
                          verbose=self._argflag['out']['verbose'])

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

def alis(modelfile=None, parlines=[], datlines=[], modlines=[], lnklines=[], data=None, fitonly=False, verbose=-1):
    """
    modelfile : This is the name of a model file
    lines : This is a three element list of the form [parlines, datlines, modlines]

    lines is ignored if modelfile is given
    """
    msgs.info("calling ALIS...", verbose=verbose)
    debug = False  # There are two instances of this (one is in main just below)
    if debug:
        argflag = alload.optarg(os.path.realpath(__file__), verbose=verbose)
        return ClassMain(argflag, parlines=parlines, datlines=datlines, modlines=modlines, lnklines=lnklines, modelfile=modelfile, data=data, fitonly=fitonly, verbose=verbose)
    else:
        try:
            argflag = alload.optarg(os.path.realpath(__file__), verbose=verbose)
            return ClassMain(argflag, parlines=parlines, datlines=datlines, modlines=modlines, lnklines=lnklines, modelfile=modelfile, data=data, fitonly=fitonly, verbose=verbose)
        except Exception:
            # There is a bug in the code, print the file and line number of the error.
            et, ev, tb = sys.exc_info()
            while tb:
                co = tb.tb_frame.f_code
                filename = str(co.co_filename)
                line_no =  str(traceback.tb_lineno(tb))
                tb = tb.tb_next
            filename=filename.split('/')[-1]
            msgs.bug("There appears to be a bug on Line "+line_no+" of "+filename+" with error:"+msgs.newline()+str(ev)+msgs.newline()+"---> please contact the author")

def initialise(alispath, verbose=-1):
    argflag = alload.optarg(alispath, verbose=verbose)
    slf = ClassMain(argflag,getinst=True)
    slf._argflag = argflag
    slf._argflag['out']['verbose'] = verbose
    slf._atomic = alload.load_atomic(slf)
    slf._isonefits = False
    slf._funcarray = [None, None, None]
    slf._funcarray[0] = alfunc_base.call(getfuncs=True, verbose=slf._argflag['out']['verbose'])
    slf._funcarray[1] = alfunc_base.call(verbose=slf._argflag['out']['verbose'])
    slf._funcarray[2] = alfunc_base.call(prgname=slf._argflag['run']['prognm'], getinst=True, verbose=slf._argflag['out']['verbose'],atomic=slf._atomic)
    return slf

