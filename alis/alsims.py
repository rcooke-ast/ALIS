from __future__ import absolute_import, division, print_function

import os
import time
import copy
import warnings
import numpy as np
# ALIS code
from alis.alcsmin import alfit
from alis import almsgs
from alis import alsave
from alis import alplot
from alis import alutils
msgs=almsgs.msgs()

try: input = raw_input
except NameError: pass

def make_directory(dirname,overwrite=False,verbose=2):
    currDIR=os.getcwd()
    newdir = "{0:s}/{1:s}".format(currDIR,dirname)
    if os.path.exists(newdir):
        if not overwrite: msgs.warn("The following directory already exists:"+msgs.newline()+newdir, verbose=verbose)
    else: os.mkdir(newdir)
    return

def perturb(slf, covar, bparams, parinfo):
    from alis.alis import myfunct_wrap
    # Decide how many characters to use for output files
    nchr = str(int(np.log10(slf._argflag['sim']['perturb']))+1)
    # Create the directory structure for the simulations
    make_directory(slf._argflag['sim']['dirname'],overwrite=slf._argflag['out']['overwrite'],verbose=slf._argflag['out']['verbose'])
    # Grab the best-fitting model
    perror = np.sqrt(np.diag(covar))
    # Store the starting parameters in an array
    outpert = np.array([slf._modpass['p0']])
    wavf, fluf, errf = np.array([]), np.array([]), np.array([])
    for sp in range(len(slf._posnfull)):
        wavf = np.append(wavf, slf._wavefit[sp].copy())
        fluf = np.append(fluf, slf._fluxfit[sp].copy())
        errf = np.append(errf, slf._fluefit[sp].copy())
    # Find the non-zero elements in the covariance matrix
    cvsize = covar.shape[0]
    cxzero, cyzero = np.where(covar==0.0)
    bxzero, byzero = np.bincount(cxzero), np.bincount(cyzero)
    wxzero, wyzero = np.where(bxzero==cvsize)[0], np.where(byzero==cvsize)[0]
    zrocol = np.intersect1d(wxzero, wyzero) # This is the list of columns (or rows), where all elements are zero
    # Create a mask for the non-zero elements
    mask=np.zeros_like(covar)
    mask[:,zrocol],mask[zrocol,:]=1,1
    cvnz=np.zeros((cvsize-zrocol.size,cvsize-zrocol.size))
    cvnz[np.where(cvnz==0.0)]=covar[np.where(mask==0.0)]
    # Generate a new set of starting parameters from the covariance matrix
    X_covar_fit=np.matrix(np.random.standard_normal((cvnz.shape[0],slf._argflag['sim']['perturb'])))
    C_covar_fit=np.matrix(cvnz)
    U_covar_fit=np.linalg.cholesky(C_covar_fit)
    Y_covar_fit=U_covar_fit * X_covar_fit
    # Run through the simulations
    for sim in range(slf._argflag['sim']['perturb']):
        ntxt="{0:0"+nchr+"d}"
        ntxt=ntxt.format(sim+slf._argflag['sim']['startid']) # Text Identifier used as output
        msgs.test("PERTURB -- Realisation {0:s}/{1:s} began {2:s}".format(str(sim+1),str(slf._argflag['sim']['perturb']),time.ctime()),verbose=slf._argflag['out']['verbose'])
        # Enter the new starting parameters
        p0new = []
        cntr=0
        for pw in range(len(slf._modpass['p0'])):
            if perror[pw] > 0.0:
                # Check that the maximum level of perturbation has not been reached
                if slf._modpass['p0'][pw] != 0.0:
                    if (Y_covar_fit[cntr,sim]).flatten()[0]/slf._modpass['p0'][pw] > slf._argflag['sim']['maxperturb']:
                        p0new.append( slf._modpass['p0'][pw]*(1.0+slf._argflag['sim']['maxperturb']) )
                        cntr+=1
                    elif (Y_covar_fit[cntr,sim]).flatten()[0]/slf._modpass['p0'][pw] < -1.0*slf._argflag['sim']['maxperturb']:
                        p0new.append( slf._modpass['p0'][pw]*(1.0-slf._argflag['sim']['maxperturb']) )
                        cntr+=1
                    else:
                        p0new.append( slf._modpass['p0'][pw]+(Y_covar_fit[cntr,sim]).flatten()[0] )
                        cntr+=1
                else:
                    p0new.append( slf._modpass['p0'][pw]+(Y_covar_fit[cntr,sim]).flatten()[0] )
                    cntr+=1
            else: p0new.append(slf._modpass['p0'][pw])
        # Make sure all of the new parameters are within parinfo limits
        for i in range(len(p0new)):
            if parinfo[i]['limited'][0] == 1:
                if parinfo[i]['limits'][0] > p0new[i]: p0new[i] = parinfo[i]['limits'][0]
            if parinfo[i]['limited'][1] == 1:
                if parinfo[i]['limits'][1] < p0new[i]: p0new[i] = parinfo[i]['limits'][1]
            parinfo[i]['value'] = p0new[i]
        #alsave.save_model(slf, p0new, mfit.perror, [(0.0 - 0.0)/3600.0, mfit.fnorm, mfit.dof, mfit.niter, mfit.status], printout=False, extratxt=[slf._argflag['sim']['dirname']+'/',".new"])
        #np.savetxt(slf._argflag['run']['modname']+'.covar',mfit.covar)
        fdict = slf.__dict__.copy()
        fa = {'x':wavf, 'y':fluf, 'err':errf, 'fdict':fdict}
        # Calculate the starting chi-squared
        start_func = myfunct_wrap(p0new,output=2,**fa)
        slf._chisq_init = np.sum(((fluf-start_func)/errf)**2)
        if np.isnan(slf._chisq_init): msgs.error("Initial chi-squared is not a number")
        if slf._chisq_init == np.inf: msgs.error("Input chi-squared is Infinite"+msgs.newline()+"Perhaps the error spectrum is zero?")
        # Fit the realisation
#		msgs.info("Using {0:d} CPUs".format(slf._argflag['run']['ncpus']),verbose=slf._argflag['out']['verbose'])
        tstart=time.time()
        mr = alfit(myfunct_wrap, p0new, parinfo=parinfo, functkw=fa,
                   verbose=1, modpass=slf._modpass, miniter=slf._argflag['chisq']['miniter'], maxiter=slf._argflag['chisq']['maxiter'],
                   atol=slf._argflag['chisq']['atol'], ftol=slf._argflag['chisq']['ftol'], gtol=slf._argflag['chisq']['gtol'], xtol=slf._argflag['chisq']['xtol'],
                   ncpus=slf._argflag['run']['ncpus'], fstep=slf._argflag['chisq']['fstep'], limpar=slf._argflag['run']['limpar'])
#		mr = alfit(myfunct_wrap, slf._modpass['p0'], parinfo=parinfo, functkw=fa,
#					verbose=0, modpass=slf._modpass, miniter=slf._argflag['chisq']['miniter'], maxiter=slf._argflag['chisq']['maxiter'],
#					ftol=slf._argflag['chisq']['ftol'], gtol=slf._argflag['chisq']['gtol'], xtol=slf._argflag['chisq']['xtol'],
#					ncpus=slf._argflag['run']['ncpus'], fstep=slf._argflag['chisq']['fstep'])
        tend=time.time()
        if mr.status <= 0:
            if mr.status == -20:
                msgs.info("Simulation was interrupted",verbose=slf._argflag['out']['verbose'])
                return
            else: msgs.error(mr.errmsg)
        else:
            msgs.info("Reason for convergence:"+msgs.newline()+alutils.getreason(mr.status,verbose=slf._argflag['out']['verbose']),verbose=slf._argflag['out']['verbose'])
        if mr.perror is None:
            msgs.bug("Errors returned from perturbed fit is None", verbose=slf._argflag['out']['verbose'])
            msgs.error("Cannot continue with the simulations")
        # Get the results and print them to file
        outpert = np.append(outpert, np.array([np.array(mr.params)]),axis=0)
        alsave.save_model(slf, mr.params, mr.perror, [(tend - tstart)/3600.0, mr.fnorm, mr.dof, mr.niter, mr.status], printout=False, extratxt=[slf._argflag['sim']['dirname']+'/',".perturb"+ntxt])
        # Plot the data (if requested)
        if slf._argflag['plot']['fits']:
            model = myfunct_wrap(mr.params,output=3,**fa)
            alplot.make_plots_all(slf, model=model)
            fileend=input(msgs.input()+"Press enter to view the fits -")
            alplot.plot_showall()
    ntxt=":0"+nchr+"d}"
    outname="{0:s}.{1:s}_{2"+ntxt+"-{3"+ntxt
    msgs.info("Saving the results from the simulations",verbose=slf._argflag['out']['verbose'])
    pertname=outname.format(slf._argflag['out']['modelname'],'perturb',slf._argflag['sim']['startid'],slf._argflag['sim']['perturb']+slf._argflag['sim']['startid'])
    np.savetxt(pertname,outpert)
    return


def sim_random(slf, covar, bparams, parinfo):
    from alis.alis import myfunct_wrap
    # Decide how many characters to use for output files
    nchr = str(int(np.log10(slf._argflag['sim']['random']))+1)
    # Create the directory structure for the simulations
    make_directory(slf._argflag['sim']['dirname'],overwrite=slf._argflag['out']['overwrite'],verbose=slf._argflag['out']['verbose'])
    # Grab the best-fitting model
    perror = np.sqrt(np.diag(covar))
    modlt = copy.deepcopy(slf._modconv_all)
    outrand, outsyst = np.array([slf._modpass['p0']]), np.array([slf._modpass['p0']]) # Store the starting parameters in an array
    wavf, errf = np.array([]), np.array([])
    fluefull, fluefit = copy.deepcopy(slf._fluefull), copy.deepcopy(slf._fluefit)
    # Check for edge effects due to convolution and create the new wave and error arrays.
    # First find out which indices correspond to convolution
    cvind=[]
    for i in range(len(slf._modpass['emab'])):
        if slf._modpass['emab'][i] == 'cv': cvind.append(i)
    edgearr=[[] for i in range(len(slf._posnfit))]
    iind=0
    for sp in range(len(slf._posnfull)):
        for sn in range(len(slf._posnfull[sp])-1):
            ll = slf._posnfull[sp][sn]
            lu = slf._posnfull[sp][sn+1]
            getstdd=[slf._argflag['sim']['edgecut'],slf._wavefull[sp][ll],slf._wavefull[sp][lu-1]]
            mtyp = slf._modpass['mtyp'][cvind[iind]]
            slf._funcarray[2][mtyp]._keywd = slf._modpass['mkey'][cvind[iind]]
            wvl, wvu = slf._funcarray[1][mtyp].set_vars(slf._funcarray[2][mtyp], bparams, slf._levadd[cvind[iind]], slf._modpass, cvind[iind], getstdd=getstdd)
            if wvl > slf._posnfit[sp][2*sn+0] or wvu < slf._posnfit[sp][2*sn+1]:
                msgs.warn("The random simulations cannot be trusted. The fitted is"+msgs.newline()+
                        "affected by edge effects from convolution. It is recommended"+msgs.newline()+
                        "that you input more data outside the fitted regions for:"+msgs.newline()+
                        slf._snipnames[sp][sn],verbose=slf._argflag['out']['verbose'])
            edgearr[sp].append([wvl,wvu])
            iind += 1
        wavf = np.append(wavf, slf._wavefit[sp].copy())
        errf = np.append(errf, slf._fluefit[sp].copy())
    # Find the non-zero elements in the covariance matrix
    cvsize = covar.shape[0]
    cxzero, cyzero = np.where(covar==0.0)
    bxzero, byzero = np.bincount(cxzero), np.bincount(cyzero)
    wxzero, wyzero = np.where(bxzero==cvsize)[0], np.where(byzero==cvsize)[0]
    zrocol = np.intersect1d(wxzero, wyzero) # This is the list of columns (or rows), where all elements are zero
    # Create a mask for the non-zero elements
    mask=np.zeros_like(covar)
    mask[:,zrocol],mask[zrocol,:]=1,1
    cvnz=np.zeros((cvsize-zrocol.size,cvsize-zrocol.size))
    cvnz[np.where(cvnz==0.0)]=covar[np.where(mask==0.0)]
    # Generate a new set of starting parameters from the covariance matrix
    if slf._argflag['sim']['newstart']:
        X_covar_fit=np.matrix(np.random.standard_normal((cvnz.shape[0],slf._argflag['sim']['random'])))
        C_covar_fit=np.matrix(cvnz)
        U_covar_fit=np.linalg.cholesky(C_covar_fit)
        Y_covar_fit=U_covar_fit * X_covar_fit
    # Run through the simulations
    for sim in range(slf._argflag['sim']['random']):
        ntxt="{0:0"+nchr+"d}"
        ntxt=ntxt.format(sim+slf._argflag['sim']['startid']) # Text Identifier used as output
        msgs.simulate("RANDOM ERRORS -- Realisation {0:s}/{1:s} began {2:s}".format(str(sim+1),str(slf._argflag['sim']['random']),time.ctime()),verbose=slf._argflag['out']['verbose'])
        # Generate a random realisation
        newfluxfull, newfluxfit = [], []
        fluf = np.array([])
        for sp in range(len(slf._posnfull)):
            newfluxfull.append(np.random.normal(modlt[sp],fluefull[sp]))
            newfluxfit.append(np.array([]))
            for sn in range(len(slf._posnfull[sp])-1):
                ll = slf._posnfull[sp][sn]
                lu = slf._posnfull[sp][sn+1]
                w = np.where((slf._wavefull[sp][ll:lu] >= slf._posnfit[sp][2*sn+0]) & (slf._wavefull[sp][ll:lu] <= slf._posnfit[sp][2*sn+1]))
                wA= np.in1d(slf._wavefull[sp][ll:lu][w], slf._wavefit[sp])
                wB= np.where(wA==True)
                newfluxfit[sp] = np.append(newfluxfit[sp], np.copy(newfluxfull[sp][ll:lu][w][wB]))
                fluf = np.append(fluf, np.copy(newfluxfull[sp][ll:lu][w][wB]))
        slf._fluxfull, slf._fluxfit = copy.deepcopy(newfluxfull), copy.deepcopy(newfluxfit)
        slf._fluefull, slf._fluefit = copy.deepcopy(fluefull), copy.deepcopy(fluefit)
        p0new = []
        if slf._argflag['sim']['newstart']:
            cntr=0
            for pw in range(len(slf._modpass['p0'])):
                if perror[pw] > 0.0:
                    p0new.append( slf._modpass['p0'][pw]+(Y_covar_fit[cntr,sim]).flatten()[0] )
                    cntr+=1
                else: p0new.append(slf._modpass['p0'][pw])
        else: p0new = slf._modpass['p0']
        #alsave.save_model(slf, p0new, mfit.perror, [(0.0 - 0.0)/3600.0, mfit.fnorm, mfit.dof, mfit.niter, mfit.status], printout=False, extratxt=[slf._argflag['sim']['dirname']+'/',".new"])
        #np.savetxt(slf._argflag['out']['modelname']+'.covar',mfit.covar)
        fdict = slf.__dict__.copy()
        fa = {'x':wavf, 'y':fluf, 'err':errf, 'fdict':fdict}
        # Calculate the starting chi-squared
        start_func = myfunct_wrap(slf._modpass['p0'],output=2,**fa)
        slf._chisq_init = np.sum(((fluf-start_func)/errf)**2)
        if np.isnan(slf._chisq_init): msgs.error("Initial chi-squared is not a number")
        if slf._chisq_init == np.inf: msgs.error("Input chi-squared is Infinite"+msgs.newline()+"Perhaps the error spectrum is zero?")
        # Fit the realisation
#		msgs.info("Using {0:d} CPUs".format(slf._argflag['run']['ncpus']),verbose=slf._argflag['out']['verbose'])
        tstart=time.time()
        mr = alfit(myfunct_wrap, slf._modpass['p0'], parinfo=parinfo, functkw=fa,
                   verbose=1, modpass=slf._modpass, miniter=slf._argflag['chisq']['miniter'], maxiter=slf._argflag['chisq']['maxiter'],
                   atol=slf._argflag['chisq']['atol'], ftol=slf._argflag['chisq']['ftol'], gtol=slf._argflag['chisq']['gtol'], xtol=slf._argflag['chisq']['xtol'],
                   ncpus=slf._argflag['run']['ncpus'], fstep=slf._argflag['chisq']['fstep'], limpar=slf._argflag['run']['limpar'])
#		mr = alfit(myfunct_wrap, slf._modpass['p0'], parinfo=parinfo, functkw=fa,
#					verbose=0, modpass=slf._modpass, miniter=slf._argflag['chisq']['miniter'], maxiter=slf._argflag['chisq']['maxiter'],
#					ftol=slf._argflag['chisq']['ftol'], gtol=slf._argflag['chisq']['gtol'], xtol=slf._argflag['chisq']['xtol'],
#					ncpus=slf._argflag['run']['ncpus'], fstep=slf._argflag['chisq']['fstep'])
        tend=time.time()
        if mr.status <= 0:
            if mr.status == -20:
                msgs.info("Random simulation was interrupted",verbose=slf._argflag['out']['verbose'])
                return
            else: msgs.error(mr.errmsg)
        else:
            msgs.info("Reason for convergence:"+msgs.newline()+alutils.getreason(mr.status,verbose=slf._argflag['out']['verbose']),verbose=slf._argflag['out']['verbose'])
        if mr.perror is None:
            msgs.bug("Errors returned from random fit is None",verbose=slf._argflag['out']['verbose'])
            msgs.error("Cannot continue with the simulations")
        # Get the results and print them to file
        outrand = np.append(outrand, np.array([np.array(mr.params)]),axis=0)
        alsave.save_model(slf, mr.params, mr.perror, [(tend - tstart)/3600.0, mr.fnorm, mr.dof, mr.niter, mr.status], printout=False, extratxt=[slf._argflag['sim']['dirname']+'/',".rand"+ntxt])
        # Plot the data (if requested)
        if slf._argflag['plot']['fits']:
            model = myfunct_wrap(mr.params,output=3,**fa)
            alplot.make_plots_all(slf, model=model)
            fileend=input(msgs.input()+"Press enter to view the fits -")
            alplot.plot_showall()
        if slf._argflag['sim']['systematics']:
            # Calculate the systematics
            msgs.simulate("SYSTEMATIC ERRORS -- Realisation {0:s}/{1:s} began {2:s}".format(str(sim+1),str(slf._argflag['sim']['random']),time.ctime()),verbose=slf._argflag['out']['verbose'])
            ms = sim_systematics(slf, p0new, parinfo, ntxt, edgearr)
            outsyst = np.append(outsyst, np.array([np.array(ms.params)]),axis=0)
    ntxt=":0"+nchr+"d}"
    outname="{0:s}.{1:s}_{2"+ntxt+"-{3"+ntxt
    msgs.info("Saving the results from the random simulations",verbose=slf._argflag['out']['verbose'])
    randname=outname.format(slf._argflag['out']['modelname'],'rand',slf._argflag['sim']['startid'],slf._argflag['sim']['random']+slf._argflag['sim']['startid'])
    np.savetxt(randname,outrand)
    if slf._argflag['sim']['systematics']:
        msgs.info("Saving the results from the systematics simulations",verbose=slf._argflag['out']['verbose'])
        systname=outname.format(slf._argflag['out']['modelname'],'syst',slf._argflag['sim']['startid'],slf._argflag['sim']['random']+slf._argflag['sim']['startid'])
        np.savetxt(systname,outsyst)
    return

def sim_systematics(slf, p0new, parinfo, ntxt, edgearr):
    """
    This implementation of Sim Systematics uses
    the information in 'systematics' (passed in as input)
    to refit the continuum of the fake data.
    'systematics' should be a keyword red in
    from the 'data read' section.
    -------------------------------
    Systematic errors include:
    + choice of continuum
    + choice of starting parameters
    -------------------------------
    """
    from alis.alis import myfunct_wrap
    # Make changes to the continuum
    wavf, fluf, errf = np.array([]), np.array([]), np.array([])
    stf, enf = [0 for all in slf._posnfull], [0 for all in slf._posnfull]
    for sp in range(len(slf._posnfull)):
        for sn in range(len(slf._posnfull[sp])-1):
            ll = slf._posnfull[sp][sn]
            lu = slf._posnfull[sp][sn+1]
            if slf._datopt['systmodule'][sp][sn] is None:
                # Don't make any systematic corrections to these spectra
                msgs.warn("Not applying any systematic corrections to file:"+msgs.newline()+slf._snipnames[sp][sn],verbose=slf._argflag['out']['verbose'])
                newfluxfull, newfluefull = np.copy(slf._fluxfull[sp][ll:lu]), np.copy(slf._fluefull[sp][ll:lu])
            elif slf._datopt['systmodule'][sp][sn] == 'default' or slf._datopt['systmodule'][sp][sn] == 'continuumpoly':
                # The default systematics routine --- fit a polynomial to the continuum
                newfluxfull, newfluefull = syst_continuumpoly(slf._wavefull[sp][ll:lu], slf._fluxfull[sp][ll:lu], slf._fluefull[sp][ll:lu], slf._systfull[sp][ll:lu], edgearr[sp][sn], slf._snipnames[sp][sn], verbose=slf._argflag['out']['verbose'])
            #elif slf._datopt['systmodule'][sp][sn] == other built-in function:
            else:
                # If the routine isn't built-in, it must be user-defined
                # Get the identifier text
                if ',' in slf._datopt['systmodule'][sp][sn]:
                    filename, idtxt = slf._datopt['systmodule'][sp][sn].split(',')
                else:
                    filename, idtxt = slf._datopt['systmodule'][sp][sn], 'systmodule'
                path, file = os.path.split(filename)
                name, ext = os.path.splitext(file)
                # Set the import loader
                try:
                    from ihooks import BasicModuleLoader as srcloader
                except ImportError:
                    msgs.error("Cannot call user-defined function without 'ihooks' module installed." + msgs.newline() +
                               "Install ihooks to continue (note: ihooks is currently only supported in python 2.*)")
                impload = srcloader()
                modu = impload.find_module_in_dir(name, path)
                if not modu: msgs.error("Could not import {0:s}".format(name))
                usrsystmod = impload.load_module(name, modu)
                newfluxfull, newfluefull = usrsystmod.loader(idtxt, slf._wavefull[sp][ll:lu], slf._fluxfull[sp][ll:lu], slf._fluefull[sp][ll:lu], slf._systfull[sp][ll:lu], edgearr[sp][sn], slf._snipnames[sp][sn])
            # Assign the adjusted data to the slf class (and multiply by the user-specified continuum).
            slf._fluxfull[sp][ll:lu], slf._fluefull[sp][ll:lu] = copy.deepcopy(newfluxfull)*slf._contfull[sp][ll:lu], copy.deepcopy(newfluefull)*slf._contfull[sp][ll:lu]
            # Make the appropriate changes to the fitted spectral region
            w = np.where((slf._wavefull[sp][ll:lu] >= slf._posnfit[sp][2*sn+0]) & (slf._wavefull[sp][ll:lu] <= slf._posnfit[sp][2*sn+1]))
            wA= np.in1d(slf._wavefull[sp][ll:lu][w], slf._wavefit[sp])
            wB= np.where(wA==True)
            enf[sp] = stf[sp] + slf._wavefull[sp][ll:lu][w][wB].size
            slf._fluxfit[sp][stf[sp]:enf[sp]] = slf._fluxfull[sp][ll:lu][w][wB]
            slf._fluefit[sp][stf[sp]:enf[sp]] = slf._fluefull[sp][ll:lu][w][wB]
            stf[sp] = enf[sp]
        # Place the fitted regions into a single array to be read by th echi-squared fitting program
        wavf = np.append(wavf, slf._wavefit[sp])
        fluf = np.append(fluf, slf._fluxfit[sp])
        errf = np.append(errf, slf._fluefit[sp])
    # Make sure all of the new parameters are within parinfo limits
    for i in range(len(p0new)):
        if parinfo[i]['limited'][0] == 1:
            if parinfo[i]['limits'][0] > p0new[i]: p0new[i] = parinfo[i]['limits'][0]
        if parinfo[i]['limited'][1] == 1:
            if parinfo[i]['limits'][1] < p0new[i]: p0new[i] = parinfo[i]['limits'][1]
        parinfo[i]['value'] = p0new[i]
    # Update functargs and fit the data!
    fdict = slf.__dict__.copy()
    fa = {'x':wavf, 'y':fluf, 'err':errf, 'fdict':fdict}
    # Calculate the starting chi-squared
    start_func = myfunct_wrap(p0new,output=2,**fa)
    slf._chisq_init = np.sum(((fluf-start_func)/errf)**2)
    if np.isnan(slf._chisq_init): msgs.error("Initial chi-squared is not a number")
    if slf._chisq_init == np.inf: msgs.error("Input chi-squared is Infinite"+msgs.newline()+"Perhaps the error spectrum is zero?")
    if slf._argflag['plot']['fits']:
        model = myfunct_wrap(p0new,output=3,**fa)
        alplot.make_plots_all(slf,model=model)
        fileend=input(msgs.input()+"Press enter to view the fits -")
        alplot.plot_showall()
    # Now fit it!
    tstart=time.time()
    ms = alfit(myfunct_wrap, p0new, parinfo=parinfo, functkw=fa,
                verbose=1, modpass=slf._modpass, miniter=slf._argflag['chisq']['miniter'], maxiter=slf._argflag['chisq']['maxiter'],
                ftol=slf._argflag['chisq']['ftol'], gtol=slf._argflag['chisq']['gtol'], xtol=slf._argflag['chisq']['xtol'],
                ncpus=slf._argflag['run']['ncpus'], fstep=slf._argflag['chisq']['fstep'])
    tend=time.time()
    if ms.status <= 0:
        if ms.status == -20:
            msgs.info("Systematics simulation was interrupted",verbose=slf._argflag['out']['verbose'])
            return
        else: msgs.error(ms.errmsg)
    else:
        msgs.info("Reason for convergence:"+msgs.newline()+alutils.getreason(ms.status,verbose=slf._argflag['out']['verbose']),verbose=slf._argflag['out']['verbose'])
    if ms.perror is None:
        msgs.bug("Errors returned from systematics fit is None",verbose=slf._argflag['out']['verbose'])
        msgs.error("Cannot continue with the simulations")
    # Plot the data (if requested)
    if slf._argflag['plot']['fits']:
        model = myfunct_wrap(ms.params,output=3,**fa)
        alplot.make_plots_all(slf,model=model)
        fileend=input(msgs.input()+"Press enter to view the fits -")
        alplot.plot_showall()
    # Get the results and print them to file
    alsave.save_model(slf, ms.params, ms.perror, [(tend - tstart)/3600.0, ms.fnorm, ms.dof, ms.niter, ms.status], printout=False, extratxt=[slf._argflag['sim']['dirname']+'/',".syst"+ntxt])
    return ms

##############################################################################
#########      BELOW HERE ARE THE BUILT-IN SYSTEMATICS MODULES      ##########
##############################################################################

def syst_continuumpoly(wave, flux, flue, syst, edgearr, filename, verbose=2):
    """
    This is the default systematics routine. It fits a polynomial to the randomly
    generated data using the same pixels as used for the real data.
    ----
    wave, flux, & flue are the wavelength, flux, and error array respectively.
    ----
    syst is the same length as these arrays, which can take only two integer
    values: -1 and n. -1 indicates the pixels that shouldn't be used, and n
    indicates the order of the polynomial. Pixels labeled with an integer `n'
    will be used to refit the continuum.
    """
    # Find the polynomial order used
    w = np.where(syst != -1.0)
    if np.size(w[0]) == 0:
        msgs.error("Systematics information provides no polynomial order or fitting regions.")
    else:
        polyord = np.unique(syst[w])
        if np.size(polyord) != 1:
            msgs.warn("Non-unique 'systematics' column in data",verbose=verbose)
        polyord = polyord[0]
    # Check for edge effects:
    wB = np.where((wave[w] > edgearr[0]) & (wave[w] < edgearr[1]))[0]
    if np.size(w) != np.size(wB):
        msgs.warn("The systematics simulations cannot be trusted."+msgs.newline()+
                "The adjusted region is affected by edge effects from convolution."+msgs.newline()+
                "It is recommended that you input more data outside the fitted regions for:"+msgs.newline()+filename,verbose=verbose)
    # Perform the polynomial fit on the fake data
    flxsz = flux.size
    coeffs = sim_weighted_polyfit(np.arange(flxsz)[w][wB],flux[w][wB],polyord,w=1.0/flue[w][wB]**2.0)
    newcont = np.polyval(coeffs[::-1],np.arange(flxsz))
#	plt.plot(wave,flux,'k-',drawstyle='steps')
#	plt.plot(wave,newcont,'r-')
#	plt.plot(wave[w],flux[w],'ro')
#	plt.plot(wave[w][wB],flux[w][wB],'bx')
#	plt.show()
    # Apply and return the new continuum
    return flux/newcont, flue/newcont

##############################################################################
#########            BELOW HERE ARE SOME USEFUL FUNCTIONS           ##########
##############################################################################

def sim_weighted_polyfit(x, y, deg, rcond=None, full=False, w=None):
    """
    Copy of the polyfit algorithm in new versions of numpy (>2.7).
    It is listed here just in case some user's don't have the most
    up-to-date version of numpy on their system.
    """
    order = int(deg) + 1
    x = np.asarray(x) + 0.0
    y = np.asarray(y) + 0.0

    # check arguments.
    if deg < 0 :
        raise ValueError("expected deg >= 0")
    if x.ndim != 1:
        raise TypeError("expected 1D vector for x")
    if x.size == 0:
        raise TypeError("expected non-empty vector for x")
    if y.ndim < 1 or y.ndim > 2 :
        raise TypeError("expected 1D or 2D array for y")
    if len(x) != len(y):
        raise TypeError("expected x and y to have same length")

    # set up the least squares matrices
    lhs = sim_polyvander(x, deg)
    rhs = y
    if w is not None:
        w = np.asarray(w) + 0.0
        if w.ndim != 1:
            raise(TypeError, "expected 1D vector for w")
        if len(x) != len(w):
            raise(TypeError, "expected x and w to have same length")
        # apply weights
        if rhs.ndim == 2:
            lhs /= w[:, np.newaxis]
            rhs /= w[:, np.newaxis]
        else:
            lhs /= w[:, np.newaxis]
            rhs /= w

    # set rcond
    if rcond is None :
        rcond = len(x)*np.finfo(x.dtype).eps

    # scale the design matrix and solve the least squares equation
    scl = np.sqrt((lhs*lhs).sum(0))
    c, resids, rank, s = np.linalg.lstsq(lhs/scl, rhs, rcond)
    c = (c.T/scl).T

    # warn on rank reduction
    if rank != order and not full:
        msg = "The fit may be poorly conditioned"
        #warnings.warn(msg, pu.RankWarning)

    if full :
        return c, [resids, rank, s, rcond]
    else :
        return c

def sim_polyvander(x, deg) :
    """
    Vandermonde matrix of given degree.
    This is taken out of the new versions of numpy (>2.7)
    """
    ideg = int(deg)
    if ideg != deg:
        raise ValueError("deg must be integer")
    if ideg < 0:
        raise ValueError("deg must be non-negative")

    x = np.array(x, copy=0, ndmin=1) + 0.0
    v = np.empty((ideg + 1,) + x.shape, dtype=x.dtype)
    v[0] = x*0 + 1
    if ideg > 0 :
        v[1] = x
        for i in range(2, ideg + 1) :
            v[i] = v[i-1]*x
    return np.rollaxis(v, 0, v.ndim)

