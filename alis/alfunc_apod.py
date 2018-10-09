import numpy as np
from alis import almsgs
from alis import alfunc_base
from scipy.signal import convolve
msgs=almsgs.msgs()

class AFWHM(alfunc_base.Base) :
    """
    Convolves the spectrum with a Gaussian with full-width at half-maxium AFWHM (in Angstroms):
    p[0] = AFWHM
    """
    def __init__(self, prgname="", getinst=False, atomic=None, verbose=2):
        self._idstr   = 'apod'			# ID string for this class
        self._pnumr   = 1				# Total number of parameters fed in
        self._keywd   = dict({'kind':'uniform', 'blind':False})		# Additional arguments to describe the model --- 'input' cannot be used as a keyword
        self._keych   = dict({'kind':1,         'blind':0})			# Require keywd to be changed (1 for yes, 0 for no)
        self._keyfm   = dict({'kind':"", 'blind':""})			# Require keywd to be changed (1 for yes, 0 for no)
        self._parid   = ['scale']		# Name of each parameter
        self._defpar  = [ 1.0 ]			# Default values for parameters that are not provided
        self._fixpar  = [ True ]		# By default, should these parameters be fixed?
        self._limited = [ [0  ,0  ] ]	# Should any of these parameters be limited from below or above
        self._limits  = [ [0.0,0.0] ]	# What should these limiting values be
        self._svfmt   = [ "{0:s}" ]	# Specify the format used to print or save output
        self._prekw   = []				# Specify the keywords to print out before the parameters
        # DON'T CHANGE THE FOLLOWING --- it tells ALIS what parameters are provided by the user.
        tempinput = self._parid+list(self._keych.keys())                             #
        self._keywd['input'] = dict(zip((tempinput),([0]*np.size(tempinput)))) #
        ########################################################################
        self._verbose = verbose
        # Set the atomic data
        self._atomic = atomic
        if getinst: return

    def call_CPU(self, x, y, p, ncpus=1):
        """
        Define the functional form of the model
        --------------------------------------------------------
        x  : array of wavelengths
        y  : model flux array
        p  : array of parameters for this model
        --------------------------------------------------------
        """
        midval = x[x.size // 2]
        deltaF = np.mean(x[1:]-x[:-1])
        kvals = (x - midval) / deltaF
        # Determine which apodization function to use
        kind = self._keywd['kind']
        # Calculate the apodization function
        if kind == 'uniform':
            instfunc = np.sinc(kvals)
        elif kind == 'hanning':
            instfunc = (np.sinc(kvals) + 0.5 * np.sinc(kvals - 1.0) + 0.5 * np.sinc(kvals + 1.0))
        elif kind == 'hamming':
            fact = (27 - 16 * (kvals / 2.0) ** 2) / 25
            instfunc = fact * (np.sinc(kvals) + 0.5 * np.sinc(kvals - 1.0) + 0.5 * np.sinc(kvals + 1.0))
        elif kind == 'bartlett':
            instfunc = np.sinc(kvals / 2.0) ** 2
        elif kind == 'blackmann':
            fact = (21 - 9 * (kvals / 2.0) ** 2) / 25
            instfunc = (fact / (1.0 - (kvals / 2.0) ** 2)) * (
                        np.sinc(kvals) + 0.5 * np.sinc(kvals - 1.0) + 0.5 * np.sinc(kvals + 1.0))
        elif kind == 'welch':
            pk = np.pi * kvals
            instfunc = 4.0 * (np.sin(pk) - pk * np.cos(pk)) / (2.0 * pk ** 3)
            ww = np.where(pk == 0.0)
            if ww[0].size != 0:
                instfunc[ww] = 0.5 * (instfunc[ww[0] - 1] + instfunc[ww[0] + 1])
        elif kind == 'new':
            pk = np.pi * kvals
            instfunc = np.cos(pk) / pk ** 2
            ww = np.where(pk == 0.0)
            if ww[0].size != 0:
                instfunc[ww] = 0.5 * (instfunc[ww[0] - 1] + instfunc[ww[0] + 1])
        else:
            return y
        # Convolve the data with the instrument function
        yb = convolve(y, instfunc, mode='same', method='fft')
        return yb

    def getfwhm(self):
        kind = self._keywd['kind']
        if kind == 'uniform':
            return 1.20671
        elif kind == 'hanning':
            return 2.0
        elif kind == 'hamming':
            return 1.81522
        elif kind == 'bartlett':
            return 1.77179
        elif kind == 'blackmann':
            return 2.29880
        elif kind == 'welch':
            return 1.59044
        elif kind == 'new':
            return 0.0

    def getminmax(self, par, fitrng, Nsig=20.0):
        """
        This definition is only used for specifying the
        FWHM Resolution of the data.
        --------------------------------------------------------
        This definition will return the additional wavelength range
        of the data to be extracted around the user-speficied fitrange
        to ensure the edges of the model near the min and max of
        fitrange aren't affected.
        --------------------------------------------------------
        par    : The input parameters which defines the FWHM of this
                 function
        fitrng : The fitrange specified by the user at input
        Nsig   : Width in number of sigma to extract either side of
                 fitrange
        """
        # Use the parameters to now calculate the sigma width
        frac = 0.1
        # Calculate the min and max extraction wavelengths
        wmin = fitrng[0]*(1.0 - frac)
        wmax = fitrng[1]*(1.0 + frac)
        return wmin, wmax

    def load(self, instr, cntr, mp, specid, forcefix=False):
        """
        Load the parameters in the input model file
        --------------------------------------------------------
        instr: input string for the parameters and keywords of
               model to be loaded (ignoring the identifier at
               the beginning of the model line).
        cntr : The line number of the model (e.g. if it's the
               first model line, cntr=0)
        mp   : modpass --> A dictionary with the details of all
               models read in so far.
        --------------------------------------------------------
        Nothing should be changed here when writing a new function.
        --------------------------------------------------------
        """
        def check_tied_param(ival, cntr, mps, iind):
            havtie = False
            tieval=ival.lstrip('+-.0123456789')
            if tieval[0:2] in ['E+', 'e+', 'E-', 'e-']: # Scientific Notation is used.
                tieval=tieval[2:].lstrip('.0123456789')
            inval=float(ival.rstrip(tieval))
            if len(tieval) == 0: # Parameter is not tied
                mps['mtie'][cntr].append(-1)
                if forcefix:
                    mps['mfix'][cntr].append(1)
                else:
                    mps['mfix'][cntr].append(0)
            else: # parameter is tied
                # Determine if this parameter is fixed
                if tieval[0].isupper() or forcefix: mps['mfix'][cntr].append(1)
                else: mps['mfix'][cntr].append(0)
                # Check if this tieval has been used before
                if len(mps['tpar']) == 0: # If it's the first known tied parameter in the model
                    mps['tpar'].append([])
                    mps['tpar'][0].append(tieval)
                    mps['tpar'][0].append(len(mps['p0']))
                    mps['mtie'][cntr].append(-1) # i.e. not tied to anything
                else:
                    for j in range(0,len(mps['tpar'])):
                        if mps['tpar'][j][0] == tieval:
                            mps['mtie'][cntr].append(j)
                            havtie = True
                    if havtie == False: # create a New tied parameter
                        mps['tpar'].append([])
                        mps['tpar'][-1].append(tieval)
                        mps['tpar'][-1].append(len(mps['p0']))
                        mps['mtie'][cntr].append(-1) # i.e. not tied to anything
            if havtie == False: mps['p0'].append(inval)
            mps['mpar'][cntr].append(inval)
            mps['mlim'][cntr].append([self._limits[iind][i] if self._limited[iind][i]==1 else None for i in range(2)])
            return mps
        ################
        isspl=instr.split()
        # Seperate the parameters from the keywords
        kywrd = []
        keywdk = list(self._keywd.keys())
        keywdk[:] = (kych for kych in keywdk if kych[:] != 'input') # Remove the keyword 'input'
        param = [None for all in range(self._pnumr)]
        parid = [i for i in range(self._pnumr)]
        for i in range(len(isspl)):
            if "=" in isspl[i]:
                kwspl = isspl[i].split('=')
                if kwspl[0] in keywdk:
                    self._keywd['input'][kwspl[0]]=1
                    kywrd.append(isspl[i])
                else: msgs.error("Keyword '"+isspl[i]+"' is unknown for -"+msgs.newline()+self._idstr+"   "+instr)
            else:
                param[i] = isspl[i]
                self._keywd['input'][self._parid[i]]=1
        # Do some quick checks
        if len(param) != self._pnumr:
            msgs.error("Incorrect number of parameters (should be "+str(self._pnumr)+"):"+msgs.newline()+self._idstr+"   "+instr)
        # Set the parameters:
        mp['mtyp'].append(self._idstr)
        mp['mpar'].append([])
        mp['mtie'].append([])
        mp['mfix'].append([])
        mp['mlim'].append([])
        for i in range(self._pnumr):
            mp = check_tied_param(param[i], cntr, mp, i)
        # Now load the keywords:
        for i in range(len(kywrd)):
            kwspl = kywrd[i].split('=')
            ksspl = kwspl[1].split(',')
            for j in range(len(ksspl)):
                if type(self._keywd[kwspl[0]]) is int:
                    typeval='integer'
                    self._keywd[kwspl[0]] = int(kwspl[1])
                elif type(self._keywd[kwspl[0]]) is str:
                    typeval='string'
                    self._keywd[kwspl[0]] = kwspl[1]
                elif type(self._keywd[kwspl[0]]) is float:
                    typeval='float'
                    self._keywd[kwspl[0]] = float(kwspl[1])
                elif type(self._keywd[kwspl[0]]) is list:
                    typeval='list'
                    self._keywd[kwspl[0]].append(kwspl[1])
                elif type(self._keywd[kwspl[0]]) is bool:
                    if kwspl[1] in ['True', 'False']:
                        typeval='boolean'
                        self._keywd[kwspl[0]] = kwspl[1] in ['True']
                    else:
                        typeval='string'
                        self._keywd[kwspl[0]] = kwspl[1]
                        msgs.warn(kwspl[0]+" should be of type boolean (True/False)", verbose=self._verbose)
                elif self._keywd[kwspl[0]] is None:
                    typeval='None'
                    self._keywd[kwspl[0]] = None
                else:
                    msgs.error("I don't understand the format on line:"+msgs.newline()+self._idstr+"   "+instr)
            self._keych[kwspl[0]] = 0 # Set keych for this keyword to zero to show that this has been changed
        # Check that all required keywords were changed
        for i in range(len(keywdk)):
            if self._keych[keywdk[i]] == 1: msgs.error(keywdk[i]+" must be set for -"+msgs.newline()+self._idstr+"   "+instr)
        # Append the final set of keywords
        mp['mkey'].append(self._keywd.copy())
        return mp, parid

    def parin(self, i, par):
        """
        This routine converts a parameter in the input model file
        to the parameter used in 'call'
        --------------------------------------------------------
        When writing a new function, one should change how each
        input parameter 'par' is converted into a parameter used
        in the function specified by 'call'
        --------------------------------------------------------
        """
        if   i == 0: pin = par
        return pin

    def set_vars(self, p, level, mp, ival, wvrng=[0.0,0.0], spid='None', levid=None, nexbin=None, ddpid=None, getinfl=False, getstdd=None):
        """
        Return the parameters for a Gaussian function to be used by 'call'
        The only thing that should be changed here is the parb values
        """
        levadd=0
        params=np.zeros(self._pnumr)
        parinf=[]
        for i in range(self._pnumr):
            lnkprm = None
            if mp['mtie'][ival][i] >= 0:
                getid = mp['tpar'][mp['mtie'][ival][i]][1]
            elif mp['mtie'][ival][i] <= -2:
                if len(mp['mlnk']) == 0:
                    lnkprm = mp['mpar'][ival][i]
                else:
                    for j in range(len(mp['mlnk'])):
                        if mp['mlnk'][j][0] == mp['mtie'][ival][i]:
                            cmd = 'lnkprm = ' + mp['mlnk'][j][1]
                            exec(cmd)
                levadd += 1
            else:
                getid = level+levadd
                levadd+=1
            if lnkprm is None:
                params[i] = self.parin(i, p[getid])
                if mp['mfix'][ival][i] == 0: parinf.append(level+levadd) # If parameter not fixed, append it to the influence array
            else:
                params[i] = lnkprm
        if ddpid is not None:
            if ddpid not in parinf: return []
        if nexbin is not None:
            if params[0] == 0: return params, 1
            if nexbin[0] == "km/s": msgs.error("bintype is set to 'km/s', when FWHM is specified in Hz.")
            elif nexbin[0] == "A" : msgs.error("bintype is set to 'A', when FWHM is specified in Hz.")
            elif nexbin[0] == "Hz" : return params, nexbin[1]
            else: msgs.bug("bintype "+nexbin[0]+" should not have been specified in model function: "+self._idstr, verbose=self._verbose)
        elif getstdd is not None:
            fact = 2.0*np.sqrt(2.0*np.log(2.0))
            return getstdd[1]*(1.0+getstdd[0]*self.getfwhm()/fact), getstdd[2]*(1.0-getstdd[0]*self.getfwhm()/fact)
        elif getinfl: return params, parinf
        else: return params

