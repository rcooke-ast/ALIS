import numpy as np
from alis import almsgs
from alis import alfunc_base
msgs=almsgs.msgs()

class MultiVFWHM(alfunc_base.Base) :
    """
    Convolves the spectrum with a sum of Gaussians with various full-width
    at half-maximum values, relative offsets. An arbitrary number of Gaussians
    can be used. The number of parameters is 2*n+1, where n is the number of
    Gaussians. The first parameter is the vFWHM of the main component. If there
    is just one Gaussian, the spectrum is convolved with a Gaussian with a
    full-width at half-maximum of vFWHM (This will be the same as the alfunc_vfwhm
    function). There are three required parameters for each additional Gaussian:
    the relative offset of the Gaussian from the main component, the relative
    amplitude of the Gaussian, and the full-width at half-maximum of the Gaussian.
    Thus, the user must specify 3*n+1 parameters, where n is the number of Gaussians.
    Note that the relative offset and FWHM are expressed as velocity in km/s.
    """
    def __init__(self, prgname="", getinst=False, atomic=None, verbose=2):
        self._idstr   = 'multivfwhm'			# ID string for this class
        self._pnumr   = 1				# Total number of parameters fed in
        self._keywd   = dict({'blind':False, 'blindseed':0,  'blindrange':[]})		# Additional arguments to describe the model --- 'input' cannot be used as a keyword
        self._keych   = dict({'blind':0, 'blindseed':0,  'blindrange':0})			# Require keywd to be changed (1 for yes, 0 for no)
        self._keyfm   = dict({'blind':"", 'blindseed':"",  'blindrange':""})			# Require keywd to be changed (1 for yes, 0 for no)
        self._parid   = ['vfwhm0']		# Name of each parameter
        self._defpar  = [ 0.0 ]			# Default values for parameters that are not provided
        self._fixpar  = [ None ]		# By default, should these parameters be fixed?
        self._limited = [ [1  ,0  ] ]	# Should any of these parameters be limited from below or above
        self._limits  = [ [0.0,0.0] ]	# What should these limiting values be
        self._svfmt   = [ "{0:.3g}" ]	# Specify the format used to print or save output
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
        vFWHM = p[0]  # This is the main component
        multipar = p[1:].reshape((-1,3))
        ngauss = multipar.shape[0]
        # Setup the arrays required for the convolution
        sigd = p[0] / ( 2.99792458E5 * ( 2.0*np.sqrt(2.0*np.log(2.0)) ) )
        ysize = y.size
        fsigd = 6.0*sigd
        dwav = 0.5*(x[2:]-x[:-2])/x[1:-1]
        dwav = np.append(np.append(dwav[0],dwav),dwav[-1])
        df= int(np.min([int(np.ceil(fsigd/dwav).max()), ysize//2 - 1]))
        yval = np.zeros(2*df+1)
        yval[:2*df+1] = (x[:2*df+1]/x[df] - 1.0)/sigd
        # Construct the multi-Gaussian convolution kernel
        gaus = np.exp(-0.5*yval*yval)
        for i in range(ngauss):
            ampl, offs, vFWHM = multipar[i,0], multipar[i,1], multipar[i,2]
            if vFWHM == 0.0:
                continue
            sigd = vFWHM / ( 2.99792458E5 * ( 2.0*np.sqrt(2.0*np.log(2.0)) ) )  # Convert to sigma in Angstroms
            offsA = x[df]*(offs/2.99792458E5)  # Convert to offset in Angstroms
            yval = ((x+offsA)/x[df]-1.0)/sigd
            gaus += ampl*np.exp(-0.5*yval*yval)
        # Perform the convolution
        size = ysize + gaus.size - 1
        fsize = 2 ** int(np.ceil(np.log2(size))) # Use this size for a more efficient computation
        conv = np.fft.fft(y, fsize)
        conv *= np.fft.fft(gaus/gaus.sum(), fsize)
        ret = np.fft.ifft(conv).real.copy()
        del conv
        return ret[df:df+ysize]

    def getminmax(self, par, fitrng, Nsig=10.0):
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
        # Convert the input parameters to the parameters used in call
        pin = [0.0 for all in par]
        for i in range(len(par)):
            tval=par[i].lstrip('+-.0123456789')
            if tval[0:2] in ['E+', 'e+', 'E-', 'e-']: # Scientific Notation is used.
                tval=tval[2:].lstrip('.0123456789')
            parsnd=float(par[i].rstrip(tval))
            pin[i] = self.parin(i, parsnd)
        # Use the parameters to now calculate the sigma width
        sigd = pin[0] / ( 2.99792458E5 * ( 2.0*np.sqrt(2.0*np.log(2.0)) ) )
        # Calculate the min and max extraction wavelengths
        wmin = fitrng[0]*(1.0 - Nsig*sigd)
        wmax = fitrng[1]*(1.0 + Nsig*sigd)
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
        # Separate the parameters from the keywords
        ptemp, kywrd = [], []
        keywdk = list(self._keywd.keys())
        keywdk[:] = (kych for kych in keywdk if kych[:] != 'input') # Remove the keyword 'input'
        numink = 0
        # Since there's no way of knowing how many coefficients and
        # keywords there are, determine the number of parameters
        pnumr = 0
        for i in range(len(isspl)):
            if "=" in isspl[i]:
                if isspl[i].split('=')[0] == self._parid[0]: pnumr += 1
            else: pnumr += 1
        # Check that the number of parameters is correct
        if pnumr == 1:
            msgs.warn("Only one parameter found in the model. You could use the standard vfwhm function instead.")
        elif (pnumr-1) % 3 != 0:
            msgs.error("The number of parameters in the model is incorrect. Please check the input model." +
                       msgs.newline() + "The number of parameters should be 1 + 3n, where n is the number of Gaussians.")
        # Now return to the default code
        param = [None for all in range(pnumr)]
        parid = [i for i in range(pnumr)]
        for i in range(len(isspl)):
            if "=" in isspl[i]:
                kwspl = isspl[i].split('=')
                if kwspl[0] in self._parid:
                    self._keywd['input'][kwspl[0]] = 2
                    for j in range(pnumr):
                        if kwspl[0] == self._parid[0]:
                            param[j] = kwspl[1]
                            numink += 1
                            break
                elif kwspl[0] in keywdk:
                    kywrd.append(isspl[i])
                    self._keywd['input'][kwspl[0]] = 2
                else:
                    msgs.error("Keyword '" + isspl[i] + "' is unknown for -" + msgs.newline() + self._idstr + "   " + instr)
            else:
                ptemp.append(isspl[i])
        # Set parameters that weren't given in param
        cntp=0
        fixpid = [0 for all in range(pnumr)]
        for i in range(pnumr):
            if param[i] is None:
                if cntp >= len(ptemp): # Use default values
                    param[i] = str(self._defpar[0])
                    fixpid[i] = 1
                else:
                    param[i] = ptemp[cntp]
                    self._keywd['input'][self._parid[0]]=1
                    cntp += 1
        # Do some quick checks
        if len(ptemp)+numink > pnumr:
            msgs.error("Incorrect number of parameters (should be "+str(pnumr)+"):"+msgs.newline()+self._idstr+"   "+instr)
        if len(specid) > 1: # Force the user to specify a spectrum ID number
            self._keych['specid'] = 1
        ngauss = (pnumr-1) // 3
        specidset=False # Detect is the user has specified specid
        # Set the parameters:
        mp['mtyp'].append(self._idstr)
        mp['mpar'].append([])
        mp['mtie'].append([])
        mp['mfix'].append([])
        mp['mlim'].append([])
        for i in range(pnumr):
            mp = check_tied_param(param[i], cntr, mp, i)
        if any(fixpid): # If default values had to be used, fix the values
            for i in range(pnumr):
                if fixpid[i] == 1: mp['mfix'][-1][i] = 1
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
        # Before returning, update the parameters of the model
        for i in range(ngauss):
            #ampl, offs, vFWHM = multipar[i,0], multipar[i,1], multipar[i,2]
            # The amplitude is the first parameter
            self._parid.append('amplitude{0:d}'.format(i+1))
            self._defpar.append(0.0)
            self._fixpar.append(None)
            self._limited.append([1, 1])
            self._limits.append([0.0,1.0])
            # The offset is the second parameter
            self._parid.append('offset{0:d}'.format(i+1))
            self._defpar.append(0.0)
            self._fixpar.append(None)
            self._limited.append([0, 0])
            self._limits.append([0.0,0.0])
            # The FWHM is the third parameter
            self._parid.append('vfwhm{0:d}'.format(i+1))
            self._defpar.append(0.0)
            self._fixpar.append(None)
            self._limited.append([1, 0])
            self._limits.append([0.0,0.0])
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
        return par


    def set_pinfo(self, pinfo, level, mp, lnk, mnum):
        """
        Place limits on the functions parameters (as specified in init)
        Nothing should be changed here.
        """
        pnumr = len(mp['mpar'][mnum])
        add = pnumr
        levadd = 0
        for i in range(pnumr):
            # First Check if there are any operations/links to perform on this parameter
            if len(lnk['opA']) != 0:
                breakit=False
                for j in range(len(mp['tpar'])):
                    if breakit: break
                    if mp['tpar'][j][1] == level+levadd: # then this is a tied parameter
                        # Check if this tied parameter is to be linked to another parameter
                        for k in range(len(lnk['opA'])):
                            if mp['tpar'][j][0] == lnk['opA'][k]:
                                ttext = lnk['exp'][k]
                                for l in lnk['opB'][k]:
                                    for m in range(len(mp['tpar'])):
                                        if mp['tpar'][m][0] == l:
                                            pn = mp['tpar'][m][1]
                                            break
                                    ttext = ttext.replace(l,'p[{0:d}]'.format(pn))
                                pinfo[level+levadd]['tied'] = ttext
                                breakit = True
                                break
            # Now set limits and fixed values
            if mp['mtie'][mnum][i] >= 0: add -= 1
            elif mp['mtie'][mnum][i] <= -2:
                pinfo[level+levadd]['limited'] = [0 if j is None else 1 for j in mp['mlim'][mnum][i]]
                pinfo[level+levadd]['limits']  = [0.0 if j is None else float(j) for j in mp['mlim'][mnum][i]]
                mp['mfix'][mnum][i] = -1
                levadd += 1
            else:
                pinfo[level+levadd]['limited'] = [0 if j is None else 1 for j in mp['mlim'][mnum][i]]
                pinfo[level+levadd]['limits']  = [0.0 if j is None else float(j) for j in mp['mlim'][mnum][i]]
                pinfo[level+levadd]['fixed']   = mp['mfix'][mnum][i]
                levadd += 1
        return pinfo, add

    def set_vars(self, p, level, mp, ival, wvrng=[0.0,0.0], spid='None', levid=None, nexbin=None, getinfl=False, ddpid=None, getstdd=None):
        """
        Return the parameters for a Gaussian function to be used by 'call'
        The only thing that should be changed here is the parb values
        """
        pnumr = len(mp['mpar'][ival])
        levadd=0
        params=np.zeros(pnumr)
        parinf=[]
        for i in range(pnumr):
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
                            namespace = dict({'p': p})
                            exec(cmd, namespace)
                            lnkprm = namespace['lnkprm']
                levadd += 1
            else:
                getid = level+levadd
                levadd+=1
            if lnkprm is None:
                params[i] = self.parin(i, p[getid])
                if mp['mfix'][ival][i] == 0: parinf.append(getid) # If parameter not fixed, append it to the influence array
            else:
                params[i] = lnkprm
        if ddpid is not None:
            if ddpid not in parinf: return []
        minwid = params[0]
        for gg in range(ngauss):
            if (params[3 + gg * 3] < minwid) and params[3 + gg * 3] != 0: minwid = params[3 + gg * 3]
        if nexbin is not None:
            if minwid == 0: return params, 1
            if nexbin[0] == "km/s":
                return params, int(round(2.0*np.sqrt(2.0*np.log(2.0))*nexbin[1]/minwid + 0.5))
            elif nexbin[0] == "A" : msgs.error("bintype is set to 'A' for Angstroms, when FWHM is specified as a velocity.")
            else: msgs.bug("bintype "+nexbin[0]+" should not have been specified in model function: "+self._idstr, verbose=self._verbose)
        elif getstdd is not None:
            fact = 2.0*np.sqrt(2.0*np.log(2.0))
            return getstdd[1]*(1.0+getstdd[0]*minwid/(fact*299792.458)), getstdd[2]*(1.0-getstdd[0]*minwid/(fact*299792.458))
        elif getinfl: return params, parinf
        else: return params

