import os
import numpy as np
from alis import almsgs
from alis import alfunc_base
from scipy.interpolate import interp1d
from IPython import embed
#import pycuda.driver as cuda
#import pycuda.autoinit
#from pycuda.compiler import SourceModule
msgs=almsgs.msgs()

class SplineLSF(alfunc_base.Base) :
    """
    Returns a Spline version of the Line Spread Function
    """
    def __init__(self, prgname="", getinst=False, atomic=None, verbose=2):
        self._idstr   = 'lsfspline'																				# ID string for this class
        self._pnumr   = 1																					# Total number of parameters fed in
        self._keywd   = dict({'blind':False, 'locations':[], 'blindseed':0,  'blindrange':[]})		# Additional arguments to describe the model --- 'input' cannot be used as a keyword
        self._keych   = dict({'blind':0,     'locations':1,  'blindseed':0,  'blindrange':0})			# Require keywd to be changed (1 for yes, 0 for no)
        self._keyfm   = dict({'blind':"",    'locations':"", 'blindseed':"", 'blindrange':""})		# Format for the keyword. "" is the Default setting
        self._parid   = ['vfwhm0', 'tau']		# Name of each parameter
        self._defpar  = [ 0.0 , 1.0]				# Default values for parameters that are not provided
        self._fixpar  = [ None,        None ]				# By default, should these parameters be fixed?
        self._limited = [ [1  ,0  ], [1, 1] ]		# Should any of these parameters be limited from below or above
        self._limits  = [ [0.0,0.0], [0.0,1.0] ]		# What should these limiting values be
        self._svfmt   = [ "{0:.7g}", "{0:.7g}"]			# Specify the format used to print or save output
        self._prekw   = []																			# Specify the keywords to print out before the parameters
        # DON'T CHANGE THE FOLLOWING --- it tells ALIS what parameters are provided by the user.
        tempinput = self._parid+list(self._keych.keys())                             #
        self._keywd['input'] = dict(zip((tempinput),([0]*np.size(tempinput)))) #
        ########################################################################
        self._verbose = verbose
        # Set the atomic data
        self._atomic = atomic
        #self._kernal  = self.GPU_kernal()										# Get the Source Module for the GPU
        if getinst: return

    def call_CPU(self, x, y, par, mkey=None, ncpus=1):
        """
        Define the functional form of the model
        --------------------------------------------------------
        x  : array of wavelengths
        y  : model flux array
        par  : array of parameters for this model
        --------------------------------------------------------
        """
        FWHM_to_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))
        vFWHM = par[0]  # This is the smoothing component in km/s
        # Setup the arrays required for the convolution
        sigd = vFWHM / (2.99792458E5 * FWHM_to_SIGMA)
        fsigd = 6.0 * sigd
        # Calculate the number of points required for the convolution
        ysize = y.size
        dwav = 0.5 * (x[2:] - x[:-2]) / x[1:-1]
        dwav = np.append(np.append(dwav[0], dwav), dwav[-1])
        df = int(np.min([int(np.ceil(fsigd / dwav).max()), ysize // 2 - 1]))
        #########################################
        # Construct the spline convolution kernel
        #########################################
        # Get the locations of the spline points
        nspl = (len(par) - 1)
        yspl = par[1:1 + nspl]
        xspl = np.array(mkey['locations'])
        if xspl.size != yspl.size:
            msgs.error("'locations' parameter must have same length as input parameters in spline function")
        splev = interp1d(xspl, yspl, kind=1, bounds_error=False, fill_value=0)
        splx = 299792.458 * (x - x[x.size//2]) / x[x.size//2]
        smth = splev(splx)
        iysize = smth.size
        idf = int(iysize / 2 - 1)
        if vFWHM > 0.0:
            # Smooth by the b-value
            isigd = (vFWHM / np.sqrt(2)) / 2.99792458E5
            ifsigd = 6.0 * isigd
            idwav = 0.5 * (x[2:] - x[:-2]) / x[1:-1]
            idwav = np.append(np.append(idwav[0], idwav), idwav[-1])
            # idf = int(np.min([int(np.ceil(ifsigd / idwav).max()), iysize / 2 - 1]))
            iyval = np.zeros(2 * idf + 1)
            iyval[idf:2 * idf + 1] = (x[idf:2 * idf + 1] / x[idf] - 1.0) / isigd
            iyval[:idf] = (x[:idf] / x[idf] - 1.0) / isigd
            igaus = np.exp(-0.5 * iyval * iyval)
            isize = iysize + igaus.size - 1
            ifsize = 2 ** int(np.ceil(np.log2(isize)))  # Use this size for a more efficient computation
            iconv = np.fft.fft(modret, ifsize)
            iconv *= np.fft.fft(igaus / igaus.sum(), ifsize)
            iret = np.fft.ifft(iconv).real.copy()
            del iconv
            smth = iret[idf:idf + iysize]

        # Perform the convolution
        size = ysize + iysize - 1
        fsize = 2 ** int(np.ceil(np.log2(size)))  # Use this size for a more efficient computation
        conv = np.fft.fft(y, fsize)
        conv *= np.fft.fft(smth / smth.sum(), fsize)
        ret = np.fft.ifft(conv).real.copy()
        del conv
        return ret[idf:idf + ysize]

    def load(self, instr, cntr, mp, specid):
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
        def check_tied_param(ival, cntr, mps, iind, parj):
            havtie = False
            tieval=ival.lstrip('-+.0123456789')
            if tieval[0:2] in ['E+', 'e+', 'E-', 'e-']: # Scientific Notation is used.
                tieval=tieval[2:].lstrip('.0123456789')
            try:
                inval=float(ival.rstrip(tieval))
            except:
                msgs.error("This is not allowed for model input: "+ival.rstrip(tieval))
            if len(tieval) == 0: # Parameter is not tied
                mps['mtie'][cntr].append(-1)
                mps['mfix'][cntr].append(0)
            else: # parameter is tied
                # Determine if this parameter is fixed
                if tieval[0].isupper(): mps['mfix'][cntr].append(1)
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
            mps['mlim'][cntr].append([self._limits[parj][i] if self._limited[parj][i]==1 else None for i in range(2)])
            return mps
        ################
        instr = instr.replace(":", "=")
        instrb = instr.split(";")  # Separate the parameters from the keywords
        isspl = instrb[0].split(",")
        # Seperate the parameters from the keywords
        ptemp, kywrd = [], instrb[1:]
        keywdk = list(self._keywd.keys())
        keywdk[:] = (kych for kych in keywdk if kych[:] != 'input') # Remove the keyword 'input'
        numink = 0  # Number of keywords in the input string
        # Since there's no way of knowing how many coefficients and
        # keywords there are, determine the number of parameters
        pnumr = len(isspl)
        # Setup the parameters
        param = [None for all in range(pnumr)]
        parid = [i for i in range(pnumr)]
        for pp in range(1,pnumr): parid[pp] = 1
        for i in range(len(isspl)):
            ptemp.append(isspl[i])
        # Set parameters that weren't given in param
        cntp = 0
        fixpid = [0 for all in range(pnumr)]
        for i in range(pnumr):
            if param[i] is None:
                param[i] = ptemp[cntp]
                if i >= 1:
                    self._keywd['input'][self._parid[0]] = 1
                else:
                    self._keywd['input'][self._parid[i]] = 1
                cntp += 1
        # Do some quick checks
        if len(ptemp)+numink > pnumr:
            msgs.error("Incorrect number of parameters (should be "+str(pnumr)+"):"+msgs.newline()+self._idstr+"   "+instr)
        if len(specid) > 1: # Force the user to specify a spectrum ID number
            self._keych['specid'] = 1
        specidset = False  # Detect is the user has specified specid
        # Set the parameters:
        mp['mtyp'].append(self._idstr)
        mp['mpar'].append([])
        mp['mtie'].append([])
        mp['mfix'].append([])
        mp['mlim'].append([])
        for i in range(pnumr):
            mp = check_tied_param(param[i], cntr, mp, i, parid[i])
        # Now load the keywords:
        # Make a copy of the default values
        cpy_keywd = self._keywd.copy()
        cpy_keych = self._keych.copy()
        for i in range(len(kywrd)):
            kwspl = kywrd[i].split('=')
            ksspl = kwspl[1].lstrip('([').rstrip(')]').split(',')
            if kwspl[0] == 'locations':
                scllist = []
                if len(ksspl) != pnumr-1:
                    msgs.error("Keyword 'locations' for function '{0:s}', must contain the same number".format(self._idstr)+msgs.newline()+"of elements as the number of specified parameters")
                for j in range(len(ksspl)): # Do some checks
                    try:
                        scllist.append(float(ksspl[j]))
                    except:
                        msgs.error("Keyword 'locations' in function '{0:s}' should be an array of floats, not:".format(self._idstr)+msgs.newline()+"{0:s}".format(ksspl[j]))
            for j in range(len(ksspl)):
                if type(cpy_keywd[kwspl[0]]) is int:
                    typeval='integer'
                    cpy_keywd[kwspl[0]] = int(kwspl[1])
                elif type(cpy_keywd[kwspl[0]]) is str:
                    typeval='string'
                    cpy_keywd[kwspl[0]] = kwspl[1]
                elif type(cpy_keywd[kwspl[0]]) is float:
                    typeval='float'
                    cpy_keywd[kwspl[0]] = float(kwspl[1])
                elif type(cpy_keywd[kwspl[0]]) is list and kwspl[0] == 'specid':
                    typeval='list'
                    cpy_keywd[kwspl[0]] = sidlist
                elif type(cpy_keywd[kwspl[0]]) is list and kwspl[0] == 'locations':
                    typeval = 'list'
                    cpy_keywd[kwspl[0]] = scllist
                elif type(cpy_keywd[kwspl[0]]) is bool:
                    if kwspl[1] in ['True', 'False']:
                        typeval='boolean'
                        cpy_keywd[kwspl[0]] = kwspl[1] in ['True']
                    else:
                        typeval='string'
                        cpy_keywd[kwspl[0]] = kwspl[1]
                        msgs.warn(kwspl[0]+" should be of type boolean (True/False)", verbose=self._verbose)
                elif cpy_keywd[kwspl[0]] is None:
                    typeval='None'
                    cpy_keywd[kwspl[0]] = None
                else:
                    msgs.error("I don't understand the format on line:"+msgs.newline()+self._idstr+"   "+instr)
            cpy_keych[kwspl[0]] = 0 # Set keych for this keyword to zero to show that this has been changed
        # Check that all required keywords were changed
        for i in range(len(keywdk)):
            if cpy_keych[keywdk[i]] == 1 and pnumr != 1: # Only check locations if it's not a column density ratio (i.e. when pnumr==1)
                msgs.error(keywdk[i]+" must be set for -"+msgs.newline()+self._idstr+"   "+instr)
        # Check that we have set a specid
        if len(specid) == 1 and not specidset:
            if len(cpy_keywd['specid']) == 0: cpy_keywd['specid'].append(specid[0])
        # Append the final set of keywords
        mp['mkey'].append(cpy_keywd)
        # Before returning, update the parameters of the model
        for i in range(pnumr-self._pnumr):
            self._parid.append("tau")
            self._defpar.append(0.0)
            self._fixpar.append(None)
            self._limited.append([1, 0])
            self._limits.append([0.0, 0.0])
        return mp, parid

    def parin(self, i, par, parb):
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

    def set_vars(self, p, level, mp, ival, wvrng=[0.0,0.0], spid='None', levid=None, nexbin=None, ddpid=None, getinfl=False):
        """
        Return the parameters for a Voigt function to be used by 'call'
        """
        pnumr = len(mp['mpar'][ival])
        params = np.zeros(pnumr)
        parinf = []
        levadd=0
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
                            namespace = dict({'p':p, 'np':np})
                            exec(cmd, namespace)
                            lnkprm = namespace['lnkprm']
                levadd += 1
            else:
                getid = level+levadd
                levadd+=1
            if lnkprm is None:
                params[i] = self.parin(i, p[getid], None)
                if mp['mfix'][ival][i] == 0: parinf.append(getid)
            else:
                params[i] = lnkprm
        if ddpid is not None:
            if ddpid not in parinf: return []
        if nexbin is not None:
            if params[0] == 0.0: return params, 1
            if nexbin[0] == "km/s": return params, int(round(np.sqrt(2.0)*nexbin[1]/params[0] + 0.5))
            elif nexbin[0] == "A" : msgs.error("bintype is set to 'A' for Angstroms, when FWHM is specified as a velocity.")
            elif nexbin[0] == "Hz" : msgs.error("bintype is set to 'Hz' for Hertz, when FWHM is specified as a velocity.")
            else:
                msgs.bug("bintype "+nexbin[0]+" should not have been specified in model function: "+self._idstr, verbose=self._verbose)
                msgs.error("Cannot proceed until this bug is fixed")
        elif getinfl: return params, parinf
        else: return params

    def parout(self, params, mp, istart, level, errs=None, reletter=False, conv=None):
        """
        Convert the parameter list to the input form for
        printing to screen or output to a file.
        --------------------------------------------------------
        pval     : List of all parameters
        mp       : modpass, which includes details of all models
        istart   : Model number
        errs     : List of all parameter errors.
        reletter : Set to true if you want to reletter the tied
                   and fixed parameters (a will be used for the
                   first parameter, b is used for the second...)
        conv     : If convergence test is being written, conv is
                   the threshold for convergence (in sigma's).
        --------------------------------------------------------
        Nothing should be changed here when writing a new function.

        I changed several self._svfmt[i] to self._svfmt[0].
        --------------------------------------------------------
        """
        if errs is None:
            errors = params
        else:
            errors = errs
        pnumr = len(mp['mpar'][istart])
        add = pnumr
        havtie = 0
        tienum = 0
        levadd = 0
        outstring = ['  %s ' % (self._idstr)]
        errstring = ['# %s ' % (self._idstr)]

        # Check if we are blinding any parameters with an offset value
        blindoffset = 0
        if 'blindrange' in mp['mkey'][istart]:
            if len(mp['mkey'][istart]['blindrange']) == 2:
                if 'blindseed' in mp['mkey'][istart]:
                    np.random.seed(mp['mkey'][istart]['blindseed'])
                else:
                    np.random.seed(0)
                blindoffset = np.random.uniform(int(mp['mkey'][istart]['blindrange'][0]),
                                                int(mp['mkey'][istart]['blindrange'][1]))
            # Reset the seed
            self.resetseed()

        for i in range(pnumr):
            if mp['mkey'][istart]['input'][self._parid[0]] == 0:  # Parameter not given as input
                outstring.append("")
                errstring.append("")
                continue
            elif mp['mkey'][istart]['input'][self._parid[0]] == 1:
                pretxt = ""  # Parameter is given as input, without parid
            else:
                pretxt = self._parid[0] + "="  # Parameter is given as input, with parid
            if mp['mtie'][istart][i] >= 0:
                if reletter:
                    newfmt = pretxt + self.gtoef(params[mp['tpar'][mp['mtie'][istart][i]][1]],
                                                 self._svfmt[0] + '{1:c}')
                    outstring.append((newfmt).format(blindoffset + params[mp['tpar'][mp['mtie'][istart][i]][1]],
                                                     97 + mp['mtie'][istart][i] - 32 * mp['mfix'][istart][1]))
                    if conv is None:
                        errstring.append((newfmt).format(errors[mp['tpar'][mp['mtie'][istart][i]][1]],
                                                         97 + mp['mtie'][istart][i] - 32 * mp['mfix'][istart][1]))
                    else:
                        if params[mp['tpar'][mp['mtie'][istart][i]][1]] < conv:
                            cvtxt = "CONVERGED"
                        else:
                            cvtxt = "!!!!!!!!!"
                        errstring.append(
                            ('--{0:s}--{1:c}    ').format(cvtxt, 97 + tienum - 32 * mp['mfix'][istart][1]))
                else:
                    newfmt = pretxt + self.gtoef(params[mp['tpar'][mp['mtie'][istart][i]][1]],
                                                 self._svfmt[0] + '{1:s}')
                    outstring.append((newfmt).format(blindoffset + params[mp['tpar'][mp['mtie'][istart][i]][1]],
                                                     mp['tpar'][mp['mtie'][istart][i]][0]))
                    if conv is None:
                        errstring.append((newfmt).format(errors[mp['tpar'][mp['mtie'][istart][i]][1]],
                                                         mp['tpar'][mp['mtie'][istart][i]][0]))
                    else:
                        if params[mp['tpar'][mp['mtie'][istart][i]][1]] < conv:
                            cvtxt = "CONVERGED"
                        else:
                            cvtxt = "!!!!!!!!!"
                        errstring.append(('--{0:s}--{1:s}    ').format(cvtxt, mp['tpar'][mp['mtie'][istart][i]][0]))
                add -= 1
            else:
                if havtie != 2:
                    if havtie == 0:  # First searching for the very first instance of a tied parameter
                        for tn in range(0, len(mp['tpar'])):
                            if mp['tpar'][tn][1] == level + levadd:
                                tienum = tn
                                havtie = 1
                                break
                    if len(mp['tpar']) != 0:
                        if mp['tpar'][tienum][1] == level + levadd:
                            if reletter:
                                newfmt = pretxt + self.gtoef(params[level + levadd], self._svfmt[0] + '{1:c}')
                                outstring.append((newfmt).format(blindoffset + params[level + levadd],
                                                                 97 + tienum - 32 * mp['mfix'][istart][1]))
                                if conv is None:
                                    errstring.append((newfmt).format(errors[level + levadd],
                                                                     97 + tienum - 32 * mp['mfix'][istart][1]))
                                else:
                                    if params[level + levadd] < conv:
                                        cvtxt = "CONVERGED"
                                    else:
                                        cvtxt = "!!!!!!!!!"
                                    errstring.append(('--{0:s}--{1:c}    ').format(cvtxt, 97 + tienum - 32 *
                                                                                   mp['mfix'][istart][1]))
                            else:
                                newfmt = pretxt + self.gtoef(params[level + levadd], self._svfmt[0] + '{1:s}')
                                outstring.append(
                                    (newfmt).format(blindoffset + params[level + levadd], mp['tpar'][tienum][0]))
                                if conv is None:
                                    errstring.append((newfmt).format(errors[level + levadd], mp['tpar'][tienum][0]))
                                else:
                                    if params[level + levadd] < conv:
                                        cvtxt = "CONVERGED"
                                    else:
                                        cvtxt = "!!!!!!!!!"
                                    errstring.append(('--{0:s}--{1:s}    ').format(cvtxt, mp['tpar'][tienum][0]))
                            tienum += 1
                            if tienum == len(
                                mp['tpar']): havtie = 2  # Stop searching for 1st instance of tied param
                        else:
                            newfmt = pretxt + self.gtoef(params[level + levadd], self._svfmt[0])
                            outstring.append((newfmt).format(blindoffset + params[level + levadd]))
                            if conv is None:
                                errstring.append((newfmt).format(errors[level + levadd]))
                            else:
                                if params[level + levadd] < conv:
                                    cvtxt = "CONVERGED"
                                else:
                                    cvtxt = "!!!!!!!!!"
                                errstring.append(('--{0:s}--    ').format(cvtxt))
                    else:  # There are no tied parameters!
                        newfmt = pretxt + self.gtoef(params[level + levadd], self._svfmt[0])
                        outstring.append((newfmt).format(blindoffset + params[level + levadd]))
                        if conv is None:
                            errstring.append((newfmt).format(errors[level + levadd]))
                        else:
                            if params[level + levadd] < conv:
                                cvtxt = "CONVERGED"
                            else:
                                cvtxt = "!!!!!!!!!"
                            errstring.append(('--{0:s}--    ').format(cvtxt))
                else:
                    newfmt = pretxt + self.gtoef(params[level + levadd], self._svfmt[0])
                    outstring.append((newfmt).format(blindoffset + params[level + levadd]))
                    if conv is None:
                        errstring.append((newfmt).format(errors[level + levadd]))
                    else:
                        if params[level + levadd] < conv:
                            cvtxt = "CONVERGED"
                        else:
                            cvtxt = "!!!!!!!!!"
                        errstring.append(('--{0:s}--    ').format(cvtxt))
                levadd += 1
        level += add
        # Now write in the keywords
        keys = list(mp['mkey'][istart].keys())
        keys[:] = (kych for kych in keys if kych[:] != 'input')  # Remove the keyword 'input'
        for i in range(len(keys)):
            if mp['mkey'][istart]['input'][keys[i]] == 0:  # This keyword wasn't provided as user input
                outstring.append("")
                errstring.append("")
                continue
            if type(mp['mkey'][istart][keys[i]]) is list:
                outkw = ','.join(map(str, mp['mkey'][istart][keys[i]]))
            else:
                outkw = mp['mkey'][istart][keys[i]]
            if self._keyfm[keys[i]] != "":
                outstring.append(('{0:s}=' + self._keyfm[keys[i]]).format(keys[i], outkw))
                errstring.append(('{0:s}=' + self._keyfm[keys[i]]).format(keys[i], outkw))
            else:
                outstring.append('%s=%s' % (keys[i], outkw))
                errstring.append('%s=%s' % (keys[i], outkw))
        #                outstring.append( '{0:s}={1:s}'.format(keys[i],outkw) )
        #                errstring.append( '{0:s}={1:s}'.format(keys[i],outkw) )
        # Now place the keywords specified in self._prekw at the beginning of the return string:
        if len(self._prekw) != 0:
            insind = 1
            for i in range(len(self._prekw)):
                delind = -1
                for j in range(len(keys)):
                    if self._prekw[i] == keys[j]:
                        delind = self._pnumr + insind + j
                        del keys[j]
                        break
                if delind == -1: msgs.bug("prekw variable for function " + self._idstr + " contains bad argument",
                                          verbose=self._verbose)
                outstring.insert(insind, outstring[delind])
                errstring.insert(insind, errstring[delind])
                del outstring[delind + 1]
                del errstring[delind + 1]
                insind += 1
        if mp['mkey'][istart]['blind'] and conv is None:
            retout = "       ------ BLIND MODEL ------\n"
            # reterr = "       ------ BLIND MODEL ------\n"
            reterr = '  '.join(errstring) + '\n'
        else:
            retout = '  '.join(outstring) + '\n'
            reterr = '  '.join(errstring) + '\n'
        # Return the strings and the new level
        if errs is not None or conv is not None:
            return retout, reterr, level
        else:
            return retout, level

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
            if mp['mtie'][mnum][i] >= 0:
                add -= 1
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

