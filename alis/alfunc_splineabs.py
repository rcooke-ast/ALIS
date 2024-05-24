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

class SplineAbs(alfunc_base.Base) :
    """
    Returns a Voigt Profile of form:
    p[0] = log10 of the column density
    p[1] = redshift
    p[2] = turbulent Doppler parameter
    p[3] = temperature
    """
    def __init__(self, prgname="", getinst=False, atomic=None, verbose=2):
        self._idstr   = 'splineabs'																				# ID string for this class
        self._pnumr   = 4																					# Total number of parameters fed in
        self._keywd   = dict({'specid':[], 'continuum':False, 'blind':False, 'ion':'', 'logN':True,  'reference':0, 'wavemin':-1.0, 'wavemax':-1.0, 'locations':[], 'blindseed':0,  'blindrange':[]})		# Additional arguments to describe the model --- 'input' cannot be used as a keyword
        self._keych   = dict({'specid':0,  'continuum':0,     'blind':0,     'ion':1,  'logN':0,  'reference':0,  'wavemin':0,    'wavemax':0,    'locations':1, 'blindseed':0,  'blindrange':0})			# Require keywd to be changed (1 for yes, 0 for no)
        self._keyfm   = dict({'specid':"", 'continuum':"",    'blind':"",    'ion':"{1:7}", 'logN':"", 'reference':"", 'wavemin':"{1:.8g}",   'wavemax':"{1:.8g}",   'locations':"", 'blindseed':"",  'blindrange':""})		# Format for the keyword. "" is the Default setting
        self._parid   = ['TotalColDens',   'redshift', 'bturb',   'tau']		# Name of each parameter
        self._defpar  = [ 8.1,         0.0,        7.0,       1.0 ]				# Default values for parameters that are not provided
        self._fixpar  = [ None,        None,       None,      None ]				# By default, should these parameters be fixed?
        self._limited = [ [0  ,0  ],  [0  ,0  ],  [1  ,0  ], [1  ,1] ]		# Should any of these parameters be limited from below or above
        self._limits  = [ [0.0,0.0],  [0.0,0.0],  [0.5,0.0], [0.0,1.0] ]		# What should these limiting values be
        self._svfmt   = [ "{0:.7g}", "{0:.12g}", "{0:.6g}", "{0:.7g}"]			# Specify the format used to print or save output
        self._prekw   = [ 'ion', 'wavemin', 'wavemax' ]																			# Specify the keywords to print out before the parameters
        # DON'T CHANGE THE FOLLOWING --- it tells ALIS what parameters are provided by the user.
        tempinput = self._parid+list(self._keych.keys())                             #
        self._keywd['input'] = dict(zip((tempinput),([0]*np.size(tempinput)))) #
        ########################################################################
        self._verbose = verbose
        # Set the atomic data
        self._atomic = atomic
        #self._kernal  = self.GPU_kernal()										# Get the Source Module for the GPU
        if getinst: return

    def call_CPU(self, wave, pin, ae='ab', mkey=None, ncpus=1):
        """
        Define the functional form of the model
        --------------------------------------------------------
        x  : array of wavelengths
        p  : array of parameters for this model
        --------------------------------------------------------
        """
        def model(par, karr):
            """
            Define the model here

            8.85282061604877e-13 = pi x r_e, where r_e = e^2 / (m_e c^2) = 2.8179403227E-13 is the classical electron radius
            """
            wavein = wave
            wv = par[3] * 1.0e-8
            if karr['logN']: cold = 10.0**par[0]
            else: cold = par[0]
            zp1 = par[1]+1.0
            cns = 8.85282061604877e-13 * 299792.458 * par[4] * wv
            cne = cold*cns
            ww = (wavein*1.0e-8)/zp1
            x = 299792.458*(ww-wv)/wv
            # Get the locations of the spline points
            nspl = (len(par)-5)//2
            yspl = par[5:5+nspl]
            xspl = par[5+nspl:]
            if xspl.size != yspl.size:
                msgs.error("'locations' parameter must have same length as input parameters in spline function")
            splev = interp1d(xspl, yspl, kind=1, bounds_error=False, fill_value=0)
            modret = splev(x)
            if par[2]>0.0:
                # Smooth by the b-value
                sigd = (par[2]/np.sqrt(2)) / 2.99792458E5
                ysize = modret.size
                fsigd = 6.0 * sigd
                dwav = 0.5 * (wave[2:] - wave[:-2]) / wave[1:-1]
                dwav = np.append(np.append(dwav[0], dwav), dwav[-1])
                df = int(np.min([int(np.ceil(fsigd / dwav).max()), ysize / 2 - 1]))
                yval = np.zeros(2 * df + 1)
                yval[df:2 * df + 1] = (wave[df:2 * df + 1] / wave[df] - 1.0) / sigd
                yval[:df] = (wave[:df] / wave[df] - 1.0) / sigd
                gaus = np.exp(-0.5 * yval * yval)
                size = ysize + gaus.size - 1
                fsize = 2 ** int(np.ceil(np.log2(size)))  # Use this size for a more efficient computation
                conv = np.fft.fft(modret, fsize)
                conv *= np.fft.fft(gaus / gaus.sum(), fsize)
                ret = np.fft.ifft(conv).real.copy()
                del conv
                modret_conv = ret[df:df + ysize]
            else:
                modret_conv = modret.copy()
            modret_conv[modret_conv < 0.0] = 0.0
            # Normalise
            norm = np.trapz(modret_conv, x=x)
            modret_conv /= norm
            # Prepare the optical depth
            tau = cne * modret_conv
            return np.exp(-1.0*tau)
        #############
        yout = np.zeros((pin.shape[0],wave.size))
        for i in range(pin.shape[0]):
            yout[i,:] = model(pin[i,:], karr=mkey[i])
        if ae == 'em': return yout.sum(axis=0)
        else: return yout.prod(axis=0)

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
        isspl=instr.split()
        # Seperate the parameters from the keywords
        ptemp, kywrd = [], []
        keywdk = list(self._keywd.keys())
        keywdk[:] = (kych for kych in keywdk if kych[:] != 'input') # Remove the keyword 'input'
        numink = 0
        # Since there's no way of knowing how many coefficients and
        # keywords there are, determine the number of parameters
        pnumr = 0
        for i in range(len(isspl)):
            if "=" in isspl[i]:
                if isspl[i].split('=')[0] == self._parid[3]: pnumr += 1
            else: pnumr += 1
        # Setup the parameters
        param = [None for all in range(pnumr)]
        parid = [i for i in range(pnumr)]
        for pp in range(3,pnumr): parid[pp] = 3
        isratio = False
        for i in range(len(isspl)):
            if "=" in isspl[i]:
                kwspl = isspl[i].split('=')
                if kwspl[0] in self._parid:
                    self._keywd['input'][kwspl[0]] = 2
                    for j in range(pnumr):
                        if kwspl[0] == self._parid[3]:
                            param[j] = kwspl[1]
                            numink += 1
                            break
                        elif kwspl[0] == self._parid[j]:
                            param[j] = kwspl[1]
                            numink += 1
                            break
                elif kwspl[0] in keywdk:
                    self._keywd['input'][kwspl[0]]=1
                    kywrd.append(isspl[i])
                    if '/' in kwspl[1]:
                        isratio = True
                        ionspl = kwspl[1].split('/')
                        if ionspl[0] in self._atomic['Ion'] and ionspl[1] in self._atomic['Ion']:
                            pnumr = 1
                        else: msgs.error("Keyword '"+isspl[i]+"' has incorrect form on line -"+msgs.newline()+self._idstr+"   "+instr)
                else: msgs.error("Keyword '"+isspl[i]+"' is unknown for -"+msgs.newline()+self._idstr+"   "+instr)
            else: ptemp.append(isspl[i])
        if isratio: # If a column density ratio is specified
            for i in range(pnumr-1):
                del param[1]
                del parid[1]
        # Set parameters that weren't given in param
        cntp=0
        fixpid = [0 for all in range(pnumr)]
        for i in range(pnumr):
            if param[i] is None:
                if cntp >= len(ptemp): # Use default values
                    if i >= 3:
                        param[i] = str(self._defpar[parid[3]])
                    else:
                        param[i] = str(self._defpar[parid[i]])
                    fixpid[i] = 1
                else:
                    param[i] = ptemp[cntp]
                    if i >= 3:
                        self._keywd['input'][self._parid[3]] = 1
                    else:
                        self._keywd['input'][self._parid[i]]=1
                    cntp += 1
        # Do some quick checks
        if len(ptemp)+numink > pnumr:
            msgs.error("Incorrect number of parameters (should be "+str(pnumr)+"):"+msgs.newline()+self._idstr+"   "+instr)
        if len(specid) > 1: # Force the user to specify a spectrum ID number
            self._keych['specid'] = 1
        specidset=False # Detect is the user has specified specid
        # Set the parameters:
        mp['mtyp'].append(self._idstr)
        mp['mpar'].append([])
        mp['mtie'].append([])
        mp['mfix'].append([])
        mp['mlim'].append([])
        for i in range(pnumr):
            mp = check_tied_param(param[i], cntr, mp, i, parid[i])
        # Hardwire in the minimum and maximum column density ratio
        if pnumr == 1 and mp['mlim'][cntr][0][0] is not None: mp['mlim'][cntr][0] = [mp['mlim'][cntr][0][0]-22.0,22.0-mp['mlim'][cntr][0][0]]
        # If default values had to be used, fix the values
        if any(fixpid):
            for i in range(pnumr):
                if fixpid[i] == 1: mp['mfix'][-1][i] = 1
        # Now load the keywords:
        # Make a copy of the default values
        cpy_keywd = self._keywd.copy()
        cpy_keych = self._keych.copy()
        for i in range(len(kywrd)):
            kwspl = kywrd[i].split('=')
            ksspl = kwspl[1].lstrip('([').rstrip(')]').split(',')
            if kwspl[0] == 'specid':
                sidlist = []
                for j in range(len(ksspl)): # Do some checks
                    if ksspl[j] in sidlist:
                        msgs.error("specid: "+ksspl[j]+" is set twice for -"+msgs.newline()+self._idstr+"   "+instr.replace('\t','  '))
                    else: sidlist.append(ksspl[j])
                    if ksspl[j] not in specid:
                        msgs.error("There is no data with specid: "+ksspl[j]+" for -"+msgs.newline()+self._idstr+"   "+instr.replace('\t','  '))
                specidset=True
            elif kwspl[0] == 'locations':
                scllist = []
                if len(ksspl) != pnumr-3:
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
        if   i == 0:
            if parb['ap_1a'][0] is None: pin = par
            else:
                if parb['ap_1a'][1]: # Num is logN
                    if parb['ap_1a'][2]: pin = parb['ap_1a'][0] + par           # Den is logN
                    else:                pin = parb['ap_1a'][0] + np.log10(par) # Den is N
                else:                # Num is N
                    if parb['ap_1a'][2]: pin = parb['ap_1a'][0] * 10.0**par     # Den is logN
                    else:                pin = parb['ap_1a'][0] * par           # Den is N
        elif i == 1: pin = par
        elif i == 2: pin = par
        # elif i == 3+parb['ap_4a']: pin = par  # TODO :: Need to set this based on the total column density
        elif i >= 3: pin = par
        else:
            msgs.error("Function "+self._idstr+" is badly defined in definition parin.")
        return pin

    def set_vars(self, p, level, mp, ival, wvrng=[0.0,0.0], spid='None', levid=None, nexbin=None, ddpid=None, getinfl=False):
        """
        Return the parameters for a Voigt function to be used by 'call'
        """
        pnumr = len(mp['mpar'][ival])
        params = np.zeros(pnumr)
        parinf = []
        # Find the minimum and maximum wavelength
        wvmin, wvmax = wvrng[0], wvrng[1]
        if wvmin > wvmax:
            # Likely, Hz is the bintype
            temp = wvmin
            wvmin = wvmax
            wvmax = temp
        # Determine if this is a column density ratio:
        if '/' in self._keywd['ion']:
            cdratio = True
            numrat, denrat = self._keywd['ion'].split('/')
            m = np.where(self._atomic['Element'].astype(str) == numrat.split('_')[0])
            if np.size(m) != 1: msgs.error("Numerator element "+numrat+" not found for -"+msgs.newline()+self._keywd['ion'])
            elmass = self._atomic['AtomicMass'][m][0]
        else: cdratio = False
        if not cdratio: # THE COLUMN DENSITY FOR A SINGLE ION HAS BEEN SPECIFIED
            pt=np.zeros(pnumr)
            levadd=0
            m = np.where(self._atomic['Element'].astype(str) == self._keywd['ion'].split('_')[0])
            if np.size(m) != 1:
                msgs.error("Element {0:s} not found in atomic data file".format(self._keywd['ion'].split('_')[0]))
            for i in range(pnumr):
                lnkprm = None
                parb = dict({'ap_1a':[None], 'ap_4a':mp['mkey'][ival]['reference'], 'ap_4b':pt[0]})
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
                    pt[i] = self.parin(i, p[getid], parb)
                    if mp['mfix'][ival][i] == 0: parinf.append(getid)
                else:
                    pt[i] = lnkprm
            if ddpid is not None:
                if ddpid not in parinf: return []
            nv = np.where(self._atomic['Ion'] == self._keywd['ion'])[0]
            nw = np.where( (self._atomic['Wavelength'][nv]*(1.0+pt[1]) >= wvmin) & (self._atomic['Wavelength'][nv]*(1.0+pt[1]) <= wvmax) )
            if self._atomic['Wavelength'][nv][nw].size == 0:
                if nexbin is not None: return [], None
                elif getinfl: return [], []
                else: return []
            params = np.zeros((self._atomic['Wavelength'][nv][nw].size,2+2*pt.size-3))
            for ln in range(0,self._atomic['Wavelength'][nv][nw].size):
                restwave = self._atomic['Wavelength'][nv][nw][ln]
                splpars = np.append(pt[3:], mp['mkey'][ival]['locations'])
                params[ln,:] = np.append(np.array([pt[0],pt[1],pt[2],restwave,self._atomic['fvalue'][nv][nw][ln]]), splpars)
        else: # THE RATIO OF TWO COLUMN DENSITIES HAS BEEN SPECIFIED
            # Find all denrat in mp with matching specid's.
            pt=[]
            tsize = 0
            for i in range(len(mp['mkey'])):
                lnkprm = None
                if mp['mtyp'][i] != self._idstr: continue   # Not a Voigt profile
                if mp['mkey'][i]['ion'] != denrat: continue # Not denrat
                if spid not in mp['mkey'][i]['specid']: continue # Not a common specid
                # Determine if the column density ratio is N or logN.
                nln = mp['mkey'][ival]['logN']
                dln = mp['mkey'][i]['logN']
                # Get the value of the column density ratio
                pt.append(np.zeros(len(mp['mpar'][i])))
                if mp['mtie'][ival][0] >= 0:
                    getid = mp['tpar'][mp['mtie'][ival][0]][1]
                elif mp['mtie'][ival][0] <= -2:
                    if len(mp['mlnk']) == 0:
                        lnkprm = mp['mpar'][ival][0]
                    else:
                        for j in range(len(mp['mlnk'])):
                            if mp['mlnk'][j][0] == mp['mtie'][ival][0]:
                                cmd = 'lnkprm = ' + mp['mlnk'][j][1]
                                namespace = dict({'p':p, 'np':np})
                                exec(cmd, namespace)
                                lnkprm = namespace['lnkprm']
                else:
                    getid = level
                if lnkprm is None:
                    pt[-1][0] = p[getid]
                    if mp['mfix'][ival][0] == 0: parinf.append(getid)
                else:
                    pt[-1][0] = lnkprm
                # Get the parameters from the matching denrat and combine these with the appropriate parameters for numrat
                levadd=0
                for j in range(len(mp['mpar'][i])):
                    lnkprm = None
                    parb = dict({'ap_1a':[pt[-1][0],nln,dln], 'ap_4a':mp['mkey'][ival]['reference'], 'ap_4b':pt[-1][0]})
                    if mp['mtie'][i][j] >= 0:
                        getid = mp['tpar'][mp['mtie'][i][j]][1]
                        getpr = p[getid]
                    elif mp['mtie'][i][j] == -1 and mp['mfix'][i][j] == 1:
                        getid = None
                        getpr = mp['mpar'][i][j]
                        levadd+=1
                    elif mp['mtie'][i][j] <= -2:
                        if len(mp['mlnk']) == 0:
                            lnkprm = mp['mpar'][i][j]
                        else:
                            for k in range(len(mp['mlnk'])):
                                if mp['mlnk'][k][0] == mp['mtie'][i][j]:
                                    cmd = 'lnkprm = ' + mp['mlnk'][k][1]
                                    namespace = dict({'p': p, 'np':np})
                                    exec(cmd, namespace)
                                    lnkprm = namespace['lnkprm']
                        levadd += 1
                    else:
                        getid = levid[i]+levadd
                        getpr = p[getid]
                        levadd+=1
                    if lnkprm is None:
                        pt[-1][j] = self.parin(j, getpr, parb)
                        if mp['mfix'][i][j] == 0: parinf.append(getid)
                    else:
                        pt[-1][j] = lnkprm
                if ddpid is not None:
                    if ddpid not in parinf: continue
                nv = np.where(self._atomic['Ion'] == numrat)[0]
                nw = np.where( (self._atomic['Wavelength'][nv]*(1.0+pt[-1][1]) >= wvmin) & (self._atomic['Wavelength'][nv]*(1.0+pt[-1][1]) <= wvmax) )
                paramst = np.zeros((self._atomic['Wavelength'][nv][nw].size,2+2*pt[-1].size-3))
                for ln in range(0,self._atomic['Wavelength'][nv][nw].size):
                    restwave = self._atomic['Wavelength'][nv][nw][ln]
                    splpars = np.append(pt[-1][3:], mp['mkey'][i]['locations'])
                    paramst[ln,:] = np.append(np.array([pt[-1][0],pt[-1][1],pt[-1][2],restwave,self._atomic['fvalue'][nv][nw][ln],]), splpars)
                if tsize == 0: params = paramst
                else: params = np.append(params, paramst, axis=0)
                tsize += self._atomic['Wavelength'][nv][nw].size
            if tsize == 0:
                if nexbin is not None: return [], None
                elif getinfl: return [], []
                else: return []
        if nexbin is not None:
            if params[:,2].min() == 0.0:
                msgs.error("Cannot calculate "+self._idstr+" subpixellation -- width = 0.0")
            if nexbin[0] == "km/s": return params, int(round(np.sqrt(2.0)*nexbin[1]/params[:,2].min() + 0.5))
            elif nexbin[0] == "A" : return params, int(round(np.sqrt(2.0)*299792.458*nexbin[1]/((1.0+params[:,1])*params[:,3]*params[:,2]).min() + 0.5))
            elif nexbin[0] == "Hz" : return params, int(round(np.sqrt(2.0)*299792.458*nexbin[1]/((1.0+params[:,1])*params[:,3]*params[:,2]).min() + 0.5))
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
        --------------------------------------------------------
        Nothing should be changed here when writing a new function.
        --------------------------------------------------------
        """
        if errs is None: errors = params
        else: errors = errs
        numpar = len(mp['mpar'][istart])
        parid = [i for i in range(numpar)]
        for pp in range(3, numpar): parid[pp] = 3
        add = numpar
        havtie = 0
        tienum = 0
        levadd = 0
        outstring = ['  %s ' % (self._idstr)]
        errstring = ['# %s ' % (self._idstr)]

        # Check if we are blinding any parameters with an offset value
        blindoffset = 0
        if 'blindrange' in mp['mkey'][istart]:
            print("Blinding the parameters")
            if len(mp['mkey'][istart]['blindrange']) == 2:
                if 'blindseed' in mp['mkey'][istart]:
                    np.random.seed(mp['mkey'][istart]['blindseed'])
                else:
                    np.random.seed(0)
                blindoffset = np.random.uniform(int(mp['mkey'][istart]['blindrange'][0]), int(mp['mkey'][istart]['blindrange'][1]))
            # Reset the seed
            self.resetseed()

        for i in range(numpar):
            ipid = i
            if i >= 3: ipid = 3
            if mp['mkey'][istart]['input'][self._parid[ipid]] == 0: # Parameter not given as input
                outstring.append( "" )
                errstring.append( "" )
                continue
            elif mp['mkey'][istart]['input'][self._parid[ipid]] == 1: pretxt = ""   # Parameter is given as input, without parid
            else: pretxt = self._parid[ipid]+"="                                    # Parameter is given as input, with parid
            if mp['mtie'][istart][i] >= 0:
                if reletter:
                    newfmt=pretxt+self.gtoef(params[mp['tpar'][mp['mtie'][istart][i]][1]],self._svfmt[parid[ipid]]+'{1:c}')
                    outstring.append( (newfmt).format(blindoffset+params[mp['tpar'][mp['mtie'][istart][i]][1]],97+mp['mtie'][istart][i]-32*mp['mfix'][istart][1]) )
                    if conv is None:
                        errstring.append( (newfmt).format(errors[mp['tpar'][mp['mtie'][istart][i]][1]],97+mp['mtie'][istart][i]-32*mp['mfix'][istart][1]) )
                    else:
                        if params[mp['tpar'][mp['mtie'][istart][i]][1]] < conv: cvtxt = "CONVERGED"
                        else: cvtxt = "!!!!!!!!!"
                        errstring.append( ('--{0:s}--{1:c}    ').format(cvtxt,97+mp['mtie'][istart][i]-32*mp['mfix'][istart][1]) )
                else:
                    newfmt=pretxt+self.gtoef(params[mp['tpar'][mp['mtie'][istart][i]][1]],self._svfmt[parid[ipid]]+'{1:s}')
                    outstring.append( (newfmt).format(blindoffset+params[mp['tpar'][mp['mtie'][istart][i]][1]],mp['tpar'][mp['mtie'][istart][i]][0]) )
                    if conv is None:
                        errstring.append( (newfmt).format(errors[mp['tpar'][mp['mtie'][istart][i]][1]],mp['tpar'][mp['mtie'][istart][i]][0]) )
                    else:
                        if params[mp['tpar'][mp['mtie'][istart][i]][1]] < conv: cvtxt = "CONVERGED"
                        else: cvtxt = "!!!!!!!!!"
                        errstring.append( ('--{0:s}--{1:s}    ').format(cvtxt,mp['tpar'][mp['mtie'][istart][i]][0]) )
                add -= 1
            else:
                if havtie != 2:
                    if havtie == 0: # First searching for the very first instance of a tied parameter
                        for tn in range(0,len(mp['tpar'])):
                            if mp['tpar'][tn][1] == level+levadd:
                                tienum=tn
                                havtie=1
                                break
                    if mp['tpar'][tienum][1] == level+levadd:
                        if reletter:
                            newfmt=pretxt+self.gtoef(params[level+levadd],self._svfmt[parid[i]]+'{1:c}')
                            outstring.append( (newfmt).format(blindoffset+params[level+levadd],97+tienum-32*mp['mfix'][istart][1]) )
                            if conv is None:
                                errstring.append( (newfmt).format(errors[level+levadd],97+tienum-32*mp['mfix'][istart][1]) )
                            else:
                                if params[level+levadd] < conv: cvtxt = "CONVERGED"
                                else: cvtxt = "!!!!!!!!!"
                                errstring.append( ('--{0:s}--{1:c}    ').format(cvtxt,97+tienum-32*mp['mfix'][istart][1]) )
                        else:
                            newfmt=pretxt+self.gtoef(params[level+levadd],self._svfmt[parid[i]]+'{1:s}')
                            outstring.append( (newfmt).format(blindoffset+params[level+levadd],mp['tpar'][tienum][0]) )
                            if conv is None:
                                errstring.append( (newfmt).format(errors[level+levadd],mp['tpar'][tienum][0]) )
                            else:
                                if params[level+levadd] < conv: cvtxt = "CONVERGED"
                                else: cvtxt = "!!!!!!!!!"
                                errstring.append( ('--{0:s}--{1:s}    ').format(cvtxt,mp['tpar'][tienum][0]) )
                        tienum += 1
                        if tienum == len(mp['tpar']): havtie = 2 # Stop searching for 1st instance of tied param
                    else:
                        newfmt=pretxt+self.gtoef(params[level+levadd],self._svfmt[parid[i]])
                        outstring.append( (newfmt).format(blindoffset+params[level+levadd]) )
                        if conv is None:
                            errstring.append( (newfmt).format(errors[level+levadd]) )
                        else:
                            if params[level+levadd] < conv: cvtxt = "CONVERGED"
                            else: cvtxt = "!!!!!!!!!"
                            errstring.append( ('--{0:s}--    ').format(cvtxt) )
                else:
                    newfmt=pretxt+self.gtoef(params[level+levadd],self._svfmt[parid[i]])
                    outstring.append( (newfmt).format(blindoffset+params[level+levadd]) )
                    if conv is None:
                        errstring.append( (newfmt).format(errors[level+levadd]) )
                    else:
                        if params[level+levadd] < conv: cvtxt = "CONVERGED"
                        else: cvtxt = "!!!!!!!!!"
                        errstring.append( ('--{0:s}--    ').format(cvtxt) )
                levadd += 1
        level += add
        # Now write in the keywords
        keys = list(mp['mkey'][istart].keys())
        keys[:] = (kych for kych in keys if kych[:] != 'input') # Remove the keyword 'input'
        for i in range(len(keys)):
            if mp['mkey'][istart]['input'][keys[i]] == 0: # This keyword wasn't provided as user input
                outstring.append( "" )
                errstring.append( "" )
                continue
            if type(mp['mkey'][istart][keys[i]]) is list: outkw = ','.join(map(str,mp['mkey'][istart][keys[i]]))
            else: outkw = mp['mkey'][istart][keys[i]]
            if self._keyfm[keys[i]] != "":
                outstring.append( ('{0:s}='+self._keyfm[keys[i]]).format(keys[i],outkw) )
                errstring.append( ('{0:s}='+self._keyfm[keys[i]]).format(keys[i],outkw) )
            else:
                outstring.append( '%s=%s' % (keys[i],outkw) )
                errstring.append( '%s=%s' % (keys[i],outkw) )
        # Now place the keywords specified in self._prekw at the beginning of the return string:
        if len(self._prekw) != 0:
            insind = 1
            for i in range(len(self._prekw)):
                delind = -1
                for j in range(len(keys)):
                    if self._prekw[i] == keys[j]:
                        delind = numpar+insind+j
                        del keys[j]
                        break
                if delind == -1: msgs.bug("prekw variable for function "+self._idstr+" contains bad argument", verbose=self._verbose)
                outstring.insert(insind,outstring[delind])
                errstring.insert(insind,errstring[delind])
                del outstring[delind+1]
                del errstring[delind+1]
                insind += 1
        if mp['mkey'][istart]['blind'] and conv is None:
            retout = "       ------ BLIND MODEL ------\n"
            #reterr = "       ------ BLIND MODEL ------\n"
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
        # Hardcode in the minimum and maximum column density ratio
        if len(mp['mpar'][mnum])==1:
            pinfo[level]['limited'] = [1,1]
            pinfo[level]['limits']  = [-14.0,14.0]
        return pinfo, add


    def tick_info(self, p, level, mp, ival, wvrng=[0.0,0.0], spid='None', levid=None):
        # Removed: nexbin=None, ddpid=None, getinfl=False
        """
        For a given model, determine the wavelengths that
        tick marks should be plotted, and the label that
        should be associated with this tick mark.
        """
        wavelns=[]
        labllns=[]
        parinf=[]
        # Determine if this is a column density ratio:
        if '/' in self._keywd['ion']:
            cdratio = True
            numrat, denrat = self._keywd['ion'].split('/')
            m = np.where(self._atomic['Element'].astype(str) == numrat.split('_')[0])
            if np.size(m) != 1: msgs.error("Numerator element "+numrat+" not found for -"+msgs.newline()+self._keywd['ion'])
            elmass = self._atomic['AtomicMass'][m][0]
        else: cdratio = False
        if not cdratio: # THE COLUMN DENSITY FOR A SINGLE ION HAS BEEN SPECIFIED
            pt=np.zeros(self._pnumr)
            levadd=0
            m = np.where(self._atomic['Element'].astype(str) == self._keywd['ion'].split('_')[0])
            if np.size(m) != 1:
                msgs.error("Element {0:s} not found in atomic data file".format(self._keywd['ion'].split('_')[0]))
            for i in range(self._pnumr):
                lnkprm = None
                parb = dict({'ap_1a':[None], 'ap_2a':pt[2], 'ap_2b':self._atomic['AtomicMass'][m][0]})
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
                    pt[i] = self.parin(i, p[getid], parb)
                    if mp['mfix'][ival][i] == 0: parinf.append(getid)
                else:
                    pt[i] = lnkprm
            nv = np.where(self._atomic['Ion'] == self._keywd['ion'])[0]
            nw = np.where( (self._atomic['Wavelength'][nv]*(1.0+pt[1]) >= wvrng[0]) & (self._atomic['Wavelength'][nv]*(1.0+pt[1]) <= wvrng[1]) )
            if self._atomic['Wavelength'][nv][nw].size == 0:
                return [], []
            for ln in range(0,self._atomic['Wavelength'][nv][nw].size):
                wavelns.append((1.0+pt[1])*self._atomic['Wavelength'][nv][nw][ln])
                labllns.append("{0:s} {1:.1f}".format(self._atomic['Ion'][nv][nw][ln].replace("_"," "),self._atomic['Wavelength'][nv][nw][ln]))
        else: # THE RATIO OF TWO COLUMN DENSITIES HAS BEEN SPECIFIED
            # Find all denrat in mp with matching specid's.
            pt=[]
            tsize = 0
            for i in range(len(mp['mkey'])):
                lnkprm = None
                if mp['mtyp'][i] != self._idstr: continue   # Not a Voigt profile
                if mp['mkey'][i]['ion'] != denrat: continue # Not denrat
                if spid not in mp['mkey'][i]['specid']: continue # Not a common specid
                # Determine if the column density ratio is N or logN.
                nln = mp['mkey'][ival]['logN']
                dln = mp['mkey'][i]['logN']
                # Get the value of the column density ratio
                pt.append(np.zeros(self._pnumr))
                if mp['mtie'][ival][0] >= 0:
                    getid = mp['tpar'][mp['mtie'][ival][0]][1]
                elif mp['mtie'][ival][0] <= -2:
                    if len(mp['mlnk']) == 0:
                        lnkprm = mp['mpar'][ival][0]
                    else:
                        for j in range(len(mp['mlnk'])):
                            if mp['mlnk'][j][0] == mp['mtie'][ival][0]:
                                cmd = 'lnkprm = ' + mp['mlnk'][j][1]
                                namespace = dict({'p':p, 'np':np})
                                exec(cmd, namespace)
                                lnkprm = namespace['lnkprm']
                else:
                    getid = level
                if lnkprm is None:
                    pt[-1][0] = p[getid]
                    if mp['mfix'][ival][0] == 0: parinf.append(getid)
                else:
                    pt[-1][0] = lnkprm
                # Get the parameters from the matching denrat and combine these with the appropriate parameters for numrat
                levadd=0
                for j in range(self._pnumr):
                    lnkprm = None
                    parb = dict({'ap_1a':[pt[-1][0],nln,dln], 'ap_2a':pt[-1][2], 'ap_2b':elmass})
                    if mp['mtie'][i][j] >= 0:
                        getid = mp['tpar'][mp['mtie'][i][j]][1]
                        getpr = p[getid]
                    elif mp['mtie'][i][j] == -1 and mp['mfix'][i][j] == 1:
                        getid = None
                        getpr = mp['mpar'][i][j]
                        levadd+=1
                    elif mp['mtie'][i][j] <= -2:
                        if len(mp['mlnk']) == 0:
                            lnkprm = mp['mpar'][i][j]
                        else:
                            for k in range(len(mp['mlnk'])):
                                if mp['mlnk'][k][0] == mp['mtie'][i][j]:
                                    cmd = 'lnkprm = ' + mp['mlnk'][k][1]
                                    namespace = dict({'p': p, 'np':np})
                                    exec(cmd, namespace)
                                    lnkprm = namespace['lnkprm']
                        levadd += 1
                    else:
                        getid = levid[i]+levadd
                        getpr = p[getid]
                        levadd+=1
                    if lnkprm is None:
                        pt[-1][j] = self.parin(j, getpr, parb)
                        if mp['mfix'][i][j] == 0: parinf.append(getid)
                    else:
                        pt[-1][j] = lnkprm
                nv = np.where(self._atomic['Ion'] == numrat)[0]
                nw = np.where( (self._atomic['Wavelength'][nv]*(1.0+pt[-1][1]) >= wvrng[0]) & (self._atomic['Wavelength'][nv]*(1.0+pt[-1][1]) <= wvrng[1]) )
                for ln in range(0,self._atomic['Wavelength'][nv][nw].size):
                    wavelns.append((1.0+pt[-1][1])*self._atomic['Wavelength'][nv][nw][ln])
                    labllns.append("{0:s} {1:.1f}".format(self._atomic['Ion'][nv][nw][ln].replace("_"," "),self._atomic['Wavelength'][nv][nw][ln]))
                tsize += self._atomic['Wavelength'][nv][nw].size
            if tsize == 0:
                return [], []
        return wavelns, labllns

