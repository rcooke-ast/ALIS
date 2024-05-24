import os
import numpy as np
from alis import almsgs
from alis import alfunc_base
#import pycuda.driver as cuda
#import pycuda.autoinit
#from pycuda.compiler import SourceModule
msgs=almsgs.msgs()


class PhotIon_CrossSection(alfunc_base.Base):
    """
    Returns absorption due to photoelectric absorption:
    p[0] = log10 of the column density
    p[1] = redshift
    """
    def __init__(self, prgname="", getinst=False, atomic=None, verbose=2):
        self._idstr   = 'phionxs'																				# ID string for this class
        self._pnumr   = 2																					# Total number of parameters fed in
        self._keywd   = dict({'specid':[], 'continuum':False, 'blind':False, 'ion':'', 'logN':True, 'blindseed':0,  'blindrange':[]})		# Additional arguments to describe the model --- 'input' cannot be used as a keyword
        self._keych   = dict({'specid':0,  'continuum':0,     'blind':0,     'ion':1,  'logN':0, 'blindseed':0,  'blindrange':0})			# Require keywd to be changed (1 for yes, 0 for no)
        self._keyfm   = dict({'specid':"", 'continuum':"",    'blind':"",    'ion':"{1:7}", 'logN':"", 'blindseed':"",  'blindrange':""})		# Format for the keyword. "" is the Default setting
        self._parid   = ['ColDens',   'redshift']		                                                    # Name of each parameter
        self._defpar  = [ 8.1,         0.0]				                                                    # Default values for parameters that are not provided
        self._fixpar  = [ None,        None]			                                                   	# By default, should these parameters be fixed?
        self._limited = [ [0  ,0  ],  [0  ,0  ] ]		                                                    # Should any of these parameters be limited from below or above
        self._limits  = [ [0.0,0.0],  [0.0,0.0] ]		                                                    # What should these limiting values be
        self._svfmt   = [ "{0:.7g}", "{0:.10g}"]		                                                    # Specify the format used to print or save output
        self._prekw   = [ 'ion' ]																			# Specify the keywords to print out before the parameters
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
        def calc_xsec(wave, zp1, vars):
            """Assuming wave is in Angstroms
            """
            engy = 6.62606957E-34 * 299792458.0 * zp1 / (wave*1.0E-10 * 1.60217657E-19)
            Et, Emx, Eo, so, ya, P, yw, yo, y1, El = vars[0], vars[1], vars[2], vars[3], vars[4], vars[5], vars[6], vars[7], vars[8], vars[9]
            xsec = np.zeros(engy.size)
            x = engy / Eo - yo
            y = np.sqrt(x ** 2 + y1 ** 2)
            Fy = ((x - 1.0) ** 2 + yw ** 2) * y ** (0.5 * P - 5.5) * (1.0 + np.sqrt(y / ya)) ** (-1.0 * P)
            w = np.where((engy >= Et) & (engy <= Emx))
            xsec[w] = 1.0E-18 * so * Fy[w]
            # Fill in the spectrum between the Lyman limit and the last line considered
            wmn = np.argmin(engy[w])
            xsecrep = xsec[w][wmn]
            w = np.where((engy >= El) & (engy < Et))
            xsec[w] = xsecrep
            return xsec

        def model(par, karr):
            """
            Define the model here
            """
            # We now need to break each pixel up into smaller units to
            # calculate the effective absorption in a given pixel. Let's
            # sample each pixel by nexpd bins per Doppler parameter.
            # This is only valid if the pixelsize is in km/s
            if karr['logN']: cold = 10.0**par[0]
            else: cold = par[0]
            zp1 = par[1]+1.0
            xsecv = calc_xsec(wave, zp1, par[2:])
            return np.exp(-1.0*xsecv*cold)
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
        numpar = self._pnumr
        param = [None for all in range(numpar)]
        parid = [i for i in range(numpar)]
        for i in range(len(isspl)):
            if "=" in isspl[i]:
                kwspl = isspl[i].split('=')
                if kwspl[0] in self._parid:
                    self._keywd['input'][kwspl[0]]=2
                    for j in range(self._pnumr):
                        if kwspl[0] == self._parid[j]:
                            param[j] = kwspl[1]
                            numink += 1
                            break
                elif kwspl[0] in keywdk:
                    self._keywd['input'][kwspl[0]]=1
                    kywrd.append(isspl[i])
                    if '/' in kwspl[1]:
                        ionspl = kwspl[1].split('/')
                        if ionspl[0] in self._atomic['Ion'] and ionspl[1] in self._atomic['Ion']:
                            numpar = 1
                        else: msgs.error("Keyword '"+isspl[i]+"' has incorrect form on line -"+msgs.newline()+self._idstr+"   "+instr)
                else: msgs.error("Keyword '"+isspl[i]+"' is unknown for -"+msgs.newline()+self._idstr+"   "+instr)
            else: ptemp.append(isspl[i])
        if numpar != self._pnumr: # If a column density ratio is specified
            for i in range(self._pnumr-numpar):
                del param[1]
                del parid[1]
        # Set parameters that weren't given in param
        cntp=0
        fixpid = [0 for all in range(numpar)]
        for i in range(numpar):
            if param[i] is None:
                if cntp >= len(ptemp): # Use default values
                    param[i] = str(self._defpar[parid[i]])
                    fixpid[i] = 1
                else:
                    param[i] = ptemp[cntp]
                    self._keywd['input'][self._parid[i]]=1
                    cntp += 1
        # Do some quick checks
        if len(ptemp)+numink > numpar:
            msgs.error("Incorrect number of parameters (should be "+str(numpar)+"):"+msgs.newline()+self._idstr+"   "+instr)
        if len(specid) > 1: # Force the user to specify a spectrum ID number
            self._keych['specid'] = 1
        specidset=False # Detect is the user has specified specid
        # Set the parameters:
        mp['mtyp'].append(self._idstr)
        mp['mpar'].append([])
        mp['mtie'].append([])
        mp['mfix'].append([])
        mp['mlim'].append([])
        for i in range(numpar):
            mp = check_tied_param(param[i], cntr, mp, i, parid[i])
        # Hardwire in the minimum and maximum column density ratio
        if numpar == 1 and mp['mlim'][cntr][0][0] is not None: mp['mlim'][cntr][0] = [mp['mlim'][cntr][0][0]-22.0,22.0-mp['mlim'][cntr][0][0]]
        # If default values had to be used, fix the values
        if any(fixpid):
            for i in range(numpar):
                if fixpid[i] == 1: mp['mfix'][-1][i] = 1
        # Now load the keywords:
        # Make a copy of the default values
        cpy_keywd = self._keywd.copy()
        cpy_keych = self._keych.copy()
        for i in range(len(kywrd)):
            kwspl = kywrd[i].split('=')
            ksspl = kwspl[1].split(',')
            if kwspl[0] == 'specid':
                sidlist = []
                for j in range(len(ksspl)): # Do some checks
                    if ksspl[j] in sidlist:
                        msgs.error("specid: "+ksspl[j]+" is set twice for -"+msgs.newline()+self._idstr+"   "+instr.replace('\t','  '))
                    else: sidlist.append(ksspl[j])
                    if ksspl[j] not in specid:
                        msgs.error("There is no data with specid: "+ksspl[j]+" for -"+msgs.newline()+self._idstr+"   "+instr.replace('\t','  '))
                specidset=True
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
            if cpy_keych[keywdk[i]] == 1: msgs.error(keywdk[i]+" must be set for -"+msgs.newline()+self._idstr+"   "+instr)
        # Check that we have set a specid
        if len(specid) == 1 and not specidset:
            if len(cpy_keywd['specid']) == 0: cpy_keywd['specid'].append(specid[0])
        # Append the final set of keywords
        mp['mkey'].append(cpy_keywd)
        return mp, parid

    def load_xsec(self, ion):
        """
        Load Dima Verner's data.
        To call, use the following call
        info = load_data(dict({}))
        print info["C IV"]
        """
        from alis import alutils
        datname = "/".join(__file__.split("/")[:-1])+"/data/phionxsec.dat"
        data = np.loadtxt(datname)
        for i in range(data.shape[0]):
            elem = alutils.numtoelem(int(data[i, 0]))
            ionstage = alutils.numtorn(int(data[i, 0]) - int(data[i, 1]), subone=True)
            elion = elem + "_" + ionstage
            if elion != ion: continue  # Don't include the unnecessary data
            params = data[i, 2:]
            break
        # Return the result
        return params

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
        if i == 0: pin = par
        elif i == 1: pin = par
        else:
            msgs.error("Function "+self._idstr+" is badly defined in definition parin.")
        return pin

    def set_vars(self, p, level, mp, ival, wvrng=[0.0,0.0], spid='None', levid=None, nexbin=None, ddpid=None, getinfl=False):
        """
        Return the parameters for a Phionxs function to be used by 'call'
        """
        levadd=0
        params=np.zeros(self._pnumr+10)
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
                            namespace = dict({'p': p})
                            exec(cmd, namespace)
                            lnkprm = namespace['lnkprm']
                levadd += 1
            else:
                getid = level+levadd
                levadd+=1
            if lnkprm is None:
                params[i] = self.parin(i, p[getid])
                if mp['mfix'][ival][i] == 0: parinf.append(getid)
            else:
                params[i] = lnkprm
        # Now get the cross-section data
        params[2:-1] = self.load_xsec(self._keywd['ion'].lstrip('0123456789'))
        # Update the ionization energy to correspond to the minimum wavelength in the linelist
        nv = np.where(self._atomic['Ion'] == self._keywd['ion'])[0]
        params[-1] = 6.62606957E-34 * 299792458.0 / (np.min(self._atomic['Wavelength'][nv])*1.0E-10 * 1.60217657E-19)
        if ddpid is not None:
            if ddpid not in parinf: return []
        if nexbin is not None:
            if nexbin[0] == "km/s": return params, 1
            elif nexbin[0] == "A" : return params, 1
            else: msgs.bug("bintype "+nexbin[0]+" should not have been specified in model function: "+self._idstr, verbose=self._verbose)
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
        if type(errs) is np.ndarray:
            errors = errs
        else:
            errors = params
        parid = [i for i in range(self._pnumr)]
        if len(mp['mpar'][istart]) == self._pnumr: numpar = self._pnumr
        else:
            numpar = len(mp['mpar'][istart])
            for i in range(self._pnumr-numpar):
                del parid[1]
        add = numpar
        havtie = 0
        tienum = 0
        levadd = 0
        outstring = ['  %s ' % (self._idstr)]
        errstring = ['# %s ' % (self._idstr)]
        for i in range(numpar):
            if mp['mkey'][istart]['input'][self._parid[i]] == 0: # Parameter not given as input
                outstring.append( "" )
                errstring.append( "" )
                continue
            elif mp['mkey'][istart]['input'][self._parid[i]] == 1: pretxt = ""   # Parameter is given as input, without parid
            else: pretxt = self._parid[i]+"="                                    # Parameter is given as input, with parid
            if mp['mtie'][istart][i] >= 0:
                if reletter:
                    newfmt=pretxt+self.gtoef(params[mp['tpar'][mp['mtie'][istart][i]][1]],self._svfmt[parid[i]]+'{1:c}')
                    outstring.append( (newfmt).format(params[mp['tpar'][mp['mtie'][istart][i]][1]],97+mp['mtie'][istart][i]-32*mp['mfix'][istart][1]) )
                    if conv is None:
                        errstring.append( (newfmt).format(errors[mp['tpar'][mp['mtie'][istart][i]][1]],97+mp['mtie'][istart][i]-32*mp['mfix'][istart][1]) )
                    else:
                        if params[mp['tpar'][mp['mtie'][istart][i]][1]] < conv: cvtxt = "CONVERGED"
                        else: cvtxt = "!!!!!!!!!"
                        errstring.append( ('--{0:s}--{1:c}    ').format(cvtxt,97+mp['mtie'][istart][i]-32*mp['mfix'][istart][1]) )
                else:
                    newfmt=pretxt+self.gtoef(params[mp['tpar'][mp['mtie'][istart][i]][1]],self._svfmt[parid[i]]+'{1:s}')
                    outstring.append( (newfmt).format(params[mp['tpar'][mp['mtie'][istart][i]][1]],mp['tpar'][mp['mtie'][istart][i]][0]) )
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
                            outstring.append( (newfmt).format(params[level+levadd],97+tienum-32*mp['mfix'][istart][1]) )
                            if conv is None:
                                errstring.append( (newfmt).format(errors[level+levadd],97+tienum-32*mp['mfix'][istart][1]) )
                            else:
                                if params[level+levadd] < conv: cvtxt = "CONVERGED"
                                else: cvtxt = "!!!!!!!!!"
                                errstring.append( ('--{0:s}--{1:c}    ').format(cvtxt,97+tienum-32*mp['mfix'][istart][1]) )
                        else:
                            newfmt=pretxt+self.gtoef(params[level+levadd],self._svfmt[parid[i]]+'{1:s}')
                            outstring.append( (newfmt).format(params[level+levadd],mp['tpar'][tienum][0]) )
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
                        outstring.append( (newfmt).format(params[level+levadd]) )
                        if conv is None:
                            errstring.append( (newfmt).format(errors[level+levadd]) )
                        else:
                            if params[level+levadd] < conv: cvtxt = "CONVERGED"
                            else: cvtxt = "!!!!!!!!!"
                            errstring.append( ('--{0:s}--    ').format(cvtxt) )
                else:
                    newfmt=pretxt+self.gtoef(params[level+levadd],self._svfmt[parid[i]])
                    outstring.append( (newfmt).format(params[level+levadd]) )
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
        add = len(mp['mpar'][mnum])
        levadd = 0
        for i in range(len(mp['mpar'][mnum])):
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

