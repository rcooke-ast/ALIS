import os
import copy
import numpy as np
import datetime
from alis import load, logger
from alis.functions import base
import astropy.io.fits as pyfits
from matplotlib import pyplot as plt
from matplotlib import colormaps as pltcmaps
from alis.utils import getreason
msgs = logger.msgs()

def file_exists(slf, filename):
    """
    Check to see if a file exists before overwriting it
    """
    if slf._argflag['out']['overwrite']: ans='y'
    else: ans=''
    if os.path.exists(filename):
        while ans != 'y' and ans != 'n' and ans !='r':
            msgs.warn("File %s exists!" % (filename), verbose=slf._argflag['out']['verbose'])
            ans = input(msgs.input()+"Overwrite? (y/n) or rename? (r) - ")
            if ans == 'r':
                fileend=input(msgs.input()+"Enter new filename - ")
                filename = fileend
                if os.path.exists(filename): ans = ''
    return ans, filename


def save_asciifits(fname, slf, arr, model):
    """
    Save the best-fitting model into an ascii file.
    """
    sp, sn, ll, lu = arr
    wfek = list(slf._datopt['columns'][sp][sn].keys())
    maxn=0
    for i in wfek:
        if slf._datopt['columns'][sp][sn][i] > maxn: maxn = slf._datopt['columns'][sp][sn][i]
    data = np.zeros((lu-ll,maxn+2))
    for i in wfek:
        if slf._datopt['columns'][sp][sn][i] == -1: continue
        num = slf._datopt['columns'][sp][sn][i]
        if   i == 'wave':
            data[:,num] = slf._wavefull[sp][ll:lu]
        elif i == 'flux':
            data[:,num] = slf._fluxfull[sp][ll:lu]
        elif i == 'error':
            data[:,num] = slf._fluefull[sp][ll:lu]
        elif i == 'continuum':
            data[:,num] = slf._contfinal[sp][ll:lu]
        elif i == 'zerolevel':
            data[:,num] = slf._zerofinal[sp][ll:lu]
        elif i == 'fitrange':
            out = np.zeros(lu-ll).astype(float)
            w = np.where((slf._wavefull[sp][ll:lu] >= slf._posnfit[sp][2*sn+0]) & (slf._wavefull[sp][ll:lu] <= slf._posnfit[sp][2*sn+1]))
            out[w] = np.isin(slf._wavefull[sp][ll:lu][w], slf._wavefit[sp]).astype(float)
            data[:,num] = out
        elif i == 'loadrange':
            data[:,num] = np.ones(lu-ll)
        elif i == 'systematics':
            data[:,num] = slf._systfull[sp][ll:lu]
        elif i == 'resolution':
            msgs.bug("I haven't completed writing out 'resolution' to file yet... sorry")
            data[:,num] = np.zeros(lu-ll)
        else:
            msgs.bug("I didn't expect the keyword '{0:s}' when saving fits file -".format(i)+msgs.newline()+fname+".dat")
    data[:, -1] = model
    # Save the file
    dirname = os.path.dirname(fname + ".dat")
    if dirname != '':
        # Check the directory exists
        if not os.path.exists(dirname):
            os.makedirs(dirname)
    np.savetxt(fname + ".dat", data)
    return


def save_fitsfits(fname, slf, arr, model):
    """
    Save the best-fitting model into fits files.
    """
    sp, sn, ll, lu = arr
    wfek = list(slf._datopt['columns'][sp][sn].keys())
    maxn=0
    for i in wfek:
        if slf._datopt['columns'][sp][sn][i] > maxn: maxn = slf._datopt['columns'][sp][sn][i]
    data = np.zeros((lu-ll,maxn+2))
    for i in wfek:
        if slf._datopt['columns'][sp][sn][i] == -1: continue
        num = slf._datopt['columns'][sp][sn][i]
        if   i == 'wave':
            data[:,num] = slf._wavefull[sp][ll:lu]
        elif i == 'flux':
            data[:,num] = slf._fluxfull[sp][ll:lu]
        elif i == 'error':
            data[:,num] = slf._fluefull[sp][ll:lu]
        elif i == 'continuum':
            data[:,num] = slf._contfinal[sp][ll:lu]
        elif i == 'zerolevel':
            data[:,num] = slf._zerofinal[sp][ll:lu]
        elif i == 'fitrange':
            out = np.zeros(lu-ll).astype(float)
            w = np.where((slf._wavefull[sp][ll:lu] >= slf._posnfit[sp][2*sn+0]) & (slf._wavefull[sp][ll:lu] <= slf._posnfit[sp][2*sn+1]))
            out[w] = np.isin(slf._wavefull[sp][ll:lu][w], slf._wavefit[sp]).astype(float)
            data[:,num] = out
        elif i == 'loadrange':
            data[:,num] = np.ones(lu-ll)
        elif i == 'systematics':
            data[:,num] = slf._systfull[sp][ll:lu]
        elif i == 'resolution':
            msgs.bug("I haven't completed writing out 'resolution' to file yet... sorry")
            data[:,num] = np.zeros(lu-ll)
        else:
            msgs.bug("I didn't expect the keyword '{0:s}' when saving fits file -".format(i)+msgs.newline()+fname+".dat")
    data[:,-1] = model
    # Save the file
    hdu = pyfits.PrimaryHDU(data.transpose())
    hdulist = pyfits.HDUList([hdu])
    hdulist[0].header['label'] = slf._datopt['label'][sp][sn]
    hdulist[0].header['alisfits'] = "fits"
    ans = 'y'
    if os.path.exists(fname+".fits"):
        if slf._argflag['out']['overwrite']:
            os.remove(fname+".fits")
        else:
            ans = ''
            while ans != 'y' and ans != 'n':
                msgs.warn("File %s exists!" % (fname+".fits"), verbose=slf._argflag['out']['verbose'])
                ans = input(msgs.input()+"Overwrite? (y/n) - ")
            if ans == 'y': os.remove(fname+".fits")
    if ans == 'y': hdulist.writeto(fname+".fits")
    return


def save_onefits(fname, slf):
    """
    Save the best-fitting model into a single fits file with multiple extensions.
    """
    # Setup the HDU
    hdu = pyfits.PrimaryHDU()
    # Get input model and place it in the fits header
    plines = ''.join(slf._parlines).replace("\t","  ")
    dlines = ''.join(slf._datlines).replace("\t","  ")
    mlines = ''.join(slf._modlines).replace("\t","  ")
    llines = ''.join(slf._lnklines).replace("\t","  ")
    pcard=pyfits.Card('parlines',','.join([str(ord(c)) for c in plines]))
    dcard=pyfits.Card('datlines',','.join([str(ord(c)) for c in dlines]))
    mcard=pyfits.Card('modlines',','.join([str(ord(c)) for c in mlines]))
    lcard=pyfits.Card('lnklines',','.join([str(ord(c)) for c in llines]))
    hdu.header.append(pcard)
    hdu.header.append(dcard)
    hdu.header.append(mcard)
    # Get output model and place it in the fits header
    fit_info=[(slf._tend - slf._tstart)/3600.0, slf._fitresults.fnorm, slf._fitresults.dof, slf._fitresults.niter, slf._fitresults.status]
    outstr = save_model(slf,slf._fitresults.params,slf._fitresults.perror,fit_info,printout=False,filename=None,getlines=True,save=False)
    ocard=pyfits.Card('output',','.join([str(ord(c)) for c in outstr]))
    hdu.header.append(ocard)
    hdulist = pyfits.HDUList([hdu]) # Insert the primary HDU (input model)
    # Now loop through all the data and put it into an HDU
    datnum = 1
    for sp in range(len(slf._posnfull)):
        for sn in range(len(slf._posnfull[sp])-1):
            ll = slf._posnfull[sp][sn]
            lu = slf._posnfull[sp][sn+1]
            # Prepare the model array:
            modelout = -9.999999999E9*np.ones(slf._wavefull[sp][ll:lu].size)
            w = np.where((slf._wavefull[sp][ll:lu] >= slf._posnfit[sp][2*sn+0]) & (slf._wavefull[sp][ll:lu] <= slf._posnfit[sp][2*sn+1]))
            modelout[w] = slf._modfinal[sp][ll:lu][w]
            # Get the columns information for this index
            wfek = list(slf._datopt['columns'][sp][sn].keys())
            maxn=0
            for i in wfek:
                if slf._datopt['columns'][sp][sn][i] > maxn: maxn = slf._datopt['columns'][sp][sn][i]
            data = np.zeros((lu-ll,maxn+2))
            ncol = 0
            colarr=[]
            for i in wfek:
                if slf._datopt['columns'][sp][sn][i] == -1: continue
                num = slf._datopt['columns'][sp][sn][i]
                if   i == 'wave':
                    data[:,num] = slf._wavefull[sp][ll:lu]
                elif i == 'flux':
                    data[:,num] = slf._fluxfull[sp][ll:lu]
                elif i == 'error':
                    data[:,num] = slf._fluefull[sp][ll:lu]
                elif i == 'continuum':
                    data[:,num] = slf._contfinal[sp][ll:lu]
                elif i == 'zerolevel':
                    data[:,num] = slf._zerofinal[sp][ll:lu]
                elif i == 'fitrange':
                    out = np.zeros(lu-ll).astype(float)
                    w = np.where((slf._wavefull[sp][ll:lu] >= slf._posnfit[sp][2*sn+0]) & (slf._wavefull[sp][ll:lu] <= slf._posnfit[sp][2*sn+1]))
                    out[w] = np.isin(slf._wavefull[sp][ll:lu][w], slf._wavefit[sp]).astype(float)
                    data[:,num] = out
                elif i == 'loadrange':
                    data[:,num] = np.ones(lu-ll)
                elif i == 'systematics':
                    data[:,num] = slf._systfull[sp][ll:lu]
                elif i == 'resolution':
                    msgs.bug("I haven't completed writing out 'resolution' to file yet... sorry")
                    data[:,num] = np.zeros(lu-ll)
                else:
                    msgs.bug("I didn't expect the keyword '{0:s}' when saving fits file -".format(i)+msgs.newline()+fname+".dat")
                coltxt = "{0:2d}".format(ncol)
                colarr.append([coltxt,"{0:s}:{1!s}".format(i,num)])
                ncol += 1
            data[:,-1] = modelout
            # Save the data into a new HDU
            hdulist.append(pyfits.ImageHDU(data.transpose())) # Add a new Image HDU
            # Insert the data options
            hdulist[datnum].header['bintype']  = slf._datopt['bintype'][sp][sn]
            for i in colarr:
                hdulist[datnum].header[i[0]]   = i[1]
            hdulist[datnum].header['filename'] = slf._snipnames[sp][sn]
            hdulist[datnum].header['fitrange'] = slf._datopt['fitrange'][sp][sn]
            hdulist[datnum].header['loadrange'] = slf._datopt['loadrange'][sp][sn]
            hdulist[datnum].header['label']    = slf._datopt['label'][sp][sn]
            hdulist[datnum].header['nsubpix']  = slf._datopt['nsubpix'][sp][sn]
            hdulist[datnum].header['plotone']  = slf._datopt['plotone'][sp][sn]
            hdulist[datnum].header['specid']   = slf._datopt['specid'][sp][sn]
            resspl = slf._resn[sp][sn].split("(")
            hdulist[datnum].header['resfunc']  = resspl[0]
            respar = resspl[1].rstrip(")").split(",")
            for i in range(len(respar)):
                restxt = "respar{0:02d}".format(i)
                hdulist[datnum].header[restxt] = respar[i]
            datnum += 1
    # Finally, append a keyword to the primary HDU to tell ALIS it's a onefits file, and save it.
    hdulist[0].header['alisfits'] = "onefits"
    hdulist[0].header['modname']  = slf._argflag['run']['modname']
    hdulist[0].header['numext']   = datnum
    ans = 'y'
    if os.path.exists(fname+".fits"):
        if slf._argflag['out']['overwrite']:
            os.remove(fname+".fits")
        else:
            ans = ''
            while ans != 'y' and ans != 'n' and ans != 'r':
                msgs.warn("File %s exists!" % (fname+".fits"), verbose=slf._argflag['out']['verbose'])
                ans = input(msgs.input()+"Overwrite? (y/n) or rename? (r) - ")
                if ans == 'r':
                    fname=input(msgs.input()+"Enter new filename (without the extension) - ")
                    if os.path.exists(fname+".fits"): ans = ''
            if ans == 'y': os.remove(fname+".fits")
    if ans == 'y': hdulist.writeto(fname+".fits")
    return


def save_modelfits(slf):
    msgs.info("Writing out the model fits", verbose=slf._argflag['out']['verbose'])
    stf = 0
    fit_fnames = np.array([]).astype(str)
    fnames = np.array([]).astype(str)
#	stf, enf = [0 for all in slf._posnfull], [0 for all in slf._posnfull]
    usdtwice, usdtwind, usdtwext = np.array([]).astype(str), np.array([]).astype(int), np.array([]).astype(str)
    if slf._argflag['out']['onefits']: wvarr, fxarr, erarr, mdarr = [], [], [], []
    # If we are generating fakedata, find the peak value of the model
    if slf._argflag['generate']['data'] and slf._argflag['generate']['peaksnr'] > 0.0:
        modmax = [0.0 for all in slf._specid]
        for sp in range(len(slf._posnfull)):
            for sn in range(len(slf._posnfull[sp])-1):
                ll = slf._posnfull[sp][sn]
                lu = slf._posnfull[sp][sn+1]
                maxval = np.max(slf._modfinal[sp][ll:lu])
                if maxval > modmax[sp]: modmax[sp] = maxval
        peakerr = 1.0/slf._argflag['generate']['peaksnr']
        if peakerr**2 < slf._argflag['generate']['skyfrac']**2:
            msgs.error("The following condition must hold for generated data:"+msgs.newline()+"skyfrac < 1/peaksnr")
        objterr = [0.0 for all in slf._specid]
        objtsnr = [0.0 for all in slf._specid]
        for sp in range(len(slf._posnfull)):
            objterr[sp] = modmax[sp]*np.sqrt(peakerr**2 - (slf._argflag['generate']['skyfrac'])**2)
            objtsnr[sp] = modmax[sp]/objterr[sp]
        slf._fluxfull = copy.deepcopy(slf._modfinal)
    # Now iterate through the spectra and save the output
    for sp in range(len(slf._posnfull)):
        for sn in range(len(slf._posnfull[sp])-1):
            ll = slf._posnfull[sp][sn]
            lu = slf._posnfull[sp][sn+1]
            # Check if the outfile already exists:
            fname = slf._snipnames[sp][sn]
            fspl = fname.split('.')
            if slf._argflag['generate']['data']:
                if os.path.exists(slf._snipnames[sp][sn]):
                    fnoext = '.'.join(fspl[:-1])+'_model'
                else:
                    fnoext = '.'.join(fspl[:-1])
            else:
                fnoext = '.'.join(fspl[:-1])+'_fit'
            an = np.where(fit_fnames == fnoext)
            if np.size(an[0]) != 0: # The same snip is used more than once as input
                un = np.where(usdtwice == fnoext)
                if np.size(un[0]) == 0: # First time this snip has been used twice
                    usdtwice = np.append(usdtwice, fnoext)
                    usdtwind = np.append(usdtwind, 2)
                    usdtwext = np.append(usdtwext, fspl[-1])
                else: # This snip is seen more than twice
                    usdtwind[un[0]] += 1
                # Now that the relevant additions have been made to the arrays, get the index
                un = np.where(usdtwice == fnoext)
                fnoext += "%02i" % (usdtwind[un][0])
            # Prepare the model array:
            modelout = -9.999999999E9*np.ones(slf._wavefull[sp][ll:lu].size)
            w = np.where((slf._wavefull[sp][ll:lu] >= slf._posnfit[sp][2*sn+0]) & (slf._wavefull[sp][ll:lu] <= slf._posnfit[sp][2*sn+1]))
            modelout[w] = slf._modfinal[sp][ll:lu][w]
            # Add noise if we are generating fakedata
            if slf._argflag['generate']['data']:
                if not os.path.exists(slf._snipnames[sp][sn]):
                    if slf._argflag['generate']['peaksnr'] > 0.0:
                        slf._fluefull[sp][ll:lu] = np.sqrt((slf._modfinal[sp][ll:lu]/objtsnr[sp])**2 + (modmax[sp]*slf._argflag['generate']['skyfrac'])**2)
                        slf._fluxfull[sp][ll:lu] += np.random.normal(0.0, slf._fluefull[sp][ll:lu])
                else:
                    if np.size(np.where(slf._fluefull[sp][ll:lu] <= 0.0)[0]) != 0:
                        if slf._argflag['generate']['peaksnr'] > 0.0:
                            msgs.warn("Couldn't add noise to generated data -"+msgs.newline()+"the error array contains zero or negative values", verbose=slf._argflag['out']['verbose'])
                    else:
                        slf._fluxfull[sp][ll:lu] += np.random.normal(0.0, slf._fluefull[sp][ll:lu])
            # Now that we have the output name, send the data away to be written to file
            if slf._argflag['out']['fits']:
                if slf._argflag['out']['onefits']:
                    # Store the fits files in an array and write them out at the end of the for loop
                    ext = '.fits'
                    #wvarr.append(slf._wavefull[sp][ll:lu])
                    #fxarr.append(slf._fluxfull[sp][ll:lu])
                    #erarr.append(slf._fluefull[sp][ll:lu])
                    #mdarr.append(modelout)
                elif fspl[-1] in ["fits", "fit"]:
                    # Write out this snip to a fits file
                    ext = '.fits'
                    save_fitsfits(fnoext, slf, [sp,sn,ll,lu], modelout)
                else:
                    # Write out the data to an ascii (.dat) file.
                    ext = '.dat'
                    save_asciifits(fnoext, slf, [sp,sn,ll,lu], modelout)
                fit_fnames = np.append(fit_fnames, fnoext)
                fnames = np.append(fnames, fnoext+ext)
    if slf._argflag['out']['fits']:
        if slf._argflag['out']['onefits']: # The user has requested that all model fits be written into a single fits file:
            outspl = slf._argflag['run']['modname'].split('.')
            outname = '.'.join(outspl[:-1])+'_fit'
            save_onefits(outname, slf)
        else: # For snips that were used twice, rename the first instance to have suffix "01"
            for i in range(len(usdtwice)):
                os.rename(usdtwice[i]+"."+usdtwext[i],usdtwice[i]+"01."+usdtwext[i])
        msgs.info("Saved absorption line fits", verbose=slf._argflag['out']['verbose'])
    # If data has been generated, return the data within slf
    if slf._argflag['generate']['data']:
        return slf, fnames
    else:
        return fnames


def print_model(params, mp, errs=None, reletter=False, blind=False, getlines=False, verbose=2, funcarray=[None,None,None]):
    function=funcarray[0]
    funccall=funcarray[1]
    funcinst=funcarray[2]
    level=0
    outstring = ""
    errstring = "#\n# Errors:\n#\n"
    cvstring  = ""
    cvastring = ""
    cvestring = "# Errors:\n#\n"
    shstring  = ""
    shastring = ""
    shestring = "# Errors:\n#\n"
    donecv, donesh, donezl = [], [], []
    lastemab=""
    for i in range(len(mp['mtyp'])):
        #if errs is not None and mp['emab'][i] == "cv": continue
        mtyp = mp['mtyp'][i]
        if mp['emab'][i] != lastemab:
            thisemabtag = mp['emab'][i]
            if mp['emab'][i]=="em":
                aetag = "emission"
            elif mp['emab'][i]=="ab":
                aetag = "absorption"
            elif mp['emab'][i]=="cv":
                aetag = "Convolution"
            elif mp['emab'][i]=="sh":
                aetag = "Shift"
            elif mp['emab'][i]=="zl":
                aetag = "zerolevel"
            else:
                thisemabtag = lastemab
            # Place the model details into a string
            if mp['emab'][i] == "cv":
                cvstring  += " "+aetag+"\n"
                cvestring += "#"+aetag+"\n"
            elif mp['emab'][i] == "sh":
                shstring  += " "+aetag+"\n"
                shestring += "#"+aetag+"\n"
            elif mp['emab'][i] == "va":
                pass
            else:
                outstring += " "+aetag+"\n"
                errstring += "#"+aetag+"\n"
            lastemab = thisemabtag
        if errs is None:
            funcinst[mtyp]._keywd = mp['mkey'][i]
            outstr, level = funccall[mtyp].parout(funcinst[mtyp], params, mp, i, level)
            if mp['emab'][i] == "cv":
                cvastring += outstr
            elif mp['emab'][i] == "sh":
                shastring += outstr
            if outstr in donecv or outstr in donesh or outstr in donezl: continue
            if mp['emab'][i] == "cv": donecv.append(outstr) # Make sure we don't print convolution parameters more than once.
            elif mp['emab'][i] == "sh": donesh.append(outstr) # Make sure we don't print shift parameters more than once.
            elif mp['emab'][i] == "zl": donezl.append(outstr) # Make sure we don't print zerolevel more than once.
            # Place the model details into a string
            if mp['emab'][i] == "cv":
                cvstring  += outstr
            elif mp['emab'][i] == "sh":
                shstring  += outstr
            else:
                outstring += outstr
        else:
            funcinst[mtyp]._keywd = mp['mkey'][i]
            outstr, errstr, level = funccall[mtyp].parout(funcinst[mtyp], params, mp, i, level, errs=errs)
            if mp['emab'][i] == "cv":
                cvastring += outstr
            elif mp['emab'][i] == "sh":
                shastring += outstr
            if outstr in donecv or outstr in donesh or outstr in donezl: continue
            if mp['emab'][i] == "cv": donecv.append(outstr) # Make sure we don't print convolution parameters more than once.
            elif mp['emab'][i] == "sh": donesh.append(outstr) # Make sure we don't print shift parameters more than once.
            elif mp['emab'][i] == "zl": donezl.append(outstr) # Make sure we don't print zerolevel more than once.
            # Place the model details into a string
            if mp['emab'][i] == "cv":
                cvstring  += outstr
                cvestring += errstr
            elif mp['emab'][i] == "sh":
                shstring  += outstr
                shestring += errstr
            else:
                outstring += outstr
                errstring += errstr
    if blind: return outstring
    if getlines:
        if errs is None:
            return outstring.split("\n"), [cvstring.split("\n"),cvastring.split("\n"),shstring.split("\n"),shastring.split("\n")]
        else:
            return outstring.split("\n"), errstring.split("\n"), [cvstring.split("\n"),cvestring.split("\n"),cvastring.split("\n"),shstring.split("\n"),shestring.split("\n"),shastring.split("\n")]
    if errs is None:
        return outstring, [cvstring,cvastring,shstring,shastring]
    else:
        return outstring, errstring, [cvstring,cvestring,cvastring,shstring,shestring,shastring]


def strip_cli_override_block(parlines):
    """
    ``parlines`` with any previously written command-line override block removed.

    Recognised by :data:`alis.load.CLI_OVERRIDE_MARK`, a trailing comment on each
    of the block's live settings. The surrounding header cannot be used: ALIS
    drops comment-only lines when it reads a file, so by the time the writer sees
    ``_parlines`` the header is gone and only the settings remain.

    Generated by RJC and Claude.
    """
    return [ln for ln in parlines if load.CLI_OVERRIDE_MARK not in ln]


def cli_override_block(overrides, previous=()):
    """
    The settings block recording what the command line changed, or ``""``.

    The settings themselves are written **live**, not commented out, so that
    ``run_alis model.mod.out`` reproduces the run that produced the file
    (Q6.12). Alongside them, commented, go the value each one replaced and the
    flags that did not persist -- a record of the whole invocation, without
    changing what a re-run does (Q6.25). Only settings marked ``persist`` are
    made live: ``plot only`` would stop the re-run fitting at all,
    ``out modelname`` would make it clobber the original output, and the
    simulation counters would make it redo the whole set (Q6.20).

    Generated by RJC and Claude.
    """
    # Overrides recorded by an earlier run are carried forward, with this run's
    # taking precedence key by key. Without that, re-fitting a .mod.out would
    # produce a file that no longer reproduces the run it describes: the second
    # generation ran with the inherited settings but would record none of them
    # (Q6.25).
    carried = {}
    for line in previous:
        parts = line.split()
        if len(parts) >= 2:
            carried[(parts[0], parts[1])] = line if line.endswith("\n") else line + "\n"
    for section, key, _old, _new, persist in overrides or []:
        if persist:
            carried.pop((section, key), None)
    live = [o for o in overrides or [] if o[4]]
    noted = [o for o in overrides or [] if not o[4]]
    if not carried and not live and not noted:
        return ""
    out = "\n" + load.CLI_OVERRIDE_HEADER + "\n"
    for line in carried.values():
        out += line
    for section, key, old, new, _persist in live:
        out += "{0:s} {1:s} {2!s}   {3:s} was {4!s}\n".format(
            section, key, new, load.CLI_OVERRIDE_MARK, old)
    if noted:
        out += "# The following were also given on the command line, but describe\n"
        out += "# that run rather than this model, so they are recorded only:\n"
        for section, key, old, new, _persist in noted:
            out += "#   {0:s} {1:s} {2!s}    (was {3!s})\n".format(section, key, new, old)
    out += load.CLI_OVERRIDE_END + "\n"
    return out


def save_model(slf,params,errors,info,printout=True,extratxt=["",""],filename=None,getlines=False,save=True, overwrite=False):
    """
    Save the input model into an output script
    that can be run as input.
    """
    verbose=slf._argflag['out']['verbose']
    msgs.info("Saving the best-fitting model parameters", verbose=verbose)
    if filename is None:
        filename = extratxt[0]+slf._argflag['out']['modelname']+extratxt[1]
    prestringA = "#\n#  Generated by ALIS on {0:s}\n#\n".format(datetime.datetime.now().strftime("%d/%m/%y at %H:%M:%S"))
    prestringA += "#   Running Time (hrs)  = {0:f}\n".format(info[0])
    prestringA += "#   Initial Chi-Squared = {0:f}\n".format(slf._chisq_init)
    prestringA += "#   Bestfit Chi-Squared = {0:f}\n".format(info[1])
    prestringA += "#   Degrees-of-Freedom  = {0:d}\n".format(info[2])
    prestringA += "#   Reduced Chi-Squared = {0:f}\n".format(info[1]/info[2])
    prestringA += "#   Num. of Iterations  = {0:d}\n".format(info[3])
    prestringA += "#   Convergence Reason  = {0:s}\n".format(getreason(info[4],verbose=verbose))
    prestringA += "\n"
    inputmodl = "#\n"
    # Any override block written by a *previous* run is dropped rather than
    # carried through: _parlines is everything the reader found, so re-fitting a
    # .mod.out would otherwise emit the old block and append the new one, and the
    # file would accumulate duplicates that warn on every read (Q6.25).
    for line in strip_cli_override_block(slf._parlines):
        prestringA += line
        inputmodl += "#   "+line
    override_block = cli_override_block(
        getattr(slf, "_cli_overrides", None),
        previous=[ln for ln in slf._parlines if load.CLI_OVERRIDE_MARK in ln],
    )
    prestringA += override_block
    for line in override_block.splitlines(True):
        inputmodl += "#   "+line
    prestringA +="\ndata read\n"
    inputmodl += "#   data read\n"
    for i in range(len(slf._datlines)):
        #prestring += slf._datlines[i]
        inputmodl += "#   "+slf._datlines[i]
    prestringB ="data end\n"
    inputmodl += "#   data end\n"
    prestringB +="\nmodel read\n"
    inputmodl += "#   model read\n"
    modcomlin=[]
    modcomind=[]
    toutstring=''
    for i in range(len(slf._modlines)):
        if len(slf._modlines[i].strip()) == 0: # Nothing on a line
            inputmodl += "#  "+slf._modlines[i]
            continue
        if slf._modlines[i].split()[0] in ["fix", "lim"]: toutstring += slf._modlines[i].replace('\t',' ')
        if slf._modlines[i].lstrip()[0] == '#':
            modcomlin.append(slf._modlines[i].rstrip('\n'))
            modcomind.append(i)
        inputmodl += "#   "+slf._modlines[i]
    outstring, errstring, arrstring = print_model(params,slf._modpass,errs=errors,verbose=slf._argflag['out']['verbose'],funcarray=slf._funcarray)
    cvstring, cvestring, cvastring = arrstring[0], arrstring[1], arrstring[2]
    shstring, shestring, shastring = arrstring[3], arrstring[4], arrstring[5]
    if printout and slf._argflag['out']['verbose'] != -1:
        print("\n####################################################")
        print(outstring)
        print(errstring)
        print("#"+"\n#".join(cvstring.replace("Convolution","Convolution Models:").split("\n")))
        print(cvestring.replace("#Convolution\n",""))
        print("#"+"\n#".join(shstring.replace("Shift","Shift Models:").split("\n")))
        print(shestring.replace("#Shift\n","")+"\n")
        print("####################################################\n")
    # Reinsert the comments at the original locations
    outstrspl = (toutstring+outstring).split('\n')
    for i in range(len(modcomlin)): outstrspl.insert(modcomind[i],modcomlin[i])
    outstring = '\n'.join(outstrspl)
    # Include an end tag for the model
    outstring += "model end\n"
    inputmodl += "#   model end\n#\n\n"
    # Include the model links
    if len(slf._lnklines) != 0:
        outstring += "\nlink read\n"
        for i in range(len(slf._lnklines)): outstring += slf._lnklines[i]
        outstring += "link end\n"
    # Update datlines for the newly derived instrument resolution
    cnum=0
    snum=0
    dstrarr = ["" for all in slf._datlines]
    for sp in range(len(slf._specid)):
        # Index of the loaded snip within this specid. One data line produces
        # exactly one snip (checked across the shipped examples and the 351-line
        # DH_orders model), so this advances in step with the matched lines,
        # like cnum/snum above.
        sn = 0
        for i in range(len(slf._datlines)):
            if slf._datlines[i].lstrip() == "": continue # This line is needed for OneFits.
            if slf._datlines[i].lstrip()[0] == "#": dstrarr[i] += slf._datlines[i]
            datspl = slf._datlines[i].split()
            spmatch = False
            for j in range(1,len(datspl)):
                dspl = datspl[j].split("=")
                if dspl[0] == "specid":
                    if dspl[1] == slf._specid[sp]:
                        spmatch = True
                    break
            if not spmatch: continue
            gotres=False
            gotshf=False
            for j in range(1,len(datspl)):
                dspl = datspl[j].split("=")
                if dspl[0] == "resolution":
                    cspl = cvastring.split("\n")[cnum].split()
                    # Sub-keywords inside the parentheses are separated with
                    # ':', not '=' (Stage 5.4). load_data splits each data-line
                    # token on '=' and keeps only field 1, so an '=' in here
                    # truncates the value -- 'resolution=lsf(name=STIS,...)'
                    # reads back as the function 'lsf(name'. The convolution
                    # loaders that take keywords (lsf, lsffile, lsfspline,
                    # apod, multivfwhm) all undo this with
                    # instr.replace(":", "=") on the way in. Purely numeric
                    # parameters contain no '=', so this is a no-op for them.
                    cpars = ",".join(cspl[1:]).replace("=", ":")
                    datspl[j] = "resolution={0:s}({1:s})".format(cspl[0],cpars)
                    gotres = True
                elif dspl[0] == "shift":
                    sspl = shastring.split("\n")[snum].split()
                    spars = ",".join(sspl[1:])
                    datspl[j] = "shift={0:s}({1:s})".format(sspl[0],spars)
                    gotshf = True
            # Record the pixel-load buffer that was actually used (Stage 5.4).
            # ALIS sizes the buffer from the resolution at load time, and the
            # line above has just been rewritten with the *fitted* resolution --
            # so without this, re-reading a .mod.out loads a different set of
            # pixels from the run that produced it.
            #
            # This is recorded as a pixel *count* rather than as a
            # 'loadrange=[wmin,wmax]', because an explicit loadrange is itself
            # widened by the resolution rule on the way back in
            # (load.load_data), so writing the loaded range simply re-inflates
            # it. A count is independent of the resolution, which is the whole
            # point. The two sides are stored separately: the resolution rule
            # extends by a wavelength, so it does not in general cover the same
            # number of pixels on each side.
            #
            # Only added when the user gave no loadrange of their own: an
            # explicit one (including 'loadrange=all', used 1231 times in this
            # repo) already round-trips, and states an intent that must not be
            # narrowed to whatever this particular fit happened to need.
            if not any(t.split("=")[0] in ("loadrange", "bufferpix")
                       for t in datspl[1:]):
                posn = slf._posnfull[sp]
                lwave = slf._wavefull[sp][posn[sn]:posn[sn+1]]
                fitlo, fithi = slf._posnfit[sp][2*sn], slf._posnfit[sp][2*sn+1]
                if lwave.size != 0:
                    nleft = int(np.sum(lwave < fitlo))
                    nright = int(np.sum(lwave > fithi))
                    datspl.append("bufferpix=[{0:d},{1:d}]".format(nleft, nright))
            sn += 1
            cnum += 1
            snum += 1

#			if not gotres:
#				dstrarr[i] += slf._datlines[i]
#			else:
#				dstrarr[i] += "  " + "  ".join(datspl) + "\n"
            dstrarr[i] += "  " + "  ".join(datspl) + "\n"
    datstring = "".join(dstrarr)
    # Save the output
    if save:
        if slf._argflag['out']['overwrite'] or overwrite: ans='y'
        else: ans=''
        if os.path.exists(filename):
            while ans != 'y' and ans != 'n' and ans !='r':
                msgs.warn("File %s exists!" % (filename), verbose=verbose)
                ans = input(msgs.input()+"Overwrite? (y/n) or rename? (r) - ")
                if ans == 'r':
                    fileend=input(msgs.input()+"Enter new filename - ")
                    filename = fileend
                    if os.path.exists(filename): ans = ''
        if ans != 'n':
            infile = open(filename,"w")
            infile.write(prestringA)
            infile.write(datstring)
            infile.write(prestringB)
            infile.write(outstring)
            infile.write("\n"+errstring+"\n")
            infile.write("#"+"\n#".join(cvstring.replace("Convolution","Convolution Models:").split("\n")))
            infile.write("\n"+cvestring.replace("#Convolution\n","")+"\n")
            infile.write("#"+"\n#".join(shstring.replace("Shift","Shift Models:").split("\n")))
            infile.write("\n"+shestring.replace("#Shift\n","")+"\n")
            infile.write("\n###################################################")
            infile.write("\n#                                                 #")
            infile.write("\n#          HERE IS A COPY OF THE INPUT MODEL      #")
            infile.write("\n#                                                 #")
            infile.write("\n###################################################\n")
            infile.write(inputmodl)
            infile.close()
            msgs.info("Saved output file successfully:"+msgs.newline()+filename, verbose=verbose)
    if getlines:
        sendstr  = prestringA + datstring + prestringB + outstring + "\n"+errstring+"\n"
        sendstr += "#"+"\n#".join(cvstring.replace("Convolution","Convolution Models:").split("\n"))
        sendstr += "\n"+cvestring.replace("#Convolution\n","")+"\n"
        sendstr += "#"+"\n#".join(shstring.replace("Shift","Shift Models:").split("\n"))
        sendstr += "\n"+shestring.replace("#Shift\n","")+"\n"
        return sendstr


def save_covar(slf, covar):
    """
    Save the covariance matrix into an output ascii file
    """
    msgs.info("Writing out the covariance matrix for the best-fitting model parameters", verbose=slf._argflag['out']['verbose'])
    if covar is None:
        msgs.warn("Covariance matrix is 'None', did you interupt the fit?", verbose=slf._argflag['out']['verbose'])
        msgs.info("Not writing out covariance matrix", verbose=slf._argflag['out']['verbose'])
        return
    if slf._argflag['out']['overwrite']: ans='y'
    else: ans=''
    filename=slf._argflag['out']['covar']
    if os.path.exists(filename) or filename == "":
        while ans != 'y' and ans != 'n' and ans !='r':
            if filename == "":
                msgs.warn("You must provide a filename to save the covariance matrix!", verbose=slf._argflag['out']['verbose'])
                ans = 'r'
            else:
                msgs.warn("File %s exists!" % (filename), verbose=slf._argflag['out']['verbose'])
                ans = input(msgs.input()+"Overwrite? (y/n) or rename? (r) - ")
            if ans == 'r':
                fileend=input(msgs.input()+"Enter new filename - ")
                filename = fileend
                if os.path.exists(filename): ans = ''
    if ans != 'n':
        fnspl = filename.split('.')
        if fnspl[-1] in ['fit','fits']:
            hdu = pyfits.PrimaryHDU(covar)
            hdulist = pyfits.HDUList([hdu])
            hdulist[0].header['alisfits'] = "covar"
            hdulist.writeto(filename)
        else:
            np.savetxt(filename, covar)
        msgs.info("Saved covariance matrix successfully:"+msgs.newline()+filename, verbose=slf._argflag['out']['verbose'])
        # Generate the correlation matrix from the covariance matrix
        outsize = int(np.sqrt(np.shape(np.where(covar!=0.0))[1]))
        if float(outsize) != np.sqrt(np.shape(np.where(covar!=0.0))[1]): msgs.bug("Error when deriving correlation matrix.", verbose=slf._argflag['out']['verbose'])
        corrM = np.zeros((outsize,outsize))
        sig    = np.sqrt(np.diag(covar))
        sigsig = np.dot(sig[:,np.newaxis],sig[np.newaxis,:])
        w = np.where(covar != 0.0)
        wc = np.where(corrM == 0.0)
        corrM[wc] = covar[w]/sigsig[w]
#		corrM = np.zeros(covar.shape)
#		corrM = covarNZ/sigsig
        # Mask out the fixed parameters (where the covariance matrix is 0.0)
#		w = np.where(covarNZ == 0.0)
#		mask = np.zeros(corrM.shape)
#		mask[w] = 1.0
        # Create a new array with the mask applied and plot it
#		cplt = np.ma.array(corrM, mask=mask)
        # Choose a colormap without white and set the bad pixels to white
        # matplotlib.cm.get_cmap was removed in matplotlib 3.9; the registry
        # form (3.6+) is the supported replacement. Cosmetic only -- this
        # colormap is used solely for the correlation-matrix png.
        cmap = pltcmaps['jet'].resampled(10)
#		cmap.set_bad('w')
        plt.imshow(corrM, interpolation="nearest", cmap=cmap, vmin=-1.0, vmax=1.0)
        plt.title("Correlation Matrix for: "+filename)
        tks=np.linspace(-1.0,1.0,11,endpoint=True)
        cbar=plt.colorbar(ticks=tks)
        # os.path.splitext, not str.rstrip: rstrip removes any trailing
        # characters *in that set*, so a covariance name with no extension
        # ('out covar mycovar') had every character stripped and the image was
        # written to a file called 'png' (Stage 5.4).
        imgname = os.path.splitext(filename)[0] + '.png'
        plt.savefig(imgname)
        msgs.info("Saved image of covariance matrix to:"+msgs.newline()+imgname, verbose=slf._argflag['out']['verbose'])
    return


def modlines(slf, params, mp, reletter=False, blind=False, verbose=2):
    level=0
    linesarr = []
    donezl=[]
    lastemab=""
    for i in range(len(mp['mtyp'])):
        if mp['emab'][i] != lastemab:
            if   mp['emab'][i]=="em": aetag = "emission"
            elif mp['emab'][i]=="ab": aetag = "absorption"
            elif mp['emab'][i]=="cv": aetag = "convolution"
            elif mp['emab'][i]=="zl": aetag = "zerolevel"
            if aetag != "convolution": linesarr += [aetag]
            lastemab = mp['emab'][i]
        mtyp = mp['mtyp'][i]
        slf._funcarray[2][mtyp]._keywd = mp['mkey'][i]
        outstr, level = slf._funcarray[1][mtyp].parout(slf._funcarray[2][mtyp], params, mp, i, level)
        if mp['emab'][i] == "zl": donezl.append(outstr) # Make sure we don't print zerolevel more than once.
        if aetag != 'convolution' and outstr not in donezl: linesarr += [outstr]
    return linesarr
