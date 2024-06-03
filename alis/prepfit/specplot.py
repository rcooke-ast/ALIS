import pdb
import os
import sys
import numpy as np
import matplotlib
from matplotlib.lines import Line2D
import matplotlib.transforms as mtransforms
matplotlib.use('Qt5Agg')
from scipy.special import wofz

from alis import alload
from alis import alis as alismain


class SelectRegions(object):
    """
    Select Regions to be included in the fit

    Key-bindings

      's' save regions file

    """

    def __init__(self, canvas, ax, spec, prop, atom, vel=500.0, include_cont=True):
        """
        vel : float
          Default +/- plotting window in km/s
        """
        self.ax = ax
        self.spec = spec
        self.prop = prop
        self.atom = atom
        self.veld = vel
        self.fb = None
        self.lineidx = 0
        self._addsub = 0  # Adding a region (1) or removing (0)
        self._changes = False
        self.annlines = []
        self.anntexts = []
        self.voigtlines = []
        self.logn = 22.0
        self.HImodel = self.voigtmodel()
        self._include_cont = include_cont

        # Unset some of the matplotlib keymaps
        matplotlib.pyplot.rcParams['keymap.fullscreen'] = ''        # toggling fullscreen (Default: f, ctrl+f)
        matplotlib.pyplot.rcParams['keymap.home'] = ''              # home or reset mnemonic (Default: h, r, home)
        matplotlib.pyplot.rcParams['keymap.back'] = ''              # forward / backward keys to enable (Default: left, c, backspace)
        matplotlib.pyplot.rcParams['keymap.forward'] = ''           # left handed quick navigation (Default: right, v)
        matplotlib.pyplot.rcParams['keymap.pan'] = ''              # pan mnemonic (Default: p)
        matplotlib.pyplot.rcParams['keymap.zoom'] = ''              # zoom mnemonic (Default: o)
        matplotlib.pyplot.rcParams['keymap.save'] = ''              # saving current figure (Default: s)
        matplotlib.pyplot.rcParams['keymap.quit'] = ''              # close the current figure (Default: ctrl+w, cmd+w)
        matplotlib.pyplot.rcParams['keymap.grid'] = ''              # switching on/off a grid in current axes (Default: g)
        matplotlib.pyplot.rcParams['keymap.yscale'] = ''            # toggle scaling of y-axes ('log'/'linear') (Default: l)
        matplotlib.pyplot.rcParams['keymap.xscale'] = ''            # toggle scaling of x-axes ('log'/'linear') (Default: L, k)
        #matplotlib.pyplot.rcParams['keymap.all_axes'] = ''          # enable all axes (Default: a)

        canvas.mpl_connect('draw_event', self.draw_callback)
        canvas.mpl_connect('button_press_event', self.button_press_callback)
        canvas.mpl_connect('key_press_event', self.key_press_callback)
        canvas.mpl_connect('button_release_event', self.button_release_callback)
        self.canvas = canvas

        # Sort the lines by decreasing probability of detection
        self.sort_lines()
        self.get_current_line()

        # Draw the first line
        self.next_spectrum()

        self.atomic_path = "/".join(__file__.split("/")[:-2])+"/data/atomic.dat"

        self.update_waverange()  # This includes a redraw of the canvas

    def voigtmodel(self):
        """
        Define the model here
        """
        wave = self.prop._wave
        logn = self.logn
        zabs = self.prop._zabs
        bval = 10.0
        lam0 = 1215.6701
        fval = 0.416400
        gama = 6.265E8
        wv = lam0 * 1.0e-8
        cold = 10.0**logn
        zp1=zabs+1.0
        bl=bval*wv/2.99792458E5
        a=gama*wv*wv/(3.76730313461770655E11*bl)
        cns=wv*wv*fval/(bl*2.002134602291006E12)
        cne=cold*cns
        ww=(wave*1.0e-8)/zp1
        v = wv * ((wv / ww) - 1) / bl
        tau = cne*wofz(v + 1j * a).real
        return np.median(self.prop._flux) * np.exp(-1.0*tau)

    def draw_lines(self):
        #annotations = [child for child in self.ax.get_children() if isinstance(child, matplotlib.text.Annotation)]
        for i in self.annlines: i.remove()
        for i in self.anntexts: i.remove()
        for i in self.voigtlines:
            for ii in i: ii.remove()
        self.annlines = []
        self.anntexts = []
        self.voigtlines = []
        if self.atom._atom_wvl[self.linecur] == 1215.6701:
            self.voigtlines.append(self.ax.plot(self.prop._wave, self.HImodel, color='r'))
        return
        molecules=False
        # Plot the lines
        xmn, xmx = self.ax.get_xlim()
        ymn, ymx = self.ax.get_ylim()
        xmn /= (1.0+self.prop._zabs)
        xmx /= (1.0+self.prop._zabs)
        w = np.where((self.atom._atom_wvl > xmn) & (self.atom._atom_wvl < xmx))[0]
        for i in range(w.size):
            dif = i%5
            self.annlines.append(self.ax.axvline(self.atom._atom_wvl[w[i]]*(1.0+self.prop._zabs), color='r'))
            txt = "{0:} {1:} {2:.1f}".format(self.atom._atom_atm[w[i]],self.atom._atom_ion[w[i]],self.atom._atom_wvl[w[i]])
            ylbl = ymn + (ymx-ymn)*(dif+1.5)/8.0
            self.anntexts.append(self.ax.annotate(txt, (self.atom._atom_wvl[w[i]]*(1.0+self.prop._zabs), ylbl), rotation=90.0, color='b', ha='center', va='bottom'))
        if molecules:
            # Plot molecules
            molecule_file = "/".join(__file__.split("/")[:-2]) + "/data/molecule.dat"
            labls, lines = np.loadtxt(molecule_file, dtype={'names': ('ion', 'wvl'), 'formats': ('S6', 'f8')}, unpack=True, usecols=(0,1))
            w = np.where((lines > xmn) & (lines < xmx))[0]
            for i in range(w.size):
                dif = i%5
                self.annlines.append(self.ax.axvline(lines[w[i]]*(1.0+self.prop._zabs), color='g'))
                txt = "{0:s} {1:.1f}".format(labls[w[i]],lines[w[i]])
                ylbl = ymn + (ymx-ymn)*(dif+1.5)/8.0
                self.anntexts.append(self.ax.annotate(txt, (lines[w[i]]*(1.0+self.prop._zabs), ylbl), rotation=90.0, color='b', ha='center', va='bottom'))
        return

    def draw_callback(self, event):
        trans = mtransforms.blended_transform_factory(self.ax.transData, self.ax.transAxes)
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        if self.fb is not None:
            self.fb.remove()
        # Find all regions
        regwhr = np.copy(self.prop._regions==1)
        # Fudge to get the leftmost pixel shaded in too
        regwhr[np.where((self.prop._regions[:-1]==0) & (self.prop._regions[1:]==1))] = True
        self.fb = self.ax.fill_between(self.prop._wave, 0, 1, where=regwhr, facecolor='green', alpha=0.5, transform=trans)
        self.ax.draw_artist(self.spec)
        self.draw_lines()
        # Set the title of the window
        idtxt = "Current transition = {0:} {1:} {2:.1f}".format(self.atom._atom_atm[self.linecur].strip(),
                                             self.atom._atom_ion[self.linecur].strip(),
                                             self.atom._atom_wvl[self.linecur])
        self.ax.set_title("Press '?' to list the available options\n{0:s}".format(idtxt))

    def get_ind_under_point(self, event):
        """
        Get the index of the spectrum closest to the cursor
        """
        ind = np.argmin(np.abs(self.prop._wave-event.xdata))
        return ind

    def sort_lines(self, method="ion"):
        """
        Sort lines by decreasing probability of detection
        """
        if method == "sig":
            coldens = 10.0**(self.atom.solar-12.0)
            ew = coldens * (self.atom._atom_wvl**2 * self.atom._atom_fvl)
            #snr = 1.0/self.prop._flue
            #indices = np.abs(np.subtract.outer(self.prop._wave, self.atom._atom_wvl*(1.0+self.prop._zabs))).argmin(0)
            sigdet = ew#*snr[indices]
            self.sortlines = np.argsort(sigdet)[::-1]
        elif method == "ion":
            self.sortlines = np.arange(self.atom._atom_wvl.size)
        elif method == "wave":
            self.sortlines = np.argsort(self.atom._atom_wvl)
        return

    def get_current_line(self):
        if self.lineidx < 0:
            self.lineidx += self.sortlines.size
        if self.lineidx >= self.sortlines.size:
            self.lineidx -= self.sortlines.size
        self.linecur = self.sortlines[self.lineidx]

    def button_press_callback(self, event):
        """
        whenever a mouse button is pressed
        """
        if event.inaxes is None:
            return
        if self.canvas.toolbar.mode != "":
            return
        if event.button == 1:
            self._addsub = 1
        elif event.button == 3:
            self._addsub = 0
        self._start = self.get_ind_under_point(event)

    def button_release_callback(self, event):
        """
        whenever a mouse button is released
        """
        if event.inaxes is None:
            return
        if self.canvas.toolbar.mode != "":
            return
        self._end = self.get_ind_under_point(event)
        if self._end != self._start:
            if self._start > self._end:
                tmp = self._start
                self._start = self._end
                self._end = tmp
            self.update_regions()
        trans = mtransforms.blended_transform_factory(self.ax.transData, self.ax.transAxes)
        self.canvas.restore_region(self.background)
        if self.fb is not None:
            self.fb.remove()
        # Find all regions
        regwhr = np.copy(self.prop._regions==1)
        # Fudge to get the leftmost pixel shaded in too
        regwhr[np.where((self.prop._regions[:-1]==0) & (self.prop._regions[1:]==1))] = True
        self.fb = self.ax.fill_between(self.prop._wave, 0, 1, where=regwhr, facecolor='green', alpha=0.5, transform=trans)
        self.canvas.draw()

    def key_press_callback(self, event):
        """
        whenever a key is pressed
        """
        if not event.inaxes:
            return
        if event.key == '?':
            print("============================================================")
            print("       MAIN OPERATIONS")
            print("p       : toggle pan/zoom with the cursor")
            print("w       : write the spectrum with the associated region")
            print("q       : exit")
            print("+/-     : Increase/decrease logN of N(HI) model")
            print("------------------------------------------------------------")
            print("       SHORTCUTS TO MOVE BETWEEN LINES")
            print("[ / ]   : go to previous/next element")
            print(", / .   : go to previous/next ion")
            print("b / n   : go to previous/next line")
            print("------------------------------------------------------------")
            print("       ATOMIC DATA OF THE CURRENT LINE")
            print("{0:} {1:}  {2:f}".format(self.atom._atom_atm[self.linecur].strip(),self.atom._atom_ion[self.linecur].strip(),self.atom._atom_wvl[self.linecur]))
            print("Observed wavelength = {0:f}".format(self.atom._atom_wvl[self.linecur]*(1.0+self.prop._zabs)))
            print("f-value = {0:f}".format(self.atom._atom_fvl[self.linecur]))
            print("------------------------------------------------------------")
            print("Current logN value = ", self.logn)
            print("------------------------------------------------------------")
        elif event.key == 'w':
            self.write_data()
        elif event.key == 'n':
            self.lineidx += 1
            self.next_spectrum()
        elif event.key == 'b':
            self.lineidx -= 1
            self.next_spectrum()
        elif event.key == ']':
            self.next_element(1)
            self.next_spectrum()
        elif event.key == '[':
            self.next_element(-1)
            self.next_spectrum()
        elif event.key == '.':
            self.next_element(1, ion=True)
            self.next_spectrum()
        elif event.key == ',':
            self.next_element(-1, ion=True)
            self.next_spectrum()
        elif event.key == '+':
            self.logn += 0.1
            self.voigtmodel()
            self.canvas.draw()
        elif event.key == '-':
            self.logn -= 0.1
            self.voigtmodel()
            self.canvas.draw()
        elif event.key == 'q':
            if self._changes:
                print("WARNING: There are unsaved changes!!")
                print("Press q again to exit")
                self._changes = False
            else:
                sys.exit()
        self.canvas.draw()

    def next_element(self, pm, ion=False):
        if ion == True:
            arrsrt = np.core.defchararray.add(self.atom._atom_atm, self.atom._atom_ion)
        else:
            arrsrt = self.atom._atom_atm
        unq, idx = np.unique(arrsrt, return_index=True)
        unq = unq[idx.argsort()]
        nxt = np.where(unq == arrsrt[self.linecur])[0][0]+pm
        if nxt >= unq.size:
            nxt = 0
        ww = np.where(arrsrt == unq[nxt])[0]
        self.lineidx = ww[0]
        return

    def next_spectrum(self):
        self.get_current_line()
        # Update the wavelength range of the spectrum being plot
        self.update_waverange()
        # See if any regions need to be loaded
        self.prop._regions[:] = 0
        idtxt = "{0:}_{1:}_{2:.1f}".format(self.atom._atom_atm[self.linecur].strip(),self.atom._atom_ion[self.linecur].strip(),self.atom._atom_wvl[self.linecur])
        tstnm = self.prop._outp + "_" + idtxt + "_reg.dat"
        if os.path.exists(tstnm):
            wv, reg = np.loadtxt(tstnm, unpack=True, usecols=(0,3))
            mtch = in1d_tol(self.prop._wave, wv, 1.0E-7)
#			mtch = np.in1d(self.prop._wave, wv, assume_unique=True)
            self.prop._regions[np.where(mtch)] = reg.copy()
            self.ax.set_xlim([np.min(wv), np.max(wv)])
        # Other stuff
        self.canvas.draw()
        return

    def update_regions(self):
        self.prop._regions[self._start:self._end] = self._addsub
        std = np.std(self.prop._flux[self._start:self._end])
        med = np.median(self.prop._flue[self._start:self._end])
        mad = 1.4826 * np.median(np.abs(self.prop._flux[self._start:self._end]-np.median(self.prop._flux[self._start:self._end])))

    def update_waverange(self):
        self.get_current_line()
        wcen = self.atom._atom_wvl[self.linecur]*(1.0+self.prop._zabs)
        xmn = wcen * (1.0 - self.veld/299792.458)
        xmx = wcen * (1.0 + self.veld/299792.458)
        self.ax.set_xlim([xmn, xmx])
        diff = xmx-xmn
        wdisp = ((self.prop._wave >= xmn-3*diff) & (self.prop._wave <= xmx+3*diff))
        try:
            medval = np.percentile(self.prop._flux[~np.isnan(self.prop._flux) & wdisp], 99.0)
        except IndexError:
            medval = 1.0
        self.ax.set_ylim([-0.1*medval, 1.1*medval])
        #print("Updated wavelength range:", xmn, xmx)
        self.canvas.draw()

    def write_data(self):
        # Plot the lines
        xmn, xmx = self.ax.get_xlim()
        wsv = np.where((self.prop._wave>xmn) & (self.prop._wave<xmx))
        idtxt = "{0:}_{1:}_{2:.1f}".format(self.atom._atom_atm[self.linecur].strip(),self.atom._atom_ion[self.linecur].strip(),self.atom._atom_wvl[self.linecur])
        outnm = self.prop._outp + "_" + idtxt + "_reg.dat"
        sclfct = 1.0
        if self._include_cont:
            np.savetxt(outnm, np.transpose((self.prop._wave[wsv], sclfct*self.prop._flux[wsv]*self.prop._cont[wsv], sclfct*self.prop._flue[wsv]*self.prop._cont[wsv], self.prop._regions[wsv])))
        else:
            np.savetxt(outnm, np.transpose((self.prop._wave[wsv], sclfct*self.prop._flux[wsv], sclfct*self.prop._flue[wsv], self.prop._regions[wsv])))
        print("Saved file:")
        print(outnm)

class props:
    def __init__(self, dla):
        # Load the data
        ifil = dla._path + dla._filename
        outf = dla._path + dla._filename.replace(".dat", "_reg.dat")
        try:
            wave, flux, flue, regions = np.loadtxt(outf, unpack=True, usecols=(0,1,2,3))
            cont = np.ones(wave.size)
            print("Loaded file:")
            print(outf)
        except:
            try:
                wave, flux, flue, cont = np.loadtxt(ifil, unpack=True, usecols=(0,1,2,4))
                # print("NEED TO DELETE THIS CONT FUDGE!")
                # cont = np.ones(wave.size)
                regions = np.zeros(wave.size)
                print("Loaded file:")
                print(ifil)
            except:
                wave, flux, flue = np.loadtxt(ifil, unpack=True, usecols=(0,1,2))
                cont = np.ones(wave.size)
                regions = np.zeros(wave.size)
        self._wave = wave
        self._flux = flux
        self._flue = flue
        self._cont = cont
        self._file = ifil
        self._regions = regions
        self._outp = dla._path + dla._filename.replace(".dat", "")
        self._outf = outf
        self._zabs = dla._zabs

    def set_regions(arr):
        self._regions = arr.copy()
        return

class atomic:
    def __init__(self, wmin=None, wmax=None):
        self._wmin = wmin
        self._wmax = wmax
        self._atom_atm=[]
        self._atom_ion=[]
        self._atom_lbl=[]
        self._atom_wvl=[]
        self._atom_fvl=[]
        self._atom_gam=[]
        self._molecule_atm=[]
        self._molecule_ion=[]
        self._molecule_lbl=[]
        self._molecule_wvl=[]
        self._molecule_fvl=[]
        self._molecule_gam=[]
        self.load_lines()

    def solar(self):
        elem = np.array(['H ', 'He','Li','Be','B ', 'C ', 'N ', 'O ', 'F ', 'Ne','Na','Mg','Al','Si','P ', 'S ', 'Cl','Ar','K ', 'Ca','Sc','Ti','V ', 'Cr','Mn','Fe','Co','Ni','Cu','Zn'])
        mass = np.array([1.0, 4.0, 7.0, 8.0, 11.0, 12.0,14.0,16.0,19.0,20.0,23.0,24.0,27.0,28.0,31.0,32.0,35.0,36.0,39.0,40.0,45.0,48.0,51.0,52.0,55.0,56.0,59.0,60.0,63.0,64.0])
        solr = np.array([12.0,10.93,3.26,1.30,2.79,8.43,7.83,8.69,4.42,7.93,6.26,7.56,6.44,7.51,5.42,7.14,5.23,6.40,5.06,6.29,3.05,4.91,3.95,5.64,5.48,7.47,4.93,6.21,4.25,4.63])
        solar = np.zeros(self._atom_atm.size)
        for i in range(elem.size):
            w = np.where(self._atom_atm==elem[i])
            solar[w] = solr[i]
        self.solar = solar
        return

    def load_lines(self, verbose=1):
        # Load the lines file
        print("Loading a list of atomic transitions...")
        alispath = __file__#"/".join(__file__.split("/")[:-2])#+"/data/atomic.dat"
        argflag = alload.optarg(alispath, verbose=verbose)
        slf = alismain.ClassMain(argflag, getinst=True)
        slf._argflag = argflag
        atmdata = alload.load_atomic(slf)

        # Convert to numpy arrays
        numtrans = atmdata['Wavelength'].size
        ionspl = np.char.split(atmdata['Ion'], sep ='_')
        self._atom_atm, self._atom_ion = [], []
        keep = np.ones(numtrans, dtype=bool)
        for ss in range(numtrans):
            if str(ionspl[ss][0]) in ['1Ly', 'FAKE']: keep[ss] = False
            if str(ionspl[ss][1]) in ['IB']: keep[ss] = False
            if keep[ss]:
                self._atom_atm.append(str(ionspl[ss][0]))
                self._atom_ion.append(str(ionspl[ss][1]))
        self._atom_atm = np.array(self._atom_atm)
        self._atom_ion = np.array(self._atom_ion)
        self._atom_lbl = atmdata['Ion'][keep]
        self._atom_wvl = atmdata['Wavelength'][keep]
        self._atom_fvl = atmdata['fvalue'][keep]
        self._atom_gam = atmdata['Gamma'][keep]
        # seen = set()
        # atmdata['Element'] = np.array([x for x in isotope if x not in seen and not seen.add(x)]).astype(str)
        # seen = set()
        # atmdata['AtomicMass'] = np.array([x for x in table.array['AtomicMass'] if x not in seen and not seen.add(x)])

        # Ignore lines outside of the specified wavelength range
        if self._wmin is not None:
            ww = np.where(self._atom_wvl > self._wmin)
            self._atom_atm = self._atom_atm[ww]
            self._atom_ion = self._atom_ion[ww]
            self._atom_lbl = self._atom_lbl[ww]
            self._atom_wvl = self._atom_wvl[ww]
            self._atom_fvl = self._atom_fvl[ww]
            self._atom_gam = self._atom_gam[ww]
        if self._wmax is not None:
            ww = np.where(self._atom_wvl < self._wmax)
            self._atom_atm = self._atom_atm[ww]
            self._atom_ion = self._atom_ion[ww]
            self._atom_lbl = self._atom_lbl[ww]
            self._atom_wvl = self._atom_wvl[ww]
            self._atom_fvl = self._atom_fvl[ww]
            self._atom_gam = self._atom_gam[ww]

        # Ignore some lines
        ign = np.where((self._atom_ion!="I*  ")&(self._atom_ion!="II* ")&(self._atom_ion!="I** ")&(self._atom_ion!="II**"))
        self._atom_atm = self._atom_atm[ign]
        self._atom_ion = self._atom_ion[ign]
        self._atom_lbl = self._atom_lbl[ign]
        self._atom_wvl = self._atom_wvl[ign]
        self._atom_fvl = self._atom_fvl[ign]
        self._atom_gam = self._atom_gam[ign]

        # Assign solar abundances to these lines
        self.solar()

        # Load the lines file
        print("Loading a list of molecular transitions...")
        molecule_file = "/".join(__file__.split("/")[:-2])+"/data/molecule.dat"
        try:
            infile = open(molecule_file, "r")
        except IOError:
            print("The lines file:\n" + "molecule.dat\ndoes not exist!")
            sys.exit()
        molecule_list=infile.readlines()
        leninfile=len(molecule_list)
        infile.close()
        infile = open(molecule_file, "r")
        for i in range(0, leninfile):
            self._molecule_atm.append(infile.read(2))
            self._molecule_ion.append(infile.read(4))
            self._molecule_lbl.append((self._molecule_atm[i]+self._molecule_ion[i]).strip())
            line2 = infile.readline()
            wfg = line2.split()
            self._molecule_wvl.append(eval(wfg[0]))
            self._molecule_fvl.append(eval(wfg[1]))
            self._molecule_gam.append(eval(wfg[2]))

        # Convert to numpy array
        self._molecule_atm = np.array(self._molecule_atm)
        self._molecule_ion = np.array(self._molecule_ion)
        self._molecule_lbl = np.array(self._molecule_lbl)
        self._molecule_wvl = np.array(self._molecule_wvl)
        self._molecule_fvl = np.array(self._molecule_fvl)
        self._molecule_gam = np.array(self._molecule_gam)

        # Now sort lines data according to wavelength
        argsrt = np.argsort(self._molecule_wvl)
        self._molecule_atm = self._molecule_atm[argsrt]
        self._molecule_ion = self._molecule_ion[argsrt]
        self._molecule_lbl = self._molecule_lbl[argsrt]
        self._molecule_wvl = self._molecule_wvl[argsrt]
        self._molecule_fvl = self._molecule_fvl[argsrt]
        self._molecule_gam = self._molecule_gam[argsrt]


def in1d_tol(avec, bvec, tol):
    dvec=np.abs(avec-bvec[:, np.newaxis])
    return np.any(dvec<=tol, axis=0)
