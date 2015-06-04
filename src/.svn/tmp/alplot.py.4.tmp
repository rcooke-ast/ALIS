import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker
import almsgs
msgs=almsgs.msgs()

def get_fitregions(wave,cont,fitted,disp,size=None):
	fsize = 40.0
	xfr, yfr = [], []
	nval = 0
	for i in range(wave.size):
		if fitted[i]:
			if nval == 0:
				xfr.append(wave[i]-disp[i])
				xfr.append(wave[i]-disp[i])
				if size is None:
					yfr.append(cont[i]-cont[i]/fsize)
					yfr.append(cont[i]+cont[i]/fsize)
				else:
					yfr.append(cont[i]-size/fsize)
					yfr.append(cont[i]+size/fsize)
				xfr.append(None)
				yfr.append(None)
				xfr.append(wave[i]-disp[i])
				yfr.append(cont[i])
				xfr.append(wave[i])
				yfr.append(cont[i])
				nval = 1
			elif nval == 1:
				xfr.append(wave[i])
				yfr.append(cont[i])
			flag = True
		elif flag:
			if nval == 1:
				xfr.append(wave[i-1]+disp[i-1])
				yfr.append(cont[i-1])
				xfr.append(None)
				yfr.append(None)
				xfr.append(wave[i-1]+disp[i-1])
				xfr.append(wave[i-1]+disp[i-1])
				if size is None:
					yfr.append(cont[i-1]-cont[i-1]/fsize)
					yfr.append(cont[i-1]+cont[i-1]/fsize)
				else:
					yfr.append(cont[i-1]-size/fsize)
					yfr.append(cont[i-1]+size/fsize)
			xfr.append(None)
			yfr.append(None)
			nval = 0
			flag = False
	# Finally, include the end point
	i = wave.size
	xfr.append(wave[i-1]+disp[i-1])
	yfr.append(cont[i-1])
	xfr.append(None)
	yfr.append(None)
	xfr.append(wave[i-1]+disp[i-1])
	xfr.append(wave[i-1]+disp[i-1])
	if size is None:
		yfr.append(cont[i-1]-cont[i-1]/fsize)
		yfr.append(cont[i-1]+cont[i-1]/fsize)
	else:
		yfr.append(cont[i-1]-size/fsize)
		yfr.append(cont[i-1]+size/fsize)
	return xfr, yfr

def make_plots_all(slf, model=None):
	msgs.info("Preparing data to be plotted", verbose=slf._argflag['out']['verbose'])
	wavearr, fluxarr, fluearr, modlarr, contarr, zeroarr = slf._wavefull, slf._fluxfull, slf._fluefull, slf._modfinal, slf._contfinal, slf._zerofinal
	if model is not None: modlarr = model
	posnarr, dims = slf._posnfull, slf._argflag['plot']['dims']
	dspl = dims.split('x')
	if len(dspl) != 2:
		msgs.error("Panel plot dimensions passed incorrectly")
		return
	try:
		dspl[0] = int(dspl[0])
		dspl[1] = int(dspl[1])
	except:
		msgs.error("Panel plot dimensions passed incorrectly")
	panppg = dspl[0]*dspl[1]
	numsub = 0
	numoplt = 0
	subids = []
	tpltlst = []
	pltlst = []
	seen = set()
	sidlst = np.array([x for x in slf._snipid if x not in seen and not seen.add(x)])
	for i in range(len(sidlst)):
		subids.append([])
		tpltlst.append(0)
		pltlst.append(0)
		for j in range(len(slf._datopt['plotone'][i])):
			if slf._datopt['plotone'][i][j]:
				subids[i].append(0)
				numoplt += 1
			else:
				numsub += 1
				subids[i].append(1)
	# For each spectrum that is read in, create an entry in snpid with the index for the sidlist
	snpid=[]
	for i in range(len(slf._snipid)): snpid.append(np.where(sidlst==slf._snipid[i])[0][0])
	pages = numoplt + int(np.ceil(numsub/float(panppg)))
	panels_left=numsub
	subpnl_done=0
	snips_done=0
	numone = 0
	pgcnt_arr = []
	p = slf._fitparams
	ps_wvarr, ps_fxarr, ps_fearr, ps_mdarr, ps_ctarr, ps_zrarr, ps_disps, ps_twarr, ps_tlarr, ps_ftarr = [], [], [], [], [], [], [], [], [], []
	po_wvarr, po_fxarr, po_fearr, po_mdarr, po_ctarr, po_zrarr, po_disps, po_twarr, po_tlarr, po_ftarr = [], [], [], [], [], [], [], [], [], []
	ps_labels, po_labels = [], []
	ps_yrange, po_yrange = [], []
	# Construct the arrays for the subplots
	if pages == 0: # Only doing single plots
		sp = snpid[snips_done]
		sn = pltlst[sp]
		while subids[sp][sn] == 0:
			po_disps.append([])
			po_wvarr.append([])
			po_fxarr.append([])
			po_fearr.append([])
			po_mdarr.append([])
			po_ctarr.append([])
			po_zrarr.append([])
			po_labels.append([])
			po_yrange.append([])
			po_twarr.append([])
			po_tlarr.append([])
			po_ftarr.append([])
			llo=posnarr[sp][sn]
			luo=posnarr[sp][sn+1]
			po_disps[numone].append(0.5*np.append( (wavearr[sp][llo+1]-wavearr[sp][llo]), (wavearr[sp][llo+1:luo]-wavearr[sp][llo:luo-1]) ))
			po_wvarr[numone].append(wavearr[sp][llo:luo])
			po_fxarr[numone].append(fluxarr[sp][llo:luo])
			po_fearr[numone].append(fluearr[sp][llo:luo])
			po_mdarr[numone].append(modlarr[sp][llo:luo])
			po_ctarr[numone].append(contarr[sp][llo:luo])
			po_zrarr[numone].append(zeroarr[sp][llo:luo])
			po_labels[numone].append(slf._datopt['label'][sp][sn])
			po_yrange[numone].append(slf._datopt['yrange'][sp][sn])
			# Load the tick marks and labels
			wvrng = [wavearr[sp][llo],wavearr[sp][luo-1]]
			p = slf._fitparams
			tickwave, ticklabl = [], []
			for i in range(0,len(slf._modpass['mtyp'])):
				if slf._modpass['emab'][i] in ['cv','sh']: continue # This is a convolution or a shift (not emission or absorption)
				if slf._specid[sp] not in slf._modpass['mkey'][i]['specid']: continue # Don't plot ticks for this model on this snip
				mtyp=slf._modpass['mtyp'][i]
				slf._funcarray[2][mtyp]._keywd = slf._modpass['mkey'][i]
				ttickwave, tticklabl = slf._funcarray[1][mtyp].tick_info(slf._funcarray[2][mtyp], p, slf._levadd[i], slf._modpass, i, wvrng=wvrng, spid=slf._specid[sp], levid=slf._levadd)
				for j in range(len(ttickwave)):
					tickwave.append(ttickwave[j])
					ticklabl.append(tticklabl[j])
			po_twarr[numone].append(tickwave)
			po_tlarr[numone].append(ticklabl)
			# Determine the fitted regions
			wft = np.where((wavearr[sp][llo:luo] >= slf._posnfit[sp][2*sn+0]) & (wavearr[sp][llo:luo] <= slf._posnfit[sp][2*sn+1]))
			wftA= np.in1d(wavearr[sp][llo:luo][wft], slf._wavefit[sp])
			po_ftarr[numone].append(wftA)
			snips_done += 1
			numone += 1
			pltlst[sp] += 1
			sn = pltlst[sp]
			# If the snip number has gone beyond the array size, go to the next specid.
			if sn == np.size(subids[sp]):
				if snips_done == np.size(snpid): break
				sp = snpid[snips_done]
				sn = pltlst[sp]
				# If the next snip for the next specid is a single plot, don't break.
#				if subids[sp][sn] != 0: break 
	else:# A combination of single + subplots (or just subplots)
		# First count the number of pages
		lesspg = 0
		for pg in range(0,pages):
			# Determine the number of panels for this page
			if snips_done == len(snpid): break
			sp = snpid[snips_done]
			sn = tpltlst[sp]
			if subids[sp][sn] == 0:
				snips_done += 1
				numone += 1
				tpltlst[sp] += 1
#				sn = tpltlst[sp]
#				# If the snip number has gone beyond the array size, go to the next specid.
#				if sn == np.size(subids[sp]):
#					if snips_done == np.size(snpid): break
#					sp = snpid[snips_done]
#					sn = tpltlst[sp]
			else:
				if panels_left <= panppg: pgcnt = numsub-subpnl_done
				else: pgcnt = panppg
				tnumone = 0
				for i in range(pgcnt):
					sp = snpid[snips_done+i]
					sn = tpltlst[sp]
					while subids[sp][sn] == 0:
						snips_done += 1
						tnumone += 1
						lesspg += 1
						tpltlst[sp] += 1
						sn = tpltlst[sp]
						# If the snip number has gone beyond the array size, go to the next specid.
						if sn == np.size(subids[sp]):
							if snips_done+i == np.size(snpid): break
							sp = snpid[snips_done+i]
							sn = tpltlst[sp]
#							print pg, sp, sn
#							print tpltlst
#							print pg, i, sp, sn, snips_done, len(subids), len(subids[sp])
							# If the next snip for the next specid is a single plot, don't break.
							if subids[sp][sn] != 0: break # Otherwise, break the while loop
					tpltlst[sp] += 1
				snips_done += pgcnt
				subpnl_done += pgcnt
				numone += tnumone
				panels_left -= panppg
		panels_left=numsub
		subpnl_done=0
		snips_done=0
		numone = 0
		pgs = 0
		for pg in range(0,pages-lesspg):
			# Determine the number of panels for this page
			sp = snpid[snips_done]
			sn = pltlst[sp]
			"""
				w = np.where((x[sp][ll:lu] >= self._posnfit[sp][2*sn+0]) & (x[sp][ll:lu] <= self._posnfit[sp][2*sn+1]))
				wA= np.in1d(x[sp][ll:lu][w], self._wavefit[sp])
				wB= np.where(wA==True)
				enf[sp] = stf[sp] + x[sp][ll:lu][w][wB]
			"""
			if subids[sp][sn] == 0:
				po_disps.append([])
				po_wvarr.append([])
				po_fxarr.append([])
				po_fearr.append([])
				po_mdarr.append([])
				po_ctarr.append([])
				po_zrarr.append([])
				po_labels.append([])
				po_yrange.append([])
				po_twarr.append([])
				po_tlarr.append([])
				po_ftarr.append([])
				llo=posnarr[sp][sn]
				luo=posnarr[sp][sn+1]
				po_disps[numone].append(0.5*np.append( (wavearr[sp][llo+1]-wavearr[sp][llo]), (wavearr[sp][llo+1:luo]-wavearr[sp][llo:luo-1]) ))
				po_wvarr[numone].append(wavearr[sp][llo:luo])
				po_fxarr[numone].append(fluxarr[sp][llo:luo])
				po_fearr[numone].append(fluearr[sp][llo:luo])
				po_mdarr[numone].append(modlarr[sp][llo:luo])
				po_ctarr[numone].append(contarr[sp][llo:luo])
				po_zrarr[numone].append(zeroarr[sp][llo:luo])
				po_labels[numone].append(slf._datopt['label'][sp][sn])
				po_yrange[numone].append(slf._datopt['yrange'][sp][sn])
				# Load the tick marks and labels
				wvrng = [wavearr[sp][llo],wavearr[sp][luo-1]]
				tickwave, ticklabl = [], []
				for j in range(0,len(slf._modpass['mtyp'])):
					if slf._modpass['emab'][j] in ['cv','sh']: continue # This is a convolution or a shift (not emission or absorption)
					if slf._specid[sp] not in slf._modpass['mkey'][j]['specid']: continue # Don't plot ticks for this model on this snip
					mtyp=slf._modpass['mtyp'][j]
					slf._funcarray[2][mtyp]._keywd = slf._modpass['mkey'][j]
					ttickwave, tticklabl = slf._funcarray[1][mtyp].tick_info(slf._funcarray[2][mtyp], p, slf._levadd[j], slf._modpass, j, wvrng=wvrng, spid=slf._specid[sp], levid=slf._levadd)
					for k in range(len(ttickwave)):
						tickwave.append(ttickwave[k])
						ticklabl.append(tticklabl[k])
				po_twarr[numone].append(tickwave)
				po_tlarr[numone].append(ticklabl)
				# Determine the fitted regions
				wft = np.where((wavearr[sp][llo:luo] >= slf._posnfit[sp][2*sn+0]) & (wavearr[sp][llo:luo] <= slf._posnfit[sp][2*sn+1]))
				wftA= np.in1d(wavearr[sp][llo:luo][wft], slf._wavefit[sp])
				po_ftarr[numone].append(wftA)
				snips_done += 1
				numone += 1
				pltlst[sp] += 1
				sn = pltlst[sp]
				# If the snip number has gone beyond the array size, go to the next specid.
#				if sn == np.size(subids[sp]):
#					if snips_done == np.size(snpid): break
#					sp = snpid[snips_done]
#					sn = pltlst[sp]
			else:
#				ps_names.append([])
#				ps_waves.append([])
				ps_disps.append([])
				ps_wvarr.append([])
				ps_fxarr.append([])
				ps_fearr.append([])
				ps_mdarr.append([])
				ps_ctarr.append([])
				ps_zrarr.append([])
				ps_labels.append([])
				ps_yrange.append([])
				ps_twarr.append([])
				ps_tlarr.append([])
				ps_ftarr.append([])
#				ps_cparr.append([])
				if panels_left <= panppg: pgcnt = numsub-subpnl_done
				else: pgcnt = panppg
				tnumone = 0
				for i in range(pgcnt):
					sp = snpid[snips_done+i]
					sn = pltlst[sp]
					while subids[sp][sn] == 0:
						po_disps.append([])
						po_wvarr.append([])
						po_fxarr.append([])
						po_fearr.append([])
						po_mdarr.append([])
						po_ctarr.append([])
						po_zrarr.append([])
						po_labels.append([])
						po_yrange.append([])
						po_twarr.append([])
						po_tlarr.append([])
						po_ftarr.append([])
						llo=posnarr[sp][sn]
						luo=posnarr[sp][sn+1]
						po_disps[numone+tnumone].append(0.5*np.append( (wavearr[sp][llo+1]-wavearr[sp][llo]), (wavearr[sp][llo+1:luo]-wavearr[sp][llo:luo-1]) ))
						po_wvarr[numone+tnumone].append(wavearr[sp][llo:luo])
						po_fxarr[numone+tnumone].append(fluxarr[sp][llo:luo])
						po_fearr[numone+tnumone].append(fluearr[sp][llo:luo])
						po_mdarr[numone+tnumone].append(modlarr[sp][llo:luo])
						po_ctarr[numone+tnumone].append(contarr[sp][llo:luo])
						po_zrarr[numone+tnumone].append(zeroarr[sp][llo:luo])
						po_labels[numone+tnumone].append(slf._datopt['label'][sp][sn])
						po_yrange[numone+tnumone].append(slf._datopt['yrange'][sp][sn])
						# Load the tick marks and labels
						wvrng = [wavearr[sp][llo],wavearr[sp][luo-1]]
						tickwave, ticklabl = [], []
						for j in range(0,len(slf._modpass['mtyp'])):
							if slf._modpass['emab'][j] in ['cv','sh']: continue # This is a convolution or a shift (not emission or absorption)
							if slf._specid[sp] not in slf._modpass['mkey'][j]['specid']: continue # Don't plot ticks for this model on this snip
							mtyp=slf._modpass['mtyp'][j]
							slf._funcarray[2][mtyp]._keywd = slf._modpass['mkey'][j]
							ttickwave, tticklabl = slf._funcarray[1][mtyp].tick_info(slf._funcarray[2][mtyp], p, slf._levadd[j], slf._modpass, j, wvrng=wvrng, spid=slf._specid[sp], levid=slf._levadd)
							for k in range(len(ttickwave)):
								tickwave.append(ttickwave[k])
								ticklabl.append(tticklabl[k])
						po_twarr[numone+tnumone].append(tickwave)
						po_tlarr[numone+tnumone].append(ticklabl)
						# Determine the fitted regions
						wft = np.where((wavearr[sp][llo:luo] >= slf._posnfit[sp][2*sn+0]) & (wavearr[sp][llo:luo] <= slf._posnfit[sp][2*sn+1]))
						wftA= np.in1d(wavearr[sp][llo:luo][wft], slf._wavefit[sp])
						po_ftarr[numone+tnumone].append(wftA)
						snips_done += 1
						tnumone += 1
						pltlst[sp] += 1
						sn = pltlst[sp]
						# If the snip number has gone beyond the array size, go to the next specid.
						if sn == np.size(subids[sp]):
							if snips_done+i == np.size(snpid): break
							sp = snpid[snips_done+i]
							sn = pltlst[sp]
							# If the next snip for the next specid is a single plot, don't break.
							if subids[sp][sn] != 0: break # Otherwise, break the while loop
					ll=posnarr[sp][sn]
					lu=posnarr[sp][sn+1]
#				if slf._argflag['plot']['xaxis'] == 'velocity': # For velocity:
#					ps_disps[pg].append(0.5*299792.458*np.append( (wavearr[ll+1]-wavearr[ll])/wavearr[ll], (wavearr[ll+1:lu]-wavearr[ll:lu-1])/wavearr[ll:lu-1]))
#					ps_wvarr[pg].append(299792.458*(wavearr[ll:lu]/(1.0+rdshft)-ps_waves[pg][i])/ps_waves[pg][i])
#				elif slf._argflag['plot']['xaxis'] == 'rest': # For rest wave:
#					ps_disps[pg].append(0.5*np.append( (wavearr[ll+1]-wavearr[ll])/(1.0+rdshft), (wavearr[ll+1:lu]-wavearr[ll:lu-1])/(1.0+rdshft) ))
#					ps_wvarr[pg].append(wavearr[ll:lu]/(1.0+rdshft))
#				else: # For observed wave:
					ps_disps[pgs].append(0.5*np.append( (wavearr[sp][ll+1]-wavearr[sp][ll]), (wavearr[sp][ll+1:lu]-wavearr[sp][ll:lu-1]) ))
					ps_wvarr[pgs].append(wavearr[sp][ll:lu])
#	
					ps_fxarr[pgs].append(fluxarr[sp][ll:lu])
					ps_fearr[pgs].append(fluearr[sp][ll:lu])
					ps_mdarr[pgs].append(modlarr[sp][ll:lu])
					ps_ctarr[pgs].append(contarr[sp][ll:lu])
					ps_zrarr[pgs].append(zeroarr[sp][ll:lu])
					ps_labels[pgs].append(slf._datopt['label'][sp][sn])
					ps_yrange[pgs].append(slf._datopt['yrange'][sp][sn])
					# Load the tick marks and labels
					wvrng = [wavearr[sp][ll],wavearr[sp][lu-1]]
					tickwave, ticklabl = [], []
					for j in range(0,len(slf._modpass['mtyp'])):
						if slf._modpass['emab'][j] in ['cv','sh']: continue # This is a convolution or a shift (not emission or absorption)
						if slf._specid[sp] not in slf._modpass['mkey'][j]['specid']: continue # Don't plot ticks for this model on this snip
						mtyp=slf._modpass['mtyp'][j]
						slf._funcarray[2][mtyp]._keywd = slf._modpass['mkey'][j]
						ttickwave, tticklabl = slf._funcarray[1][mtyp].tick_info(slf._funcarray[2][mtyp], p, slf._levadd[j], slf._modpass, j, wvrng=wvrng, spid=slf._specid[sp], levid=slf._levadd)
						for k in range(len(ttickwave)):
							tickwave.append(ttickwave[k])
							ticklabl.append(tticklabl[k])
					ps_twarr[pgs].append(tickwave)
					ps_tlarr[pgs].append(ticklabl)
					# Determine the fitted regions
					wft = np.where((wavearr[sp][ll:lu] >= slf._posnfit[sp][2*sn+0]) & (wavearr[sp][ll:lu] <= slf._posnfit[sp][2*sn+1]))
					wftA= np.in1d(wavearr[sp][ll:lu][wft], slf._wavefit[sp])
					ps_ftarr[pgs].append(wftA)
					#				ps_cparr[pg].append(comparr[sp][panels_done+i])
					pltlst[sp] += 1
				pgs += 1
				snips_done += pgcnt
				subpnl_done += pgcnt
				numone += tnumone
				panels_left -= panppg
				pgcnt_arr.append(pgcnt)
#	ps_nw = [ps_names, ps_waves]
	ps_wfemc = [ps_wvarr, ps_fxarr, ps_fearr, ps_mdarr, ps_ctarr, ps_zrarr, ps_twarr, ps_tlarr, ps_ftarr]
	po_wfemc = [po_wvarr, po_fxarr, po_fearr, po_mdarr, po_ctarr, po_zrarr, po_twarr, po_tlarr, po_ftarr]
	msgs.info("Prepared {0:d} panels in subplots".format(subpnl_done), verbose=slf._argflag['out']['verbose'])
	msgs.info("Prepared {0:d} panels in single plots".format(numone), verbose=slf._argflag['out']['verbose'])
	pticks=[slf._argflag['plot']['ticks'],slf._argflag['plot']['ticklabels']]
	if slf._argflag['plot']['fits']:
		numpagesA = plot_drawplots(pages-numone, ps_wfemc, pgcnt_arr, ps_disps, dspl, slf._argflag, labels=ps_labels, verbose=slf._argflag['out']['verbose'],plotticks=pticks,yrange=ps_yrange)
		numpagesB = plot_drawplots(numone, po_wfemc, np.ones(numone).astype(np.int), po_disps, [1,1], slf._argflag, labels=po_labels, numpages=pages-numone, verbose=slf._argflag['out']['verbose'],plotticks=pticks,yrange=po_yrange)
	if slf._argflag['plot']['residuals']:
		numpagesA = plot_drawresiduals(pages-numone, ps_wfemc, pgcnt_arr, ps_disps, dspl, slf._argflag, labels=ps_labels, verbose=slf._argflag['out']['verbose'],plotticks=pticks,yrange=ps_yrange)
		numpagesB = plot_drawresiduals(numone, po_wfemc, np.ones(numone).astype(np.int), po_disps, [1,1], slf._argflag, labels=po_labels, numpages=pages-numone, verbose=slf._argflag['out']['verbose'],plotticks=pticks,yrange=po_yrange)
	if slf._argflag['plot']['fits'] and slf._argflag['plot']['residuals']:
		numpagesA *= 2
		numpagesB *= 2
	msgs.info("Plotted {0:d} pages".format(numpagesA+numpagesB), verbose=slf._argflag['out']['verbose'])

def plot_drawplots(pages, wfemcarr, pgcnt, disp, dims, argflag, labels=None, numpages=0, verbose=2, plotticks=[True,False], yrange=None):
	"""
	Plot the fitting results in mxn panels.
	"""
	fig = []
	# Determine which pages should be plotted
	plotall = False
	if argflag['plot']['pages'] == 'all': plotall = True
	else: pltpages = argflag['plot']['pages'].split(',')
#	mmpltx = np.array([-120.0,120.0])
	pgnum = 0
	for pg in range(pages):
		if not plotall:
			if str(pg+1+numpages) not in pltpages:
				msgs.info("Skipping plot page number {0:d}".format(pg+1+numpages), verbose=argflag['out']['verbose'])
				continue
		fig.append(plt.figure(figsize=(12.5,10), dpi=80))
		fig[pgnum].subplots_adjust(hspace=0.1, wspace=0.1, bottom=0.07, top=0.98, left=0.04, right=0.98)
		for i in range(pgcnt[pg]):
			w = np.where(wfemcarr[3][pg][i] > -0.5)
			modl_min, modl_max = np.min(wfemcarr[3][pg][i][w]), np.max(wfemcarr[3][pg][i][w])
			flue_med = 3.0*np.median(wfemcarr[2][pg][i])
			res_size = 0.05*(modl_max-modl_min)
			shift = np.min([modl_min-2.0*res_size, np.min(wfemcarr[5][pg][i])-np.max(wfemcarr[2][pg][i][w])-2.0*res_size])
			if np.size(w[0]) == 0:
#				msgs.warn("There was no model data found for plot page {0:d}, panel {1:d}".format(pg+1,i+1), verbose=argflag['out']['verbose'])
				modl_min, modl_max = np.min(wfemcarr[1][pg][i]), np.max(wfemcarr[1][pg][i])
				flue_med = 3.0*np.median(wfemcarr[2][pg][i])
				res_size = 0.05*(modl_max-modl_min)
				shift = np.min([modl_min-2.0*res_size, np.min(wfemcarr[5][pg][i])-np.median(wfemcarr[2][pg][i])-2.0*res_size])
			else:
				modl_min, modl_max = np.min(wfemcarr[3][pg][i][w]), np.max(wfemcarr[3][pg][i][w])
				flue_med = 3.0*np.median(wfemcarr[2][pg][i])
				res_size = 0.05*(modl_max-modl_min)
				shift = np.min([modl_min-2.0*res_size, np.min(wfemcarr[5][pg][i])-np.median(wfemcarr[2][pg][i][w])-2.0*res_size])
			ymax = np.max([modl_max+flue_med, 1.2*np.max(wfemcarr[4][pg][i])])
			ymin = shift - 2.0*res_size
			ax = fig[pgnum].add_subplot(dims[0],dims[1],i+1)
			# Plot the error spectrum
			ax.fill_between(wfemcarr[0][pg][i],wfemcarr[5][pg][i]-wfemcarr[2][pg][i],wfemcarr[5][pg][i]+wfemcarr[2][pg][i],facecolor='0.7')
			# Plot the residual region
			ax.fill_between(wfemcarr[0][pg][i],shift+res_size,shift-res_size,facecolor='0.3')
##			ax.plot(wfemcarr[0][pg][i]+disp[pg][i],wfemcarr[2][pg][i], 'b-', drawstyle='steps')
			# Plot the data
			ax.plot(wfemcarr[0][pg][i]+disp[pg][i],wfemcarr[1][pg][i], 'k-', drawstyle='steps')
			if np.size(w[0]) != 0:
				# Plot the fitted regions
				if argflag['plot']['fitregions']:
					xfr, yfr = get_fitregions(wfemcarr[0][pg][i][w],wfemcarr[4][pg][i][w],wfemcarr[8][pg][i],disp[pg][i][w],size=(ymax-ymin)*dims[0]/3.0)
					ax.plot(xfr,yfr,'g-',linewidth=2)
				# Plot the model
				ax.plot(wfemcarr[0][pg][i][w],wfemcarr[3][pg][i][w], 'r-')
				# Plot the continuum
				ax.plot(wfemcarr[0][pg][i][w],wfemcarr[4][pg][i][w], 'b--')
				# Plot the residuals
				ax.plot(wfemcarr[0][pg][i][w]+disp[pg][i][w],(wfemcarr[5][pg][i][w]+wfemcarr[1][pg][i][w]-wfemcarr[3][pg][i][w])*res_size/wfemcarr[2][pg][i][w] + shift, 'b-', drawstyle='steps', alpha=0.5)
				# Plot the zero level
				ax.plot(wfemcarr[0][pg][i],wfemcarr[5][pg][i], 'g--')				
#			if argx == 2: # For velocity:
#				wmin=np.min([mmpltx[0],1.2*np.min(wfemcarr[0][pg][i][w])])
#				wmax=np.max([mmpltx[1],1.2*np.max(wfemcarr[0][pg][i][w])])
#			elif argx == 1: # For rest wave:
#				xfacm = elnw[1][pg][i]*(1.0+mmpltx/299792.458)
#				wmin=np.min([xfacm[0],1.2*np.min(wfemcarr[0][pg][i][w])-0.2*elnw[1][pg][i]])
#				wmax=np.max([xfacm[1],1.2*np.min(wfemcarr[0][pg][i][w])-0.2*elnw[1][pg][i]])
#			else: # For observed wave:
#				xfacm = (1.0+rdshft)*elnw[1][pg][i]*(1.0 + mmpltx/299792.458)
#			wmin=np.min([xfacm[0],1.2*np.min(wfemcarr[0][pg][i][w])-0.2*(1.0+rdshft)*elnw[1][pg][i]])
#			wmax=np.max([xfacm[1],1.2*np.max(wfemcarr[0][pg][i][w])-0.2*(1.0+rdshft)*elnw[1][pg][i]])
			if np.size(w[0]) != 0:
				wmin=1.3*np.min(wfemcarr[0][pg][i][w])-0.3*np.mean(wfemcarr[0][pg][i][w])
				wmax=1.3*np.max(wfemcarr[0][pg][i][w])-0.3*np.mean(wfemcarr[0][pg][i][w])
			else:
				msgs.warn("No model to plot for page {0:d} panel {1:d}".format(pg+1+numpages,i+1)+msgs.newline()+"You might need to check the fitrange for this parameter?", verbose=verbose)
				wmin=np.min(wfemcarr[0][pg][i])
				wmax=np.max(wfemcarr[0][pg][i])
			# Plot tick marks
			if plotticks[0] == True:
				for j in range(len(wfemcarr[6][pg][i])):
					wtc = wfemcarr[6][pg][i][j]
					wmt = np.argmin(np.abs(wfemcarr[0][pg][i][w]-wtc))
					ytkmin = wfemcarr[4][pg][i][w][wmt] + 0.05*(ymax-ymin)
					ytkmax = wfemcarr[4][pg][i][w][wmt] + 0.12*(ymax-ymin)
					ax.plot([wtc,wtc],[ytkmin,ytkmax], 'r-')
					if plotticks[1] == True: # Also plot the tick labels
						ytkmax = wfemcarr[4][pg][i][w][wmt] + 0.15*(ymax-ymin)
						ax.text(wtc,ytkmax,wfemcarr[7][pg][i][j],horizontalalignment='center',rotation='vertical',clip_on=True)

#			for j in range(0,len(comparr[pg][i])/3):
#				if comparr[pg][i][3*j] == '0': cstr = 'k-'
#				else: cstr = 'r-'
#				if argx == 2: # For velocity:
#					xfact = float(comparr[pg][i][3*j+1])
#				elif argx == 1: # For rest wave:
#					xfact = elnw[1][pg][i]*(1.0+float(comparr[pg][i][3*j+1])/299792.458)
#				else: # For observed wave:
#					xfact = (1.0+rdshft)*elnw[1][pg][i]*(1.0+float(comparr[pg][i][3*j+1])/299792.458)
#				ax.plot([xfact,xfact],[1.05,1.15], cstr)
#				if flags['labels']: ax.text(xfact,1.2,comparr[pg][i][3*j+2],horizontalalignment='center',rotation='vertical',clip_on=True)
			ax.set_xlim(wmin,wmax)
			ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
			ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
#			if argx == 2: ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%5.1f'))
#			else: ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%6.2f'))
			ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%6.2f'))
#			if argflag['plot']['labels']: ymax = np.max([1.0+2.0*flue_med, 2.0])
#			else: ymax = np.max([1.0+2.0*flue_med, 1.2])
			# Check if the user has specified the yrange
			if "=" in yrange[pg][i]:
				tempyrange = yrange[pg][i].split("=")
				yrngspl = tempyrange[1].strip('[]()').split(',')
				if len(yrngspl) == 2:
					if yrngspl[0].lower() == "none":
						pass
					else:
						ymin = np.float64(yrngspl[0])
					if yrngspl[1].lower() == "none":
						pass
					else:
						ymax = np.float64(yrngspl[1])
			ax.set_ylim(ymin,ymax)
			#ax.set_yticks((0,0.5,1.0))
			# Plot the label
			if argflag['plot']['labels'] and labels is not None:
				if labels[pg][i] != "": ax.text(0.02*dims[0],0.04*dims[1],labels[pg][i],transform=ax.transAxes)
		pgnum += 1
	return pgnum

def plot_drawresiduals(pages, wfemcarr, pgcnt, disp, dims, argflag, labels=None, numpages=0, verbose=2, plotticks=[True,False], yrange=None):
	"""
	Plot the fitting residuals in mxn panels.
	"""
	fig = []
	# Determine which pages should be plotted
	plotall = False
	if argflag['plot']['pages'] == 'all': plotall = True
	else: pltpages = argflag['plot']['pages'].split(',')
#	mmpltx = np.array([-120.0,120.0])
	pgnum = 0
	for pg in range(pages):
		if not plotall:
			if str(pg+1+numpages) not in pltpages:
				msgs.info("Skipping plot page number {0:d}".format(pg+1+numpages), verbose=argflag['out']['verbose'])
				continue
		fig.append(plt.figure(figsize=(12.5,10), dpi=80))
		fig[pgnum].subplots_adjust(hspace=0.1, wspace=0.1, bottom=0.07, top=0.98, left=0.04, right=0.98)
		for i in range(pgcnt[pg]):
			w = np.where(wfemcarr[3][pg][i] > -0.5)
			if np.size(w[0]) == 0:
#				msgs.warn("There was no model data found for plot page {0:d}, panel {1:d}".format(pg+1,i+1), verbose=argflag['out']['verbose'])
				modl_min, modl_max = np.min(wfemcarr[1][pg][i]), np.max(wfemcarr[1][pg][i])
				flue_med = 3.0*np.median(wfemcarr[2][pg][i])
				res_size = 0.05*(modl_max-modl_min)
				shift = np.min([modl_min-2.0*res_size, np.min(wfemcarr[5][pg][i])-np.median(wfemcarr[2][pg][i])-2.0*res_size])
			else:
				modl_min, modl_max = np.min(wfemcarr[3][pg][i][w]), np.max(wfemcarr[3][pg][i][w])
				flue_med = 3.0*np.median(wfemcarr[2][pg][i])
				res_size = 0.05*(modl_max-modl_min)
				shift = np.min([modl_min-2.0*res_size, np.min(wfemcarr[5][pg][i])-np.median(wfemcarr[2][pg][i][w])-2.0*res_size])
			ax = fig[pgnum].add_subplot(dims[0],dims[1],i+1)
			# Plot +/- 3sigma, 2sigma, and 1sigma
			szarr = wfemcarr[0][pg][i].size
			ax.fill_between(wfemcarr[0][pg][i],-3.0*np.ones(szarr),3.0*np.ones(szarr),facecolor='0.75')
			ax.fill_between(wfemcarr[0][pg][i],-2.0*np.ones(szarr),2.0*np.ones(szarr),facecolor='0.50')
			ax.fill_between(wfemcarr[0][pg][i],-1.0*np.ones(szarr),1.0*np.ones(szarr),facecolor='0.25')
			# Plot the data
			if np.size(w[0]) != 0:
				# Plot the fitted regions
				if argflag['plot']['fitregions']:
					xfr, yfr = get_fitregions(wfemcarr[0][pg][i][w],np.zeros(np.size(w[0])),wfemcarr[8][pg][i],disp[pg][i][w],size=3.0*dims[0])
					ax.plot(xfr,yfr,'g-',linewidth=2)
				# Plot the residuals
				ax.plot(wfemcarr[0][pg][i][w]+disp[pg][i][w],(wfemcarr[5][pg][i][w]+wfemcarr[1][pg][i][w]-wfemcarr[3][pg][i][w])/wfemcarr[2][pg][i][w], 'b-', drawstyle='steps', linewidth=1.5, alpha=0.75)
			if np.size(w[0]) != 0:
				wmin=1.3*np.min(wfemcarr[0][pg][i][w])-0.3*np.mean(wfemcarr[0][pg][i][w])
				wmax=1.3*np.max(wfemcarr[0][pg][i][w])-0.3*np.mean(wfemcarr[0][pg][i][w])
			else:
				msgs.warn("No model to plot for page {0:d} panel {1:d}".format(pg+1+numpages,i+1)+msgs.newline()+"You might need to check the fitrange for this parameter?", verbose=verbose)
				wmin=np.min(wfemcarr[0][pg][i])
				wmax=np.max(wfemcarr[0][pg][i])
			# Plot tick marks
			if plotticks[0] == True:
				for j in range(len(wfemcarr[6][pg][i])):
					wtc = wfemcarr[6][pg][i][j]
					ytkmin = -3.0
					ytkmax = +3.5
					ax.plot([wtc,wtc],[ytkmin,ytkmax], 'r-')
					if plotticks[1] == True: # Also plot the tick labels
						ytkmax += 0.1
						ax.text(wtc,ytkmax,wfemcarr[7][pg][i][j],horizontalalignment='center',verticalalignment='bottom',rotation='vertical',clip_on=True)

#			for j in range(0,len(comparr[pg][i])/3):
#				if comparr[pg][i][3*j] == '0': cstr = 'k-'
#				else: cstr = 'r-'
#				if argx == 2: # For velocity:
#					xfact = float(comparr[pg][i][3*j+1])
#				elif argx == 1: # For rest wave:
#					xfact = elnw[1][pg][i]*(1.0+float(comparr[pg][i][3*j+1])/299792.458)
#				else: # For observed wave:
#					xfact = (1.0+rdshft)*elnw[1][pg][i]*(1.0+float(comparr[pg][i][3*j+1])/299792.458)
#				ax.plot([xfact,xfact],[1.05,1.15], cstr)
#				if flags['labels']: ax.text(xfact,1.2,comparr[pg][i][3*j+2],horizontalalignment='center',rotation='vertical',clip_on=True)
			ax.set_xlim(wmin,wmax)
			ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
			ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
#			if argx == 2: ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%5.1f'))
#			else: ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%6.2f'))
			ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%6.2f'))
#			if argflag['plot']['labels']: ymax = np.max([1.0+2.0*flue_med, 2.0])
#			else: ymax = np.max([1.0+2.0*flue_med, 1.2])
			# Check if the user has specified the yrange
			ymin, ymax = -3.5,4.5
			ax.set_ylim(ymin,ymax)
			#ax.set_yticks((0,0.5,1.0))
			# Plot the label
			if argflag['plot']['labels'] and labels is not None:
				if labels[pg][i] != "": ax.text(0.02*dims[0],0.04*dims[1],labels[pg][i],transform=ax.transAxes)
		pgnum += 1
	return pgnum

def plot_showall():
	plt.show()

def plot_pdf(slf):
	msgs.info("Saving a pdf of the data and models", verbose=slf._argflag['out']['verbose'])
	from matplotlib.backends.backend_pdf import PdfPages
	import alsave
	if (slf._argflag['out']['plots'].lower() == 'true'):
		tfn = slf._argflag['run']['modname']+'.pdf'
	else:
		tfn = slf._argflag['out']['plots']
		if tfn[-4:] != ".pdf": tfn += ".pdf"
	# Check that the file does not already exist
	ans, fn = alsave.file_exists(slf, tfn)
	if ans == 'n':
		msgs.info("PDF file was not saved", verbose=slf._argflag['out']['verbose'])
	else:
		pp = PdfPages(fn)
		for i in plt.get_fignums():
			plt.figure(i)
			pp.savefig()
		pp.close()
		msgs.info("Saved a pdf with filename:"+msgs.newline()+fn, verbose=slf._argflag['out']['verbose'])

def prep_arrs(snip_ions, snip_detl, posnfit, verbose=2):
	"""
	Not presently used in ALIS
	"""
	elnames = np.array([])
	elwaves = np.array([])
	comparr = []
	rdshft=0.0
	max_CD = 0.0
	testrds=1.0
	for sn in range(0,len(slf._snip_ions)):
		comparr.append([])
		max_elm = None
		max_col = 0.0
		max_wav = 0.0
		max_fvl = 0.0
		for ln in range(0,len(slf._snip_ions[sn])):
			wavl = (1.0+slf._snip_detl[sn][ln][3])*slf._snip_detl[sn][ln][0]
			if wavl >= slf._posnfit[2*sn] and wavl <= slf._posnfit[2*sn+1]:
				if slf._snip_detl[sn][ln][1] > max_col:
					max_elm = slf._snip_ions[sn][ln]
					if slf._snip_detl[sn][ln][2] > max_fvl:
						max_wav = slf._snip_detl[sn][ln][0]
						max_fvl = slf._snip_detl[sn][ln][2]
					max_col = slf._snip_detl[sn][ln][1]
					tmp_rds = slf._snip_detl[sn][ln][3]
		if max_elm is None:
			max_elm = "None"
			max_wav = 0.5*(slf._posnfit[2*sn] + slf._posnfit[2*sn+1])
		elnames = np.append(elnames, max_elm)
		elwaves = np.append(elwaves, max_wav)
		if max_col > max_CD:
			max_CD = max_col
			rdshft = tmp_rds
	testrds=rdshft
	tri = np.where(elnames == "None")
	if np.size(tri) != 0: elwaves[tri] /= (1.0+testrds)
	if rdshft == 0.0:
		msgs.warn("Couldn't find the redshift of the main component for plotting", verbose=verbose)
		msgs.info("Assuming z=0", verbose=verbose)
	for sn in range(0,len(slf._snip_ions)):
		for ln in range(0,len(slf._snip_ions[sn])):
			compvel = 299792.458 * (slf._snip_detl[sn][ln][0]*(1.0+slf._snip_detl[sn][ln][3])/(elwaves[sn]*(1.0+rdshft)) - 1.0)
			if slf._snip_ions[sn][ln] == elnames[sn]: comparr[sn].append('1')
			else: comparr[sn].append('0')
			elnameID = '%s %5.1f' % (slf._snip_ions[sn][ln],slf._snip_detl[sn][ln][0])
			comparr[sn].append( '%8.3f' % (compvel) )
			comparr[sn].append( '%s' % (elnameID) )
	return elnames, elwaves, rdshft, comparr

