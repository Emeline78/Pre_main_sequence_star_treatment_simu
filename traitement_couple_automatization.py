import pandas as pd
from math import *
import numpy as np
from scipy import *
import matplotlib.pyplot as plt
import glob
import os
import matplotlib
matplotlib.interactive(True)
from magic.libmagic import anelprof
from matplotlib.ticker import ScalarFormatter
import re
formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((0, 0))
from pathlib import Path
from statsmodels.tsa.stattools import adfuller

"""
git add traitement_couple_automatization.py
git commit -m "modifications"
git push
"""

matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 24
matplotlib.rcParams['ytick.labelsize']=20
matplotlib.rcParams['ytick.direction']='in'
matplotlib.rcParams['ytick.right']='True'
matplotlib.rcParams['xtick.labelsize']=20
matplotlib.rcParams['xtick.direction']='in'
matplotlib.rcParams['xtick.top']='True'
matplotlib.rcParams['figure.figsize']= [8, 6]

def parse_p_number(s):
	if 'p' in s:
		if s.startswith('p'):
	    		return float('0.' + s[1:])
		else:
			return float(s.replace('p', '.'))
	return float(s)

def extract_params(path):
	parts = Path(path).parts
	params = {}
	for p in parts:
		if p in ["gr", "gr2", "gr_gr2_Louis"]:
			params["config"] = p

		match_nr = re.search(r'Nr([0-9p]+)', p)
		if match_nr:
			params["Nr"] = parse_p_number(match_nr.group(1))

		elif p.startswith("xi"):
			match_xi = re.search(r'xi[_]?([p\d]+)', p)
			if match_xi:
				params["xi"] = parse_p_number(match_xi.group(1))
				
			match_pm = re.search(r'pm(\d+)', p)
			if match_pm:
				params["Pm"] = float(match_pm.group(1))

		elif p.startswith("ra"):
			val = p.split("_")[1]  # ex: 5e6
			params["ra"] = float(parse_p_number(val))

		elif p.startswith("om"):
			params["om"] = float(p[2:])

	return params

def make_case_name(path):
	p = Path(path)
	parts = list(p.parts)

	# trouver l'index de "gr", "gr2" ou "gr_gr2_Louis"
	for i, part in enumerate(parts):
		if part in ["gr", "gr2", "gr_gr2_Louis"]:
			relevant_parts = parts[i:]
			break
	else:
		raise ValueError("No valid config folder found in path")

	# enlever slash final implicite et reconstruire
	case_name = "_".join(relevant_parts)

	return case_name


def save_snapshots(save_dir, case_name, r, times, RS, MS, MC, Visc):
	np.savez(save_dir + "/"+ f"{case_name}.npz",r=r,times=times,RS=RS,MS=MS,MC=MC,Visc=Visc)

def load_snapshots(file):
	data = np.load(file)
	return data["r"], data["times"], data["RS"], data["MS"], data["MC"], data["Visc"]

from magic import *
mu0 = 4*np.pi*1e-7

liste = []
index = []
snap_dir = "snapshots1"

all_dirs = (list(Path("/travail/dynconv/multiscale_dyno/anelasticCouette/gr").glob("Nr*/ra*/om*")) +list(Path("/travail/dynconv/multiscale_dyno/anelasticCouette/gr2").glob("xi*/ra*/om*")) +list(Path("/travail/dynconv/multiscale_dyno/anelasticCouette/gr_gr2_Louis").glob("ra*/om*")))

valid_dirs = []

for path in all_dirs:
    last_part = Path(path).name
    
    # accepte uniquement "om" suivi uniquement de chiffres
    if re.fullmatch(r'om\d+', last_part):
        valid_dirs.append(path)
    else:
        print(f"Skipping invalid case: {last_part}")

all_dirs = valid_dirs

for path in all_dirs:
	a = str(path)
	print(a)
	params = extract_params(path)
	case_name = make_case_name(path)
	snap_file = Path(snap_dir) / f"{case_name}.npz"
	
	ts = MagicTs(datadir = a, field='e_kin', all=True, iplot = False) 	# verification que le regime ne change pas dans le temps pour pouvoir faire l'integration en temps 
	CV = np.std(ts.ekin_pol) / np.mean(ts.ekin_pol)
	p_adf = adfuller(ts.ekin_pol)[1]

	if CV < 0.3 and p_adf < 0.05:
	    status = True
	else:
	    status = False
	
	if snap_file.exists():

		print(f"Loading {case_name}")
		r, times, RS_snap, MS_snap, MC_snap, Visc_snap = \
		load_snapshots(snap_file)
		stp = MagicSetup(datadir=a)
		n = stp.polind
		Pm = stp.prmag
		ki = stp.radratio
		Nrho = stp.strat 
		Ek = stp.ek
		g0 = stp.g0
		g1 = stp.g1
		g2 = stp.g2
		om = 1/Ek
	else : 	
		stp = MagicSetup(datadir = a)
		
		n = stp.polind
		Pm = stp.prmag
		ki = stp.radratio
		Nrho = stp.strat 
		Ek = stp.ek
		g0 = stp.g0
		g1 = stp.g1
		g2 = stp.g2
		om = 1/Ek

		files = glob.glob(os.path.join(a,'G_[0-9]*.rot01'))
		files.sort(key=lambda f: int(os.path.basename(f).split('_')[1].split('.')[0]))

		times = []
		RS_snap = []
		MS_snap = []
		Visc_snap = []
		MC_snap = []

		# pour les moyennes sur phi j'aurais juste pu faire mean vu que c'est espace regulierement
		for j in range(1,len(files)+1): 
			gr = MagicGraph(datadir=a,tag='rot01',ivar = j)
			times.append(gr.time)

			if j == 1:
				r = gr.radius
				th = np.linspace(0,np.pi,gr.ntheta)
				phi = np.linspace(0,2*np.pi,gr.nphi-1)

				dphi = 2*np.pi/(gr.nphi-2)
				dtheta = np.pi/(gr.ntheta-1)

				w_theta = dtheta * np.sin(th)
				w_phi = dphi / (2* np.pi)

			# fluctuations
			vr = gr.vr - gr.vr.mean(axis=0)
			vp = gr.vphi - gr.vphi.mean(axis=0)

			# def de tau
			dvphi = np.gradient(gr.vphi, r, axis=2)
			tau_rphi = dvphi - gr.vphi/r[None,None,:] 

			# Reynolds
			prodR = (vr * vp * w_phi).sum(axis=0)	# flux
			RS = (prodR * np.sin(th)[:,None] * w_theta[:,None]).sum(axis=0) * r # integrated flux over a spherical surface

			# Maxwell
			prodM = -(gr.Br * gr.Bphi * w_phi).sum(axis=0)
			MS = (prodM * np.sin(th)[:,None] * w_theta[:,None]).sum(axis=0) * r  

			# Ecoulement meridional
			vr_mean = (gr.vr * w_phi).sum(axis=0)
			vphi_mean = (gr.vphi * w_phi).sum(axis=0)
			MC = (vr_mean * (vphi_mean + r[None,:] * np.sin(th)[:,None] * 1/Ek) * np.sin(th)[:,None] * w_theta[:,None]).sum(axis = 0) * r

			# Viscosite
			mean_tau = (tau_rphi * w_phi).sum(axis = 0)
			Visc = - (mean_tau * np.sin(th)[:,None] * w_theta[:,None]).sum(axis=0) * r

			Visc_snap.append(Visc)
			RS_snap.append(RS)
			MS_snap.append(MS)
			MC_snap.append(MC)

		times = np.array(times)
		
		RS_snap = np.array(RS_snap) 
		MS_snap = np.array(MS_snap) 
		Visc_snap = np.array(Visc_snap) 
		MC_snap = np.array(MC_snap) 
	
		save_snapshots(snap_dir,case_name,r,times,RS_snap,MS_snap,MC_snap,Visc_snap)

	t_total = times[-1] - times[0]
	dt = np.diff(times)

	RS = np.zeros_like(RS_snap[0])
	MS = np.zeros_like(MS_snap[0])
	MC = np.zeros_like(MC_snap[0])
	Visc = np.zeros_like(Visc_snap[0])
	for i in range(len(dt)):
		RS += 0.5*(RS_snap[i] + RS_snap[i+1])*dt[i]
		MS += 0.5*(MS_snap[i] + MS_snap[i+1])*dt[i]
		MC += 0.5*(MC_snap[i] + MC_snap[i+1])*dt[i]
		Visc += 0.5*(Visc_snap[i] + Visc_snap[i+1])*dt[i]
	
	L = 1		# pas 1 - ki car r0 n'est pas egale a 1 mais a 1/(1-ki)
	nu = Ek * om * L**2
	tau = L**2/nu
	eta = nu/Pm
	temp, rho, drho = anelprof(r, strat = Nrho, polind = n, g0=g0, g1=g1, g2=g2)
	rho0 = rho[0]
	rho = rho / rho0  
	B0car = eta * om * mu0 	* rho0
	
	RS = RS / t_total * rho * L**3 / tau**2 * 2 * np.pi * r**2
	MS = MS / t_total * L * B0car / mu0 * 2 * np.pi * r**2
	Visc = Visc / t_total * rho * L**3 / tau**2 * 2 * np.pi * r**2
	MC = MC / t_total * rho * L**3 / tau**2 * 2 * np.pi * r**2

	params = extract_params(path)
	res = pd.DataFrame({"r": r,"RS": RS, "MC": MC, "MS": MS, "Visc": Visc,"name": str(case_name), "status": status})
	for key, value in params.items():
        	res[key] = value
	liste.append(res)

df_final = pd.concat(liste, ignore_index=True)

df_final.loc[df_final["name"].str.startswith("gr_"), "xi"] = 0.2
df_final.loc[df_final["name"].str.startswith("gr_gr2_Louis"), "xi"] = 0.35

df_final.loc[df_final["name"].str.startswith("gr_gr2_Louis"), "Pm"] =4

def encode_config(name):
     if "gr_gr2_Louis" in name:
         return 2
     elif "gr2" in name:
         return 1
     elif "gr" in name:
         return 0
     else:
         return -1

df_final["config_code"] = df_final["config"].apply(encode_config)

mapping = {
    "gr_Nr2p5_Pm4_ra_1p6e7": 2500,
    "gr_Nr2p5_Pm4_ra_8e6": 250,
    "gr_Nr2p5_Pm6_ra_8e6": 1,
    "gr2_xi_p2_pm4_ra_1e6": 350,
    "gr2_xi_p2_pm6_ra_1p5e6": 175,
    "gr2_xi_p1_pm4_ra_5e5": 850,
    "gr2_xi_p1_pm6_ra_5p5e6": 355,
    "gr2_xi_p35_pm4_ra_2e6": 42.5,
    "gr2_xi_p35_pm4_ra_5e6": 135,
    "gr2_xi_p35_pm6_ra_1p5e6": 17.5,
    "gr_gr2_Louis_ra_1p5e7": 125,
    "gr_gr2_Louis_ra_1e7": 87.5,
}

df_final["om_lim"] = np.nan

for pattern, value in mapping.items():
    mask = df_final["name"].str.startswith(pattern)
    df_final.loc[mask, "om_lim"] = value
    
mapping = {
    "gr_Nr2p5_Pm4_ra_1p6e7": 0.062,
    "gr_Nr2p5_Pm4_ra_8e6": 0.029,
    "gr2_xi_p2_pm4_ra_1e6": 0.025,
    "gr2_xi_p2_pm6_ra_1p5e6": 0.040,
    "gr2_xi_p1_pm4_ra_5e5": 0.015,
    "gr2_xi_p1_pm6_ra_5p5e6": 0.016,
    "gr2_xi_p35_pm4_ra_2e6": 0.044,
    "gr2_xi_p35_pm4_ra_5e6": 0.127,
    "gr2_xi_p35_pm6_ra_1p5e6": 0.026,
    "gr_gr2_Louis_ra_1p5e7": 0.121,
    "gr_gr2_Louis_ra_1e7": 0.084,
}

df_final["Ro_conv"] = np.nan

for pattern, value in mapping.items():
    mask = df_final["name"].str.startswith(pattern)
    df_final.loc[mask, "Ro_conv"] = value
    
mapping = {
    "gr_Nr2p5_Pm4_ra_1p6e7": 15.09,
    "gr_Nr2p5_Pm4_ra_8e6": 1.44,
    "gr2_xi_p2_pm4_ra_1e6": 1.68,
    "gr2_xi_p2_pm6_ra_1p5e6": 19.93,
    "gr2_xi_p1_pm4_ra_5e5": 1.49,
    "gr2_xi_p1_pm6_ra_5p5e6": 1.49,
    "gr2_xi_p35_pm4_ra_2e6": 9.35,
    "gr2_xi_p35_pm4_ra_5e6": 29.58,
    "gr2_xi_p35_pm6_ra_1p5e6": 9.75,
    "gr_gr2_Louis_ra_1p5e7": 31.42,
    "gr_gr2_Louis_ra_1e7": 20.06,
}

df_final["Elsasser"] = np.nan

for pattern, value in mapping.items():
    mask = df_final["name"].str.startswith(pattern)
    df_final.loc[mask, "Elsasser"] = value

df_final.to_parquet("transport_profiles1.parquet")
