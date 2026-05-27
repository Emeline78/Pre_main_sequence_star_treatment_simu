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
git add fohm.py
git commit -m "modifications"
git push
"""

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


from magic import *
mu0 = 4*np.pi*1e-7

liste = []
index = []
snap_dir = "snapshots2"

all_dirs = (list(Path("/travail/dynconv/multiscale_dyno/anelasticCouette/gr").glob("Nr*/ra*/om*")) +list(Path("/travail/dynconv/multiscale_dyno/anelasticCouette/gr2").glob("xi*/ra*/om*")) +list(Path("/travail/dynconv/multiscale_dyno/anelasticCouette/gr_gr2_Louis").glob("ra*/om*")))

valid_dirs = []

for path in all_dirs:
    last_part = Path(path).name
    if re.fullmatch(r'om\d+', last_part):
        valid_dirs.append(path)
    else:
        print(f"Skipping invalid case: {last_part}")

all_dirs = valid_dirs


df = pd.read_parquet("transport_profiles_adim.parquet")
df["fohm"] = np.nan

    


for path in all_dirs:
	a = str(path)
	print(a)
	params = extract_params(path)
	case_name = make_case_name(path)
	snap_file = Path(snap_dir) / f"{case_name}.npz"
	
	ts = MagicTs(datadir = a, field='power', all=True, iplot = False) 	# verification que le regime ne change pas dans le temps pour pouvoir faire l'integration en temps 
	fohm = ts.fohm
	mask = df["name"].str.startswith(case_name)
	df.loc[mask, "fohm"] = fohm


df.to_parquet("transport_profiles_adim.parquet")
