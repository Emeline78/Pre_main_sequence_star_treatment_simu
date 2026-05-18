import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit

"""
git add plot.py
git commit -m "modifications"
git push
"""

df = pd.read_parquet("transport_profiles_adim_sep.parquet")

names = df.groupby("name").mean().index.to_numpy(dtype = str)
om = (df.groupby("name")["om"].first()).to_numpy()
om_lim = (df.groupby("name")["om_lim"].first()).to_numpy()
mask = (om < om_lim) & (df.groupby("name")["status"].first().to_numpy()) & (np.char.find(names, "wrong") == -1)

for target in names[mask] : 
	sim = df[df["name"] == target]
	 
	r    = sim["r"].values
	MS   = sim["MS"].values
	MT   = sim["MT"].values

	Tot = MS + MT
	plt.figure()
	plt.plot(r,MS, label = "Maxwell stress")
	plt.plot(r,MT, label = "Magnetic tension")
	plt.plot(r, Tot,"k", label = "Sum", linewidth=3)
	plt.xlabel(r"$r$")
	plt.ylabel(r"Fluxes")
	plt.title(target)
	plt.grid()
	plt.legend()
	plt.tight_layout()
	plt.show()
