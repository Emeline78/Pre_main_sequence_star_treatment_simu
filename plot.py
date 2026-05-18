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
for target in names : 
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
