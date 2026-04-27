import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
"""
git add res_phys.py
git commit -m "modifications"
git push
"""


df = pd.read_parquet("transport_profiles.parquet")

r_phys = 1e9
Omega_phys = 1e-5
rho_ref = 1e-8
mu0 = 4*np.pi*1e-7
L_phys = r_phys * df["xi"]

df["r_phys"] = df["r"] * r_phys / (1 - df["xi"])

scale = rho_ref * Omega_phys**2 * L_phys**5

df["RS_SI"]   = df["RS"]   * scale
df["MC_SI"]   = df["MC"]   * scale
df["Visc_SI"] = df["Visc"] * scale
df["MS_SI"]   = df["MS"]   * scale

target = "gr_Nr2p5_Pm4_ra_8e6_om100"

sim = df[df["name"] == target]
 
r    = sim["r_phys"].values
RS   = sim["RS_SI"].values
MC   = sim["MC_SI"].values
Visc = sim["Visc_SI"].values
MS   = sim["MS_SI"].values

F = RS + MS + MC + Visc
plt.figure()
plt.plot(r, RS,   label="Reynolds stress")
plt.plot(r, MC,   label="Meridional circulation with Coriolis part")
plt.plot(r, Visc, label="Viscous stress")
plt.plot(r,MS, label = "Maxwell stress")
plt.plot(r, F,"k", label = "Radial flux", linewidth=3)
plt.xlabel(r"$r$")
plt.ylabel(r"Fluxes")
plt.title(target)
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()


