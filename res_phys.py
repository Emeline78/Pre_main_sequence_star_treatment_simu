import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

"""
git add res_phys.py
git commit -m "modifications"
git push
"""
Ek = 1e-4
target = "gr_Nr2p5_Pm4_ra_8e6_om100"
df_stay = pd.read_parquet("transport_profiles.parquet")
param = np.loadtxt("0.55msun.dat", skiprows=1)		#Age [Myrs]   R [Rsun]     rho(0.9R)[g/cm^3]
Omega_phys = 2 * np.pi/ (10*24*3600) * Ek
liste = []
Rsun = 6.957e8

for t,r_phys,rho_ref in param:
	r_phys *= 0.9 * Rsun
	rho_ref *= 1e3
	df = df_stay.copy()
	L_phys = r_phys * (1-df["xi"])

	df["r_phys"] = df["r"] * r_phys * (1 - df["xi"])

	scale = rho_ref * Omega_phys**2 * L_phys**5

	df["RS_SI"]   = df["RS"]   * scale
	df["MC_SI"]   = df["MC"]   * scale
	df["Visc_SI"] = df["Visc"] * scale
	df["MS_SI"]   = df["MS"]   * scale
	df["date"] = t
	
	liste.append(df)

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

df_final = pd.concat(liste, ignore_index=True)
df_final.to_parquet("transport_profiles_SI.parquet")

target = "gr_Nr2p5_Pm4_ra_8e6_om100"
df_stay = pd.read_parquet("transport_profiles.parquet")
param = np.loadtxt("0.55msun.dat", skiprows=1)		#Age [Myrs]   R [Rsun]     rho(0.9R)
Omega_phys = 2 * np.pi/ (10*24*3600) * Ek 
liste = []
Rsun = 6.957e10

for t,r_phys,rho_ref in param:
	r_phys *= 0.9 * Rsun
	#rho_ref *= 1e3
	df = df_stay.copy()
	L_phys = r_phys * (1-df["xi"])

	df["r_phys"] = df["r"] * r_phys * (1 - df["xi"])

	scale = rho_ref * Omega_phys**2 * L_phys**5
	#nu_phys = Ek * Omega_phys * L_phys**2
	#scale = rho_ref * nu_phys**2 / L_phys**3

	df["RS_SI"]   = df["RS"]   * scale
	df["MC_SI"]   = df["MC"]   * scale
	df["Visc_SI"] = df["Visc"] * scale
	df["MS_SI"]   = df["MS"]   * scale
	df["date"] = t
	
	liste.append(df)


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

df_final = pd.concat(liste, ignore_index=True)
df_final.to_parquet("transport_profiles_CGS.parquet")
