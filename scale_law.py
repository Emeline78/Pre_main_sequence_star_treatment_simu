import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
git add scale_law.py
git commit -m "modifications"
git push
"""

df = pd.read_parquet("transport_profiles.parquet")

MS_mean = (df.groupby("name")["MS"].mean()).to_numpy()
names = df.groupby("name").mean().index.to_numpy()

Ra = (df.groupby("name")["ra"].first()).to_numpy()
g = (df.groupby("name")["config_code"].first()).to_numpy()
Ro_sh = (df.groupby("name")["om"].first()).to_numpy() * 1e-4

MS_min = np.full(len(names),np.nan)
MS_max = np.full(len(names),np.nan)
for i,namefile in enumerate(names): 
	data = np.load("snapshots/"+namefile+".npz")
	MS_snap = data["MS"]
	MS_min[i] = np.min(np.mean(MS_snaps,axis = 1))
	MS_max[i] = np.max(np.mean(MS_snaps,axis = 1))

plt.figure()
plt.errorbar(Ra, MS_mean, yerr=[MS_min, MS_max], fmt='o')
plt.xlabel("Rayleigh number")
plt.ylabel("Radial mean of the Maxwell stress of each run")

plt.figure()
plt.errorbar(g, MS_mean, yerr=[MS_min, MS_max], fmt='o')
plt.xlabel("Gravity configuration")
plt.ylabel("Radial mean of the Maxwell stress of each run")

plt.figure()
plt.errorbar(Ro_sh, MS_mean, yerr=[MS_min, MS_max], fmt='o')
plt.xlabel("Rossby shear number")
plt.ylabel("Radial mean of the Maxwell stress of each run")

plt.figure()
plt.scatter(Ra, g, c=MS_mean, cmap='viridis')
plt.colorbar(label='Radial mean of the Maxwell stress of each run')
plt.xlabel('Rayleigh number')
plt.ylabel('Gravity configuration')

plt.figure()
plt.scatter(Ra, Ro_sh, c=MS_mean, cmap='viridis')
plt.colorbar(label='Radial mean of the Maxwell stress of each run')
plt.xlabel('Rayleigh number')
plt.ylabel('Rossby shear number')
plt.show()
