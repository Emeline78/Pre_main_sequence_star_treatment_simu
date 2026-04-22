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
#RS_mean = (df.groupby("name")["RS"].mean()).to_numpy()
MS_max = df.groupby("name")["MS"].apply(lambda x: x.iloc[x.abs().argmax()]).to_numpy()
#RS_max = (df.groupby("name")["RS"].max()).to_numpy()

names = df.groupby("name").mean().index.to_numpy()

Ra = (df.groupby("name")["ra"].first()).to_numpy()
g = (df.groupby("name")["config_code"].first()).to_numpy()
om = (df.groupby("name")["om"].first()).to_numpy()
om_lim = (df.groupby("name")["om_lim"].first()).to_numpy()
Els = (df.groupby("name")["Elsasser"].first()).to_numpy()
Ro_conv = (df.groupby("name")["Ro_conv"].first()).to_numpy()
mask = om < om_lim
#mask = (df.groupby("name")["status"].first()).to_numpy()

MS_mean_dist = np.full(len(names),np.nan)
MS_max_dist = np.full(len(names),np.nan)
for i,namefile in enumerate(names): 
	data = np.load("snapshots/"+namefile+".npz")
	MS_snap = data["MS"]
	times = data["times"]
	r = data["r"]

	MS_snap_mean = np.trapz(np.mean(MS_snap, axis=1), times) / (times[-1]-times[0]) * MS_fact
	print(MS_snap_mean,MS_mean[i])
	MS_mean_dist[i] = np.sqrt(np.mean((x - MS_mean[i])**2)) / np.sqrt(len(x))
	
	x = MS_snap[np.arange(len(MS_snap)), np.abs(MS_snap).argmax(axis=1)]
	print(np.mean(x),MS_max[i])
	MS_max_dist[i] = np.sqrt(np.mean((x - MS_max[i])**2)) / np.sqrt(len(x))

plt.figure()
plt.errorbar(Ro_conv[mask], MS_mean[mask], yerr=MS_mean_dist[mask], fmt='o', label = "Radial mean")
plt.errorbar(Ro_conv[mask], MS_max[mask], yerr=MS_max_dist[mask], fmt='o', label = "Radial max")
plt.xlabel("Rossby convectif")
plt.ylabel("MS of each run")
plt.show()

plt.figure()
plt.errorbar(Els[mask], MS_mean[mask], yerr=MS_mean_dist[mask], fmt='o', label = "Radial mean")
plt.errorbar(Els[mask], MS_max[mask], yerr=MS_max_dist[mask], fmt='o', label = "Radial max")
plt.xlabel("Elsasser number")
plt.ylabel("MS of each run")
plt.show()


"""
plt.figure()
plt.scatter(Ra, Ro_sh, c=MS_mean, cmap='viridis')
plt.colorbar(label='Radial mean of the Maxwell stress of each run')
plt.xlabel('Rayleigh number')
plt.ylabel('Rossby shear number')
"""


