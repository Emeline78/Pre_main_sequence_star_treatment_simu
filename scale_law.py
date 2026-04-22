import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
Rm = (df.groupby("name")["rm"].first()).to_numpy()
Ro_sh = om*1e-4
mask = om < om_lim
#mask = (df.groupby("name")["status"].first()).to_numpy()

MS_mean_dist = np.full(len(names),np.nan)
MS_max_dist = np.full(len(names),np.nan)
for i,namefile in enumerate(names): 
	data = np.load("snapshots/"+namefile+".npz")
	MS_snap = data["MS"]
	times = data["times"]
	r = data["r"]
	
	x = np.mean(MS_snap, axis=1)
	MS_snap_mean = np.trapz(x, times) / (times[-1]-times[0])
	#print(MS_snap_mean,MS_mean[i])
	MS_mean_dist[i] = np.sqrt(np.mean((x - MS_mean[i])**2)) / np.sqrt(len(x))
	
	x = MS_snap[np.arange(len(MS_snap)), np.abs(MS_snap).argmax(axis=1)]
	#print(np.mean(x),MS_max[i])
	MS_max_dist[i] = np.sqrt(np.mean((x - MS_max[i])**2)) / np.sqrt(len(x))



color_values = g[mask]
norm = plt.Normalize(vmin=color_values.min(), vmax=color_values.max())
cmap1 = cm.inferno  
colors1 = cmap1(norm(color_values))

cmap2 = cm.cool  
colors2 = cmap(norm(color_values))

plt.figure()
plt.errorbar(Ro_conv[mask], MS_mean[mask],yerr=MS_mean_dist[mask],fmt='none',ecolor=colors1)
sc = plt.scatter(Ro_conv[mask], MS_mean[mask],c=color_values, cmap='inferno', norm=norm,zorder=5, label="Radial mean")
plt.colorbar(sc, label="Gravity configuration")

plt.errorbar(Ro_conv[mask], MS_max[mask],yerr=MS_max_dist[mask],fmt='none',ecolor=colors2)
sc = plt.scatter(Ro_conv[mask], MS_max[mask],c=color_values, cmap='cool', norm=norm,zorder=5, label="Radial max")
plt.colorbar(sc)

plt.xlabel("Rossby convectif")
plt.ylabel("MS of each run")
plt.legend()

plt.figure()
plt.errorbar(Els[mask], MS_mean[mask],yerr=MS_mean_dist[mask],fmt='none',ecolor=colors1)
sc = plt.scatter(Els[mask], MS_mean[mask],c=color_values, cmap='inferno', norm=norm,zorder=5, label="Radial mean")
plt.colorbar(sc, label="Gravity configuration")

plt.errorbar(Els[mask], MS_max[mask],yerr=MS_max_dist[mask],fmt='none',ecolor=colors2)
sc = plt.scatter(Els[mask], MS_max[mask],c=color_values, cmap='cool', norm=norm,zorder=5, label="Radial max")
plt.colorbar(sc)
plt.xlabel("Elsasser number")
plt.ylabel("MS of each run")
plt.legend()

plt.figure()
plt.errorbar(Ro_sh[mask], MS_mean[mask],yerr=MS_mean_dist[mask],fmt='none',ecolor=colors1)
sc = plt.scatter(Ro_sh[mask], MS_mean[mask],c=color_values, cmap='inferno', norm=norm,zorder=5, label="Radial mean")
plt.colorbar(sc, label="Gravity configuration")

plt.errorbar(Ro_sh[mask], MS_max[mask],yerr=MS_max_dist[mask],fmt='none',ecolor=colors2)
sc = plt.scatter(Ro_sh[mask], MS_max[mask],c=color_values, cmap='cool', norm=norm,zorder=5, label="Radial max")
plt.colorbar(sc)
plt.xlabel("Rossby shear")
plt.ylabel("MS of each run")
plt.legend()

plt.figure()
plt.errorbar(Rm[mask], MS_mean[mask],yerr=MS_mean_dist[mask],fmt='none',ecolor=colors1)
sc = plt.scatter(Rm[mask], MS_mean[mask],c=color_values, cmap='inferno', norm=norm,zorder=5, label="Radial mean")
plt.colorbar(sc, label="Gravity configuration")

plt.errorbar(Rm[mask], MS_max[mask],yerr=MS_max_dist[mask],fmt='none',ecolor=colors2)
sc = plt.scatter(Rm[mask], MS_max[mask],c=color_values, cmap='cool', norm=norm,zorder=5, label="Radial max")
plt.colorbar(sc)
plt.xlabel("Reynolds magnetic")
plt.ylabel("MS of each run")
plt.legend()


"""
plt.figure()
plt.scatter(Ra, Ro_sh, c=MS_mean, cmap='viridis')
plt.colorbar(label='Radial mean of the Maxwell stress of each run')
plt.xlabel('Rayleigh number')
plt.ylabel('Rossby shear number')
"""
target = "gr_Nr2p5_Pm4_ra_8e6_om100"
sim = df[df["name"] == target]
 
r    = sim["r"].values
RS   = sim["RS"].values
MC   = sim["MC"].values
Visc = sim["Visc"].values
MS   = sim["MS"].values

F = RS + MS + MC + Visc
plt.figure()
plt.plot(r, RS,   label="Reynolds stress")
plt.plot(r, MC,   label="Meridional circulation with Coriolis part")
plt.plot(r, Visc, label="Viscous stress")
plt.plot(r,MS, label = "Maxwell stress")
plt.plot(r, F,"k", label = "Radial flux", linewidth=3)
plt.xlabel(r"$r$")
plt.ylabel(r"Torque")
plt.title(target)
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()




