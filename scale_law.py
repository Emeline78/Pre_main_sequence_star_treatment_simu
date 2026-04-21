import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
git add scale_law.py
git commit -m "modifications"
git push

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

df["om_lim"] = np.nan

for pattern, value in mapping.items():
    mask = df["name"].str.startswith(pattern)
    df.loc[mask, "om_lim"] = value
    
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

df["Ro_conv"] = np.nan

for pattern, value in mapping.items():
    mask = df["name"].str.startswith(pattern)
    df.loc[mask, "Ro_conv"] = value
    
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

df["Elsasser"] = np.nan

for pattern, value in mapping.items():
    mask = df["name"].str.startswith(pattern)
    df.loc[mask, "Elsasser"] = value
"""


df = pd.read_parquet("transport_profiles.parquet")

MS_mean = (df.groupby("name")["MS"].mean()).to_numpy()
RS_mean = (df.groupby("name")["RS"].mean()).to_numpy()
MS_max = df["MS"].abs().groupby(df["name"]).max().to_numpy()
RS_max = (df.groupby("name")["RS"].max()).to_numpy()

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
	x = np.mean(MS_snap,axis = 1)
	print(np.mean(x),)
	plt.figure()
	plt.plot(x)
	plt.axhline(MS_mean[i], color='r', linestyle='--')
	plt.show()
	
	MS_mean_dist[i] = np.sqrt(np.mean((x - MS_mean[i])**2)) / np.sqrt(len(x))
	
	x = np.max(np.abs(MS_snap),axis = 1)
	print(np.mean(x),MS_max[i])
	plt.figure()
	plt.plot(x)
	plt.axhline(MS_max[i], color='r', linestyle='--')
	plt.show()
	MS_max_dist[i] = np.sqrt(np.mean((x - MS_max[i])**2)) / np.sqrt(len(x))

plt.figure()
plt.errorbar(Ro_conv[mask], MS_mean[mask], yerr=MS_mean_dist[mask], fmt='o')
plt.xlabel("Rossby convectif")
plt.ylabel("Radial mean of the Maxwell stress of each run")

plt.figure()
plt.errorbar(Ro_conv[mask], MS_max[mask], yerr=MS_max_dist[mask], fmt='o')
plt.xlabel("Rossby convectif")
plt.ylabel("Radial max of the Maxwell stress of each run")
plt.show()
"""
plt.figure()
plt.scatter(Ra, Ro_sh, c=MS_mean, cmap='viridis')
plt.colorbar(label='Radial mean of the Maxwell stress of each run')
plt.xlabel('Rayleigh number')
plt.ylabel('Rossby shear number')
"""


