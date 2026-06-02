import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score
from matplotlib.colors import LogNorm
"""
git add plot_scalelaw.py
git commit -m "modifications"
git push
"""

a = "transport_profiles_adim.parquet"
df = pd.read_parquet(a)
datadir = "snapshots1/"

names = df.groupby("name").mean().index.to_numpy(dtype = str)
Ra = (df.groupby("name")["ra"].first()).to_numpy()
g = (df.groupby("name")["config_code"].first()).to_numpy()
om = (df.groupby("name")["om"].first()).to_numpy()
om_lim = (df.groupby("name")["om_lim"].first()).to_numpy()
Els = (df.groupby("name")["Elsasser"].first()).to_numpy()
Ro = (df.groupby("name")["Ro_conv"].first()).to_numpy()
Rm = (df.groupby("name")["rm"].first()).to_numpy()
Nu = (df.groupby("name")["Nu"].first()).to_numpy()
xi = (df.groupby("name")["xi"].first()).to_numpy()
Pm = (df.groupby("name")["Pm"].first()).to_numpy()
fohm = (df.groupby("name")["fohm"].first()).to_numpy()

E = 1e-4
Ro_sh = om * E
Pr = 1
Nu_mod = (Nu - 1) * E / Pr
Ra_mod = Ra * (Nu - 1) * E**3 / Pr**2
Lo_fohm = ((Els * E / Pm) / fohm)**(1/2) 

Ra_mod_al = Ra_mod.copy()
Lo_fohm_al = Lo_fohm.copy()

mask = (g==1) & (om < om_lim)  & (np.char.find(names, "wrong") == -1) & (df.groupby("name")["status"].first().to_numpy())

added_df = pd.read_csv('added_data.dat', sep='\s+', header=0)

Ra_added = added_df["Ra"].to_numpy()
Nu_added = added_df["Nu"].to_numpy()
E_added = added_df["E"].to_numpy()
Pr_added = added_df["Pr"].to_numpy()
Ro_added = added_df["Ro"].to_numpy()
Pm_added = added_df["Pm"].to_numpy()
ki_added = added_df["ki"].to_numpy()
Rosh_added = np.zeros(len(Pm_added))
g_added = np.ones(len(Pm_added))

Lo_fohm_added = added_df["Lo"].to_numpy()/(added_df["fohm"].to_numpy())**(1/2)
Ra_mod_added = Ra_added * (Nu_added - 1) * E_added**3 / Pr_added**2 * (1-ki_added)**2
Nu_mod_added = (Nu_added - 1) * E_added / Pr_added


Ra_mod_both = np.concatenate([Ra_mod[mask], Ra_mod_added])
Lo_fohm_both = np.concatenate([Lo_fohm[mask], Lo_fohm_added])
Ro_both = np.concatenate([Ro[mask], Ro_added])
Pm_both = np.concatenate([Pm[mask], Pm_added])
Ro_sh_both = np.concatenate([Ro_sh[mask], Rosh_added])
g_both = np.concatenate([g[mask], g_added])


mine = lambda x: 0.99*x**(0.31)
christensen = lambda x: 0.92*x**(0.34)
schrinner = lambda x: 1.58*x**(0.35)	
mine_both = lambda x: 1.34*x**(0.34)

x = np.linspace(Ra_mod_both.min(),Ra_mod_both.max(),1000)

plt.figure()
plt.plot(x,christensen(x),"r-",label ="Christensen's law")
plt.plot(x,mine(x),"g-",label ="Law for my data")
plt.plot(x,mine_both(x),"g-",label ="Law for both data set")
plt.plot(Ra_mod_added,Lo_fohm_added,"k*",label ="Schrinner's data")
sc = plt.scatter(Ra_mod[mask],Lo_fohm[mask],c = Ro_sh[mask],s=60, norm=LogNorm(vmin=Ro_sh[mask].min(), vmax=Ro_sh[mask].max()),label="My data")
plt.xlabel(r"$Ra_{Q}$")
plt.ylabel(r"$\frac{Lo}{f_{ohm}^{1/2}}$ from simulations")
plt.title(r"Scale law of $\frac{Lo}{f_{ohm}^{1/2}}$ for $g \propto 1/r^2$")
plt.colorbar(sc)
plt.grid()
plt.legend()
plt.show()


