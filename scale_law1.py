import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
"""
git add scale_law1.py
git commit -m "modifications"
git push
"""

df = pd.read_parquet("transport_profiles.parquet")

#MS_mean = (df.groupby("name")["MS"].mean()).to_numpy()
MS_rms = df.groupby("name").apply(lambda g: np.sqrt(np.mean(g["MS"]**2))).to_numpy()
MS_int = df.groupby("name").apply(lambda g: np.trapz(g["MS"], g["r"])).to_numpy()

#RS_mean = (df.groupby("name")["RS"].mean()).to_numpy()
MS_max = df.groupby("name")["MS"].apply(lambda x: x.iloc[x.abs().argmax()]).to_numpy()
#RS_max = (df.groupby("name")["RS"].max()).to_numpy()

names = df.groupby("name").mean().index.to_numpy(dtype = str)

Ra = (df.groupby("name")["ra"].first()).to_numpy()
g = (df.groupby("name")["config_code"].first()).to_numpy()
om = (df.groupby("name")["om"].first()).to_numpy()
om_lim = (df.groupby("name")["om_lim"].first()).to_numpy()
Els = (df.groupby("name")["Elsasser"].first()).to_numpy()
Ro_conv = (df.groupby("name")["Ro_conv"].first()).to_numpy()
Rm = (df.groupby("name")["rm"].first()).to_numpy()
xi = (df.groupby("name")["xi"].first()).to_numpy()
Ro_sh = om*1e-4

mask = (om < om_lim) & (df.groupby("name")["status"].first().to_numpy()) & (np.char.find(names, "wrong") == -1)
#mask = (om > om_lim) & (df.groupby("name")["status"].first().to_numpy()) & (np.char.find(names, "wrong") == -1)

MS_mean_err = np.full(len(names),np.nan)
MS_max_err = np.full(len(names),np.nan)
MS_rms_err = np.full(len(names),np.nan)
MS_int_err = np.full(len(names),np.nan)

for i,namefile in enumerate(names): 
	data = np.load("snapshots/"+namefile+".npz")
	MS_snap = data["MS"]
	times = data["times"]
	r = data["r"]
	
	x = np.mean(MS_snap, axis=1)
	MS_mean_err[i] = np.std(x) / np.sqrt(len(x))
	
	x = np.sqrt(np.mean(MS_snap**2, axis=1))
	MS_rms_err[i] = np.std(x) / np.sqrt(len(x))
	
	x = np.array([np.trapz(MS_snap[i], r) for i in range(len(MS_snap))])
	MS_int_err[i] = np.std(x) / np.sqrt(len(x))
	
	x = MS_snap[np.arange(len(MS_snap)), np.abs(MS_snap).argmax(axis=1)]
	#print(np.mean(x),MS_max[i])
	MS_max_err[i] = np.std(x) / np.sqrt(len(x))


mask1 = mask & (g == 0)
mask2 = mask & (g == 1)
mask3 = mask & (g == 2)
g_labels = {0: r"$g \propto r$", 1: r"$g \propto 1/r^2$", 2: "CESAM 2k20"}
plt.figure()
plt.subplot(1,3,1)
plt.errorbar(Ro_sh[mask1],MS_rms[mask1], yerr=MS_rms_err[mask1], fmt='o',label = r"$g \propto r$")
plt.xlabel("Rossby shear")
plt.ylabel("MS mean")
plt.legend()
plt.grid()

plt.subplot(1,3,2)
plt.errorbar(Ro_sh[mask2],MS_rms[mask2], yerr=MS_rms_err[mask2], fmt='o', label = r"$g \propto 1/r^2$")
plt.xlabel("Rossby shear")
plt.ylabel("MS mean")
plt.legend()
plt.grid()

plt.subplot(1,3,3)
plt.errorbar(Ro_sh[mask3],MS_rms[mask3], yerr=MS_rms_err[mask3], fmt='o', label = "CESAM 2k20")
plt.xlabel("Rossby shear")
plt.ylabel("MS mean")
plt.suptitle("Radial mean of MS as a function of the Rossby shear")
plt.legend()
plt.grid()

plt.figure()
plt.subplot(1,3,1)
plt.errorbar(Ro_sh[mask1],MS_max[mask1], yerr=MS_max_err[mask1], fmt='o',label = r"$g \propto r$")
plt.xlabel("Rossby shear")
plt.ylabel("MS mean")
plt.legend()
plt.grid()

plt.subplot(1,3,2)
plt.errorbar(Ro_sh[mask2],MS_max[mask2], yerr=MS_max_err[mask2], fmt='o', label = r"$g \propto 1/r^2$")
plt.xlabel("Rossby shear")
plt.ylabel("MS mean")
plt.legend()
plt.grid()

plt.subplot(1,3,3)
plt.errorbar(Ro_sh[mask3],MS_max[mask3], yerr=MS_max_err[mask3], fmt='o', label = "CESAM 2k20")
plt.xlabel("Rossby shear")
plt.ylabel("MS mean")
plt.suptitle("Radial max of MS as a function of the Rossby shear")
plt.legend()
plt.grid()


mask1 = mask & (xi == 0.1)
mask2 = mask & (xi == 0.2)
mask3 = mask & (xi == 0.35)
plt.figure()
plt.errorbar(Ro_conv[mask1],MS_rms[mask1], yerr=MS_rms_err[mask1], fmt='o')
plt.errorbar(Ro_conv[mask2],MS_rms[mask2], yerr=MS_rms_err[mask2], fmt='o')
plt.errorbar(Ro_conv[mask3],MS_rms[mask3], yerr=MS_rms_err[mask3], fmt='o')
plt.xlabel("Convective Rossby")
plt.ylabel("MS mean")
plt.grid()
plt.show()


from scipy.optimize import least_squares


def residuals(params, x, y, err):
    a, b = params
    return (y - linear_model(x, a, b)) / err
    
def linear_model(x, a, b):
	return a * x + b
    
def interp(x,y,yerr):
	valid = (x > 0) & (y > 0)
	x, y, yerr = x[valid], y[valid], yerr[valid]

	logx = np.log10(x)
	logy = np.log10(y)
	logy_err = yerr / (y * np.log(10))

	#params, cov = curve_fit(linear_model, logx, logy) #, sigma=logy_err)
	#a, b = params
	
	res = least_squares(residuals,x0=[1, 0],args=(logx, logy, logy_err),loss='soft_l1')
	a, b = res.x
	
	resi = logy - (a*logx + b)
	print(np.std(resi))
	
	x_plot = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
	y_plot = 10**b * x_plot**a
	return(a,b,x_plot,y_plot)


MS_int_sign = np.sign(MS_int)
MS_int_amp  = np.abs(MS_int)
# ======================== ROSSBY CONVECTIF =========================
a_mean,b_mean,x_plot,y_plot = interp(Ro_conv[mask],MS_rms[mask],MS_rms_err[mask])

plt.figure()
plt.subplot(1,2,1)
plt.plot(x_plot, y_plot, color='red')
plt.errorbar(Ro_conv[mask],MS_rms[mask], yerr=MS_rms_err[mask], fmt='o')
plt.xlabel("Convective Rossby")
plt.ylabel("MS root mean square")
plt.grid()
plt.title(rf"$MS_{{rms}} = 10^{{{b_mean:.2f}}} \cdot Ro_{{conv}}^{{{a_mean:.2f}}}$")
print(rf"$MS_{{rms}} = 10^{{{b_mean:.2f}}} \cdot Ro_{{conv}}^{{{a_mean:.2f}}}$")

a_max,b_max,x_plot,y_plot = interp(Ro_conv[mask],MS_max[mask],MS_max_err[mask])
plt.subplot(1,2,2)
plt.plot(x_plot, y_plot, color='red')
plt.errorbar(Ro_conv[mask],MS_max[mask], yerr=MS_max_err[mask], fmt='o')
plt.xlabel("Convective Rossby")
plt.ylabel("MS max")
plt.title(rf"$MS_{{max}} = 10^{{{b_max:.2f}}} \cdot Ro_{{conv}}^{{{a_max:.2f}}}$")
plt.grid()
print(rf"$MS_{{max}} = 10^{{{b_max:.2f}}} \cdot Ro_{{conv}}^{{{a_max:.2f}}}$")

a_mean,b_mean,x_plot,y_plot = interp(Ro_conv[mask],MS_int_amp[mask],MS_int_err[mask])
plt.figure()
plt.plot(x_plot, y_plot, color='red')
plt.errorbar(Ro_conv[mask],MS_int_amp[mask], yerr=MS_int_err[mask], fmt='o')
plt.xlabel("Convective Rossby")
plt.ylabel("MS integrated")
plt.grid()
plt.title(rf"$MS_{{int}} = 10^{{{b_mean:.2f}}} \cdot Ro_{{conv}}^{{{a_mean:.2f}}}$")
print(rf"$MS_{{int}} = 10^{{{b_mean:.2f}}} \cdot Ro_{{conv}}^{{{a_mean:.2f}}}$")

# ======================== ELSASSER =========================
a_mean,b_mean,x_plot,y_plot = interp(Els[mask],MS_rms[mask],MS_rms_err[mask])
plt.figure()
plt.subplot(1,2,1)
plt.errorbar(Els[mask],MS_rms[mask], yerr=MS_rms_err[mask], fmt='o')
plt.plot(x_plot, y_plot, color='red')
plt.xlabel("Elsasser number")
plt.ylabel("MS root mean square")
plt.grid()
plt.title(rf"$MS_{{rms}} = 10^{{{b_mean:.2f}}} \cdot \Lambda^{{{a_mean:.2f}}}$")
print(rf"$MS_{{rms}} = 10^{{{b_mean:.2f}}} \cdot \Lambda^{{{a_mean:.2f}}}$")

a_max,b_max,x_plot,y_plot = interp(Els[mask],MS_max[mask],MS_max_err[mask])
plt.subplot(1,2,2)
plt.errorbar(Els[mask],MS_max[mask], yerr=MS_max_err[mask], fmt='o')
plt.plot(x_plot, y_plot, color='red')
plt.xlabel("Elsasser number")
plt.ylabel("MS max")
plt.title(rf"$MS_{{max}} = 10^{{{b_max:.2f}}} \cdot \Lambda^{{{a_max:.2f}}}$")
plt.grid()
print(rf"$MS_{{max}} = 10^{{{b_max:.2f}}} \cdot \Lambda^{{{a_max:.2f}}}$")

a_mean,b_mean,x_plot,y_plot = interp(Els[mask],MS_int_amp[mask],MS_int_err[mask])
plt.figure()
plt.errorbar(Els[mask],MS_int_amp[mask], yerr=MS_int_err[mask], fmt='o')
plt.plot(x_plot, y_plot, color='red')
plt.xlabel("Elsasser number")
plt.ylabel("MS integrated")
plt.grid()
plt.title(rf"$MS_{{int}} = 10^{{{b_mean:.2f}}} \cdot \Lambda^{{{a_mean:.2f}}}$")
print(rf"$MS_{{int}} = 10^{{{b_mean:.2f}}} \cdot \Lambda^{{{a_mean:.2f}}}$")

# ======================== RM =========================
a_mean,b_mean,x_plot,y_plot = interp(Rm[mask],MS_rms[mask],MS_rms_err[mask])
plt.figure()
plt.subplot(1,2,1)
plt.errorbar(Rm[mask],MS_rms[mask], yerr=MS_rms_err[mask], fmt='o')
plt.plot(x_plot, y_plot, color='red')
plt.xlabel("Reynolds magnetic")
plt.ylabel("MS root mean square")
plt.grid()
plt.title(rf"$MS_{{rms}} = 10^{{{b_mean:.2f}}} \cdot Rm^{{{a_mean:.2f}}}$")
print(rf"$MS_{{rms}} = 10^{{{b_mean:.2f}}} \cdot Rm^{{{a_mean:.2f}}}$")

a_max,b_max,x_plot,y_plot = interp(Rm[mask],MS_max[mask],MS_max_err[mask])
plt.subplot(1,2,2)
plt.errorbar(Rm[mask],MS_max[mask], yerr=MS_max_err[mask], fmt='o')
plt.plot(x_plot, y_plot, color='red')
plt.xlabel("Reynolds magnetic")
plt.ylabel("MS max")
plt.title(rf"$MS_{{max}} = 10^{{{b_max:.2f}}} \cdot Rm^{{{a_max:.2f}}}$")
plt.grid()
print(rf"$MS_{{max}} = 10^{{{b_max:.2f}}} \cdot Rm^{{{a_max:.2f}}}$")

a_mean,b_mean,x_plot,y_plot = interp(Rm[mask],MS_int_amp[mask],MS_int_err[mask])
plt.figure()
plt.errorbar(Rm[mask],MS_int_amp[mask], yerr=MS_int_err[mask], fmt='o')
plt.plot(x_plot, y_plot, color='red')
plt.xlabel("Reynolds magnetic")
plt.ylabel("MS integrated")
plt.grid()
plt.title(rf"$MS_{{int}} = 10^{{{b_mean:.2f}}} \cdot Rm^{{{a_mean:.2f}}}$")
print(rf"$MS_{{int}} = 10^{{{b_mean:.2f}}} \cdot Rm^{{{a_mean:.2f}}}$")
plt.show()


"""
# ===================== OBSERVATIONAL DATA =====================

B_obs_kG = np.array([1,3])   # kG
Prot_days = np.array([11.55,0.91,0.79,3.70,3.48,11.00]) # days
tau_conv_days = np.array([268.3,363.1,357.9,351.9,238.0,334.1]) # days

# --- Unit conversions ---
B_obs = B_obs_kG * 1e3 * 1e-4  # Tesla
Omega_obs = 2 * np.pi / (Prot_days * 86400)

# --- ASSUMPTIONS ---
mu0 = 4 * np.pi * 1e-7

rho = 100     # kg/m^3 (take the ones for the right age of Cesam)
eta = 1e8     # m^2/s (hope for the best)

#Els_obs = B_obs**2 / (rho * mu0 * eta * Omega_obs)
Ro_conv_obs = Prot_days / tau_conv_days

MS_obs_lim = B_obs**2/mu0

#=======================================================================================================
df_tot = pd.read_parquet("transport_profiles_SI.parquet")
for date, df in df_tot.groupby('date'):
	if date > 5:
		print(f"Computation for age = {date} Myr")
		r0 = df.groupby("name")["r_phys"].apply(lambda x: x.iloc[x.abs().argmax()]).to_numpy()
		MS_mean = (df.groupby("name")["MS_SI"].mean()).to_numpy() / r0**3
		MS_max = df.groupby("name")["MS_SI"].apply(lambda x: x.iloc[x.abs().argmax()]).to_numpy() / r0**3

		names = df.groupby("name").mean().index.to_numpy(dtype = str)
		
		scale = (df.groupby("name")["scale"].first()).to_numpy()
		Ra = (df.groupby("name")["ra"].first()).to_numpy()
		g = (df.groupby("name")["config_code"].first()).to_numpy()
		om = (df.groupby("name")["om"].first()).to_numpy()
		om_lim = (df.groupby("name")["om_lim"].first()).to_numpy()
		Els = (df.groupby("name")["Elsasser"].first()).to_numpy()
		Ro_conv = (df.groupby("name")["Ro_conv"].first()).to_numpy()
		Rm = (df.groupby("name")["rm"].first()).to_numpy()
		Ro_sh = om*1e-4

		mask = (om < om_lim) & (df.groupby("name")["status"].first().to_numpy()) & (np.char.find(names, "wrong") == -1)
		#mask = (om > om_lim) & (df.groupby("name")["status"].first().to_numpy()) & (np.char.find(names, "wrong") == -1)


		MS_mean_dist = np.full(len(names),np.nan)
		MS_max_dist = np.full(len(names),np.nan)
		for i,namefile in enumerate(names): 
			data = np.load("snapshots/"+namefile+".npz")
			MS_snap = data["MS"] * scale[i] / r0[i]**3
			times = data["times"]
			
			x = np.mean(MS_snap, axis=1) 
			MS_snap_mean = np.trapz(x, times) / (times[-1]-times[0])
			#print(MS_snap_mean,MS_mean[i])
			MS_mean_dist[i] = np.sqrt(np.mean((x - MS_mean[i])**2)) / np.sqrt(len(x))
			
			x = MS_snap[np.arange(len(MS_snap)), np.abs(MS_snap).argmax(axis=1)]
			#print(np.mean(x),MS_max[i])
			MS_max_dist[i] = np.sqrt(np.mean((x - MS_max[i])**2)) / np.sqrt(len(x))

		MS_sign = np.sign(MS_mean)
		MS_mean = np.abs(MS_mean)
		
		# ======================== ROSSBY CONVECTIF =========================
		a_mean,b_mean,x_plot,y_plot = interp(Ro_conv[mask],MS_mean[mask],MS_mean_dist[mask])
		MS_obs = 10**b_mean * Ro_conv_obs**a_mean
		
		plt.figure()
		plt.errorbar(Ro_conv[mask],MS_mean[mask], yerr=MS_mean_dist[mask], fmt='o')
		plt.scatter(Ro_conv_obs, MS_obs, marker='*', s=120, color='red', edgecolor='black',label="Observations")
		plt.axhspan(MS_obs_lim[0],MS_obs_lim[1],color = "gray", alpha = 0.3)
		plt.plot(x_plot, y_plot, color='black')
		plt.xlabel("Convective Rossby")
		plt.ylabel("MS mean")
		#plt.xscale('log')
		#plt.yscale('log')
		plt.grid()
		plt.title(rf"$MS_{{mean}} = 10^{{{b_mean:.2f}}} \cdot Ro_{{conv}}^{{{a_mean:.2f}}}$")
		
		plt.figure()
		plt.plot(Ro_conv[mask], MS_sign[mask],"+")
		plt.xlabel("Convective Rossby")
		plt.ylabel("MS sign")
		plt.grid()
		plt.title("Sign of MS as a function of the Rossby")

		a_max,b_max,x_plot,y_plot = interp(Ro_conv[mask],MS_max[mask],MS_max_dist[mask])
		MS_obs = 10**b_max * Ro_conv_obs**a_max
		plt.figure()
		plt.errorbar(Ro_conv[mask],MS_max[mask], yerr=MS_max_dist[mask], fmt='o')
		plt.scatter(Ro_conv_obs, MS_obs, marker='*', s=120, color='red', edgecolor='black',label="Observations")
		plt.axhspan(MS_obs_lim[0],MS_obs_lim[1],color = "gray", alpha = 0.3)
		plt.plot(x_plot, y_plot, color='black')
		plt.xlabel("Convective Rossby")
		plt.ylabel("MS max")
		#plt.xscale('log')
		#plt.yscale('log')
		plt.title(rf"$MS_{{max}} = 10^{{{b_max:.2f}}} \cdot Ro_{{conv}}^{{{a_max:.2f}}}$")
		plt.grid()
		plt.show()

"""
