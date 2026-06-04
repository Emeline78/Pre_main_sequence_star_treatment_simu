import pandas as pd
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from scipy.optimize import curve_fit
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score
from scipy.optimize import least_squares
from matplotlib.colors import LogNorm
"""
git add scale_law_fin.py
git commit -m "modifications"
git push
"""

a = "transport_profiles_adim.parquet"
df = pd.read_parquet(a)
datadir = "snapshots1/"

if a == "transport_profiles_SI.parquet" or a == "transport_profiles_CGS.parquet" :
	df = df[(df["date"] > 5) & (df["date"] < 6)]
	

MS_mean = (df.groupby("name")["MS"].mean()).to_numpy()
MS_rms = df.groupby("name").apply(lambda g: np.sqrt(np.mean(g["MS"]**2))).to_numpy()
MS_int = df.groupby("name").apply(lambda g: np.trapz(g["MS"], g["r"])).to_numpy()
MS_min = df.groupby("name")["MS"].apply(lambda x: x.iloc[x.argmin()]).to_numpy()
MS_max = df.groupby("name")["MS"].apply(lambda x: x.iloc[x.argmax()]).to_numpy()

neg_width = (df.groupby("name")["MS"].apply(lambda x: np.sum(x < 0)).to_numpy())
Nmin = 10

mask = (MS_min < 0) & (neg_width > Nmin)
mask_min = MS_min < 0

names = df.groupby("name").mean().index.to_numpy(dtype = str)
Ra = (df.groupby("name")["ra"].first()).to_numpy()
g = (df.groupby("name")["config_code"].first()).to_numpy()
om = (df.groupby("name")["om"].first()).to_numpy()
om_lim = (df.groupby("name")["om_lim"].first()).to_numpy()
Els = (df.groupby("name")["Elsasser"].first()).to_numpy()
Ro_conv = (df.groupby("name")["Ro_conv_l"].first()).to_numpy()
Rm = (df.groupby("name")["rm"].first()).to_numpy()
xi = (df.groupby("name")["xi"].first()).to_numpy()
Ro_sh = om*1e-4

mask = (om < om_lim)  & (np.char.find(names, "wrong") == -1) & (df.groupby("name")["status"].first().to_numpy())
#mask = (om > om_lim) & (df.groupby("name")["status"].first().to_numpy()) & (np.char.find(names, "wrong") == -1)

MS_mean_err = np.full(len(names),np.nan)
MS_max_err = np.full(len(names),np.nan)
MS_min_err = np.full(len(names),np.nan)
MS_rms_err = np.full(len(names),np.nan)
MS_int_err = np.full(len(names),np.nan)

for i,namefile in enumerate(names): 
	data = np.load(datadir+namefile+".npz")
	MS_snap = data["MS"]
	times = data["times"]
	r = data["r"]
	
	x = np.mean(MS_snap, axis=1)
	MS_mean_err[i] = np.std(x) / np.sqrt(len(x))
	
	x = np.sqrt(np.mean(MS_snap**2, axis=1))
	MS_rms_err[i] = np.std(x) / np.sqrt(len(x))
	
	x = np.array([np.trapz(MS_snap[i], r) for i in range(len(MS_snap))])
	MS_int_err[i] = np.std(x) / np.sqrt(len(x))
	
	x = MS_snap[np.arange(len(MS_snap)), MS_snap.argmax(axis=1)]
	MS_max_err[i] = np.std(x) / np.sqrt(len(x))
	
	x = MS_snap[np.arange(len(MS_snap)), MS_snap.argmin(axis=1)]
	MS_min_err[i] = np.std(x) / np.sqrt(len(x))
	
MS_int_sign = np.sign(MS_int)
MS_int_amp  = np.abs(MS_int)

def MS_at_middle(g):
    r_mid = 0.5 * (g["r"].min() + g["r"].max())
    idx = (g["r"] - r_mid).abs().argmin()
    return g.iloc[idx]["MS"]

MS_mid = df.groupby("name").apply(MS_at_middle).to_numpy()

def model_func(X_flat, *params):
	n_vars = X_flat.shape[0]

	a = params[:-1]
	b = params[-1]

	Y_model = 10**b

	for i in range(n_vars):
		Y_model *= X_flat[i]**a[i]

	return Y_model

def model_func_signed(X_flat, *params):

	n_vars = X_flat.shape[0]

	a = params[:-1]
	A = params[-1]

	Y_model = A
	with np.errstate(over='ignore'):
		for i in range(n_vars):
			Y_model *= X_flat[i]**a[i]

	return Y_model
	
def residuals_signed(params, X_stack, Y, Yerr):
	Y_model = model_func_signed(X_stack, *params)
	return (Y - Y_model) / Yerr

def evaluate_scaling_realspace(X_vars, Y, Yerr, signed = True):
	if signed :
		mask_fit = np.ones_like(Y, dtype=bool)

		for v in X_vars:
			mask_fit &= (v > 0)

		mask_fit &= np.isfinite(Y)
		mask_fit &= np.isfinite(Yerr)

		X_vars = [v[mask_fit] for v in X_vars]
		Y = Y[mask_fit]
		Yerr = Yerr[mask_fit]
		Yerr = np.maximum(Yerr, 1e-12)

		# ---------------- Initial guess from log fit ----------------

		logX = np.column_stack([np.log10(v) for v in X_vars])
		p0 = [0.5]*len(X_vars) + [np.mean(Y)]
		p0[-1] = np.mean(Y)
		
		bounds_lower = [-5]*len(X_vars) + [-np.inf]
		bounds_upper = [5]*len(X_vars) + [ np.inf]
		
		# ---------------- Real-space nonlinear fit ----------------

		X_stack = np.vstack(X_vars)

		params, cov = curve_fit(model_func_signed, X_stack, Y, sigma=Yerr, absolute_sigma=True, bounds=(bounds_lower, bounds_upper), p0=p0, maxfev=20000)
		#result = least_squares(residuals_signed,x0=p0,args=(X_stack, Y, Yerr), bounds=(bounds_lower, bounds_upper), max_nfev=50000)
		#params = result.x

		coefs = params[:-1]
		intercept = params[-1]
		Y_model = model_func_signed(X_stack, *params)
		
	else :
		mask_fit = np.ones_like(Y, dtype=bool)

		for v in X_vars:
			mask_fit &= (v > 0)

		mask_fit &= (Y > 0)
		mask_fit &= np.isfinite(Y)
		mask_fit &= np.isfinite(Yerr)

		X_vars = [v[mask_fit] for v in X_vars]
		Y = Y[mask_fit]
		Yerr = Yerr[mask_fit]

		# ---------------- Initial guess from log fit ----------------

		logX = np.column_stack([np.log10(v) for v in X_vars])
		logY = np.log10(Y)

		lin_model = LinearRegression().fit(logX, logY)
		p0 = np.append(lin_model.coef_,lin_model.intercept_)

		# ---------------- Real-space nonlinear fit ----------------

		X_stack = np.vstack(X_vars)

		params, cov = curve_fit(model_func,X_stack,Y,sigma=Yerr,absolute_sigma=True,p0=p0,maxfev=20000)

		coefs = params[:-1]
		intercept = 10**params[-1]
		
		Y_model = model_func(X_stack, *params)

	# ---------------- Predictions ----------------

	residuals = Y - Y_model

	ss_res = np.sum(residuals**2)
	ss_tot = np.sum((Y - np.mean(Y))**2)

	R2 = 1 - ss_res / ss_tot

	# ---------------- Adjusted R2 ----------------

	n = len(Y)
	p = len(coefs)

	adj_R2 = 1 - (1 - R2)*(n - 1)/(n - p - 1)

	# ---------------- Diagnostic parameters ----------------

	cond = np.linalg.cond(logX)
	pca = PCA().fit(logX)
	corr = np.corrcoef(logX.T)

	return {"mask_fit": mask_fit,"R2": R2,"adj_R2": adj_R2,"coefs": coefs,"intercept": intercept, "condition_number": cond,"PCA_variance": pca.explained_variance_ratio_, "correlation_matrix": corr,"Y_model": Y_model,"Y": Y,"residuals": residuals}


def loo_score(X_vars, Y, signed=False):

	loo = LeaveOneOut()

	preds = []
	truths = []

	X_stack = np.vstack(X_vars)

	for train_idx, test_idx in loo.split(Y):

		Xtr = X_stack[:, train_idx]
		Xte = X_stack[:, test_idx]

		Ytr = Y[train_idx]
		Yte = Y[test_idx]

		try:
			if signed:
				p0 = np.ones(Xtr.shape[0] + 1)
				p0[-1] = np.mean(Ytr)

				params, _ = curve_fit(model_func_signed,Xtr,Ytr,p0=p0,maxfev=20000)
				pred = model_func_signed(Xte, *params)[0]
			else:
				logX = np.column_stack([np.log10(v) for v in Xtr])
				logY = np.log10(Ytr)

				lin_model = LinearRegression().fit(logX, logY)
				p0 = np.append(lin_model.coef_,lin_model.intercept_)

				params, _ = curve_fit(model_func,Xtr,Ytr,p0=p0,maxfev=20000)
				pred = model_func(Xte, *params)[0]

			preds.append(pred)
			truths.append(Yte[0])

		except:
			continue

	return r2_score(truths, preds)
L_eta = 0.62 * Rm**(-1/2) + 0.014
Els_prime = Els/(Rm*L_eta)

#models = {"Els_prime": [Els_prime], "Ro_conv": [Ro_conv], "ELs": [Els], "Ro_sh": [Ro_sh], "Ro_conv_xi": [Ro_conv, xi], "Els_prime_Ro_conv": [Els_prime,Ro_conv], "Ro_conv_Els": [Ro_conv, Els], "Ro_conv_Ro_sh": [Ro_conv, Ro_sh], "Ro_conv_xi_Rosh": [Ro_conv, xi, Ro_sh], "Ro_conv_xi_Els": [Ro_conv, xi, Els],"Ro_conv_Els_Rosh": [Ro_conv, Els, Ro_sh], "Els_prime_Ro_conv_Ro_sh": [Els_prime, Ro_conv, Ro_sh], "Els_prime_Ro_conv_Ro_sh_xi": [Els_prime, Ro_conv, Ro_sh,xi]}
#models = {"Ro_conv": [Ro_conv],"Ro_conv_Els": [Ro_conv, Els],"Ro_conv_Rosh": [Ro_conv, Ro_sh],"Ro_conv_Els_Rosh": [Ro_conv, Els, Ro_sh],}
models = {"Els_prime_Ro_conv": [Els_prime,Ro_conv],"Ro_conv_Els_prime_Ro_sh": [Ro_conv, Els_prime, Ro_sh],"Els_prime_Ro_conv_Ro_sh_xi": [Els_prime, Ro_conv, Ro_sh,xi]}
for g_code in np.unique(g):

	mask_g = mask & (g == g_code)

	npts = np.sum(mask_g)
	print()
	print("====================================================")
	print(f"GRAVITY PROFILE g = {g_code}")
	print(f"N points = {npts}")
	print("====================================================")

	if g_code != 1 : 	#npts < 8:
		print("Too few points")
		continue

	plt.figure()
	plt.scatter(np.log10(Ro_conv[mask_g]),np.log10(Els[mask_g]),s=60)
	plt.xlabel("log10(Ro_conv)")
	plt.ylabel("log10(Els)")
	plt.title(f"Parameter coverage | g={g_code}")
	
	plt.figure()
	plt.scatter(np.log10(Ro_conv[mask_g]),np.log10(xi[mask_g]),s=60)
	plt.xlabel("log10(Ro_conv)")
	plt.ylabel("log10(xi)")
	plt.title(f"Parameter coverage | g={g_code}")
	
	plt.figure()
	plt.scatter(np.log10(Ro_conv[mask_g]),np.log10(Ro_sh[mask_g]),s=60)
	plt.xlabel("log10(Ro_conv)")
	plt.ylabel("log10(Ro_sh)")
	plt.title(f"Parameter coverage | g={g_code}")
	
	plt.figure()
	plt.scatter(np.log10(xi[mask_g]),np.log10(Ro_sh[mask_g]),s=60)
	plt.xlabel("log10(xi)")
	plt.ylabel("log10(Ro_sh)")
	plt.title(f"Parameter coverage | g={g_code}")
	
	plt.figure()
	plt.scatter(np.log10(Els[mask_g]),np.log10(Ro_sh[mask_g]),s=60)
	plt.xlabel("log10(Els)")
	plt.ylabel("log10(Ro_sh)")
	plt.title(f"Parameter coverage | g={g_code}")
	
	plt.figure()
	plt.scatter(np.log10(Els[mask_g]),np.log10(xi[mask_g]),s=60)
	plt.xlabel("log10(Els)")
	plt.ylabel("log10(xi)")
	plt.title(f"Parameter coverage | g={g_code}")
	plt.show()

	# ========================================================
	# FITS
	# ========================================================

	for model_name, variables in models.items():

		print()
		print("--------------------------------------------")
		print(model_name)
		print("--------------------------------------------")

		for MS, MS_err, case, sign in [(MS_rms, MS_rms_err, "MS_rms",False),(MS_int, MS_int_err, "MS_int",True),(MS_max, MS_max_err, "MS_max",True),(MS_mid, MS_mean_err, "MS_mean",True)]:
			print()
			print(f"===== {case} =====")
			
			#if case == "MS_min":
			#	mask_g &= mask_min
				
			vars_fit = [v[mask_g] for v in variables]
			res = evaluate_scaling_realspace(vars_fit,MS[mask_g],MS_err[mask_g],signed=sign)

			print("R2                 :", res["R2"])
			print("adj_R2             :", res["adj_R2"])
			print("coefs              :", res["coefs"])
			print("intercept          :", res["intercept"])
			print("condition_number   :", res["condition_number"])
			print("PCA_variance       :", res["PCA_variance"])
			print("correlation_matrix :")
			print(res["correlation_matrix"])
			print("LOO score:",loo_score(vars_fit,MS[mask_g],signed=sign))
			#print(res["Y_model"])
			
			if model_name == "Ro_conv_Els_prime_Ro_sh":
				A = res["intercept"]
				a,b,c = res["coefs"]
				plt.figure()
				plt.scatter(res["Y_model"],res["Y"], c= Ro_sh[mask_g],s=60, norm=LogNorm(vmin=Ro_sh[mask_g].min(), vmax=Ro_sh[mask_g].max()))
				plt.colorbar()
				xmin = min(res["Y_model"].min(), res["Y"].min())
				xmax = max(res["Y_model"].max(), res["Y"].max())
				x = np.linspace(xmin, xmax, 100)
				plt.plot(x, x, 'r--')
				plt.xlabel(rf"$ {A:.2f} \cdot Ro_{{conv}}^{{{b:.2f}}} \cdot \Lambda'^{{{a:.2f}}} $")
				if case == "MS_rms":
					plt.ylabel(r"$MS_{rms}$ from simulations")
					plt.title(r"Scale law of $MS_{rms}$ for $g \propto 1/r^2$")
				if case == "MS_int":
					plt.ylabel(r"$MS_{int}$ from simulations")
					plt.title(r"Scale law of $MS_{int}$ for $g \propto 1/r^2$")
				if case == "MS_max":
					plt.ylabel(r"$MS_{max}$ from simulations")
					plt.title(r"Scale law of $MS_{max}$ for $g \propto 1/r^2$")
				if case == "MS_mean":
					plt.ylabel(r"$MS_{mean}$ from simulations")
					plt.title(r"Scale law of $MS_{mean}$ for $g \propto 1/r^2$")	
				plt.grid()
				
				

plt.show()




