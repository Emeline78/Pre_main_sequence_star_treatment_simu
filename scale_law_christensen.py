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
git add scale_law_christensen.py
git commit -m "modifications"
git push
"""

a = "transport_profiles_adim.parquet"
df = pd.read_parquet(a)
datadir = "snapshots1/"

"""
mapping = {"gr_Nr2p5_Pm4_ra_1p6e7": 2.98,"gr_Nr2p5_Pm4_ra_8e6": 1.42,"gr2_xi_p2_pm4_ra_1e6": 1.78,"gr2_xi_p2_pm6_ra_1p5e6": 3.75,"gr2_xi_p1_pm4_ra_5e5": 2.07,"gr2_xi_p1_pm6_ra_5p5e6": 2.07,"gr2_xi_p35_pm4_ra_2e6": 2.53,"gr2_xi_p35_pm4_ra_5e6": 7.26,"gr2_xi_p35_pm6_ra_1p5e6": 3.75,"gr_gr2_Louis_ra_1p5e7": 5.78,"gr_gr2_Louis_ra_1e7": 4.14}

df["Nu"] = np.nan
for pattern, value in mapping.items():
    mask = df["name"].str.startswith(pattern)
    df.loc[mask, "Nu"] = value

df.to_parquet("transport_profiles_adim.parquet")

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

from magic import *

a = "/travail/dynconv/multiscale_dyno/anelasticCouette/gr2/xi_p35_pm4/ra_5e6/om50/"
ts = MagicTs(datadir = a,field='par', all=True, iplot = False)
Rm = np.mean(ts.rm)
Ro_conv = np.mean(ts.ro)
Ro_conv_l = np.mean(ts.rossby_l)

print(Rm, Ro_conv, Ro_conv_l)

"""

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

mask = (om < om_lim)  & (np.char.find(names, "wrong") == -1) & (df.groupby("name")["status"].first().to_numpy())

added_df = pd.read_csv('added_data.dat', sep='\s+', header=0)

Ra_added = added_df["Ra"].to_numpy()
Nu_added = added_df["Nu"].to_numpy()
E_added = added_df["E"].to_numpy()
Pr_added = added_df["Pr"].to_numpy()
Ro_added = added_df["Ro"].to_numpy()
Pm_added = added_df["Pm"].to_numpy()
Rosh_added = np.zeros(len(Pm_added))
g_added = np.ones(len(Pm_added))

Lo_fohm_added = added_df["Lo"].to_numpy()/(added_df["fohm"].to_numpy())**(1/2)
Ra_mod_added = Ra_added * (Nu_added - 1) * E_added**3 / Pr_added**2
Nu_mod_added = (Nu_added - 1) * E_added / Pr_added


Nu_mod = np.concatenate([Nu_mod[mask], Nu_mod_added])
Ra_mod = np.concatenate([Ra_mod[mask], Ra_mod_added])
Lo_fohm = np.concatenate([Lo_fohm[mask], Lo_fohm_added])
Ro = np.concatenate([Ro[mask], Ro_added])
Pm = np.concatenate([Pm[mask], Pm_added])
Ro_sh = np.concatenate([Ro_sh[mask], Rosh_added])
g = np.concatenate([g[mask], g_added])



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
	
def residuals_signed(params, X_stack, Y):
	Y_model = model_func_signed(X_stack, *params)
	return (Y - Y_model)

def evaluate_scaling_realspace(X_vars, Y, signed = True):
	if signed :
		mask_fit = np.ones_like(Y, dtype=bool)

		for v in X_vars:
			mask_fit &= (v > 0)

		mask_fit &= np.isfinite(Y)

		X_vars = [v[mask_fit] for v in X_vars]
		Y = Y[mask_fit]
		print(Y.shape)

		# ---------------- Initial guess from log fit ----------------

		logX = np.column_stack([np.log10(v) for v in X_vars])
		p0 = [0.5]*len(X_vars) + [np.mean(Y)]
		p0[-1] = np.mean(Y)
		
		bounds_lower = [-5]*len(X_vars) + [-np.inf]
		bounds_upper = [5]*len(X_vars) + [ np.inf]
		
		# ---------------- Real-space nonlinear fit ----------------

		X_stack = np.vstack(X_vars)

		params, cov = curve_fit(model_func_signed, X_stack, Y, absolute_sigma=True, bounds=(bounds_lower, bounds_upper), p0=p0, maxfev=20000)

		coefs = params[:-1]
		intercept = params[-1]
		Y_model = model_func_signed(X_stack, *params)
		
	else :
		mask_fit = np.ones_like(Y, dtype=bool)

		for v in X_vars:
			mask_fit &= (v > 0)

		mask_fit &= (Y > 0)
		mask_fit &= np.isfinite(Y)

		X_vars = [v[mask_fit] for v in X_vars]
		Y = Y[mask_fit]
		print(Y.shape)

		# ---------------- Initial guess from log fit ----------------

		logX = np.column_stack([np.log10(v) for v in X_vars])
		logY = np.log10(Y)

		lin_model = LinearRegression().fit(logX, logY)
		p0 = np.append(lin_model.coef_,lin_model.intercept_)

		# ---------------- Real-space nonlinear fit ----------------

		X_stack = np.vstack(X_vars)

		params, cov = curve_fit(model_func,X_stack,Y,absolute_sigma=True,p0=p0,maxfev=20000)

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

models = {"Ra_Q": [Ra_mod], "Ra_Q_Pm": [Ra_mod, Pm]}

for g_code in np.unique(g):

	mask_g = (g == g_code) #& mask
	mask_plot = np.concatenate([np.ones(len(Nu_mod[mask][mask_g]),dtype=bool), np.zeros(len(Nu_added),dtype=bool)])[mask_g]

	npts = np.sum(mask_g)
	print()
	print("====================================================")
	print(f"GRAVITY PROFILE g = {g_code}")
	print(f"N points = {npts}")
	print("====================================================")

	if g_code != 1 : 	#npts < 8:
		print("Too few points")
		continue


	# ========================================================
	# FITS
	# ========================================================

	for model_name, variables in models.items():

		print()
		print("--------------------------------------------")
		print(model_name)
		print("--------------------------------------------")

		for X, case, sign in [(Ro, "Ro",False),(Lo_fohm, "Lo_fohm",False),(Nu_mod, "Nu_mod",True)]:
			print()
			print(f"===== {case} =====")
				
			print(X[mask_g].shape)
			vars_fit = [v[mask_g] for v in variables]
			res = evaluate_scaling_realspace(vars_fit, X[mask_g], signed=sign)

			print("R2                 :", res["R2"])
			print("adj_R2             :", res["adj_R2"])
			print("coefs              :", res["coefs"])
			print("intercept          :", res["intercept"])
			print("condition_number   :", res["condition_number"])
			print("PCA_variance       :", res["PCA_variance"])
			print("correlation_matrix :")
			print(res["correlation_matrix"])
			print("LOO score:",loo_score(vars_fit,X[mask_g],signed=sign))
			
			if model_name == "Ra_Q" and case == "Lo_fohm":
				A = res["intercept"]
				a = res["coefs"][0]
				plt.figure()
				sc = plt.scatter(res["Y_model"][mask_plot],res["Y"][mask_plot],c = Ro_sh[mask_g][mask_plot],s=60, norm=LogNorm(vmin=Ro_sh[mask_g][mask_plot].min(), vmax=Ro_sh[mask_g][mask_plot].max()))
				xmin = min(res["Y_model"].min(), res["Y"].min())
				xmax = max(res["Y_model"].max(), res["Y"].max())
				x = np.linspace(xmin, xmax, 100)
				plt.plot(x, x, 'r--')
				plt.plot(res["Y_model"][~mask_plot],res["Y"][~mask_plot],"k*")
				plt.xlabel(rf"$ {A:.2f} \cdot Ra_{{Q}}^{{*{a:.2f}}} $")
				plt.ylabel(r"$\frac{Lo}{f_{ohm}^{1/2}}$ from simulations")
				plt.title(r"Scale law of $\frac{Lo}{f_{ohm}^{1/2}}$ for $g \propto 1/r^2$")
				plt.colorbar(sc)
				plt.grid()
			
			"""
			if len(variables) == 1:
				A = res["intercept"]
				a = res["coefs"][0]
				plt.figure()
				plt.scatter(res["Y_model"],res["Y"],s=60)
				plt.colorbar()
				xmin = min(res["Y_model"].min(), res["Y"].min())
				xmax = max(res["Y_model"].max(), res["Y"].max())
				x = np.linspace(xmin, xmax, 100)
				plt.plot(x, x, 'r--')
				plt.ylabel(f"{case} from simulations")
				plt.xlabel(f"{model_name}, "rf"$A={A:.2e},\ a={a:.2f}$")
				plt.title(f"Scale law of {case} for $g \propto 1/r^2$")
				plt.grid()
			
			if len(variables) == 2:
				A = res["intercept"]
				a,b = res["coefs"]
				plt.figure()
				plt.scatter(res["Y_model"],res["Y"],s=60)
				xmin = min(res["Y_model"].min(), res["Y"].min())
				xmax = max(res["Y_model"].max(), res["Y"].max())
				x = np.linspace(xmin, xmax, 100)
				plt.plot(x, x, 'r--')
				plt.colorbar()
				plt.ylabel(f"{case} from simulations")
				plt.xlabel(f"{model_name}, "rf"$A={A:.2e},\ a={a:.2f},\ b={b:.2f}$")
				plt.title(f"Scale law of {case} for $g \propto 1/r^2$")
				plt.grid()
			"""

plt.show()




