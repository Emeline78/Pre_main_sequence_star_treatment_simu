import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from magic import *
import pandas as pd
import numpy as np
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

"""
git add scale_law_christensen.py
git commit -m "modifications"
git push
"""

a = "transport_profiles_adim.parquet"
df = pd.read_parquet(a)
datadir = "snapshots1/"

mapping = {
	"gr_Nr2p5_Pm4_ra_1p6e7": 2.98,
	"gr_Nr2p5_Pm4_ra_8e6": 1.42,
	"gr2_xi_p2_pm4_ra_1e6": 1.78,
	"gr2_xi_p2_pm6_ra_1p5e6": 3.75,
	"gr2_xi_p1_pm4_ra_5e5": 2.07,
	"gr2_xi_p1_pm6_ra_5p5e6": 2.07,
	"gr2_xi_p35_pm4_ra_2e6": 2.53,
	"gr2_xi_p35_pm4_ra_5e6": 7.26,
	"gr2_xi_p35_pm6_ra_1p5e6": 3.75,
	"gr_gr2_Louis_ra_1p5e7": 5.78,
	"gr_gr2_Louis_ra_1e7": 4.14
}

df["Nu"] = np.nan

for pattern, value in mapping.items():
	mask = df["name"].str.startswith(pattern)
	df.loc[mask, "Nu"] = value
	
mapping = {
	"gr_Nr2p5_Pm4_ra_1p6e7": 4,
	"gr_Nr2p5_Pm4_ra_8e6": 4,
	"gr2_xi_p2_pm4_ra_1e6": 4,
	"gr2_xi_p2_pm6_ra_1p5e6": 6,
	"gr2_xi_p1_pm4_ra_5e5": 4,
	"gr2_xi_p1_pm6_ra_5p5e6": 6,
	"gr2_xi_p35_pm4_ra_2e6": 4,
	"gr2_xi_p35_pm4_ra_5e6": 4,
	"gr2_xi_p35_pm6_ra_1p5e6": 6,
	"gr_gr2_Louis_ra_1p5e7": 4,
	"gr_gr2_Louis_ra_1e7": 4
}

df["Pm"] = np.nan

for pattern, value in mapping.items():
	mask = df["name"].str.startswith(pattern)
	df.loc[mask, "Pm"] = value


names = df.groupby("name").mean().index.to_numpy(dtype = str)
Ra = (df.groupby("name")["ra"].first()).to_numpy()
g = (df.groupby("name")["config_code"].first()).to_numpy()
om = (df.groupby("name")["om"].first()).to_numpy()
om_lim = (df.groupby("name")["om_lim"].first()).to_numpy()
Els = (df.groupby("name")["Elsasser"].first()).to_numpy()
Ro_conv = (df.groupby("name")["Ro_conv"].first()).to_numpy()
Rm = (df.groupby("name")["rm"].first()).to_numpy()
Nu = (df.groupby("name")["Nu"].first()).to_numpy()
xi = (df.groupby("name")["xi"].first()).to_numpy()
Pm = (df.groupby("name")["Pm"].first()).to_numpy()

E = 1e-4
Ro_sh = om * E
Pr = 1
Nu_mod = (Nu - 1) * E / Pr
Ra_mod = Ra * (Nu - 1) * E**3 / Pr**2

mask = (om < om_lim)  & (np.char.find(names, "wrong") == -1) & (df.groupby("name")["status"].first().to_numpy())

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

def evaluate_scaling_realspace(X_vars, Y, signed = True):
	if signed :
		mask_fit = np.ones_like(Y, dtype=bool)

		for v in X_vars:
			mask_fit &= (v > 0)

		mask_fit &= np.isfinite(Y)
		mask_fit &= np.isfinite(Yerr)

		X_vars = [v[mask_fit] for v in X_vars]
		Y = Y[mask_fit]

		# ---------------- Initial guess from log fit ----------------

		logX = np.column_stack([np.log10(v) for v in X_vars])
		p0 = [0.5]*len(X_vars) + [np.mean(Y)]
		p0[-1] = np.mean(Y)
		
		bounds_lower = [-5]*len(X_vars) + [-np.inf]
		bounds_upper = [5]*len(X_vars) + [ np.inf]
		
		# ---------------- Real-space nonlinear fit ----------------

		X_stack = np.vstack(X_vars)

		params, cov = curve_fit(model_func_signed, X_stack, Y, absolute_sigma=True, bounds=(bounds_lower, bounds_upper), p0=p0, maxfev=20000)
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


	# ========================================================
	# FITS
	# ========================================================

	for model_name, variables in models.items():

		print()
		print("--------------------------------------------")
		print(model_name)
		print("--------------------------------------------")

		for X, case, sign in [(Ro_conv, "Ro_conv",False),(Nu_mod, "Nu_mod",True)]:
			print()
			print(f"===== {case} =====")
				
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
			print("LOO score:",loo_score(vars_fit,MS[mask_g],signed=sign))
			
			if len(variables) == 1:
				A = res["intercept"]
				a = res["coefs"]
				plt.figure()
				plt.scatter(res["Y_model"],res["Y"],s=60)
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
				plt.ylabel(f"{case} from simulations")
				plt.xlabel(f"{model_name}, "rf"$A={A:.2e},\ a={a:.2f},\ b={b:.2f}$")
				plt.title(f"Scale law of {case} for $g \propto 1/r^2$")
				plt.grid()

plt.show()





