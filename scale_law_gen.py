import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

"""
git add scale_law_gen.py
git commit -m "modifications"
git push
"""

a = "transport_profiles_adim.parquet"
df = pd.read_parquet(a)
datadir = "snapshots1/"

if a == "transport_profiles_SI.parquet" or a == "transport_profiles_CGS.parquet" :
	df = df[(df["date"] > 5) & (df["date"] < 6)]
	

#MS_mean = (df.groupby("name")["MS"].mean()).to_numpy()
MS_rms = df.groupby("name").apply(lambda g: np.sqrt(np.mean(g["MS"]**2))).to_numpy()
MS_int = df.groupby("name").apply(lambda g: np.trapz(g["MS"], g["r"])).to_numpy()
MS_max = df.groupby("name")["MS"].apply(lambda x: x.iloc[x.abs().argmax()]).to_numpy()


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
	
	x = MS_snap[np.arange(len(MS_snap)), np.abs(MS_snap).argmax(axis=1)]
	MS_max_err[i] = np.std(x) / np.sqrt(len(x))
	
MS_int_sign = np.sign(MS_int)
MS_int_amp  = np.abs(MS_int)

def evaluate_scaling(X_vars, Y, Yerr, n_boot=100):
	"""
	X_vars : liste de tableaux
	Y : tableau 
	"""

	mask = np.ones_like(Y, dtype=bool)
	for v in X_vars:
		mask &= (v > 0)
	mask &= (Y > 0) & (Yerr > 0)

	X_vars = [v[mask] for v in X_vars]
	Y = Y[mask]
	Yerr = Yerr[mask]

	logX = np.column_stack([np.log10(v) for v in X_vars])
	logY = np.log10(Y)
	logY_err = Yerr / (Y * np.log(10))
	
	print(logY.shape)
	print(logY_err.shape)
	# ===================== R2 et Regression lineaire =======================
	def residuals(params, X, Y, Yerr):
		a = params[:-1]
		b = params[-1]
		model = X @ a + b
		return (Y - model) / Yerr

	res = least_squares(residuals, x0=np.zeros(logX.shape[1] + 1), args=(logX, logY, logY_err), loss='soft_l1')
	coefs = res.x[:-1]
	intercept = res.x[-1]
	model_logY = logX @ coefs + intercept
	weights = 1 / (logY_err**2)

	def weighted_R2(y, y_model, weights):
		y_mean = np.average(y, weights=weights)
		ss_res = np.sum(weights * (y - y_model)**2)
		ss_tot = np.sum(weights * (y - y_mean)**2)
		return 1 - ss_res / ss_tot

	R2 = weighted_R2(logY, model_logY, weights)

	# ===================== Stabilite =======================
	boot_coefs = []
	for i in range(n_boot):
		idx = (np.random.rand(len(logY)) < 0.6)
		Xb = logX[idx]
		Yb = logY[idx]

		m = LinearRegression().fit(Xb, Yb)
		boot_coefs.append(m.coef_)
		

	boot_coefs = np.array(boot_coefs)
	std_coefs = np.std(boot_coefs, axis=0)

	stable = []
	for a, s in zip(coefs, std_coefs):
		if np.abs(a) > 0:
			stable.append(s < 0.2 * np.abs(a))
		else:
			stable.append(False)

	n_stable = sum(stable)

	# ========================== PCA ===============================
	pca = PCA().fit(logX)
	var_ratio = pca.explained_variance_ratio_

	# dimension effective
	if var_ratio[0] > 0.9:
		dim = 1
	elif var_ratio[0] + var_ratio[1] > 0.9:
		dim = 2
	else:
		dim = len(coefs)

	# ============================ Correlations =============================
	corr = np.corrcoef(logX)

	return {"R2": R2,"coefs": coefs,"coef_std": std_coefs, "n_stable": n_stable, "PCA_variance": var_ratio,"dim": dim,"correlation_matrix": corr}
    
    
models = {"Ro": [Ro_conv], "Ro_xi": [Ro_conv, xi], "Ro_xi_Rosh": [Ro_conv, xi, Ro_sh]}
for MS,MS_err in [(MS_rms,MS_rms_err), (MS_int_amp,MS_int_err), (MS_max,MS_max_err)]:
	for name, var in models.items():
		res = evaluate_scaling(var, MS, MS_err)
		print(name, res.items)

