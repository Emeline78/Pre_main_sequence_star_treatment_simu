from math import *
import numpy as np
from scipy import *
import matplotlib.pyplot as plt
import glob
import os
import matplotlib
matplotlib.interactive(True)
from magic.libmagic import anelprof
from matplotlib.ticker import ScalarFormatter
from statsmodels.tsa.stattools import adfuller
from scipy import stats

formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((0, 0))


"""
git add traitement_couple.py
git commit -m "modifications"
git push
"""

matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 24
matplotlib.rcParams['ytick.labelsize']=20
matplotlib.rcParams['ytick.direction']='in'
matplotlib.rcParams['ytick.right']='True'
matplotlib.rcParams['xtick.labelsize']=20
matplotlib.rcParams['xtick.direction']='in'
matplotlib.rcParams['xtick.top']='True'
matplotlib.rcParams['figure.figsize']= [8, 6]

from magic import *
mu0 = 4*np.pi*1e-7

a = input("directory : ")
#a = "/travail/dynconv/multiscale_dyno/anelasticCouette/gr/Nr2p5_Pm4/ra_8e6/om50/"
#a = "/travail/dynconv/multiscale_dyno/anelasticCouette/gr2/xi_p35_pm4/ra_5e6/om50/"
#a = "/travail/dynconv/multiscale_dyno/anelasticCouette/gr_gr2_Louis/ra_1p5e7"
#a = "/travail/dynconv/multiscale_dyno/anelasticCouette/gr2/xi_p2_pm4/ra_1e6"
stp = MagicSetup(datadir = a)

n = stp.polind
Pm = stp.prmag
ki = stp.radratio
Nrho = stp.strat 
Ek = stp.ek
g0 = stp.g0
g1 = stp.g1
g2 = stp.g2
om = 1/Ek


ts = MagicTs(datadir = a, field='e_kin', all=True) 	# verification que le regime ne change pas dans le temps pour pouvoir faire l'integration en temps 
CV = np.std(ts.ekin_pol) / np.mean(ts.ekin_pol)
p_adf = adfuller(ts.ekin_pol)[1]

if CV < 0.2 and p_adf < 0.05:
    print("ekin_pol stationnaire, hypothèse constante valide")
else:
    print(f"Attention : CV={CV:.1%}, ADF p={p_adf:.3f}")

files = glob.glob(os.path.join(a,'G_[0-9]*.rot01'))
files.sort(key=lambda f: int(os.path.basename(f).split('_')[1].split('.')[0]))

times = []
RS_snap = []
MS_snap = []
Visc_snap = []
MC_snap = []
MS1_snap = []
l_snap = []

# pour les moyennes sur phi j'aurais juste pu faire mean vu que c'est espace regulierement
for j in range(1,len(files)+1): 
    gr = MagicGraph(datadir=a,tag='rot01',ivar = j)
    times.append(gr.time)

    if j == 1:
        r = gr.radius
        th = np.linspace(0,np.pi,gr.ntheta)
        phi = np.linspace(0,2*np.pi,gr.nphi-1)

        dphi = 2*np.pi/(gr.nphi-2)
        dtheta = np.pi/(gr.ntheta-1)
        
        w_theta = dtheta * np.sin(th)
        w_phi = dphi / (2* np.pi)

    # fluctuations
    vr = gr.vr - (gr.vr * w_phi).sum(axis=0)
    vp = gr.vphi - (gr.vphi * w_phi).sum(axis=0)

    Br = gr.Br - (gr.Br * w_phi).sum(axis=0)
    Bp = gr.Bphi - (gr.Bphi * w_phi).sum(axis=0)
    
    # def de tau
    dvphi = np.gradient(gr.vphi, r, axis=2)
    tau_rphi = dvphi - gr.vphi/r[None,None,:] 

    # Reynolds
    prodR = (vr * vp * w_phi).sum(axis=0)	# flux
    RS = (prodR * np.sin(th)[:,None] * w_theta[:,None]).sum(axis=0) * r * 2 * np.pi * r**2 # integrated flux over a spherical surface

    # Maxwell
    prodM = -(Br * Bp * w_phi).sum(axis=0)
    #prodM = -(gr.Br * gr.Bphi * w_phi).sum(axis=0)
    MS = (prodM * np.sin(th)[:,None] * w_theta[:,None]).sum(axis=0) * r  * 2 * np.pi * r**2
    
    # Moy champ mag
    Br_mean = (gr.Br * w_phi).sum(axis=0)
    Bphi_mean = (gr.Bphi * w_phi).sum(axis=0)
    MS1 = -(Br_mean * Bphi_mean * np.sin(th)[:,None] * w_theta[:,None]).sum(axis = 0) * r* 2 * np.pi * r**2
    
    # Ecoulement meridional
    vr_mean = (gr.vr * w_phi).sum(axis=0)
    vphi_mean = (gr.vphi * w_phi).sum(axis=0)
    MC = (vr_mean * (vphi_mean + r[None,:] * np.sin(th)[:,None] * 1/Ek) * np.sin(th)[:,None] * w_theta[:,None]).sum(axis = 0) * r* 2 * np.pi * r**2
    
    # Viscosite
    mean_tau = (tau_rphi * w_phi).sum(axis = 0)
    Visc = - (mean_tau * np.sin(th)[:,None] * w_theta[:,None]).sum(axis=0) * r * 2 * np.pi * r**2
    #Visc = - (r[None,:]**2 * np.sin(th)[:,None] * np.gradient(vphi_mean/r[None,:],r,axis =1)* w_theta[:,None]).sum(axis=0)
    
    # moment angulaire
    l = (r[None,None,:]*np.sin(th)[:,None]**2*gr.vphi).mean(axis = (0,1)) * 2 * np.pi * r**2
    l_snap.append(l)
    
    Visc_snap.append(Visc)
    RS_snap.append(RS)
    MS_snap.append(MS)
    MS1_snap.append(MS1)
    MC_snap.append(MC)

times = np.array(times)

RS_snap = np.array(RS_snap)
MS_snap = np.array(MS_snap)
MS1_snap = np.array(MS1_snap)
Visc_snap = np.array(Visc_snap)
MC_snap = np.array(MC_snap)
l_snap = np.array(l_snap)

t_total = times[-1] - times[0]

"""
l_snap = np.array(l_snap)
plt.figure()
for i,l in enumerate(l_snap):
     plt.plot(r,l,label = str(i))
plt.plot(r, np.mean(l_snap, axis = 0),"k", linewidth=3, label= "mean")  
plt.legend(loc = "lower left")
"""
dt = np.diff(times)

RS = np.zeros_like(RS_snap[0])
MS = np.zeros_like(MS_snap[0])
MC = np.zeros_like(MC_snap[0])
Visc = np.zeros_like(Visc_snap[0])
MS1 = np.zeros_like(MS1_snap[0])
l = np.zeros_like(l_snap[0])
for i in range(len(dt)):
    RS += 0.5*(RS_snap[i] + RS_snap[i+1])*dt[i]
    MS += 0.5*(MS_snap[i] + MS_snap[i+1])*dt[i]
    MS1 += 0.5*(MS1_snap[i] + MS1_snap[i+1])*dt[i]
    MC += 0.5*(MC_snap[i] + MC_snap[i+1])*dt[i]
    Visc += 0.5*(Visc_snap[i] + Visc_snap[i+1])*dt[i]
    l += 0.5*(l_snap[i] + l_snap[i+1])*dt[i]

L = 1		# pas 1 - ki car r0 n'est pas egale a 1 mais a 1/(1-ki)
nu = Ek * om * L**2
tau = L**2/nu		# savoir quoi prendre entre temps visqueux (L**2/nu) ou de rotation (1/om)
eta = nu/Pm
temp, rho, drho = anelprof(r, strat = Nrho, polind = n, g0=g0, g1=g1, g2=g2)
rho0 = rho[0]
rho = rho / rho0  
B0car = eta * om * mu0 	* rho0


#print(f"rho(ri)/rho(ro) = {rho.max()/rho.min():.4f}")
#print(f"attendu         = {np.exp(Nrho):.4f}")

RS = RS / t_total * rho * L**3 / tau**2 
MS = MS / t_total * L * B0car / mu0 
MS1 = MS1 / t_total * L * B0car / mu0 
Visc = Visc / t_total * rho * L**3 / tau**2 
MC = MC / t_total * rho * L**3 / tau**2 

F = (MC + MS + RS + Visc + MS1)
plt.figure()
plt.plot(r,RS, label = "Reynolds stress")  
plt.plot(r,MC,label ="Meridional circulation with Coriolis part") 
plt.plot(r,Visc, label = "Viscous stress") 
plt.plot(r, MS, label ="Maxwell stress")
plt.plot(r,MS1, label ="Contribution from mean of magnetic field")
plt.plot(r,F,"k", linewidth=3,label = "total flux")
#plt.plot(r,np.mean(np.gradient(l_snap,times, axis = 0),axis = 0)* L**2 / tau**2 * rho, label ="rho dl/dt")
plt.xlabel("r") 
plt.ylabel("Stresses") 
plt.legend() 
plt.show()

"""
plt.figure()
plt.plot(r,F)
plt.plot(r,r**2*F)
plt.xlabel("r")
plt.ylabel("Radial flux of angular momentum")
plt.show()
"""

mask = (r > r[-5])  (r < r[5])
dFdr = np.gradient(F,r)
print(dFdr,dFdr[mask])
print(np.max(np.abs(dFdr[mask])))

"""  
# Calculer F pour chaque snapshot individuellement
F_snaps = RS_snap * rho * L**3 / tau**2  + MS_snap * L * B0car / mu0 + MS1_snap * L * B0car / mu0 + Visc_snap * rho * L**3 / tau**2  + MC_snap * rho * L**3 / tau**2 

# Moyenne et écart-type sur les snapshots
F_mean = np.mean(F_snaps, axis=0)
F_std  = np.std(F_snaps, axis=0)
F_sem  = F_std / np.sqrt(len(F_snaps))  # erreur sur la moyenne

ratio = np.abs(F_mean - np.mean(F_mean)) / F_sem
print(f"Max |F - <F>| / SEM = {ratio.max():.2f}")

mask = ratio > 2
print(f"r min = {r[mask].min():.3f}, r max = {r[mask].max():.3f}")
print(f"Nombre de points : {mask.sum()} / {len(r)}")

plt.figure()
plt.plot(r, ratio)
plt.axhline(2, color='r', linestyle='--', label='seuil 2σ')
plt.xlabel('r')
plt.ylabel('|F - <F>| / SEM')
plt.legend()
plt.show()

# Si < 2, F "constant"

mask_bulk = r > 0.42  # exclure la région problématique
F_bulk    = F_mean[mask_bulk]
F_sem_bulk = F_sem[mask_bulk]
ratio_bulk = np.abs(F_bulk - np.mean(F_bulk)) / F_sem_bulk
print(f"Max ratio bulk : {ratio_bulk.max():.2f}")
"""

