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

a = "/travail/dynconv/multiscale_dyno/anelasticCouette/gr/Nr2p5_Pm4/ra_8e6/om50/"
#a = "/travail/dynconv/multiscale_dyno/anelasticCouette/gr2/xi_p35_pm4/ra_5e6/om50/"
stp = MagicSetup(datadir = a)

if stp.nRotMa == 0 :	# rotation implicite dans les unites ( = 1 rotation explicite)
    om = 1
n = stp.polind
Pm = stp.prmag
ki = stp.radratio
Nrho = stp.strat 
Ek = stp.ek
g0 = stp.g0
g1 = stp.g1
g2 = stp.g2

ts = MagicTs(datadir = a, field='e_kin', all=True) 	# verification que le regime ne change pas dans le temps pour pouvoir faire l'integration en temps 
print(ts)

files = glob.glob(os.path.join(a,'G_[0-9]*.rot03'))
files.sort(key=lambda f: int(os.path.basename(f).split('_')[1].split('.')[0]))

times = []
RS_snap = []
MS_snap = []
Visc_snap = []
MC_snap = []
l_snap = []

for j in range(1,len(files)+1): 
    gr = MagicGraph(datadir=a,tag='rot03',ivar = j)
    times.append(gr.time)

    if j == 1:
        r = gr.radius
        th = np.linspace(0,np.pi,gr.ntheta)

        dphi = 2*np.pi/gr.nphi
        dtheta = np.pi/(gr.ntheta-1)
        
        w_theta = dtheta * np.sin(th)
        w_phi = dphi / (2* np.pi)

    # fluctuations
    vr = gr.vr - gr.vr.mean(axis=0)
    vp = gr.vphi - gr.vphi.mean(axis=0)

    Br = gr.Br - gr.Br.mean(axis=0)
    Bp = gr.Bphi - gr.Bphi.mean(axis=0)
    
    # def de tau
    dvphi = np.gradient(gr.vphi, r, axis=2)
    tau_rphi = dvphi - gr.vphi/r[None,None,:] 

    # Reynolds
    prodR = (vr * vp * w_phi).sum(axis=0)	# flux
    RS = (prodR * np.sin(th)[:,None] * w_theta[:,None]).sum(axis=0) * r # integrated flux over a spherical surface

    # Maxwell
    prodM = -(Br * Bp * w_phi).sum(axis=0)
    MS = (prodM * np.sin(th)[:,None] * w_theta[:,None]).sum(axis=0) * r  
    
    # Ecoulement meridional
    vr_mean = (gr.vr * w_phi).sum(axis=0)
    vphi_mean = (gr.vphi * w_phi).sum(axis=0)
    MC = (vr_mean * vphi_mean * np.sin(th)[:,None] * w_theta[:,None]).sum(axis = 0) * r
    
    # Viscosite
    mean_tau = (tau_rphi * w_phi).sum(axis = 0)
    Visc = - (mean_tau * np.sin(th)[:,None] * w_theta[:,None]).sum(axis=0) * r
    
    # moment angulaire
    l = (r[None,None,:]*np.sin(th)[:,None]*gr.vphi).mean(axis = (0,1))
    l_snap.append(l)
    
    Visc_snap.append(Visc)
    RS_snap.append(RS)
    MS_snap.append(MS)
    MC_snap.append(MC)


times = np.array(times)
print(times)
RS_snap = np.array(RS_snap)
MS_snap = np.array(MS_snap)
Visc_snap = np.array(Visc_snap)
MC_snap = np.array(MC_snap)
l_snap = np.array(l_snap)
t_total = times[-1] - times[0]

plt.figure()
for i,l in enumerate(l_snap):
     plt.plot(r,l,label = str(i))
plt.plot(r, np.mean(l_snap, axis = 0),"k", linewidth=3, label= "mean")  
plt.legend(loc = "lower left")

dt = np.diff(times)

RS = np.zeros_like(RS_snap[0])
MS = np.zeros_like(MS_snap[0])
MC = np.zeros_like(MC_snap[0])
Visc = np.zeros_like(Visc_snap[0])

for i in range(len(dt)):
    RS += 0.5*(RS_snap[i] + RS_snap[i+1])*dt[i]
    MS += 0.5*(MS_snap[i] + MS_snap[i+1])*dt[i]
    MC += 0.5*(MC_snap[i] + MC_snap[i+1])*dt[i]
    Visc += 0.5*(Visc_snap[i] + Visc_snap[i+1])*dt[i]

L = 1		# pas 1 - ki car r0 n'est pas egale a 1 mais a 1/(1-ki)
nu = Ek * om * L**2
tau = L**2/nu		# savoir quoi prendre entre temps visqueux (L**2/nu) ou de rotation (1/om)
eta = nu/Pm
temp, rho, drho = anelprof(r, strat = Nrho, polind = n, g0=g0, g1=g1, g2=g2)
rho0 = rho[0]
rho = rho / rho0  
B0car = eta * om * mu0 	* rho0

"""
print(B0car/mu0, rho[30]/ tau**2)
plt.figure() 
plt.subplot(2,1,1)
plt.plot(r,RS/t_total, label = "Reynolds stress")  
plt.plot(r,MC/t_total,label ="Meridional circulation") 
plt.plot(r,Visc/t_total, label = "Viscous stress") 
plt.legend() 
plt.ylabel("Stresses")
plt.subplot(2,1,2)
plt.plot(r,MS/t_total, label ="Maxwell stress")
plt.xlabel("r") 
plt.ylabel("Stresses") 
plt.legend() 
plt.show()
"""
#print(f"rho(ri)/rho(ro) = {rho.max()/rho.min():.4f}")
#print(f"attendu         = {np.exp(Nrho):.4f}")

RS = RS / t_total * rho * L**3 / tau**2 * 2 * np.pi * r**2
MS = MS / t_total * L * B0car / mu0 * 2 * np.pi * r**2
Visc = Visc / t_total * rho * L**3 / tau**2 * 2 * np.pi * r**2
MC = MC / t_total * rho * L**3 / tau**2 * 2 * np.pi * r**2

plt.figure()
plt.plot(r,RS, label = "Reynolds stress")  
plt.plot(r,MC,label ="Meridional circulation") 
plt.plot(r,Visc, label = "Viscous stress") 
plt.plot(r,MS, label ="Maxwell stress")
plt.xlabel("r") 
plt.ylabel("Stresses") 
plt.legend() 
plt.show()

F = (MC + MS + RS + Visc) 
plt.figure()
plt.plot(r,F)
plt.xlabel("r")
plt.ylabel("Radial flux of angular momentum")
plt.show()

print("Relative variation:", (F.max() - F.min()) / np.mean(F))

