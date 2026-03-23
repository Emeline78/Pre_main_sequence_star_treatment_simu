from math import *
import numpy as np
from scipy import *
import matplotlib.pyplot as plt
import glob
import os
import matplotlib
matplotlib.interactive(True)
from magic.libmagic import anelprof
from matplotlib.ticker import FuncFormatter
formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))

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
stp = MagicSetup(datadir = a)

if stp.nRotMa == 0 :
    om = 1
n = stp.polind
Pm = stp.prmag
ki = stp.radratio
Nrho = stp.strat 
Ek = stp.ek
g0 = stp.g0
g1 = stp.g1
g2 = stp.g2

ts = MagicTs(datadir = a, field='e_kin', all=True)

files = glob.glob(os.path.join(a,'G_[0-9]*.rot01'))
files.sort(key=lambda f: int(os.path.basename(f).split('_')[1].split('.')[0]))

times = []
RS_snap = []
MS_snap = []
Visc_snap = []

for j in range(1,len(files)+1): 
    gr = MagicGraph(datadir=a,tag='rot01',ivar = j)
    times.append(gr.time)

    if j == 1:
        r = gr.radius
        th = np.linspace(0,np.pi,gr.ntheta)

        dphi = 2*np.pi/gr.nphi
        dtheta = np.pi/(gr.ntheta-1)

        weight = np.sin(th)*dtheta*dphi/(4*np.pi)
        ur_snap = np.zeros((len(files),gr.ntheta,gr.nr))
        l_snap = np.zeros((len(files),gr.ntheta,gr.nr))

    # fluctuations
    vr = gr.vr - gr.vr.mean(axis=0)
    vp = gr.vphi - gr.vphi.mean(axis=0)

    Br = gr.Br - gr.Br.mean(axis=0)
    Bp = gr.Bphi - gr.Bphi.mean(axis=0)
    
    dvphi_droite = np.ones_like(vr)*np.nan
    dvphi_droite[:,:,:-1] = (gr.vphi[:,:,1:] - gr.vphi[:,:,:-1])/((r[1:] - r[:-1])[None,None,:])
    tau_rphi = dvphi_droite - gr.vphi/r[None,None,:]

    # Reynolds
    prodR = vr*vp
    RS = (prodR* np.sin(th)[:,None]*weight[:,None]).sum(axis=(0,1))*r

    # Maxwell
    prodM = Br*Bp
    MS = -(prodM* np.sin(th)[:,None]*weight[:,None]).sum(axis=(0,1))*r  
    
    # Ecoulement meridional
    ur_snap[j-1] = (gr.vr*dphi).sum(axis=0)/(2*np.pi)
    l_snap[j-1] = (gr.vphi*r[None,None,:]*np.sin(th)[None,:,None]*dphi).sum(axis=0)/(2*np.pi)
    
    # Viscosite
    Visc = -((tau_rphi * np.sin(th)[:,None] * weight[:,None]).sum(axis=(0,1))) * r
    
    Visc_snap.append(Visc)
    RS_snap.append(RS)
    MS_snap.append(MS)

times = np.array(times)
RS_snap = np.array(RS_snap)
MS_snap = np.array(MS_snap)
Visc_snap = np.array(Visc_snap)
t_total = times[-1] - times[0]

dt = np.diff(times)

RS = np.zeros_like(RS_snap[0])
MS = np.zeros_like(MS_snap[0])
ur = np.zeros_like(ur_snap[0])
l = np.zeros_like(l_snap[0])
Visc = np.zeros_like(Visc_snap[0])

for i in range(len(dt)):
    RS += 0.5*(RS_snap[i] + RS_snap[i+1])*dt[i]
    MS += 0.5*(MS_snap[i] + MS_snap[i+1])*dt[i]
    ur += 0.5*(ur_snap[i] + ur_snap[i+1])*dt[i]
    l += 0.5*(l_snap[i] + l_snap[i+1])*dt[i]
    Visc += 0.5*(Visc_snap[i] + Visc_snap[i+1])*dt[i]

L = 1 - ki
nu = Ek * om * L**2
tau = L**2/nu
eta = nu/Pm
temp0, rho0, drho = anelprof(r, strat = Nrho, polind = n, g0=g0, g1=g1, g2=g2)
rho = rho0
B0car = rho * mu0 * eta * om

#print(f"rho(ri)/rho(ro) = {rho.max()/rho.min():.4f}")
#print(f"attendu         = {np.exp(Nrho):.4f}")

RS = RS / t_total * rho * L**3 / tau**2
MS = MS / t_total * L * B0car / mu0
ur /= t_total
l /= t_total
Visc = Visc / t_total * rho * L**3 / tau**2
    
MC = (ur*l*dtheta*np.sin(th)[:,None]).sum(axis =0)/2 * rho * L**3 / tau**2

plt.figure()
plt.plot(r,RS, label = "Reynolds stress")
plt.plot(r,MS, label ="Maxwell stress")
plt.plot(r,MC,label ="Meridional circulation")
plt.plot(r,Visc, label = "Viscous stress")
plt.xlabel("r")
plt.ylabel("Stresses")
plt.legend()
plt.show()

F = MC + MS + RS + Visc
plt.figure()
plt.plot(r,F)
plt.xlabel("r")
plt.ylabel("Radial flux of angular momentum")
plt.show()


