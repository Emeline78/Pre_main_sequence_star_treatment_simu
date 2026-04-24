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

"""
git add mag_field.py
git commit -m "modifications"
git push
"""

a = input("directory : ")
#a = "/travail/dynconv/multiscale_dyno/anelasticCouette/gr/Nr2p5_Pm4/ra_8e6/om50/"
#a = "/travail/dynconv/multiscale_dyno/anelasticCouette/gr/Nr2p5_Pm6/ra_8e6/om50/"
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

files = glob.glob(os.path.join(a,'G_[0-9]*.rot01'))
files.sort(key=lambda f: int(os.path.basename(f).split('_')[1].split('.')[0]))

times = []
B = []

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
	
    idx = 0
    Br = gr.Br[:,:,idx]
    Bp = gr.Bphi[:,:,idx]
    Bth = gr.Btheta[:,:,idx]
    
    
    B_mean = np.sqrt(Br**2 + Bp**2 + Bth**2)
    B_snap = (B_mean * w_phi * w_theta[None,:]).sum(axis=(0,1)) * 1/2 * r[0]**2
    B.append(B_snap)

times = np.array(times)

L = 1		# pas 1 - ki car r0 n'est pas egale a 1 mais a 1/(1-ki)
nu = Ek * om * L**2
tau = L**2/nu		# savoir quoi prendre entre temps visqueux (L**2/nu) ou de rotation (1/om)
eta = nu/Pm
temp, rho, drho = anelprof(r, strat = Nrho, polind = n, g0=g0, g1=g1, g2=g2)
rho0 = rho[0]
rho = rho / rho0  
B0car = eta * om * mu0 	* rho0


B = np.array(B) * np.sqrt(B0car)
t_total = times[-1] - times[0]
dt = np.diff(times)

B_tot = np.zeros_like(B[0])
for i in range(len(dt)):
    B_tot += 0.5*(B[i] + B[i+1])*dt[i]

B_tot = B_tot / t_total 
print(B_tot)

