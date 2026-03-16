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

a = "/travail/dynconv/multiscale_dyno/anelasticCouette/gr/Nr2p5_Pm4/ra_8e6/om50/"

# ------ Reynolds stress -------
g1 = MagicGraph(datadir=a,tag='rot01',ivar = 1)

files = glob.glob(os.path.join(a, 'G_[0-9]*.rot01'))
prod_tot = np.zeros((len(files),g1.nr))
times = []
print(files)
for j in range(1,len(files)+1):
	gr = MagicGraph(datadir=a,tag='rot01',ivar = j)
	times.append(gr.time)
	r = gr.radius
	thlin = np.linspace(0., np.pi, gr.ntheta)
	dthet = np.pi / (gr.ntheta - 1) * np.sin(thlin) 
	dphi   = 2 * np.pi / gr.nphi
	vr = gr.vr.copy()- gr.vr.mean(axis=0)
	vp = gr.vphi.copy()- gr.vphi.mean(axis=0) 
	
	prod = vr*vp
	int_phi = (prod*dphi).sum(axis=0)
	int_theta =  (int_phi*dthet[:, np.newaxis]).sum(axis = 0)
	prod_tot[j-1] = int_theta/(4*np.pi) *r

times = np.array(times)
dt = np.diff(times)
print(prod_tot.shape,times.shape)

t_total = times[-1] - times[0]

RS = (prod_tot*dt[:, np.newaxis]).sum(axis=0) / t_total
print(RS.shape)
plt.figure()
plt.plot(RS)
plt.show()

#temp0, rho0, beta = anelprof(gr.radius,strat=gr.strat, polind=gr.polind,g0=gr.g0, g1=gr.g1, g2=gr.g2)

#RS = rho0 * RS

prod_tot = np.zeros((len(files),g1.nr))
for j in range(1,len(files)+1):
	gr = MagicGraph(datadir=a,tag='rot01',ivar = j)
	r = gr.radius
	thlin = np.linspace(0., np.pi, gr.ntheta)
	dthet = np.pi / (gr.ntheta - 1) * np.sin(thlin) 
	dphi   = 2 * np.pi / gr.nphi
	Br = gr.Br.copy()- gr.Br.mean(axis=0)
	Bp = gr.Bphi.copy()- gr.Bphi.mean(axis=0) 
	prod = Br*Bp
	int_phi = (prod*dphi).sum(axis=0)
	int_theta =  (int_phi*dthet[:, np.newaxis]).sum(axis = 0)
	prod_tot[j-1] = int_theta/(4*np.pi) *r

MS = (prod_tot*dt).sum(axis=0) / t_total * (-1/(4* np.pi * 1e-7))
print(MS.shape)
plt.figure()
plt.plot(MS)
plt.show()



