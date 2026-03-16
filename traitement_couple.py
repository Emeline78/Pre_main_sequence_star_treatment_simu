from math import *
import numpy as np
from scipy import *
from matplotlib.pyplot import *
import glob
import os
matplotlib.interactive(True)
from magic.libmagic import anelprof
from matplotlib.ticker import FuncFormatter
formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))

matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
rcParams['font.size'] = 24
rcParams['ytick.labelsize']=20
rcParams['ytick.direction']='in'
rcParams['ytick.right']='True'
rcParams['xtick.labelsize']=20
rcParams['xtick.direction']='in'
rcParams['xtick.top']='True'
rcParams['figure.figsize']= [8, 6]

from magic import *

a = "gr/Nr2p5_Pm4/ra_8e6/om50"

# ------ Useful values -------

g1 = MagicGraph(datadir=a,tag='rot01',ivar=1)
g2 = MagicGraph(datadir=a,tag='rot01',ivar=2)
dt = g2.time - g1.time
print(dt)

g_last  = MagicGraph(datadir=a,tag='rot01')
t_total = g_last.time - g1.time
print(t_total)

files = glob.glob(os.path.join(a, 'G_[0-9]*.rot01'))
prod_tot = np.zeros((len(files),g1.nr))

for j in range(1,len(files)+1):
	gr = MagicGraph(datadir=a,tag='rot01',ivar = j)
	r = gr.radius
	thlin = np.linspace(0., np.pi, gr.ntheta)
	dthet = np.pi / (gr.ntheta - 1) * np.sin(thlin) 
	dphi   = 2 * np.pi / gr.nphi
	vr = gr.vr.copy()- gr.vr.mean(axis=0)
	vp = gr.vphi.copy()- gr.vphi.mean(axis=0) 
	prod = vr*vp
	int_phi = (prod*dphi).sum(axis=0)
	int_theta =  (int_phi*dthet[:, np.newaxis]).sum(axis = 0)
	prod_tot[j-1] = int_theta/(4*np.pi)

RS = (prod_tot*dt).sum(axis=0) / t_total

temp0, rho0, beta = anelprof(gr.radius,strat=gr.strat, polind=gr.polind,g0=gr.g0, g1=gr.g1, g2=gr.g2)

RS = rho0 * RS


