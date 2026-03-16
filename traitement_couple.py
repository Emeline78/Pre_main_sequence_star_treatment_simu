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

files = glob.glob(os.path.join(a,'G_[0-9]*.rot01'))

times = []
RS_snap = []
MS_snap = []

for j in range(1,len(files)+1): 
    gr = MagicGraph(datadir=a,tag='rot01',ivar = j)
    times.append(gr.time)

    if j == 1:
        r = gr.radius
        th = np.linspace(0,np.pi,gr.ntheta)

        dphi = 2*np.pi/gr.nphi
        dtheta = np.pi/(gr.ntheta-1)

        weight = np.sin(th)*dtheta*dphi/(4*np.pi)

    # fluctuations
    vr = gr.vr - gr.vr.mean(axis=0)
    vp = gr.vphi - gr.vphi.mean(axis=0)

    Br = gr.Br - gr.Br.mean(axis=0)
    Bp = gr.Bphi - gr.Bphi.mean(axis=0)

    # Reynolds
    prodR = vr*vp
    RS = (prodR*weight[:,None]).sum(axis=(0,1))*r

    # Maxwell
    prodM = Br*Bp
    MS = -(prodM*weight[:,None]).sum(axis=(0,1))*r/(4*np.pi)	# 4pi = mu0 si en SI

    RS_snap.append(RS)
    MS_snap.append(MS)

times = np.array(times)
RS_snap = np.array(RS_snap)
MS_snap = np.array(MS_snap)
t_total = times[-1] - times[0]

dt = np.diff(times)

RS = np.zeros_like(RS_snap[0])
MS = np.zeros_like(MS_snap[0])

for i in range(len(dt)):
    RS += 0.5*(RS_snap[i] + RS_snap[i+1])*dt[i]
    MS += 0.5*(MS_snap[i] + MS_snap[i+1])*dt[i]

RS /= t_total
MS /= t_total

plt.figure()
plt.plot(RS)
plt.plot(MS)
plt.show()



