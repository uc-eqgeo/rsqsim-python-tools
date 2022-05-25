import numpy as np
import matplotlib.pyplot as plt
from tde import tde

# Script to test optimization
# tri dislocation parameters
pr = 0.25
ss = -1.
ts = 0.
ds = 0.
N = 30

sx, sy, sz = np.meshgrid(np.linspace(0, 100, N), np.linspace(0, 100, N), 0)

sxr = sx.ravel(order='F')
syr = sy.ravel(order='F')
szr = sz.ravel(order='F')

X = np.array([40., 60., 40.])
Y = np.array([50., 50., 30.])
Z = np.array([0., 0., 20.])

# S = tde.calc_tri_strains(sxr, syr, szr, X, Y, Z, pr, ss, ts, ds)

for i in range(100):
    U = tde.calc_tri_displacements(sxr, syr, szr, X, Y, Z, pr, ss, ts, ds)

