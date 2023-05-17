#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 16:04:34 2023.

@author: lisandro
"""
import numpy as np
import matplotlib.pyplot as plt
from hermite import hermite_function
from basis import HermiteBasis, FourierBasis
from state import State
from scheme import (AffineSym4, Strang, Ruth, DOPRI853, AffineSym6,
                     AffineSym8, DOPRI5, Neri, Yoshida6) 
from solver import Solver
from models import GeneralModel1D, NLSE3Model1D

sch = AffineSym6()

N = 2048
scaling = 1.25
dt = 2.5e-2
T = 10.
# Set the parameters of the NLSE3 soliton solution
c = 0.5
eta = 1.0
omega = 0.5*(c**2 - eta**2)
x0 = 0.


# D = 1.0
# alpha = 1.1
# beta = 0.1
# gamma = -1.0
# delta = -0.2
# epsilon = 1.7
# nu = -0.115
# mu = -1.0

D = 1.0
alpha = 2.0
beta = 0.08
gamma = -1.0
delta = -0.1
epsilon = 0.75
nu = -0.07
mu = -0.1

# Define the soliton function
soliton = lambda x: eta / np.cosh(eta*(x+x0)) * np.exp(1j*(c*(x+x0))) # + eta / np.cosh(eta*(x-x0)) * np.exp(1j*(-c*(x-x0)))
# soliton = lambda x: hermite_function(0, (x - x0)/1.)

# soliton = lambda x: 1.2 * np.exp(-x**2 / 2)
# model = GeneralModel1D(D=D, epsilon=epsilon, delta=delta, beta=beta,
#                        gamma=gamma, mu=mu, nu=nu, s=alpha/2)
model = NLSE3Model1D()
hb = FourierBasis('hb', N, (-10*np.pi, 10*np.pi))
u0 = State('u0', hb, u=soliton)
t0 = 0.0
tf = T

scheme = sch
solver = Solver(model, scheme)
trajectory = [(t0, u0)]
solver.start(u0, t0, tf)

while solver.active:
    u = solver.step(dt)
    t = solver.sim_time
    trajectory.append((t, u))


x = []
y = []
z = []

for point in trajectory:
    x.append(point[0])
    y.append(point[1].grid)
    z.append(point[1].values)

x = np.array(x)
y = np.array(y)
z = np.abs(np.array(z))

xlim = (0, 30)
ylim = (-15., 15.)

#x_cropped = y[:, :]
#x_cropped[x_cropped <= xlim[0]] = np.nan
#x_cropped[x_cropped >= xlim[1]] = np.nan

y_cropped = y.copy()
y_cropped[y_cropped <= ylim[0]] = np.nan
y_cropped[y_cropped >= ylim[1]] = np.nan

plt.style.use('classic')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_ylim(-15, 15)
ax.set_zlim(1e-16, 1.0)
ax.set_xlabel("$t$", fontsize=18)
ax.set_ylabel("$x$", fontsize=18)
ax.set_zlabel("$|u|$", fontsize=18)
for i in range(0, len(x), 1):
        # ax.plot(x[i] * np.ones_like(y[i,:]), y[i, :], z[i, :], lw=0, color='b', zdir='z', zorder=i)
    # ax.add_collection3d(plt.fill_between(y[i,:], z[i, :], lw=0.5, color='b', fc='c', alpha=0.5, zorder=len(y)-i), zs=x[i], zdir='x')
    ax.plot(x[i] * np.ones_like(y[i,:]), y_cropped[i, :], z[i, :], lw=0, color='b', zdir='z', zorder=i)
    ax.add_collection3d(plt.fill_between(y_cropped[i, :], z[i, :], lw=0.5, color='b', fc='c', alpha=0.5, zorder=len(y)-i), zs=x[i], zdir='x')
ax.view_init(elev=30, azim=-30)
# Show the plot
plt.show()
# plt.savefig('breather.pdf')


# # num_traj = np.zeros((len(trajectory), N), dtype=np.complex128)
# # for i in range(len(trajectory)):
# #     state = trajectory[i][1]
# #     num_traj[i] = state.values
fig = plt.figure()
plt.xlabel("$t$", fontsize=20)
plt.ylabel("$x$", fontsize=20)
plt.ylim(-15,15)
plt.pcolormesh(x, y[-1], z.T, shading='gouraud', cmap='plasma')
plt.colorbar()
# plt.savefig("./experiments/output/fCGLE5/hermite/figures/soliton_alpha_1.1.pdf")
# # fig = plt.figure()
# plt.contour(x, y[-1], z.T, levels=np.linspace(0,1.35, 50))
