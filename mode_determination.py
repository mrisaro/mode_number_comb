#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 11:21:37 2024
Mode integer determination
@author: mati
"""

import numpy as np
import matplotlib.pyplot as plt
import allantools

# laser parameters
c = 3e8
nu_laser = 474e12

# frequency comb parameters
frep = 1e9+10e3
f0 = 300e6

# list of comb tooth
nu_i = c/1000e-9
nu_f = c/500e-9

Ni = int(nu_i/frep)
Nf = int(nu_f/frep)

f_comb = [f0+frep*i for i in range(0, Nf)]

f_comb = np.array(f_comb)

#%%

kk = np.where(np.abs(nu_laser-f_comb)<frep/2)
fb = nu_laser-f_comb[kk[0]]
f_opt = kk[0]*frep+f0+fb

qq = np.where(np.abs(nu_laser-f_comb)<50e9)

fig = plt.figure('Spectrum Ti:Sa',figsize=(10,6))
ax  = fig.add_subplot(111)
ax.vlines(f_comb[qq[0]],0,0.8,label=r'Spectrum Comb')
ax.vlines(nu_laser,0,1.2,color='C1',linewidth=2,label=r'$\nu_{laser}$')
ax.set_xlim(474e12-3e9,474e12+3e9)
ax.set_ylim(-0.1,1.5)
ax.set_xlabel(r'$f$ (GHz)',fontsize=12)
ax.set_ylabel(r'Counts (#)',fontsize=12)
ax.grid(linestyle='--')
ax.legend(loc='best',fontsize=12)
fig.tight_layout()

#%% Noise winters

tau_w = np.array([1.08,2.17,5.42,10.84,21.68,54.2,108.41,216.81,542.03,
                  1084.05])

sig_w = np.array([7.21,5.04,3.30,2.31,1.67,1.04,0.653,0.591,0.413,
                  0.201])*1e-12

p1,cov = np.polyfit(np.log10(tau_w), np.log10(sig_w), 1,cov=True)

h0 = 2*(7.5e-12**2)

nu_laser_sim = allantools.noise.white(num_points=2**10+1, 
                                      b0=h0*nu_laser**2, fs=1.0)+474e12

fig,axs = plt.subplots(2,1,num='allan simu winters',figsize=(10,8))
axs[0].loglog(tau_w,sig_w,'-o',label=r'Winters')
axs[0].loglog(tau_w,7.5e-12/np.sqrt(tau_w),'--',label=r'fit')
#ax.set_xlim(474e12-3e9,474e12+3e9)
axs[0].set_ylim(0.9e-13,1e-11)
axs[0].set_xlabel(r'$\tau$ (s)',fontsize=12)
axs[0].set_ylabel(r'$\sigma (y)$',fontsize=12)
axs[0].grid(which='both',linestyle='--')
axs[0].legend(loc='best',fontsize=12)

axs[1].plot(nu_laser_sim,'-')
axs[1].set_xlabel(r'$t$ (s)',fontsize=12)
axs[1].set_ylabel(r'$freq$ (Hz)',fontsize=12)
axs[1].grid(which='both',linestyle='--')
fig.tight_layout()

nu_laser_sim_2 = allantools.noise.brown(num_points=2**10+1, 
                                      b2=10e8, fs=1.0)+474e12

(tau,adev,inu,inu) = allantools.oadev(nu_laser_sim_2/474e12,rate=1.0,
                                      data_type='freq',taus=None)

fig,axs = plt.subplots(2,1,num='simu laser mephisto',figsize=(10,8))
axs[0].plot(nu_laser_sim_2,'-', label=r'fb  Mephisto')
axs[0].set_xlabel(r'$t$ (s)',fontsize=12)
axs[0].set_ylabel(r'$freq$ (Hz)',fontsize=12)
axs[0].grid(which='both',linestyle='--')

axs[1].loglog(tau,adev,'-o',label=r'Mephisto Teo')
#ax.set_xlim(474e12-3e9,474e12+3e9)
#axs[0].set_ylim(0.9e-13,1e-11)
axs[1].set_xlabel(r'$\tau$ (s)',fontsize=12)
axs[1].set_ylabel(r'$\sigma (y)$',fontsize=12)
axs[1].grid(which='both',linestyle='--')
axs[1].legend(loc='best',fontsize=12)
fig.tight_layout()

#%% now simulate the beatnote detection

def create_comb(nu_0,nu_rep):
    Nf = 500000
    f_comb = [nu_0+nu_rep*i for i in range(0, Nf)]
    return np.array(f_comb)

def create_fb(nu_0,nu_rep, nu_laser):
    fb = []
    f_comb = create_comb(nu_0, nu_rep)
    for ii in nu_laser:
        kk = np.where(np.abs(ii-f_comb)<frep/2)
        beat = float(ii-f_comb[kk[0]])
        fb.append(beat)
        
    return fb,kk

F_comb = []
F_beat = []
N_comb = []
frep_test = np.array([0,2e3])+1e9

for i in frep_test:
    n_laser = 474e12
    F_comb.append(create_comb(f0,i))
    nu_laser_simu = allantools.noise.white(num_points=2**11+1, 
                                          b0=h0*n_laser**2, fs=1.0)+n_laser
    fb_simu,mode_comb = create_fb(f0, i, nu_laser_simu)
    
    F_beat.append(fb_simu)
    N_comb.append(mode_comb[0][0])

#%%
    
fig,axs = plt.subplots(2,2,num='Mode simu winters',figsize=(14,8))

axs[0,0].vlines(F_comb[0][474000-4:474000+4]-n_laser,0,0.8,label=r'Spec Comb')
axs[0,0].vlines(0,0,1.2,color='C1',linewidth=2,label=r'$\nu_{laser}$')
axs[0,0].set_xlim(-3e9,3e9)
axs[0,0].set_ylim(-0.1,1.5)
axs[0,0].set_xlabel(r'$f$ (Hz)',fontsize=12)
axs[0,0].set_ylabel(r'Counts (#)',fontsize=12)
axs[0,0].grid(linestyle='--')
axs[0,0].legend(loc='best',fontsize=12)

axs[1,0].plot(F_beat[0],'-',label=r'$f_{b} winters$')
axs[1,0].set_title('Mode:'+str(N_comb[0]))
axs[1,0].set_xlabel(r'$t$ (s)',fontsize=12)
axs[1,0].set_ylabel(r'$freq$ (Hz)',fontsize=12)
axs[1,0].grid(linestyle='--')
axs[1,0].legend(loc='best',fontsize=12)

axs[0,1].vlines(F_comb[1][474000-4:474000+4]-n_laser,0,0.8,label=r'Spec Comb')
axs[0,1].vlines(0,0,1.2,color='C1',linewidth=2,label=r'$\nu_{laser}$')
axs[0,1].set_xlim(-3e9,3e9)
axs[0,1].set_ylim(-0.1,1.5)
axs[0,1].set_xlabel(r'$f$ (Hz)',fontsize=12)
axs[0,1].set_ylabel(r'Counts (#)',fontsize=12)
axs[0,1].grid(linestyle='--')
axs[0,1].legend(loc='best',fontsize=12)

axs[1,1].plot(F_beat[1],'-',label=r'$f_{b} winters$')
axs[1,1].set_title('Mode:'+str(N_comb[1]))
axs[1,1].set_xlabel(r'$t$ (s)',fontsize=12)
axs[1,1].set_ylabel(r'$freq$ (Hz)',fontsize=12)
axs[1,1].grid(which='both',linestyle='--')
axs[1,1].legend(loc='best',fontsize=12)

fig.tight_layout()


#%% Mode determination

m = N_comb[1]-N_comb[0]

N_exp = (m*frep_test[1]+np.mean(F_beat[1])+np.mean(F_beat[0]))/(-1*np.diff(frep_test))
print(N_exp)
N_exp = (m*frep_test[1]+np.mean(F_beat[1])-np.mean(F_beat[0]))/(-1*np.diff(frep_test))
print(N_exp)
N_exp = (m*frep_test[1]-np.mean(F_beat[1])+np.mean(F_beat[0]))/(-1*np.diff(frep_test))
print(N_exp)
N_exp = (m*frep_test[1]-np.mean(F_beat[1])-np.mean(F_beat[0]))/(-1*np.diff(frep_test))
print(N_exp)

from tabulate import tabulate
from prettytable import PrettyTable
table = [['\u03A9', 'col 2', 'col 3', 'col 4'], [1, 2222, 30, 500], [4, 55, 6777, 1]]
tab = PrettyTable(table[0])
tab.add_rows(table[1:])
