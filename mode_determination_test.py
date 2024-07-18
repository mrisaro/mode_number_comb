#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 11:26:47 2024
Mode determination test
@author: mati
"""

#%% importing libraries
import numpy as np
import matplotlib.pyplot as plt
import allantools
from prettytable import PrettyTable

def create_comb(nu_0,nu_rep):
    # function to create equidistant frequencies
    Nf = 500000
    f_comb = [nu_0+nu_rep*i for i in range(0, Nf)]
    return np.array(f_comb)

def create_fb(nu_0,nu_rep, nu_laser):
    # function to create the beat-note respect to the comb
    fb = []
    f_comb = create_comb(nu_0, nu_rep)
    for ii in nu_laser:
        kk = np.where(np.abs(ii-f_comb)<frep/2)
        beat = float(ii-f_comb[kk[0]])
        fb.append(beat)
        
    return fb,kk

def comb_close_nu(F_comb,nu_laser,D_nu):
    
    qq_nu = np.where(np.abs(nu_laser-F_comb)<D_nu)
    f_comb = F_comb[qq_nu[0]]-nu_laser
    return f_comb,qq_nu[0]

def plot_v_lines_labels(ax, x_positions, labels, y_max=0.7):
    """
    Plotea líneas verticales con puntas de flecha y cuadros de texto numerados en un eje dado.

    Parámetros:
    ax (matplotlib.axes.Axes): El eje en el que se plotean las líneas.
    x_positions (array-like): Posiciones x de las líneas verticales.
    labels (array-like): Etiquetas para cada línea vertical.
    y_max (float): Altura fija para todas las líneas. Por defecto es 1.0.
    """
    # Añadir líneas verticales con puntas de flecha y cuadros de texto numerados
    for x, label in zip(x_positions, labels):
        ax.vlines(x, ymin=0, ymax=y_max, color='C0')
        ax.annotate('', xy=(x, y_max), xytext=(x, 0),
                    arrowprops=dict(arrowstyle='->', color='C0'))
        ax.text(x, y_max + 0.1, f'{label}', ha='center', fontsize=8, 
                bbox=dict(facecolor='white', alpha=0.6))

    # Ajustar los límites del gráfico
    #ax.set_ylim(0, y_max + 0.5)
    #ax.set_xlim(np.min(x_positions) - 1e9, np.max(x_positions) + 1e9)

def calc_Nexp(m,frep_pack,fb_pack):
    # the function uses: m, (frep', frep), (fb',fb)
    # I define frep_pack = (frep', frep), fb_pack = (fb', fb)
    # we take frep as the nominal
    dif = float(np.diff(frep_pack))
    # possible integers
    N1  = float(m*frep_pack[0] + fb_pack[1] + fb_pack[0])/dif
    N2  = float(m*frep_pack[0] + fb_pack[1] - fb_pack[0])/dif
    N3  = float(m*frep_pack[0] - fb_pack[1] + fb_pack[0])/dif
    N4  = float(m*frep_pack[0] - fb_pack[1] - fb_pack[0])/dif
    
    return [N1, N2, N3, N4]
#%% Start with the simulation
# Comb parameters
f0     = 350e6              # f_CEO value
frep   = 1e9                # f_rep nominal
nu_las = 474e12             # nu_laser nominal
N_beat = 2**10+1            # number points simulate beat
h0     = 2*(7.5e-10**2)     # h0 from Enricos chart (white-freq noise)

F_comb = []
F_beat = []
N_comb = []
frep_test = np.array([0,-3.5e3,2.5e3,6e3])+frep

for i in frep_test:
    F_comb.append(create_comb(f0,i))
    nu_laser_simu = allantools.noise.white(num_points=N_beat, 
                                          b0=h0*nu_las**2, fs=1.0)+nu_las
    fb_simu,mode_comb = create_fb(f0, i, nu_laser_simu)
    
    F_beat.append(fb_simu)
    N_comb.append(mode_comb[0][0])

#%%
N_col = len(frep_test)

fig,axs = plt.subplots(2,N_col,num='Mode simu winters II',figsize=(17,5))

for i in range(N_col):
    x_comb,n_comb = comb_close_nu(F_comb[i],nu_las,1.5e9)
    plot_v_lines_labels(axs[0,i], x_comb*1e-9, n_comb, y_max=0.7)
    axs[0,i].vlines(0,0,1.4,color='C1',linewidth=2,label=r'$\nu_{laser}$')
    axs[0,i].set_xlim(-1.8,1.8)
    axs[0,i].set_ylim(-0.1,1.5)
    axs[0,i].set_xlabel(r'$f$ (GHz)',fontsize=10)
    #axs[0,i].set_ylabel(r'Counts (#)',fontsize=12)
    axs[0,i].grid(linestyle='--')
    axs[0,i].legend(loc='best',fontsize=10)
    
    axs[1,i].plot(F_beat[i],'-',label=r'$f_{b} winters$')
    axs[1,i].set_title('Mode:'+str(N_comb[i]),fontsize=10)
    axs[1,i].set_xlabel(r'$t$ (s)',fontsize=10)
    axs[1,i].set_ylabel(r'$f_{b}$ (Hz)',fontsize=10)
    axs[1,i].grid(linestyle='--')
    axs[1,i].legend(loc='best',fontsize=10)
    
fig.tight_layout()
#%% Calculate the N_exp
N_exp = []      # experimental N in each mode change
M     = []      # allmode change
col_tab = []

for i in range(len(frep_test[1:])):
    col_tmp = []
    frep_pack = [frep_test[i+1],frep]
    fb_pack   = [np.mean(F_beat[i+1]),np.mean(F_beat[0])]
    m = N_comb[i+1]-N_comb[0]     
    n_exp = calc_Nexp(m,frep_pack,fb_pack)
    print(n_exp)
    N_exp.append(n_exp)
    M.append(m)
    
    col_tmp.append(m)
    col_tmp = col_tmp+n_exp
    col_tab.append(col_tmp)
#%%

tab_head = ['m', 'fb1+fb0', 'fb1-fb0', '-fb1+fb0', '-fb1-fb0']

table = [tab_head, col_tab[0], col_tab[1], col_tab[2]]
tab = PrettyTable(table[0])
table_col = ['633 nm','633 nm','633 nm']
tab.add_rows(table[1:])
tab.add_column('\u03BB', table_col, valign='m')
print(tab)
