#! /usr/bin/env python

'''
Este script resuelve el pendulo simple usando RK2.
ARREGLADO
'''

import numpy as np
import matplotlib.pyplot as plt


A = np.pi / 30
w = np.sqrt(10)

plt.figure(1)
plt.clf()

t = np.linspace(0, 5 * 2 * np.pi / w, 400)

#se tenia: plt.plot(t, A * np.sin(w * t))
#se cambia ya que la funcion que cumple las condiciones
#iniciales(t=0): phi0=A, dphi0/dt=0 es con coseno.
plt.plot(t, A * np.cos(w * t))


def f(phi, w):
    return w, -10 * np.sin(phi)

def get_k1(phi_n, w_n, h, f):
    f_eval = f(phi_n, w_n)
    return h * f_eval[0], h * f_eval[1]

def get_k2(phi_n, w_n, h, f):
    k1 = get_k1(phi_n, w_n, h, f)
    f_eval = f(phi_n + k1[0]/2, w_n + k1[1]/2)
    return h * f_eval[0], h * f_eval[1]

def rk2_step(phi_n, w_n, h, f):
    k2 = get_k2(phi_n, w_n, h, f)
    #se tenia: phi_n1 = phi_n + k2[0] * h
    phi_n1 = phi_n + k2[0]
    #se tenia: w_n1 = w_n + k2[1] * h
    w_n1 = w_n + k2[1]
    #en ambos casos se estaba multiplicando por h,
    #cuando en verdad eso ya habia ocurrido en la definicion
    #de get_k2.
    return phi_n1, w_n1

N_steps = 40000
h = 10. / N_steps
phi = np.zeros(N_steps)
w = np.zeros(N_steps)

phi[0] = A
w[0] = 0
for i in range(1, N_steps):
    phi[i], w[i] = rk2_step(phi[i-1], w[i-1], h, f)



t_rk = [h * i for i in range(N_steps)]

plt.plot(t_rk, phi, 'g')




plt.xlabel('tiempo')
plt.ylabel('$\phi(t)$', fontsize=18)
plt.show()
plt.draw()
