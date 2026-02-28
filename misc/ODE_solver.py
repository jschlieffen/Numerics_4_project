#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 20:12:00 2025

@author: jschlieffen
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
"""
# Define the system of ODEs
def model(y, t):
    # constants: [S1->S2, S2->S1, S1->I1, S2->I2, I1->I2, I2->I1, I1->R = I2->R]
    # array y: [susceptible unconcerned (1), sus. con. (2), infected unc., inf con., recovered]
    c = np.array([1, 10, 0.1, 0.01, 2, 1, 10])
    dydt =          [-c[0] * (y[2]+y[3]) * y[0] + c[1] * y[1] - c[2]*y[0]*y[2],
                     + c[0] * (y[2]+y[3]) * y[0] - c[1] * y[1] - c[3]*y[1]*y[3],
                     - c[4] * y[2] + c[5] * y[3] + c[2]*y[0]*y[2] - c[6] * y[2],
                     + c[4] * y[2] - c[5] * y[3] + c[3]*y[1]*y[3] - c[6] * y[3],
                       c[6] * y[2] + c[6] * y[3]]
    return dydt

# Create a NumPy array for initial conditions
y0 = np.array([1000,0.0,10,0,0])
"""
def model(y, t):
    # constants: [S1->S2, S2->S1, S1->I1, S2->I2, I1->I2, I2->I1, I1->R = I2->R]
    # array y: [susceptible unconcerned (1), sus. con. (2), infected unc., inf con., recovered]
    #c = np.array([500000, 1500, 10, 0.005, 0.01, 0.01, 0.1])
    c = np.array([1, 0.01, 0.7, 0.1, 0.01, 0.3, 0.3])

    dydt =          [-c[0] * (y[2]+y[3]) * y[0] + c[1] * y[1] - c[2]*y[0]*y[2],
                     + c[0] * (y[2]+y[3]) * y[0] - c[1] * y[1] - c[3]*y[1]*y[3],
                     - c[4] * y[2] + c[5] * y[3] + c[2]*y[0]*y[2] - c[6] * y[2],
                     + c[4] * y[2] - c[5] * y[3] + c[3]*y[1]*y[3] - c[6] * y[3],
                       c[6] * y[2] + c[6] * y[3]]
    return dydt

# Create a NumPy array for initial conditions
y0 = np.array([0.6,0.39,0.01,0,0])
# Create a time array using NumPy
t = np.linspace(0, 200, 1000000)



# Use SciPy's odeint function to solve the ODEs
solution = odeint(model, y0, t)
"""
for i in range(0,10000,100):
    if sum(solution[i] != 1):
        print(sum(solution[i]))
        #print('populations zuwachs')
"""
fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
# Susceptible subplot
axs[0].plot(t, solution[:,0], label='Susceptible Unconcerned')
axs[0].plot(t, solution[:,1], label='Susceptible Concerned')
axs[0].set_ylabel('S(t)')
axs[0].legend()
axs[0].set_title('Susceptible Population')

# Infected subplot
axs[1].plot(t, solution[:,2], label='Infected Unconcerned')
axs[1].plot(t, solution[:,3], label='Infected Concerned')
axs[1].set_ylabel('I(t)')
axs[1].legend()
axs[1].set_title('Infected Population')

# Recovered subplot
axs[2].plot(t, solution[:,4], label='Recovered', color='green')
axs[2].set_xlabel('Time')
axs[2].set_ylabel('R(t)')
axs[2].legend()
axs[2].set_title('Recovered Population')

plt.tight_layout()
plt.show()