#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 21:15:06 2025

@author: jschlieffen
"""

import numpy as np
import matplotlib.pyplot as plt

class reaction:
    
    def __init__(self, start_A, start_B, start_C, start_D,start_E,
                       k1,k2,k3,k4,k5,k6,k7,k8):
        self.A = start_A
        self.B = start_B
        self.C = start_C
        self.D = start_D,
        self.E = start_E,
        self.k_1 = k1
        self.k_2 = k2
        self.k_3 = k3
        self.k_4 = k4
        self.k_5 = k5
        self.k_6 = k6
        self.k_7 = k7
        self.k_8 = k8
        #self.k_3 = start_trans_C_B
        self.state = np.array([self.A,self.B,self.C, self.D,self.E], dtype=int)
        self.stoichiometry = np.array([
                [-1,0,1,0,0],
                [0,-1,0,1,0],
                [0,0,-1,0,1],
                [0,0,0,-1,1],
                [0,0,-1,1,0],
                [-1,1,0,0,0],
                [1,-1,0,0,0]
            ])
        self.t = 0.0
        self.t_max = 50000.0
        self.time_trace = [self.t]
        self.state_trace = [self.state.copy()]
    
    def Gillespie(self):
        while self.t < self.t_max:
            a1 = self.k_1 * self.state[0] * self.state[1]  # A + B -> C
            a2 = self.k_2 * self.state[1]             # C -> A
            #a3 = self.k_3 * self.state[2]             # C -> B
            a0 = a1 + a2 
            if a0 == 0:
                break
            tau = np.random.exponential(1 / a0)
            r = np.random.rand() * a0
            if r < a1:
                reaction_index = 0
            elif r < a1 + a2:
                reaction_index = 1
            else:
                reaction_index = 2
            self.state += self.stoichiometry[reaction_index]
            self.t += tau
            self.time_trace.append(self.t)
            self.state_trace.append(self.state.copy())
        self.state_trace = np.array(self.state_trace)
        
        
        
    def plot(self, ax, title=None):
        ax.plot(self.time_trace, self.state_trace[:,0], label='susceptible')
        ax.plot(self.time_trace, self.state_trace[:,1], label='infectious')
        ax.plot(self.time_trace, self.state_trace[:,2], label='recoverd')
        ax.set_xlabel("Time")
        ax.set_ylabel("#individuals")
        ax.legend()
        if title:
            ax.set_title(title)
        
        
def main_V2():
    
    react_cls = reaction(100,1,0,1,10)
    react_cls.Gillepsie()
    react_cls.plot()
    

def main():
    num_runs = 4  # choose number of subplots
    runs = []

    for i in range(num_runs):
        r = reaction(100, 1, 0, 1, 10)
        r.Gillespie()
        runs.append(r)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i, r in enumerate(runs):
        r.plot(axes[i], title=f"Run {i+1}")

    plt.tight_layout()
    plt.show()
    
if __name__=='__main__':
    main()
    