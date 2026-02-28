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
                       k1,k2,k3,k4,k5,k6,k7,k8,k9 = 0,k10 = 0):
        self.A = start_A
        self.B = start_B
        self.C = start_C
        self.D = start_D
        self.E = start_E
        self.k_1 = k1
        self.k_2 = k2
        self.k_3 = k3
        self.k_4 = k4
        self.k_5 = k5
        self.k_6 = k6
        self.k_7 = k7
        self.k_8 = k8
        self.k_9 = k9
        self.k_10 = k10
        #self.k_3 = start_trans_C_B
        #print(self.A)
        #print(self.B)
        #print(self.C)
        #print(self.D)
        #print(self.E)
        self.state = np.array([self.A,self.B,self.C,self.D,self.E], dtype=int)
        self.stoichiometry_absorb = np.array([
                [-1,0,1,0,0],
                [0,-1,0,1,0],
                [0,0,-1,0,1],
                [0,0,0,-1,1],
                [0,0,-1,1,0],
                [0,0,1,-1,0],
                [-1,1,0,0,0],
                [1,-1,0,0,0],
            ])
        self.stoichiometry = np.array([
                [-1,0,1,0,0],
                [0,-1,0,1,0],
                [0,0,-1,0,1],
                [0,0,0,-1,1],
                [0,0,-1,1,0],
                [0,0,1,-1,0],
                [-1,1,0,0,0],
                [1,-1,0,0,0],
                [1,0,0,0,-1],
                [0,1,0,0,-1]
            ])
        self.t = 0.0
        self.t_max_absorb = 1
        self.t_max = 1
        self.time_trace = [self.t]
        self.state_trace = [self.state.copy()]
    
    def Gillespie_absorb(self):
        while self.t < self.t_max_absorb:
            a1 = self.k_1 * self.state[0] * self.state[2]                   #S_C + I_C -> 2*I_C
            a2 = self.k_2 * self.state[1] * self.state[3]                   #S_U + I_U -> 2*I_U
            a3 = self.k_3 * self.state[2]                                   #S_C -> R
            a4 = self.k_4 * self.state[3]                                   #S_U -> R
            a5 = self.k_5 * self.state[2]                                   #I_C -> I_U
            a6 = self.k_6 * self.state[3]                                   #I_U -> I_C
            a7 = self.k_7 * self.state[0]                                   #S_C -> S_U
            a8 = self.k_8 * self.state[1] * self.state[2] * self.state[3]   #S_U + I_U + I_C -> S_C + I_U + I_C
            #a8 = self.k_8 * self.state[1]   #S_U + I_U + I_C -> S_C + I_U + I_C

            a0 = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 
            #print(a0)
            if self.t % 1 == 0:
                print(f"iteration index: {self.t}")
            if a0 <= 0:
                break
            tau = np.random.exponential(1 / a0)
            r = np.random.rand() * a0
            if r < a1:
                reaction_index = 0
            elif r < a1 + a2:
                reaction_index = 1
            elif r < a1 + a2 + a3:
                reaction_index = 2
            elif r < a1 + a2 + a3 + a4:
                reaction_index = 3
            elif r < a1 + a2 + a3 + a4 + a5:
                reaction_index = 4
            elif r < a1 + a2 + a3 + a4 + a5 + a6:
                reaction_index = 5
            elif r < a1 + a2 + a3 + a4 + a5 + a6 + a7:
                reaction_index = 6
            else:
                reaction_index = 7
            self.state += self.stoichiometry_absorb[reaction_index]
            self.t += tau
            self.time_trace.append(self.t)
            self.state_trace.append(self.state.copy())
        self.state_trace = np.array(self.state_trace)
        
    def Gillespie(self):
        while self.t < self.t_max:
            a1 = self.k_1 * self.state[0] * self.state[2]                   #S_C + I_C -> 2*I_C
            a2 = self.k_2 * self.state[1] * self.state[3]                   #S_U + I_U -> 2*I_U
            a3 = self.k_3 * self.state[2]                                   #S_C -> R
            a4 = self.k_4 * self.state[3]                                   #S_U -> R
            a5 = self.k_5 * self.state[2]                                   #I_C -> I_U
            a6 = self.k_6 * self.state[3]                                   #I_U -> I_C
            a7 = self.k_7 * self.state[0]                                   #S_C -> S_U
            a8 = self.k_8 * self.state[1] * self.state[2] * self.state[3]   #S_U + I_U + I_C -> S_C + I_U + I_C
            a9 = self.k_9 * self.state[4]                                   #R -> S_U
            a10 = self.k_10* self.state[4]                                   #R -> S_C
            #a8 = self.k_8 * self.state[1]   #S_U + I_U + I_C -> S_C + I_U + I_C
            a0 = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10
            #a0 = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 
            #print(a0)
            if self.t % 1 == 0:
                print(f"iteration index: {self.t}")
            if a0 <= 0:
                break
            tau = np.random.exponential(1 / a0)
            r = np.random.rand() * a0
            if r < a1:
                reaction_index = 0
            elif r < a1 + a2:
                reaction_index = 1
            elif r < a1 + a2 + a3:
                reaction_index = 2
            elif r < a1 + a2 + a3 + a4:
                reaction_index = 3
            elif r < a1 + a2 + a3 + a4 + a5:
                reaction_index = 4
            elif r < a1 + a2 + a3 + a4 + a5 + a6:
                reaction_index = 5
            elif r < a1 + a2 + a3 + a4 + a5 + a6 + a7:
                reaction_index = 6
            elif r < a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8:
                reaction_index = 7
            elif r < a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9:
                reaction_index = 8
            else:
                reaction_index = 9
            self.state += self.stoichiometry[reaction_index]
            self.t += tau
            self.time_trace.append(self.t)
            self.state_trace.append(self.state.copy())
        self.state_trace = np.array(self.state_trace)
            
        
    def plot(self, axes, title=None):
        axes[0].plot(self.time_trace, self.state_trace[:,0], label='concerned susceptible')
        axes[0].plot(self.time_trace, self.state_trace[:,1], label='unconcerned susceptible')
        axes[1].plot(self.time_trace, self.state_trace[:,2], label='concerned infectious')
        axes[1].plot(self.time_trace, self.state_trace[:,3], label='unconcerned infectious')
        axes[2].plot(self.time_trace, self.state_trace[:,4], label='recovered')
    
        for ax, state in zip(axes, ['Susceptible', 'Infectious', 'Recovered']):
            ax.set_xlabel("Time")
            ax.set_ylabel("#Individuals")
            ax.legend()
        if title:
            for ax in axes:
                ax.set_title(title)
        
        
def main_V2():
    
    react_cls = reaction(100,1,0,1,10)
    react_cls.Gillepsie()
    react_cls.plot()
    
def run_R_can_S():
    num_runs = 3 
    runs = []

    for i in range(num_runs):
        r = reaction(0,1000, 0, 1, 0,
                     0.01,0.1,10,10,2,1,10,3,5,1)
        r.Gillespie()
        runs.append(r)

    fig, axes = plt.subplots(num_runs, 3, figsize=(15, 4 * num_runs))
    axes = axes.reshape(num_runs, 3)
    
    for i, r in enumerate(runs):
        r.plot(axes[i], title=f"Run {i+1}")

    plt.tight_layout()
    plt.title("Recoverd can be infected again")
    plt.show()

def run_R_absorbing():
    num_runs = 3 
    runs = []

    for i in range(num_runs):
        r = reaction(0,1000, 0, 10, 0,
                     0.01,0.1,10,10,2,1,30,3)
        r.Gillespie_absorb()
        runs.append(r)

    fig, axes = plt.subplots(num_runs, 3, figsize=(15, 4 * num_runs))
    axes = axes.reshape(num_runs, 3)
    
    for i, r in enumerate(runs):
        r.plot(axes[i], title=f"Run {i+1}")

    plt.tight_layout()
    #fig.title("Recovered cannot be infected again")
    plt.show()
    
def main():
    #run_R_can_S()
    run_R_absorbing()
    
if __name__=='__main__':
    main()
    