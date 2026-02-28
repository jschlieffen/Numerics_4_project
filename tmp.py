#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 16:13:54 2026

@author: jschlieffen
"""


import numpy as np
import matplotlib.pyplot as plt
from log_msg import *
# =============================================================================
# TODO: make V and W params.
# =============================================================================
class opinion_dynamics:
    
    def __init__(self,num_grid_points, max_t,initial_opinions, noise_strength, interaction_distance,L=2*np.pi):
        self.t = np.linspace(0,max_t,num_grid_points)
        self.dt = self.t[1] - self.t[0]
        self.X0 = initial_opinions
        self.X = self.X0.copy()
        self.sigma = noise_strength
        self.L = L
        self.d = interaction_distance
        self.history = [self.X.copy()]
        
    #External pot.
    def grad_V(self,x,t):
        #return x
        return 0*x
        
    def grad_V_V2(self, x, t):
        eps = 0.1
        x0 = np.pi
        return eps * np.sin(x - x0)
    
    #Interaction pot.
    
    def grad_W_V2(self,x):
        return 1 if np.abs(x) < self.d else 0
    
    def grad_W(self, x):
        return np.sin(x) + np.sin(4*x)
    
    def wrap(self, x):
        return x % self.L
    
    def torus_diff(self, x, y):
        return (x - y + self.L/2) % self.L - self.L/2
    
    def algo(self):
        N,d = self.X.shape
        for _ in self.t[1:]:
            #calc. of V
            drift_V = np.array([self.grad_V(self.X[k],_) for k in range(N)])
            #drift_V = np.zeros(N,N)
            drift_W = np.zeros_like(self.X)
            #calc. of sum W
            for k in range(N):
                for j in range(N):
                    if j !=k:
                        dx = self.torus_diff(self.X[k], self.X[j])
                        drift_W[k] += self.grad_W(self.X[k] - self.X[j])
            drift_W /= N
            # Brownian increment
            dB = np.sqrt(self.dt) * np.random.randn(N, d)
            self.X += -drift_V * self.dt - drift_W * self.dt + self.sigma * dB
            self.X = self.wrap(self.X)
            self.history.append(self.X.copy())
        
        self.history = np.array(self.history)





class Plots_cls:
    def __init__(self, opinions_model):
        self.model = opinions_model
        self.history = opinions_model.history
        self.t = opinions_model.t

    def plot_trajectories(self):
        """
        Plot opinion evolution over time
        """
        num_steps, N, d = self.history.shape
        
        plt.figure(figsize=(8, 5))
        for i in range(N):
            plt.plot(self.t, self.history[:, i, 0], alpha=0.7)
        
        plt.xlabel("Time")
        plt.ylabel("Opinion")
        plt.title("Opinion trajectories")
        plt.grid(True)
        #plt.show()
        plt.savefig('plots/trajectories.png')

    def plot_final_histogram(self, bins=20):
        """
        Histogram of final opinions
        """
        final_opinions = self.history[-1, :, 0]
        
        plt.figure(figsize=(6, 4))
        plt.hist(final_opinions, bins=bins, density=True, alpha=0.7)
        plt.xlabel("Opinion")
        plt.ylabel("Density")
        plt.title("Final opinion distribution")
        plt.grid(True)
        plt.show()
        
    def plot_opinion_time_scatter(self):
        """
        Scatter plot:
        x-axis = opinion
        y-axis = time
        each dot = one agent
        """
        T, N, d = self.history.shape

        # Flatten data
        opinions = self.history[:, :, 0].flatten()
        times = np.repeat(self.t, N)

        plt.figure(figsize=(7, 6))
        plt.scatter(opinions, times, s=8, alpha=0.6)
        plt.xlabel("Opinion")
        plt.ylabel("Time")
        plt.title("Opinion evolution (agent-time scatter)")
        plt.grid(True)
        #plt.show()
        plt.savefig('plots/dots.png')
        
def main():
    # Parameters
    N = 128
    dim = 1
    #initial_opinions = np.random.uniform(-1, 1, size=(N, 1))
    initial_opinions = np.random.uniform(0, 2*np.pi, size=(N, 1))
    model = opinion_dynamics(
        num_grid_points=200,
        max_t=50.0,
        initial_opinions=initial_opinions,
        noise_strength=0.4,
        interaction_distance=0.001
    )
    logger.status('start algo')
    model.algo()
    logger.success('finish algo')
    logger.status('start plots')
    plots = Plots_cls(model)
    plots.plot_trajectories()
    #plots.plot_final_histogram()
    plots.plot_opinion_time_scatter()
    logger.success('finish plots')
if __name__ == '__main__':
    main()
