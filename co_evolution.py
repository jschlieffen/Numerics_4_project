#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 18:47:31 2026

@author: jschlieffen
"""

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
    
    def __init__(self,num_grid_points, max_t,initial_opinions, noise_strength, interaction_distance,L, y0, stochiomatric_vector):
        self.t = np.linspace(0,max_t,num_grid_points)
        self.dt = self.t[1] - self.t[0]
        self.X0 = initial_opinions
        self.X = self.X0.copy()
        self.sigma = noise_strength
        self.L = L
        self.d = interaction_distance
        self.history = [self.X.copy()] #for plot_cls
        self.y = np.array(y0, dtype=float)
        self.nu = np.array(stochiomatric_vector)
        self.y_history = [self.y.copy()] #for plot_cls
        
    def grad_V(self, x, y):
        return 0.1 * np.sin(x - y)
    
    def grad_W(self, x_diff, y):
        return (np.sin(x_diff) + np.sin(4*x_diff)) * (1 + 0.2*y)

    
    def wrap(self, x):
        return x % self.L
    
    def torus_diff(self, x, y):
        return (x - y + self.L/2) % self.L - self.L/2
    
    def algo(self):
        N,d = self.X.shape
        for _ in self.t[1:]:
            #updating the infection numbers
            self.update_y()
            #calc. of V
            drift_V = np.array([self.grad_V(self.X[k],self.y) for k in range(N)])
            #drift_V = np.zeros(N,N)
            drift_W = np.zeros_like(self.X)
            #calc. of sum W
            for k in range(N):
                for j in range(N):
                    if j !=k:
                        dx = self.torus_diff(self.X[k], self.X[j])
                        drift_W[k] += self.grad_W(dx,self.y)
            drift_W /= N
            # Brownian increment
            dB = np.sqrt(self.dt) * np.random.randn(N, d)
            self.X += -drift_V * self.dt - drift_W * self.dt + self.sigma * dB
            self.X = self.wrap(self.X)
            self.history.append(self.X.copy())
            self.y_history.append(self.y.copy())
        self.history = np.array(self.history)
        self.y_history = np.array(self.y_history)


    def alpha(self, X, y):
        beta0 = 0.5
        gamma = 0.25    
        y_star = 20
        delta = 0.1     
        suppression = 1.0 / (1.0 + delta * max(y - y_star, 0.0))
    
        alpha_infect = beta0 * y * suppression
        alpha_recover = gamma * y
    
        return np.array([alpha_infect, alpha_recover])


    def alpha_V3(self, X, y):
        return np.array([
            0.2,        
            0.1 * y       
        ])

    def alpha_V2(self, X, y):
        mean_x = np.mean(X)
        return np.array([
            0.5 + 0.1 * np.abs(mean_x - y),
            0.2 + 0.05 * y**2
        ])

    def update_y(self):
        alphas = self.alpha(self.X, self.y)
        
        for k, alpha_k in enumerate(alphas):
            n_jumps = np.random.poisson(alpha_k * self.dt)  # Poisson process
            self.y += n_jumps * self.nu[k]


class Plots_cls:
    def __init__(self, opinions_model):
        self.model = opinions_model
        self.history = opinions_model.history
        self.t = opinions_model.t       
        self.y_history = opinions_model.y_history

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
        
        
    def plot_infection_time_series_V2(self):
        plt.figure(figsize=(7, 4))
        plt.plot(self.t, self.y_history, lw=2)
        plt.xlabel("Time")
        plt.ylabel("Infection level")
        plt.title("Infection dynamics")
        plt.grid(True)
        plt.show()
        plt.savefig("plots/infection_time_series.png")
        
    def plot_infection_time_series(self):
        y = self.y_history
    
        if y.ndim == 1:
            y = y[:, None]
    
        m = y.shape[1]
    
        plt.figure(figsize=(7, 4))
        for i in range(m):
            plt.plot(self.t, y[:, i], label=f"$y_{i}$")
    
        plt.xlabel("Time")
        plt.ylabel("Infection level")
        plt.title("Infection dynamics")
        plt.legend()
        plt.grid(True)
        #plt.show()
        plt.savefig("plots/infection_time_series.png")
        
    def plot_infection_steps(self):
        y = self.y_history
        if y.ndim == 1:
            y = y[:, None]
    
        plt.figure(figsize=(7, 4))
        for i in range(y.shape[1]):
            plt.step(self.t, y[:, i], where="post", label=f"$y_{i}$")
    
        plt.xlabel("Time")
        plt.ylabel("Infection level")
        plt.title("Infection jump process")
        plt.legend()
        plt.grid(True)
        #plt.show()
        plt.savefig("plots/infection_steps.png")

    def plot_infection_vs_mean_opinion(self):
        mean_opinion = self.history[:, :, 0].mean(axis=1)
        y = self.y_history
    
        fig, ax1 = plt.subplots(figsize=(7, 4))
    
        ax1.plot(self.t, mean_opinion, color="tab:blue")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Mean opinion", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
    
        ax2 = ax1.twinx()
        ax2.step(self.t, y, where="post", color="tab:red")
        ax2.set_ylabel("Infection level", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")
    
        plt.title("Coupling between infection and opinion dynamics")
        fig.tight_layout()
        #plt.show()
        plt.savefig("plots/infection_vs_opinion.png")

        
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
        interaction_distance=0.001,
        L = 2*np.pi,
        y0 = 25.0,
        stochiomatric_vector = [1.0, -1.0]
    )
    logger.status('start algo')
    model.algo()
    logger.success('finish algo')
    logger.status('start plots')
    plots = Plots_cls(model)
    plots.plot_trajectories()
    #plots.plot_final_histogram()
    plots.plot_opinion_time_scatter()
    plots.plot_infection_time_series()
    plots.plot_infection_steps()
    plots.plot_infection_vs_mean_opinion()
    logger.success('finish plots')
if __name__ == '__main__':
    main()
