import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 18:47:31 2026

@author: jschlieffen
"""

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 16:13:54 2026

@author: jschlieffen
"""

# =============================================================================
# TODO: make V and W params.
# =============================================================================


class opinion_dynamics:
    
    INF_RATE_MAX = 0.35
    INF_RATE_MIN = 0.0225
    INF_RATE_MAX_PLUS_MIN = INF_RATE_MAX + INF_RATE_MIN
    INF_RATE_MAX_MINUS_MIN = INF_RATE_MAX - INF_RATE_MIN
    REC_RATE = 0.1

    def __init__(self, num_grid_points: int, max_t: int, initial_opinions: np.ndarray, noise_strength: float, N:int,
                 interaction_distance: float, y0: np.ndarray, stochiomatric_vectors: np.ndarray, grad_V: Callable):
        self.t = np.linspace(0, max_t, num_grid_points)
        self.dt = max_t/num_grid_points
        
        self.opinions_curr = initial_opinions.copy()
        self.opinion_array = np.zeros((num_grid_points, len(initial_opinions)), dtype=float)
        self.opinion_array[0] = initial_opinions.copy()
        
        self.infection_curr = y0.copy()
        self.infection_array = np.zeros((num_grid_points, 3), dtype=float)
        self.infection_array[0] = y0.copy()
        self.nu = np.array(stochiomatric_vectors)
        
        self.sigma = noise_strength
        self.N = N
        self.d = interaction_distance
        self.grad_V = grad_V

    def algo(self):
        for i in range(1, len(self.t)):
            # Updating the Infection numbers
            self.update_infected()
            self.infection_array[i] = self.infection_curr.copy()
            # Apply grad_V to each agent with current infected numbers
            drift_V = self.grad_V(self.opinions_curr, self.infection_curr[1]/self.N)
            self.opinions_curr += -drift_V * self.dt
            self.opinion_array[i] = self.opinions_curr.copy()

    def infection_propensity(self):
        # TODO: Using the mean of opinion is really a band-aid solution, combinatorial assumption for Gillespie 
        # is that the agents are identical s.t. the choice from susceptible/infected for reaction is simple
        # when the agents' opinion affected the infection rate, the formula for propensity does not hold 
        propensity = (self.INF_RATE_MAX_PLUS_MIN - self.INF_RATE_MAX_MINUS_MIN * np.tanh(2 * self.opinions_curr.mean())) / 2
        return propensity * self.infection_curr[0] * self.infection_curr[1] / self.N
    
    def recovery_propensity(self):
        return self.REC_RATE * self.infection_curr[1]

    def update_infected(self):
        # Tau-leaping algorithm (variant of Gillepsie for fixed delta t)
        inf_jumps = np.random.poisson(self.infection_propensity() * self.dt)
        rec_jumps = np.random.poisson(self.recovery_propensity() * self.dt)
        self.infection_curr += inf_jumps * self.nu[0]
        self.infection_curr += rec_jumps * self.nu[1]
        

    def opinion_history(self):
        return self.opinion_array

    def infection_history(self):
        return self.infection_array

def main():
    # Parameters
    N = 10000
    initial_opinions = np.random.uniform(-1, 1, size=N)
    grad_V = np.vectorize(lambda opinion, infected: infected * opinion**2 / 100)
    model = opinion_dynamics(
        num_grid_points=100,
        max_t=100,
        initial_opinions=initial_opinions,
        noise_strength=0,
        N=N,
        interaction_distance=0,
        y0=np.array([9550, 50, 0]),
        stochiomatric_vectors=np.array([[-1, 1, 0], [0, -1, 1]]),
        grad_V=grad_V
    )
    model.algo()
    plt.hist(model.opinion_history()[0], bins=5)
    plt.savefig('plots/other/initial_opinions.png')
    plt.close()
    plt.hist(model.opinion_history()[-1], bins=5)
    plt.savefig('plots/other/final_opinions.png')
    plt.close()
    plt.plot(model.t, model.infection_history()[:, 0], label='susceptible')
    plt.plot(model.t, model.infection_history()[:, 1], label='infected')
    plt.plot(model.t, model.infection_history()[:, 2], label='recovered')
    plt.legend()
    plt.savefig('plots/other/infection_history.png')
    plt.close()

if __name__ == '__main__':
    main()
