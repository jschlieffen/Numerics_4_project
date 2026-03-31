import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from typing import Callable
from scipy.special import expit
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

global HIST_BINS;
HIST_BINS = np.array([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5])

def opinion_sampler(initial_opinion_data: np.ndarray, num_agents: int) -> np.ndarray:
    # Japan data is in 7 bins, we can force the opinions
    # Bound the opinions to be between -3.5 and 3.5, sample using initial opinion percentages
    edges = HIST_BINS
    probs = np.array(initial_opinion_data, dtype=float)
    probs = probs/probs.sum()
    
    sampled_opinions = np.random.choice(len(probs), size=num_agents, p=probs)
    left_edges = edges[sampled_opinions]
    right_edges = edges[sampled_opinions + 1]
    # Sample uniformly between the edges to get continuous opinions
    return np.random.uniform(left_edges, right_edges)

class opinion_dynamics:
    
    inf_rate_max_plus_min: float
    inf_rate_max_minus_min: float
    opinion_array: np.ndarray
    infection_array: np.ndarray
    opinions_curr: np.ndarray
    infection_curr: np.ndarray


    def __init__(self, num_grid_points: int, max_t: int, initial_opinions: np.ndarray, y0: np.ndarray, N:int,
                 interaction_distance: float, noise_strength: float, stochiomatric_vectors: np.ndarray, grad_V: str,
                 grad_V_params: tuple = (0, 0), inf_rate_max: float = 0.35, inf_rate_min: float = 0.0225, rec_rate: float = 0.01):
        self.t = np.linspace(0, max_t, num_grid_points)
        self.dt = max_t/num_grid_points
        self.inf_rate_max_plus_min = inf_rate_max + inf_rate_min
        self.inf_rate_max_minus_min = inf_rate_max - inf_rate_min
        self.rec_rate = rec_rate

        self.opinions_curr = np.copy(initial_opinions)
        self.opinion_array = np.zeros((num_grid_points, len(initial_opinions)), dtype=float)
        self.opinion_array[0] = np.copy(initial_opinions)
        
        self.infection_curr = np.copy(y0)
        self.infection_array = np.zeros((num_grid_points, 3), dtype=float)
        self.infection_array[0] = np.copy(y0)
        self.infection_num_array = np.zeros(num_grid_points)
        self.nu = np.array(stochiomatric_vectors)
        
        self.sigma = noise_strength
        self.N = N
        self.d = interaction_distance
        self.grad_V = grad_V
        self.grad_V_params = grad_V_params

    def algo(self):
        for i in range(1, len(self.t)):
            # Updating the Infection numbers
            self.infection_num_array[i] = self.update_infected()
            self.infection_array[i] = np.copy(self.infection_curr)
            # Apply grad_V to each agent with current infected numbers
            # drift_V = self.grad_V(self.opinions_curr, self.infection_curr[1]/self.N)
            # self.opinions_curr += -drift_V * self.dt
            
            infected_frac = self.infection_curr[1] / self.N
            if self.grad_V == "sigmoid":
                self.opinions_curr -= self.grad_V_params[0] * (1.0 - infected_frac) - self.grad_V_params[1] * infected_frac * expit(self.opinions_curr) * self.dt
            elif self.grad_V == "polynomial":
                alpha = (1 - 2 * self.grad_V_params[0])/(self.grad_V_params[0]**2 - self.grad_V_params[0])
                inf_func = alpha * infected_frac**2 + (2 - alpha) * infected_frac - 1
                self.opinions_curr -= 4 * self.opinions_curr * (self.opinions_curr**2 + 1) - 8 * self.grad_V_params[1] * inf_func(infected_frac) * self.opinions_curr**2 * self.dt
            
            self.opinions_curr = np.clip(self.opinions_curr, -3.5, 3.5)
            self.opinion_array[i] = np.copy(self.opinions_curr)

    def infection_propensity(self):
        # TODO: Using the mean of opinion is really a band-aid solution, combinatorial assumption for Gillespie is 
        # that the agents (molecules) are identical s.t. the choice from susceptible/infected for reaction is simple.
        # When the agents' opinion affects the infection rate, the formula for propensity does not work anymore
        propensity = (self.inf_rate_max_plus_min - self.inf_rate_max_minus_min * np.tanh(2 * self.opinions_curr.mean())) / 2
        return propensity * self.infection_curr[0] * self.infection_curr[1] / self.N
    
    def recovery_propensity(self):
        return self.rec_rate * self.infection_curr[1]

    def update_infected(self) -> float:
        # Tau-leaping algorithm (variant of Gillepsie for fixed delta t)
        self.infection_curr = np.clip(self.infection_curr, 0, self.N)
        inf_jumps = np.random.poisson(self.infection_propensity() * self.dt)
        rec_jumps = np.random.poisson(self.recovery_propensity() * self.dt)
        # Consequence of using Tau-leaping is that recovery numbers could outgrow
        # number of infected, hence make all infected recover in the case that
        # more people have recovered than there are new infected + already infected
        self.infection_curr += inf_jumps * self.nu[0]
        self.infection_curr += rec_jumps * self.nu[1]
        return inf_jumps

    def opinion_history(self):
        return self.opinion_array

    def infection_history(self):
        return self.infection_array
    
    def infection_num_history(self):
        return self.infection_num_array

def main():
    # Data loading
    start_date = datetime(2020, 3, 12)
    end_date = datetime(2020, 6, 26)
    infection_data = pd.read_csv("data/japan-covid-data.csv", index_col="date", parse_dates=["date"])
    opinion_data = pd.read_csv("data/zib/Opinion_data/survey_data_original_JPN.csv", index_col="Date", parse_dates=["Date"])
    infection_data = infection_data.loc[start_date:end_date]
    opinion_data = opinion_data.loc[start_date:end_date]
    
    # Total cases at the start date is used as initial number of infected for the simulation
    initial_infected = infection_data['total_cases'].iloc[0]
    # Reindex opinions to match the overall timescale
    opinion_data = opinion_data.reindex(infection_data.index)
    opinion_data = opinion_data.to_numpy()
    # Save indices where opinion data was recorded originally
    opinion_idx = np.where(~np.isnan(opinion_data).all(axis=1))[0]
    # Save indices where infection data is recorded (0 means not recorded)
    infection_idx = (infection_data["new_cases"] > 0).to_numpy().nonzero()[0]
    infection_data = (infection_data['new_cases']).to_numpy()
    
    # Parameters
    N = 300000
    sim_to_data_ratio = 3
    sim_length = len(opinion_data) * sim_to_data_ratio
    initial_opinion = opinion_sampler(opinion_data[0], N)
    params = (0, 0)
    
    model = opinion_dynamics(
        num_grid_points=sim_length,
        max_t=sim_length,
        initial_opinions=initial_opinion,
        N=N,
        y0=np.array([N - initial_infected, initial_infected, 0]),
        interaction_distance=0,
        noise_strength=0,
        stochiomatric_vectors=np.array([[-1, 1, 0], [0, -1, 1]]),
        grad_V="sigmoid",
        grad_V_params=params,
        inf_rate_max=0.4
    )
    model.algo()
    
    plt.hist(model.opinion_history()[0], bins=tuple(HIST_BINS))
    plt.savefig('plots/other/initial_opinions.png')
    plt.close()
    plt.hist(model.opinion_history()[-1], bins=tuple(HIST_BINS))
    plt.savefig('plots/other/final_opinions.png')
    plt.close()
    plt.plot(model.t, model.infection_history()[:, 1]/model.N)
    plt.title('Infection history (percentage of population)')
    plt.savefig('plots/other/infected_history.png')
    plt.close()
    plt.plot(model.t[infection_idx * sim_to_data_ratio], infection_data[infection_idx], label='Data')
    plt.plot(model.t, model.infection_num_history(), label='Simulated')
    plt.title('Daily new infections (data vs simulated)')
    plt.legend()
    plt.savefig('plots/other/daily_infected_history.png')
    plt.close()

if __name__ == '__main__':
    main()
