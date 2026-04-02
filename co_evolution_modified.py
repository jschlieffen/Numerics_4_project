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
        
        
        self.infection_curr = np.array(y0, dtype=int)
        self.infection_array = np.zeros((num_grid_points, 3), dtype=int)
        self.agent_state = np.zeros(len(initial_opinions), dtype=int) # (0, 1, 2) for (S, I, R)
        inital_inf_idx = np.random.choice(len(initial_opinions), size=self.infection_curr[1], replace=False)
        self.agent_state[inital_inf_idx] = 1
        self.bin_counts = np.zeros(len(HIST_BINS) - 1, dtype=int)
        self.infection_array[0] = np.copy(self.infection_curr)
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
                self.opinions_curr -= 4 * self.opinions_curr * (self.opinions_curr**2 + 1) - 8 * self.grad_V_params[1] * inf_func * self.opinions_curr**2 * self.dt
            
            self.opinions_curr = np.clip(self.opinions_curr, -3.5, 3.5)
            self.opinion_array[i] = np.copy(self.opinions_curr)

    def infection_propensity(self):
        bin_idx = np.clip((self.opinions_curr - HIST_BINS[0]).astype(int), 0, len(HIST_BINS) - 2)
        bin_susceptible_counts = np.bincount(bin_idx[self.agent_state == 0], minlength=len(HIST_BINS) - 1)
        self.bin_counts = np.bincount(bin_idx, minlength=len(HIST_BINS) - 1)
        bin_sum = np.bincount(bin_idx, weights=self.opinions_curr, minlength=len(HIST_BINS) - 1)
        bin_mean = bin_sum / np.maximum(self.bin_counts, 1)
        propensities = (self.inf_rate_max_plus_min - self.inf_rate_max_minus_min * np.tanh(2 * bin_mean)) / 2
        return bin_idx, propensities * bin_susceptible_counts * self.infection_curr[1] / self.N
    
    def recovery_propensity(self):
        return self.rec_rate * self.infection_curr[1]

    def update_infected(self) -> float:
        # Tau-leaping algorithm (variant of Gillepsie for fixed delta t)
        self.infection_curr = np.clip(self.infection_curr, 0, self.N)
        bin_idx, propensities = self.infection_propensity()
        infected_bins = np.random.poisson(propensities * self.dt)
        for bin_i in range(len(infected_bins)):
            if infected_bins[bin_i] > 0:
                # Randomly choose bin_i agents from the susceptible population to become infected
                candidates = np.flatnonzero((self.agent_state == 0) & (bin_idx == bin_i))
                chosen = np.random.choice(candidates, size=min(infected_bins[bin_i], len(candidates)), replace=False)
                self.agent_state[chosen] = 1
        inf_jumps = np.sum(infected_bins)
        rec_jumps = np.random.poisson(self.recovery_propensity() * self.dt)
        candidates = np.flatnonzero((self.agent_state == 1))
        chosen = np.random.choice(candidates, size=min(rec_jumps, len(candidates)), replace=False)
        self.agent_state[chosen] = 2
        # Consequence of using Tau-leaping is that recovery numbers could outgrow
        # number of infected, hence make all infected recover in the case that
        # more people have recovered than there are new infected + already infected
        self.infection_curr += inf_jumps * self.nu[0]
        self.infection_curr = np.clip(self.infection_curr, 0, self.N)
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
    initial_infected = infection_data.loc[start_date]["total_cases"]
    infection_data = infection_data.loc[start_date:end_date]
    opinion_data = pd.read_csv("data/zib/Opinion_data/survey_data_original_JPN.csv", index_col="Date", parse_dates=["Date"])
    opinion_data = opinion_data.loc[start_date:end_date]
    
    # Total cases at the start date is used as initial number of infected for the simulation
    # initial_infected = infection_data['total_cases'].iloc[0]
    # Reindex opinions to match the overall timescale
    opinion_data = opinion_data.reindex(infection_data.index)
    opinion_data = opinion_data.to_numpy()
    # Save indices where opinion data was recorded originally
    opinion_idx = np.where(~np.isnan(opinion_data).all(axis=1))[0]
    # Save indices where infection data is recorded (0 means not recorded)
    infection_idx = (infection_data["new_cases"] > 0).to_numpy().nonzero()[0]
    infection_data = (infection_data['new_cases']).to_numpy()
    
    # Parameters
    NUM_AGENTS = 125000
    sim_length = len(opinion_data)
    initial_opinion = opinion_sampler(opinion_data[0], NUM_AGENTS)
    params = (0, 0)
    print("Simulation parameters:")
    print(f"Number of agents: {NUM_AGENTS}")
    print(f"Length of simulation: {sim_length} time steps")

    model = opinion_dynamics(
        num_grid_points=sim_length,
        max_t=sim_length,
        initial_opinions=initial_opinion,
        N=NUM_AGENTS,
        y0=np.array([NUM_AGENTS - initial_infected, initial_infected, 0]),
        interaction_distance=0,
        noise_strength=0,
        stochiomatric_vectors=np.array([[-1, 1, 0], [0, -1, 1]]),
        grad_V="none",
        grad_V_params=params,
        inf_rate_max=0.3,
        inf_rate_min=0.1
    )
    model.algo()
    
    plt.hist(model.opinion_history()[0], bins=tuple(HIST_BINS))
    plt.savefig('plots/other/initial_opinions.png')
    plt.close()
    plt.hist(model.opinion_history()[-1], bins=tuple(HIST_BINS))
    plt.savefig('plots/other/final_opinions.png')
    plt.close()
    plt.plot(model.t, model.infection_history()[:, 0]/model.N)
    plt.title('Susceptible history (percentage of population)')
    plt.savefig('plots/other/susceptible_history.png')
    plt.close()
    plt.plot(model.t, model.infection_history()[:, 1]/model.N)
    plt.title('Infection history (percentage of population)')
    plt.savefig('plots/other/infected_history.png')
    plt.close()
    plt.plot(model.t, model.infection_history()[:, 2]/model.N)
    plt.title('Recovered history (percentage of population)')
    plt.savefig('plots/other/recovered_history.png')
    plt.close()
    plt.plot(model.t[infection_idx], infection_data[infection_idx], label='Data', ls="None", marker='o')
    plt.plot(model.t, model.infection_num_history(), label='Simulated')
    plt.title('Daily new infections (data vs simulated)')
    plt.legend()
    plt.savefig('plots/other/daily_infected_history.png')
    plt.close()

if __name__ == '__main__':
    main()
