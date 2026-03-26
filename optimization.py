from scipy.optimize import differential_evolution, OptimizeResult
from scipy.stats import wasserstein_distance
from co_evolution_modified import opinion_dynamics
import numpy as np
import pandas as pd
from itertools import repeat
from concurrent import futures
from datetime import datetime

global NUM_WORKERS; global NUM_PATH_PER_WORKER; global BINS
NUM_WORKERS = 6
NUM_PATH_PER_WORKER = 1
NUM_AGENTS = 1000000
HIST_BINS = np.array([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5])
OPIN_BINS = (-3, -2, -1, 0, 1, 2, 3)
INF_RATE = 0.35

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

def run_simulation(parameters: tuple[float, float, int], initial_opinion: np.ndarray,
                   initial_infected: int, max_t: int) -> tuple[np.ndarray, np.ndarray]:
    """Run a single simulation with given parameters

    Args:
        parameters (tuple[float, float, int]): Tuple of parameters representing 
            the coefficient of model function for coupling opinions with infections,
            maximum infection rate for the simulation,
            initial number of infected for simulation
        initial_opinion (np.ndarray): Inital opinion for each agent sampled from PDF

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple of time-series of infection numbers (per timestep) and opinions over time
    """
    # TODO: Implement the actual model function for coupling opinions with infections, and its gradient
    grad_V = np.vectorize(lambda opinion, infected: 0)
    model = opinion_dynamics(
        num_grid_points=max_t,
        max_t=max_t,
        initial_opinions=initial_opinion,
        N=NUM_AGENTS,
        y0=np.array((NUM_AGENTS - initial_infected, initial_infected, 0)),
        interaction_distance=0,
        noise_strength=0,
        stochiomatric_vectors=np.array([[-1, 1, 0], [0, -1, 1]]),
        grad_V=grad_V,
        inf_rate_max=parameters[0]
    )
    model.algo()
    return model.infection_num_history(), model.opinion_history()

def composite_loss(parameters: tuple[float, float, int], opinion_data: np.ndarray, opinion_idx: np.ndarray, infection_data: np.ndarray,
                   infection_idx: np.ndarray, initial_opinion: np.ndarray,  initial_infected: int, number_of_simulations: int) -> float:
    """Run multiple simulations and calculate total loss w.r.t. both opinions and infection numbers

    Args:
        opinion_data (np.ndarray): Opinion data for population (in discrete bins for mock data)
        opinion_idx (np.ndarray): Indices where opinion data is recorded
        infection_data (np.ndarray): Infection data for population (per million for mock data)
        infection_idx (np.ndarray): Indices where infection data is recorded
        initial_opinion (np.ndarray): Initial opinion for each agent sampled from PDF
        initial_infected (int): Initial number of infected from data for simulation
        number_of_simulations (int): Number of simulation per worker

    Returns:
        float: Total scaled error from all simulation for this worker
    """
    total_err = 0
    total_opinion_err = 0
    total_infection_err = 0
    # Scale of opinions error is bounded by the support of opinions, 
    # and scale of infection error is bounded by the mean of infections
    opinion_err_scale = HIST_BINS[-1] - HIST_BINS[0]
    infection_err_scale = infection_data[infection_idx].max()
    
    for _ in range(number_of_simulations):
        # Each simulation gives new_infected and opinions at each time step
        sim_infection, sim_opinion = run_simulation(parameters, initial_opinion, initial_infected, len(opinion_data))
        
        # Discretize simulation opinions into bins and calculate wasserstein metric between actual opinions
        disc_opinions = [(np.histogram(sim_opinion[i], bins=HIST_BINS)[0])/NUM_AGENTS for i in range(len(sim_opinion))]
        opinion_err = np.mean(
            [wasserstein_distance(OPIN_BINS, OPIN_BINS, u_weights=opinion_data[i], v_weights=disc_opinions[i]) for i in opinion_idx])
        
        # Scale the daily new infected numbers to scale of actual infectino data (per million)
        # if the agent number is too small, then Poisson jump processes resulting in only discrete
        # number of cases will blow up this error because of the ratio below
        infection_err = np.mean(np.abs(sim_infection[infection_idx] - infection_data[infection_idx]))
        
        # Scale opinion error by bounded support of opinions, and infection error by mean of infections
        # Distinction into different errors is used for debugging
        total_opinion_err += opinion_err/opinion_err_scale
        total_infection_err += infection_err/infection_err_scale
        total_err +=  opinion_err/opinion_err_scale + infection_err/infection_err_scale
    # Print parameter and error of each evaluation for progress tracking
    print(f"Parameters: {parameters}| Infection Error: {total_infection_err} | Opinion Error: {total_opinion_err}")
    return total_err

def loss(parameters, opinion_data, opinion_idx, infection_data, infection_idx, initial_infected, pool):
    """ Loss function that goes into optimizer while also taking care of parallelism of simulations

    Returns:
        float: Total loss from each worker
    """
    # Approxmating contionus opinion data using histogram weights of initial opinions, each sample from histogram is
    # mapped to intervals (-3.5, -2.5), (-2.5, -1.5), (-1.5, -0.5), (-0.5, 0.5), (0.5, 1.5), (1.5, 2.5), (2.5, 3.5)
    initial_opinion = opinion_sampler(opinion_data[0], NUM_AGENTS)
    
    # Total error collected from each worker
    total_err = 0
    for res in pool.map(composite_loss, repeat(parameters), repeat(opinion_data), repeat(opinion_idx), repeat(infection_data),
                        repeat(infection_idx), repeat(initial_opinion), repeat(initial_infected), repeat(NUM_PATH_PER_WORKER, NUM_WORKERS)):
        total_err += res
    
    return total_err

if __name__ == '__main__':
    # TODO: Either use the fixed maximum infection rate of 0.35 or include it as a parameter to optimize
    # TODO: Perform the optimization over the parameters of the model function for coupling opinions with infections, and its gradient
    initial_guess = [0.35]
    
    # Choose start and end dates for infection data and opinion data
    start_date = datetime(2020, 3, 12)
    end_date = datetime(2020, 6, 26)
    
    # TODO: March 2020 to June 2020 roughly covers the first peak of the pandemic
    # The peaks hereon have unreliable number of initial infected due to total cases being different than "active" cases
    # If other peaks are to be included, we can subtract the total cases "average recovery period" days ago from the starting date
    
    # Reading infection and opinion data, limiting dates to start_date and end_date, extracting initial number infected
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
    
    with futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
        result = differential_evolution(func=loss, bounds=[(0.1, 1)], x0=initial_guess,
                                        args=(opinion_data, opinion_idx, infection_data, infection_idx, initial_infected, pool),
                                        strategy='best1bin', disp=True, popsize=3, maxiter=3, tol=0.1, workers=1)
    print(result)