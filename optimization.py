from scipy.optimize import differential_evolution, OptimizeResult
from scipy.stats import wasserstein_distance, gaussian_kde
from co_evolution_modified import opinion_dynamics
import numpy as np
import pandas as pd
from itertools import repeat
from concurrent import futures

global NUM_WORKERS; global NUM_PATH_PER_WORKER; global BINS
NUM_WORKERS = 6
NUM_PATH_PER_WORKER = 3
NUM_AGENTS = 100000
HIST_BINS = (-np.inf, -1.5, -0.5, 0.5, 1.5, np.inf)
OPIN_BINS = (-2, -1, 0, 1, 2)
INF_RATE = 0.35
INT_INFECTED = 100

def run_simulation(parameters: tuple[float, float, int], initial_opinion: np.ndarray, max_t:int) -> tuple[np.ndarray, np.ndarray]:
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
    grad_V = np.vectorize(lambda opinion, infected: infected * parameters[0])
    model = opinion_dynamics(
        num_grid_points=max_t,
        max_t=max_t,
        initial_opinions=initial_opinion,
        N=NUM_AGENTS,
        y0=np.array((NUM_AGENTS - parameters[2], parameters[2], 0)),
        interaction_distance=0,
        noise_strength=0,
        stochiomatric_vectors=np.array([[-1, 1, 0], [0, -1, 1]]),
        grad_V=grad_V
    )
    model.INF_RATE_MAX = parameters[1]
    model.algo()
    return model.infection_num_history(), model.opinion_history()

def composite_loss(parameters: tuple[float, float, int], opinion_data:np.ndarray, infection_data:np.ndarray,
                   initial_opinion:np.ndarray, number_of_simulations: int) -> float:
    """Run multiple simulations and calculate total loss w.r.t. both opinions and infection numbers

    Args:
        opinion_data (np.ndarray): Opinion data for population (in discrete bins for mock data)
        infection_data (np.ndarray): Infection data for population (per million for mock data)
        number_of_simulations (int): Number of simulation per worker

    Returns:
        float: Total scaled error from all simulation for this worker
    """

    total_err = 0
    total_opinion_err = 0
    total_infection_err = 0
    for _ in range(number_of_simulations):
        # Each simulation gives new_infected and opinions at each time step
        sim_infection, sim_opinion = run_simulation(parameters, initial_opinion, len(opinion_data))
        
        # Discretize simulation opinions into bins and calculate wasserstein metric between actual opinions
        disc_opinions = [(np.histogram(sim_opinion[i], bins=HIST_BINS)[0])/NUM_AGENTS for i in range(len(sim_opinion))]
        opinion_err = np.sum(
            [wasserstein_distance(OPIN_BINS, OPIN_BINS, u_weights=opinion_data[i], v_weights=disc_opinions[i]) for i in range(len(opinion_data))])
        
        # Scale the daily new infected numbers to scale of actual infectino data (per million)
        # TODO: Small point is for mock data the daily cases range from (30, 2300) meaning that
        # if the agent number is too small, then Poisson jump processes resulting in only discrete
        # number of cases will blow up this error because of the ratio below
        sim_infection = sim_infection * (1000000/NUM_AGENTS)
        infection_err = np.sum(np.abs(sim_infection - infection_data))
        
        # Scale each error by their maximum (theoretical maximum for Wasserstein and empiral max for scalar array)
        total_opinion_err += opinion_err/4
        total_infection_err += infection_err/np.max(infection_data)
        total_err +=  opinion_err/4 + infection_err/np.max(infection_data)
    print(f"Tested parameters: {parameters} with opinion error: {total_opinion_err} and infection error: {total_infection_err}")
    return total_err

def loss(parameters, opinion_data, infection_data, pool):
    """ Loss function that goes into optimizer while also taking care of parallelism of simulations

    Returns:
        float: Total loss from each worker
    """
    # Approxmating initial discrete opinions with Gaussian kernel,
    # and sampling opinion for each agent and clipping it to acceptable range
    weights = opinion_data[0]
    kde = gaussian_kde((-2, -1, 0, 1, 2), weights=weights, bw_method=0.75)
    initial_opinion = np.clip(kde.resample(NUM_AGENTS), -2, 2)[0]
    
    # Total error collected from each worker
    total_err = 0
    for res in pool.map(composite_loss, repeat(parameters), repeat(opinion_data),
                        repeat(infection_data), repeat(initial_opinion), repeat(NUM_PATH_PER_WORKER, NUM_WORKERS)):
        total_err += res
    return total_err

if __name__ == '__main__':
    # Initial guess for coefficent of model function, max infection rate and initial number of infected
    initial_guess = [0, 0.3, 100]
    
    # Reading mock data and converting them to numpy array for opinions and daily infections
    data = pd.read_csv("data/mock_data_with_cases.csv", index_col="Day")
    opinion_data = (data.drop(["Cases"], axis=1)).to_numpy()
    infection_data = (data['Cases']).to_numpy()
    
    with futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
        # Global minimization procedure, bounds on coefficient (first parameter) is arbitrary
        result = differential_evolution(func=loss, bounds=[(-100, 100), (0.2, 0.9), (10, NUM_AGENTS - 10)],
                                        x0=initial_guess, args=(opinion_data, infection_data, pool), strategy='randtobest1bin',
                                        popsize=5, maxiter=50, tol=0.05, workers=1)
    print(result)
        