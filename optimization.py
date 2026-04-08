from scipy.optimize import differential_evolution
from scipy.stats import wasserstein_distance
from co_evolution_modified import opinion_dynamics
from helpers import opinion_sampler, HIST_BINS
import numpy as np
import pandas as pd
from itertools import repeat
from concurrent import futures
from datetime import datetime

NUM_WORKERS = 6
NUM_PATH_PER_WORKER = 3
NUM_AGENTS = 125000
OPIN_BINS = (-3, -2, -1, 0, 1, 2, 3)
MAX_INF_RATE = 0.3
MIN_INF_RATE = 0.1


def run_simulation(
    parameters: tuple[float, float],
    initial_opinion: np.ndarray,
    initial_infected: int,
    choice: str,
    max_t: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a single simulation with given parameters

    Args:
        parameters (tuple[float, float]): Tuple of parameters for parametrizing the model function for coupling infections through external potential
            For choice="sigmoid", we have model function(infected, opinion):
            For choice="polynomial", we have  I_crit=parameters[0] and a=parameters[1] (refer to model decribed in paper by Schütte)
        initial_opinion (np.ndarray): Inital opinion for each agent sampled from PDF
        initial_infected (int): Initial number of infected from data for simulation
        choice (str): Choice of coupling algorithm ("polynomial", "sigmoid")
        max_t (int): Number of time steps to simulate

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple of time-series of infection numbers (per timestep) and opinions over time
    """

    if choice == "sigmoid":
        # For sigmoid coupling, parameters are on log scale
        # Coefficient of sigmoid is negative while others two are positive
        parameters = tuple(
            np.array((-1, 1, 1)) * (10 ** np.array(parameters, dtype=float))
        )

    model = opinion_dynamics(
        num_grid_points=max_t,
        max_t=max_t,
        initial_opinions=initial_opinion,
        N=NUM_AGENTS,
        y0=np.array((NUM_AGENTS - initial_infected, initial_infected, 0)),
        interaction_distance=0,
        noise_strength=0,
        stochiomatric_vectors=np.array([[-1, 1, 0], [0, -1, 1]]),
        grad_V=choice,
        grad_V_params=parameters,
        inf_rate_max=MAX_INF_RATE,
        inf_rate_min=MIN_INF_RATE,
    )
    model.algo()
    return model.infection_num_history(), model.opinion_history()


def composite_loss(
    parameters: tuple[float, float],
    opinion_data: np.ndarray,
    opinion_idx: np.ndarray,
    infection_data: np.ndarray,
    infection_idx: np.ndarray,
    initial_opinion: np.ndarray,
    initial_infected: int,
    choice: str,
    number_of_simulations: int,
) -> tuple[float, float]:
    """Run multiple simulations and calculate total loss w.r.t. both opinions and infection numbers

    Args:
        opinion_data (np.ndarray): Opinion data for population (in discrete bins for mock data)
        opinion_idx (np.ndarray): Indices where opinion data is recorded
        infection_data (np.ndarray): Infection data for population (per million for mock data)
        infection_idx (np.ndarray): Indices where infection data is recorded
        initial_opinion (np.ndarray): Initial opinion for each agent sampled from PDF
        initial_infected (int): Initial number of infected from data for simulation
        choice (str): Choice of coupling algorithm ("linear", "polynomial", "tanh", "absolute", "none")
        number_of_simulations (int): Number of simulation per worker

    Returns:
        tuple[float, float]: Total scaled error from all simulation for this worker
    """
    total_opinion_err = 0
    total_infection_err = 0
    # Scale of opinions error is bounded by the support of opinions,
    # and scale of infection error is bounded by the mean of infections
    opinion_err_scale = HIST_BINS[-1] - HIST_BINS[0]
    # Two options for scaling the infection error, mean seems to dominate the opinion error, while max-min seems to give more balanced errors
    infection_err_scale = (
        infection_data[infection_idx].max() - infection_data[infection_idx].min()
    )
    # infection_err_scale = infection_data[infection_idx].mean()

    for _ in range(number_of_simulations):
        # Each simulation gives new_infected and opinions at each time step
        sim_infection, sim_opinion = run_simulation(
            parameters, initial_opinion, initial_infected, choice, len(infection_data)
        )

        # Discretize simulation opinions into bins and calculate wasserstein metric between actual opinions
        disc_opinions = [
            (np.histogram(sim_opinion[i], bins=HIST_BINS)[0]) / NUM_AGENTS
            for i in range(len(sim_opinion))
        ]
        opinion_err = np.mean(
            [
                wasserstein_distance(
                    OPIN_BINS,
                    OPIN_BINS,
                    u_weights=opinion_data[i],
                    v_weights=disc_opinions[i],
                )
                for i in opinion_idx
            ]
        )

        # Scale the daily new infected numbers to scale of actual infectino data (per million)
        # if the agent number is too small, then Poisson jump processes resulting in only discrete
        # number of cases will blow up this error because of the ratio below
        infection_err = np.mean(
            np.abs(sim_infection[infection_idx] - infection_data[infection_idx])
        )

        # Scale opinion error by bounded support of opinions, and infection error by mean of infections
        # Distinction into different errors is used for debugging
        total_opinion_err += opinion_err / opinion_err_scale
        total_infection_err += infection_err / infection_err_scale

    return total_infection_err, total_opinion_err


def loss(
    parameters,
    opinion_data,
    opinion_idx,
    infection_data,
    infection_idx,
    initial_infected,
    choice,
    pool,
):
    """Loss function that goes into optimizer while also taking care of parallelism of simulations

    Returns:
        float: Total loss from each worker after averaging over number of simulations per worker
    """
    # Approxmating contionus opinion data using histogram weights of initial opinions, each sample from histogram is
    # mapped to intervals (-3.5, -2.5), (-2.5, -1.5), (-1.5, -0.5), (-0.5, 0.5), (0.5, 1.5), (1.5, 2.5), (2.5, 3.5)
    initial_opinion = opinion_sampler(opinion_data[0], NUM_AGENTS)

    # Total error collected from each worker
    avg_opinion_err = 0
    avg_infection_err = 0

    for res in pool.map(
        composite_loss,
        repeat(parameters),
        repeat(opinion_data),
        repeat(opinion_idx),
        repeat(infection_data),
        repeat(infection_idx),
        repeat(initial_opinion),
        repeat(initial_infected),
        repeat(choice),
        repeat(NUM_PATH_PER_WORKER, NUM_WORKERS),
    ):
        inf_err, opin_err = res
        avg_infection_err += inf_err
        avg_opinion_err += opin_err

    avg_infection_err /= NUM_WORKERS * NUM_PATH_PER_WORKER
    avg_opinion_err /= NUM_WORKERS * NUM_PATH_PER_WORKER

    # Print parameter and error of each evaluation for progress tracking
    print(
        f"Parameters: {parameters}| Average Infection Error: {avg_infection_err:.5f} | Average Opinion Error: {avg_opinion_err:.5f}"
    )
    # return avg_infection_err + avg_opinion_err
    return avg_infection_err + avg_opinion_err


if __name__ == "__main__":
    # Perform the optimization over the parameters of the model function for coupling opinions with infections, and its gradient
    initial_guess = [-6.5, -2.5, -1.5]

    # Choose start and end dates for infection data and opinion data
    start_date = datetime(2020, 3, 12)
    end_date = datetime(2020, 6, 26)

    # TODO: March 2020 to June 2020 roughly covers the first peak of the pandemic
    # The peaks hereon have unreliable number of initial infected due to total cases being different than "active" cases
    # If other peaks are to be included, we can subtract the total cases "average recovery period" days ago from the starting date

    # Reading infection and opinion data, limiting dates to start_date and end_date, extracting initial number infected
    infection_data = pd.read_csv(
        "data/japan-covid-data.csv", index_col="date", parse_dates=["date"]
    )
    initial_infected = infection_data.loc[start_date]["total_cases"]
    infection_data = infection_data.loc[start_date:end_date]

    opinion_data = pd.read_csv(
        "data/zib/Opinion_data/survey_data_original_JPN.csv",
        index_col="Date",
        parse_dates=["Date"],
    )
    opinion_data = opinion_data.loc[start_date:end_date]
    opinion_data = opinion_data.reindex(
        infection_data.index
    )  # Reindex opinions to match the overall timescale of infections
    opinion_data = opinion_data.to_numpy()
    opinion_idx = np.where(~np.isnan(opinion_data).all(axis=1))[
        0
    ]  # Save indices where opinion data was recorded originally

    infection_idx = (
        (infection_data["new_cases"] > 0).to_numpy().nonzero()[0]
    )  # Save indices where infection data is recorded (0 means not recorded)
    infection_data = (infection_data["new_cases"]).to_numpy()

    # Choice for model function for coupling opinions with infections
    # choices = ["polynomial", "sigmoid"]
    # When one of the above choices is not selected, grad_V is set to 0 and opinions stay constant
    choice = "sigmoid"

    # Differential evolution convergence condition is determined by population energies (losses of population)
    # with formula np.std(population_energies) <= atol + tol * np.abs(np.mean(population_energies))
    # atol (absolute) and tol (relative) put a lower bounded on convergence condition
    with futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
        result = differential_evolution(
            func=loss,
            bounds=[(-8, -5), (-4, -1), (-3, 0)],
            x0=initial_guess,
            args=(
                opinion_data,
                opinion_idx,
                infection_data,
                infection_idx,
                initial_infected,
                choice,
                pool,
            ),
            strategy="best1bin",
            disp=True,
            polish=True,
            popsize=10,
            maxiter=20,
            init="latinhypercube",
            atol=1e-3,  # type: ignore[arg-type]
            tol=1e-3,
            workers=1,
        )
    print(result)
