import numpy as np

HIST_BINS = np.array([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5])


def opinion_sampler(initial_opinion_data: np.ndarray, num_agents: int) -> np.ndarray:
    # Japan data is in 7 bins, we can force the opinions
    # Bound the opinions to be between -3.5 and 3.5, sample using initial opinion percentages
    edges = HIST_BINS
    probs = np.array(initial_opinion_data, dtype=float)
    probs = probs / probs.sum()
    sampled_opinions = np.random.choice(len(probs), size=num_agents, p=probs)
    left_edges = edges[sampled_opinions]
    right_edges = edges[sampled_opinions + 1]
    # Sample uniformly between the edges to get continuous opinions
    return np.random.uniform(left_edges, right_edges)
