import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joypy
from helpers import opinion_sampler


def opinion_joy_plot(opinion_data, opinion_simulation, opinion_idx):
    sim_records = []
    real_records = []
    for i in opinion_idx:
        sim_records.append(
            pd.Series(
                np.random.choice(opinion_simulation[i], size=20000, replace=False),
                name=f"Day {i}",
            )
        )
        real_records.append(
            pd.Series(opinion_sampler(opinion_data[i], 20000), name=f"Day {i}")
        )
    df1 = pd.concat(sim_records, axis=1)
    fig, axes = joypy.joyplot(df1, overlap=0.5, colormap=plt.cm.plasma)
    plt.savefig("plots/other/sim_opinion_joyplot.png")
    plt.close()
    df2 = pd.concat(real_records, axis=1)
    fig, axes = joypy.joyplot(df2, overlap=0.5, colormap=plt.cm.plasma)
    plt.savefig("plots/other/real_opinion_joyplot.png")
    plt.close()
