import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joypy
from helpers import opinion_sampler, HIST_BINS


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


def ridge_plot_ensemble_hist(
    opinion_histories,
    real_opinion_data,
    opinion_idx,
    colormap=plt.cm.plasma,  # type: ignore[arg-type]
    spacing=0.3,
):
    n = len(opinion_histories)
    colors = colormap(np.linspace(0.1, 0.9, len(opinion_idx)))
    bin_centres = (HIST_BINS[:-1] + HIST_BINS[1:]) / 2  # [-3, -2, -1, 0, 1, 2, 3]
    bin_width = HIST_BINS[1] - HIST_BINS[0]  # 1.0

    fig, (ax_sim, ax_real) = plt.subplots(1, 2, figsize=(6, len(opinion_idx) * 0.8 + 2))

    for ax, title in zip((ax_sim, ax_real), ("Simulated", "Real")):
        for rank, i in enumerate(reversed(opinion_idx)):
            offset = rank * spacing
            color = colors[-(rank + 1)]

            if ax == ax_sim:
                # Bin each trajectory back to HIST_BINS as percentages
                for traj in opinion_histories:
                    counts, _ = np.histogram(traj[-(rank + 1)], bins=HIST_BINS)
                    percentages = counts / counts.sum()
                    ax.bar(
                        bin_centres,
                        percentages,
                        width=bin_width * 0.9,
                        bottom=offset,
                        alpha=1 / n,
                        color=color,
                        align="center",
                    )
            else:
                # Real data is already percentages
                percentages = real_opinion_data[i] / real_opinion_data[i].sum()
                ax.bar(
                    bin_centres,
                    percentages,
                    width=bin_width * 0.9,
                    bottom=offset,
                    alpha=0.9,
                    color=color,
                    align="center",
                )

            ax.text(-3.8, offset, f"Day {i}", ha="right", va="bottom", fontsize=8)

        ax.set_xlim(-3.5, 3.5)
        ax.set_xticks(bin_centres)
        ax.set_xticklabels([str(int(c)) for c in bin_centres])
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.set_xlabel("Opinion")
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig("plots/other/opinion_ridge_hist.png", dpi=150)
    plt.close()


def plot_infection_histories(
    timescale,
    num_agents,
    infection_histories,  # list of (T, 3) arrays, one per trajectory
    daily_infection_histories,  # list of (T,) arrays, one per trajectory
    real_daily_infections,
    real_infection_idx,
):
    n = len(infection_histories)
    titles = ["Susceptible", "Infected", "Recovered"]
    colors = ["steelblue", "red", "green"]

    # SIR plots side by side
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for traj in infection_histories:
        for col, ax in zip(range(3), axes):
            ax.plot(
                timescale,
                traj[:, col] / num_agents,
                color=colors[col],
                alpha=1 / n,
                lw=1,
            )
    for col, ax in zip(range(3), axes):
        mean_traj = np.mean([t[:, col] for t in infection_histories], axis=0)
        ax.plot(timescale, mean_traj / num_agents, color=colors[col], lw=2)
        ax.set_title(f"{titles[col]} (% of population)")
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("plots/other/sir_history.png", dpi=150)
    plt.close()

    # Daily infections vs data
    fig, ax = plt.subplots(figsize=(8, 5))
    for daily in daily_infection_histories:
        ax.plot(timescale, daily, color="red", alpha=1 / n, lw=1)
    mean_daily = np.mean(daily_infection_histories, axis=0)
    ax.plot(timescale, mean_daily, color="red", lw=1, label="Simulated mean")
    ax.plot(
        timescale[real_infection_idx],
        real_daily_infections[real_infection_idx],
        ls="None",
        marker="o",
        color="red",
        label="Data",
    )
    ax.set_title("Daily new infections (data vs simulated)")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("plots/other/daily_infections.png", dpi=150)
    plt.close()
