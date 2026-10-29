"""
MULTI-SEED Comparison (Mean ± Standard Deviation) on Subplots.
Plot 1: REINFORCE (Without and With Baseline)
Plot 2: Actor-Critic
Both share the same Y-axis scale for a direct visual comparison,
using the aggregated style (line + shaded area).
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_rewards(path):
    episodes, rewards = [], []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            episodes.append(int(row[0]))
            rewards.append(float(row[1]))
    return episodes, rewards


def moving_average(values, window):
    avg = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start:i + 1]
        avg.append(sum(chunk) / len(chunk))
    return avg


def main():
    # RUN Definition: (base_tag, label, color, subplot_index)
    # subplot_index: 0 (Left: REINFORCE), 1 (Right: Actor-Critic)
    RUNS = [
        ('reinforce_nobaseline', 'REINFORCE (No Baseline)', 'tab:blue', 0),
        ('reinforce_baseline20', 'REINFORCE (Baseline 20)', 'tab:orange', 0),
        ('actor_critic', 'Actor-Critic', 'tab:green', 1)
    ]

    SEEDS = [10, 20, 30, 40, 50]
    WINDOW = 100

    # Create the figure with 2 side-by-side subplots sharing the Y-axis
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), sharey=True)

    for base_tag, label, color, ax_idx in RUNS:
        ax = axes[ax_idx]
        all_smoothed = []

        for s in SEEDS:
            filename = f'rewards_{base_tag}_seed{s}.csv'
            path = os.path.join('rewards', filename)

            try:
                eps, rews = load_rewards(path)
                smooth_rews = moving_average(rews, WINDOW)
                all_smoothed.append((eps, smooth_rews))
            except FileNotFoundError:
                print(f"Warning: Missing file {path}")

        if not all_smoothed:
            continue

        # Safe length handling and statistical computation
        min_length = min(len(rews) for _, rews in all_smoothed)
        truncated_rews = [rews[:min_length] for _, rews in all_smoothed]
        all_smoothed_np = np.array(truncated_rews)
        mean_rewards = np.mean(all_smoothed_np, axis=0)
        std_rewards = np.std(all_smoothed_np, axis=0)
        mean_episodes = all_smoothed[0][0][:min_length]

        ax.plot(mean_episodes, mean_rewards, color=color, label=label, linewidth=2.5)
        ax.fill_between(mean_episodes,
                        mean_rewards - std_rewards,
                        mean_rewards + std_rewards,
                        color=color, alpha=0.2)


    # Titles for individual panels
    axes[0].set_title('REINFORCE Variants Comparison', fontsize=14, pad=15)
    axes[1].set_title('Actor-Critic Performance', fontsize=14, pad=15)

    # Shared settings for both panels
    for ax in axes:
        ax.set_xlabel('Episodes', fontsize=12)
        ax.grid(True, linestyle='-', alpha=0.3)
        ax.legend(loc='upper left', fontsize=11, framealpha=0.9)

    # Y-axis label only on the first panel (since it is shared)
    axes[0].set_ylabel(f'Average Reward (Moving Avg = {WINDOW})', fontsize=12)

    # Global Title for the entire image
    plt.suptitle('Multi-Seed Analysis: REINFORCE vs Actor-Critic (Mean ± Std Dev)',
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Save the plot in the current script directory (part1/)
    out_path = 'learning_curve_plots.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f'\nPlot successfully saved to: {os.path.abspath(out_path)}')


if __name__ == '__main__':
    main()