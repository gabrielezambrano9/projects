"""
Bar chart for Time Consumption.
Reads data from 'times/training_times.csv', calculates the mean over 5 seeds
for the 3 algorithms, and generates a bar chart with error bars,
saved in the current script directory.
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    # Reading from the 'times' subdirectory
    csv_path = os.path.join('times', 'training_times.csv')

    times_data = {
        'reinforce_nobaseline': [],
        'reinforce_baseline20': [],
        'actor_critic': []
    }

    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                alg_tag = row['tag']
                # Strip the "_seedX" suffix to group them into the correct category
                base_tag = alg_tag.rsplit('_seed', 1)[0]

                if base_tag in times_data:
                    times_data[base_tag].append(float(row['time_min']))
    except FileNotFoundError:
        print(f"Error: The file {csv_path} does not exist. Please check the 'times/' directory!")
        return

    if not any(times_data.values()):
        print("Error: The CSV is empty or does not contain the correct tags.")
        return

    # Calculation of Means and Standard Deviations
    means = [
        np.mean(times_data['reinforce_nobaseline']),
        np.mean(times_data['reinforce_baseline20']),
        np.mean(times_data['actor_critic'])
    ]

    stds = [
        np.std(times_data['reinforce_nobaseline']),
        np.std(times_data['reinforce_baseline20']),
        np.std(times_data['actor_critic'])
    ]

    labels = ['REINFORCE\n(No Baseline)', 'REINFORCE\n(Baseline 20)', 'Actor-Critic']
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    # Bar Chart Creation
    plt.figure(figsize=(8, 6))

    bars = plt.bar(labels, means, yerr=stds, color=colors, capsize=8,
                   alpha=0.8, edgecolor='black', linewidth=1.2)

    # Adding the Mean above each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max(stds) * 0.2) + 0.5,
                 f"{yval:.1f} min", ha='center', va='bottom', fontweight='bold')

    # Formatting
    plt.ylabel('Average Training Time (Minutes)', fontsize=12)
    plt.title('Time Consumption: Computational Comparison (5 seeds)', fontsize=14, fontweight='bold', pad=15)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    plt.ylim(0, max(means) + max(stds) + 10)
    plt.tight_layout()

    # Saving
    out_path = 'time_consumption_plot.png'
    plt.savefig(out_path, dpi=300)
    print(f"\n[+] Chart calculated and saved in: {os.path.abspath(out_path)}")


if __name__ == '__main__':
    main()