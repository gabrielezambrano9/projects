"""
SEQUENTIAL and MULTI-SEED Training on the Hopper-v4 Environment

This script executes the training of 3 algorithms sequentially:
1. REINFORCE (Without Baseline)
2. REINFORCE (With Baseline = 20.0)
3. ACTOR-CRITIC (1-Step TD)

Each algorithm is evaluated across 5 different random seeds (15 sequential runs total).
Models will be saved in 'models/', data in 'rewards/',
and execution runtimes in 'times/'.
"""
import os
import argparse
import csv
import random
import time
import numpy as np
import torch
import gymnasium as gym

from agent import Agent, Policy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=10000,
                        help='Number of training episodes per run')
    parser.add_argument('--print-every', type=int, default=100,
                        help='Interval of episodes to print the average reward')
    parser.add_argument('--device', type=str, default='cpu',
                        help='cpu or cuda')
    parser.add_argument('--algo', type=str, default='both', choices=['reinforce', 'ac', 'both'],
                        help='Choose which algorithm to train: reinforce, ac, or both')
    return parser.parse_args()


def set_seed(env, seed):
    """Freezes all random number generators to ensure reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    env.action_space.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # Placeholder environment for the initial configuration log
    env = gym.make('Hopper-v4')
    print('State space:', env.observation_space)  # state-space
    print('Action space:', env.action_space)  # action-space
    env.close()

    args = parse_args()

    # Automatically create all required logging directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('rewards', exist_ok=True)
    os.makedirs('times', exist_ok=True)

    # The 3 architectures required for the complete experiment
    all_algorithms = [
        {'algorithm': 'reinforce',    'baseline': None, 'base_tag': 'reinforce_nobaseline'},
        {'algorithm': 'reinforce',    'baseline': 20.0, 'base_tag': 'reinforce_baseline20'},
        {'algorithm': 'actor_critic', 'baseline': None, 'base_tag': 'actor_critic'}
    ]

    # Filter based on user choice
    if args.algo == 'reinforce':
        base_algorithms = [alg for alg in all_algorithms if alg['algorithm'] == 'reinforce']
    elif args.algo == 'ac':
        base_algorithms = [alg for alg in all_algorithms if alg['algorithm'] == 'actor_critic']
    else:
        base_algorithms = all_algorithms

    seeds_to_test = [10, 20, 30, 40, 50]

    # Generate all configurations
    configurations = []
    for alg in base_algorithms:
        for s in seeds_to_test:
            configurations.append({
                'algorithm': alg['algorithm'],
                'baseline': alg['baseline'],
                'seed': s,
                'tag': f"{alg['base_tag']}_seed{s}"
            })

    global_start_time = time.time()

    times_results = []

    for config in configurations:
        algorithm = config['algorithm']
        baseline = config['baseline']
        current_seed = config['seed']
        tag = config['tag']

        log_prefix = f"[{tag.upper():<28}]"
        print(f"\n{log_prefix} RUN STARTING | {algorithm} (Baseline: {baseline}) | Seed: {current_seed}")

        # Start timer for the current single run
        run_start_time = time.time()

        env = gym.make('Hopper-v4')
        set_seed(env, current_seed)

        observation_space_dim = env.observation_space.shape[-1]
        action_space_dim = env.action_space.shape[-1]

        # Initialize policy and agent for this specific run
        policy = Policy(observation_space_dim, action_space_dim)
        agent = Agent(policy, device=args.device, baseline=baseline)
        agent.algorithm = algorithm

        reward_history = []
        best_avg = float('-inf')
        best_path = os.path.join('models', f'{tag}_best.mdl')

        for episode in range(args.episodes):
            done = False

            # Synchronize environment with the specific seed on the first reset
            if episode == 0:
                state, _ = env.reset(seed=current_seed)
            else:
                state, _ = env.reset()

            episode_reward = 0

            while not done:
                action, action_log_prob = agent.get_action(state)
                previous_state = state

                state, reward, terminated, truncated, _ = env.step(action.detach().cpu().numpy())
                done = terminated or truncated

                agent.store_outcome(previous_state, state, action_log_prob, reward, done)
                episode_reward += reward

                # TEMPORAL DIFFERENCE: Step-by-step weight updates for Actor-Critic
                if agent.algorithm == 'actor_critic':
                    agent.update_policy()

            # MONTE CARLO: End of episode policy optimization for REINFORCE
            if agent.algorithm == 'reinforce':
                agent.update_policy()

            reward_history.append(episode_reward)

            window_size = 100
            if len(reward_history) >= window_size:
                current_avg = sum(reward_history[-window_size:]) / window_size

                if current_avg > best_avg:
                    best_avg = current_avg
                    torch.save(agent.policy.state_dict(), best_path)

            if (episode + 1) % args.print_every == 0:
                print_avg = sum(reward_history[-args.print_every:]) / args.print_every
                print(f"{log_prefix} Episode {episode + 1:5d} | Avg Reward: {print_avg:.2f} | Record: {best_avg:.2f}")

        # Save final trained network weights for the run
        save_path = os.path.join('models', f'{tag}.mdl')
        torch.save(agent.policy.state_dict(), save_path)


        csv_path = os.path.join('rewards', f'rewards_{tag}.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'reward'])
            for i, r in enumerate(reward_history):
                writer.writerow([i + 1, r])

        # Calculate and display run performance and duration metrics
        run_end_time = time.time()
        run_duration_min = (run_end_time - run_start_time) / 60

        print(f"{log_prefix} RUN COMPLETED in {run_duration_min:.1f} min. Max Avg Reward = {best_avg:.2f}")


        times_results.append([tag, algorithm, current_seed, run_duration_min])

        env.close()

    # export runtime analysis to 'times/training_times.csv' =====
    times_csv_path = os.path.join('times', 'training_times.csv')
    with open(times_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['tag', 'algorithm', 'seed', 'time_min'])
        writer.writerows(times_results)

    print(f"\n[+] Runtime data tracking sheet successfully saved to: {times_csv_path}")

    # Process and format full benchmark duration metrics
    global_end_time = time.time()
    total_duration_sec = global_end_time - global_start_time
    hours, remainder = divmod(total_duration_sec, 3600)
    minutes, seconds = divmod(remainder, 60)


if __name__ == '__main__':
    main()