import argparse
import os
import random

import numpy as np
from stable_baselines3.common.callbacks import EvalCallback

from rand_wrapper import RandomizationWrapper
import gymnasium as gym
import panda_gym  # type: ignore[import-not-found]  # noqa: F401 - registra gli env Panda
from stable_baselines3 import PPO, SAC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO/SAC on PandaPush-v3")
    parser.add_argument("--algo", type=str, default="both",
                        choices=["ppo", "sac", "both"], help="RL algorithm")
    parser.add_argument("--env-type", type=str, default="source",
                        choices=["source", "target"], help="Type of environment")
    parser.add_argument("--timesteps", type=int, default=500_000,
                        help="training timesteps")
    parser.add_argument(
        "--sampling-strategy",
        type=str,
        default="none",
        choices=["none", "udr", "adr"],
        help="domain randomization strategy",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    all_algorithms = ["sac", "ppo"]

    if args.algo == 'sac':
        algorithms_to_train = ["sac"]
    elif args.algo == 'ppo':
        algorithms_to_train = ["ppo"]
    else:  # 'both'
        algorithms_to_train = all_algorithms

    seeds = [10, 20, 30]

    print(f" Type: {args.env_type} | Timesteps: {args.timesteps} | Randomization: {args.sampling_strategy}")

    for algo in algorithms_to_train:

        algo_dir = algo.upper()
        os.makedirs(algo_dir, exist_ok=True)

        for current_seed in seeds:

            np.random.seed(current_seed)
            random.seed(current_seed)

            # Creazione ambiente di Training
            env = gym.make(
                "PandaPush-v3",
                render_mode="rgb_array",
                type=args.env_type,
                reward_type="dense",
            )
            # Domain Randomization: avvolge l'env col wrapper
            if args.sampling_strategy != "none":
                env = RandomizationWrapper(
                    env,
                    mass_range=(0.5, 6.0),
                    mode=args.sampling_strategy,
                )
                print(f"Randomization: {args.sampling_strategy}, range=(0.5, 6.0)")

            print(f"Algoritmo: {algo.upper()}")
            print(f"Env type:  {args.env_type}")
            print(f"Timesteps: {args.timesteps}")

            if algo == "sac":
                model = SAC(
                     "MultiInputPolicy",
                    env,
                    learning_rate=0.001,
                    learning_starts=10000,
                    ent_coef="auto_0.1",
                    gamma=0.95,
                    batch_size=256,
                    buffer_size=1_000_000,
                    verbose=1,
                    seed = current_seed,
                )
            else:  # ppo
                model = PPO(
                    "MultiInputPolicy",
                    env,
                    learning_rate=0.0003,
                    n_steps=2048,
                    batch_size=256,
                    gamma=0.95,
                    ent_coef=0.01,
                    policy_kwargs=dict(net_arch=[256, 256, 256]),
                    verbose=1,
                    seed = current_seed,
                )


            # Crete a new env for evaluation
            eval_env = gym.make(
                "PandaPush-v3",
                render_mode="rgb_array",
                type=args.env_type,
                reward_type="dense",
            )

            # save log and best model
            log_dir = os.path.join(algo_dir, f"logs_{args.env_type}_{args.sampling_strategy}_seed{current_seed}")
            os.makedirs(log_dir, exist_ok=True)

            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=log_dir,
                log_path=log_dir,
                eval_freq=10_000,  # evaluation every 10'000 steps
                n_eval_episodes=10,  # mean return every 10 ep
                deterministic=True,  # takes best actions, without random walks
                render=False
            )

            model.learn(total_timesteps=args.timesteps, callback=eval_callback)

            save_name = f"{algo}_push_{args.env_type}_{args.sampling_strategy}_{args.timesteps // 1000}k_seed{current_seed}"
            save_path = os.path.join(algo_dir, save_name)
            model.save(save_path)

            print(f"Model saved in: {save_name}.zip")
            print(f"BEST MODEL (Record) is saved in: '{log_dir}/best_model.zip'")

            env.close()
            eval_env.close()

if __name__ == "__main__":
    main()