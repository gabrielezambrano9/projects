"""Genera un video di un modello SAC addestrato su PandaPush."""
import gymnasium as gym
import imageio
import panda_gym  # noqa: F401
from stable_baselines3 import SAC

MODEL_PATH = "sac_push_source_300k.zip"
N_EPISODES = 5
OUT_PATH = "sac_push_demo.gif"

env = gym.make("PandaPush-v3", render_mode="rgb_array",
               type="source", reward_type="dense")
model = SAC.load(MODEL_PATH)

frames = []
for ep in range(N_EPISODES):
    obs, info = env.reset()
    terminated = truncated = False
    while not (terminated or truncated):
        frames.append(env.render())
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
    frames.append(env.render())
    print(f"Episodio {ep + 1} completato")

env.close()

imageio.mimsave(OUT_PATH, frames, fps=20)
print(f"Video salvato in {OUT_PATH}")