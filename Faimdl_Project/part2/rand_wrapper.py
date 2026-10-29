import gymnasium as gym
import numpy as np
import random

class RandomizationWrapper(gym.Wrapper):
    """
    Wrapper to apply Domain Randomization to cube mass.
    Types: "none", "udr", "adr".
    """

    def __init__(
        self,
        env,
        mass_range=(1.0, 1.0),
        mode="none",
        source_mass=1.0,
        adr_step=0.1,
        adr_perf_threshold=-2.0,
    ):
        super().__init__(env)
        self.mode = mode
        self.mass_range = mass_range
        self.mass_min_limit, self.mass_max_limit = mass_range

        # Range of sampling
        # UDR: fixed = global range.  ADR: starts narrow and then it expands.
        if self.mode == "adr":
            self.mass_min = source_mass
            self.mass_max = source_mass
        else:
            self.mass_min = self.mass_min_limit
            self.mass_max = self.mass_max_limit

        # ADR
        self.adr_step = adr_step
        self.adr_perf_threshold = adr_perf_threshold

        # Internal state
        self.last_sample_type = "none"
        self.episode_return = 0.0

    # Mass Sampling
    def _sample_mass(self):
        if self.mode == "none":
            return None

        elif self.mode == "udr":
            self.last_sample_type = "uniform"
            return np.random.uniform(self.mass_min, self.mass_max)

        elif self.mode == "adr":
            if np.random.rand() < 0.5:
                self.last_sample_type = "boundary"
                return self.mass_min if np.random.rand() < 0.5 else self.mass_max
            else:
                self.last_sample_type = "uniform"
                return np.random.uniform(self.mass_min, self.mass_max)

        else:
            raise NotImplementedError(
                f"Sampling strategy '{self.mode}' is not implemented yet."
            )


    # Step
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_return += reward

        # ADR: if the agent performs well, widen the domain
        if (terminated or truncated) and self.mode == "adr":
            if self.episode_return > self.adr_perf_threshold:
                self.mass_min = max(
                    self.mass_min_limit, self.mass_min - self.adr_step
                )
                self.mass_max = min(
                    self.mass_max_limit, self.mass_max + self.adr_step
                )

        return obs, reward, terminated, truncated, info

    # Reset
    def reset(self, **kwargs):
        self.episode_return = 0.0

        new_mass = self._sample_mass()

        if new_mass is not None:
            sim = self.env.unwrapped.task.sim
            object_body_id = sim._bodies_idx["object"]

            sim.physics_client.changeDynamics(
                bodyUniqueId=object_body_id,
                linkIndex=-1,
                mass=float(new_mass),
            )

            print(
                f"[{self.mode}] mass={new_mass:.2f} "
                f"range=[{self.mass_min:.2f},{self.mass_max:.2f}] "
                f"type={self.last_sample_type}"
            )

        return super().reset(**kwargs)