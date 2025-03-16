# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from src import utils
from src.envs import MouseFollowingCartPole

# %%
vec_env = make_vec_env(MouseFollowingCartPole, n_envs=1, env_kwargs={"max_episode_steps": 500, "render_mode": "rgb_array"})
vec_env = utils.RenderWrapper(vec_env)

model = PPO.load("ppo_mouse_following_1jveckpl.zip", device="cpu")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render()
