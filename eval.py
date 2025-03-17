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
from stable_baselines3.common.vec_env import VecFrameStack
from src import utils
from src.envs import MouseFollowingCartPole
import wandb

# %%
run_id = "n1bod0k9"
vec_env = make_vec_env(MouseFollowingCartPole, n_envs=1, env_kwargs={"max_episode_steps": 200, "render_mode": "rgb_array"})
try:
    api = wandb.Api()
    run = api.run(f"cartpole/{run_id}")
    print(run.config)
    n_stack = run.config['n_stack']
    vec_env = VecFrameStack(vec_env, n_stack=n_stack)
except Exception as e:
    print(f"Could not load n_stack from wandb: {e}")
vec_env = utils.RenderWrapper(vec_env)

model = PPO.load(f"ppo_mouse_following_{run_id}.zip", device="cpu")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render()
