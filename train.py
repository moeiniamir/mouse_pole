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
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from src.envs import MouseFollowingCartPole
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
import wandb
from wandb.integration.sb3 import WandbCallback
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import torch
OmegaConf.register_new_resolver("eval", eval)

# %%
@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Initialize wandb with the Hydra config
    
    # Only initialize wandb on the main process
    if os.environ.get("RANK", "0") == "0":
        run = wandb.init(
            project="cartpole",
            config=OmegaConf.to_container(cfg, resolve=True),
            sync_tensorboard=True,
            monitor_gym=False,
            save_code=True,
            mode="disabled" if not cfg.wandb.enabled else "online",
        )
    else:
        run = None
    
    # Parallel environments
    vec_env = make_vec_env(
        MouseFollowingCartPole, 
        n_envs=cfg.n_envs, 
        env_kwargs={"max_episode_steps": cfg.max_episode_steps}, 
        vec_env_cls=SubprocVecEnv
    )
    
    # Wrap with VecFrameStack to stack observations
    vec_env = VecFrameStack(vec_env, n_stack=cfg.n_stack)

    # Handle net_arch parameter - ensure it's a list
    net_arch = OmegaConf.to_container(cfg.policy_kwargs.net_arch)
    assert isinstance(net_arch, list), "net_arch must be a list"
    
    # Create policy_kwargs dictionary
    policy_kwargs = {"net_arch": net_arch, "activation_fn": torch.nn.SiLU}


    model = PPO(
        cfg.policy_type, 
        vec_env, 
        verbose=1, 
        device="cuda", 
        tensorboard_log=f"runs/{wandb.run.id}",
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        n_epochs=cfg.n_epochs,
        policy_kwargs=policy_kwargs,
        )
    model.learn(
        total_timesteps=cfg.total_timesteps, 
        callback=WandbCallback(),
        )
    
    # Save model in the original working directory
    model_path = os.path.join(hydra.utils.get_original_cwd(), f"ppo_mouse_following_{wandb.run.id}")
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    if os.environ.get("RANK", "0") == "0":
        run.finish()

if __name__ == "__main__":
    main()
