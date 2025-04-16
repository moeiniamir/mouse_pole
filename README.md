# Mouse Following CartPole

`CartPole-v1` but the cart follows the mouse. Demo on my [personal webpage](https://moeiniamir.github.io/).

## Installation

Install from `requirements.txt`

```bash
uv pip install -r requirements.txt
```

## Training

Sweeps use `hydra` with `ray` as the multi-process launcher. Below is an example sweep command. Note that the vectorized environment is also multi-process, so the number of CPUs for `ray.remote` should be set accordingly.

```bash
python train.py --multirun \
  hydra/launcher=ray \
  +hydra.launcher.ray.init.num_cpus=64 \
  +hydra.launcher.ray.init.num_gpus=2 \
  +hydra.launcher.ray.remote.num_cpus=8 \
  +hydra.launcher.ray.remote.num_gpus=0.25 \
  n_steps=32\
  n_epochs=10,15 \
  policy_kwargs.net_arch=[256,256],[512,512],[64,64,64],[128,128,128],[256,128,64] \
  n_stack=3,5,8
```

## Evaluation

`eval.py` is written to be used as a jupyter notebook. Rendering is done with `rgb_array` mode to allow rendering over ssh tunnels.
