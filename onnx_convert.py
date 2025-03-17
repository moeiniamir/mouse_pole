import torch as th
from typing import Tuple
import os

# Suppress thread affinity warnings
os.environ["OMP_NUM_THREADS"] = "1"

from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy


class OnnxableSB3Policy(th.nn.Module):
    def __init__(self, policy: BasePolicy):
        super().__init__()
        self.policy = policy

    def forward(self, observation: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        # NOTE: Preprocessing is included, but postprocessing
        # (clipping/inscaling actions) is not,
        # If needed, you also need to transpose the images so that they are channel first
        # use deterministic=False if you want to export the stochastic policy
        # policy() returns `actions, values, log_prob` for PPO
        return self.policy(observation, deterministic=True)


id = "m5qrhy1t"
model = PPO.load(f"ppo_mouse_following_{id}.zip", device="cpu")

onnx_policy = OnnxableSB3Policy(model.policy)

observation_size = model.observation_space.shape
dummy_input = th.randn(1, *observation_size)
th.onnx.export(
    onnx_policy,
    dummy_input,
    f"ppo_mouse_following_{id}.onnx",
    opset_version=17,
    input_names=["input"],
)

##### Load and test with onnx

import onnx
import onnxruntime as ort
import numpy as np

# Configure ONNX runtime to use a single thread
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 1
sess_options.inter_op_num_threads = 1

onnx_path = f"ppo_mouse_following_{id}.onnx"
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)

observation = np.zeros((1, *observation_size)).astype(np.float32)
ort_sess = ort.InferenceSession(onnx_path, sess_options=sess_options)
onnx_actions, onnx_values, onnx_log_prob = ort_sess.run(None, {"input": observation})

print("ONNX model outputs:", onnx_actions, onnx_values, onnx_log_prob)

# Check that the predictions are the same
with th.no_grad():
    torch_actions, torch_values, torch_log_prob = model.policy(th.as_tensor(observation), deterministic=True)
    print("PyTorch model outputs:", torch_actions.numpy(), torch_values.numpy(), torch_log_prob.numpy())
    
    # Compare outputs
    actions_match = np.allclose(onnx_actions, torch_actions.numpy())
    values_match = np.allclose(onnx_values, torch_values.numpy())
    log_prob_match = np.allclose(onnx_log_prob, torch_log_prob.numpy())
    
    print("\nOutputs match?")
    print(f"Actions: {actions_match}")
    print(f"Values: {values_match}")
    print(f"Log probabilities: {log_prob_match}")
