import gymnasium as gym
import matplotlib
# Set inline backend for Jupyter notebooks
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from typing import Optional, Union, Any, Dict, Tuple, List
from stable_baselines3.common.vec_env import VecEnvWrapper
import numpy as np
import os
from IPython.display import display, clear_output

class RenderWrapper(VecEnvWrapper):
    """
    A wrapper for stable-baselines3 vector environments that displays
    the environment in Jupyter notebooks.
    """
    
    def __init__(self, venv, figsize: tuple = (8, 8)):
        """
        Initialize the wrapper.
        
        Args:
            venv: The vector environment to wrap
            figsize: The size of the matplotlib figure
        """
        super().__init__(venv)
        self.figsize = figsize
        self.fig = None
        self.ax = None
    
    def reset(self) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Reset all environments and return initial observations and info.
        
        Returns:
            Tuple of initial observations and info dictionaries
        """
        return self.venv.reset()
    
    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Wait for the step taken with step_async() to finish and return step results.
        
        Returns:
            Tuple of observations, rewards, dones, truncated, and info dictionaries
        """
        return self.venv.step_wait()
    
    def render(self, mode: str = 'human'):
        """
        Render the environment and display in notebook.
        
        Args:
            mode: The render mode
            
        Returns:
            The rendered image
        """
        # Get the RGB array from the environment (first env in the vector)
        img = self.venv.render(mode='rgb_array')
        
        if mode == 'rgb_array':
            return img
        
        # Create figure and axes if they don't exist
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=self.figsize)
        
        # Display the image
        self.ax.clear()
        self.ax.imshow(img)
        self.ax.axis('off')
        plt.tight_layout()
        
        # Display in notebook
        clear_output(wait=True)
        display(self.fig)
        
        return img
    
    def close(self):
        """Close the environment and the matplotlib figure."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
        return self.venv.close()
