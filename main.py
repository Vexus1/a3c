import os 
import argparse
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

import torch
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp

from constants import *
from src.wrappers import wrap_env

def make_env() -> gym.Env:
    return wrap_env(gym.make(ENV_NAME))

@dataclass(frozen=True)
class TotalReward:
    reward: float




if __name__ == "__main__":
    pass
