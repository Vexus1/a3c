from typing import Sequence, Callable
from dataclasses import dataclass

import torch
import torch.multiprocessing as mp
import numpy as np
import gymnasium as gym

from src.models.a3c import A3C
from src.experience_source import ExperienceFirstLast, ExperienceSourceFirstLast
from src.policy_agent import PolicyAgent
from constants import GAMMA, REWARD_STEPS, MICRO_BATCH_SIZE, NUM_ENVS

@dataclass(frozen=True)
class TotalReward:
    reward: float


def unpack_batch(
    batch: Sequence[ExperienceFirstLast], 
    net: A3C, 
    last_val_gamma: float, 
    device: str = 'cpu'
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    states = []
    actions = []
    rewards = []
    not_done_idx = []
    last_states = []
    for idx, exp in enumerate(batch):
        states.append(np.asarray(exp.state))
        actions.append(int(exp.action))
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(idx)
            last_states.append(np.asarray(exp.last_state))
    states_v = torch.FloatTensor(np.asarray(states)).to(device)
    actions_t = torch.LongTensor(actions).to(device)
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        last_states_v = torch.FloatTensor(np.asarray(last_states)).to(device)
        last_vals_v = net(last_states_v)[1]
        last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
        rewards_np[not_done_idx] += last_val_gamma * last_vals_np
    ref_vals_v = torch.FloatTensor(rewards_np).to(device)
    return states_v, actions_t, ref_vals_v


def data_func(
    env: Callable[[], gym.Env], 
    net: A3C, 
    device: str, 
    train_queue: mp.Queue
) -> None:
    envs = [env() for _ in range(NUM_ENVS)]
    agent = PolicyAgent(lambda x: net(x)[0], device=device, apply_softmax=True)
    exp_source = ExperienceSourceFirstLast(
        envs, agent, gamma=GAMMA, steps_count=REWARD_STEPS)
    micro_batch = []
    for exp in exp_source:
        new_rewards = exp_source.pop_rewards_steps()
        if new_rewards:
            data = TotalReward(reward=np.mean(new_rewards))
            train_queue.put(data)
        micro_batch.append(exp)
        if len(micro_batch) < MICRO_BATCH_SIZE:
            continue
        data = unpack_batch(micro_batch, net, device=device,
                            last_val_gamma=GAMMA ** REWARD_STEPS)
        train_queue.put(data)
        micro_batch.clear()
