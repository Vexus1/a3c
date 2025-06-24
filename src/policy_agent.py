from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from src.wrappers import LazyFrames

def states_preprocessor(
    states: Sequence[np.ndarray | LazyFrames]
) -> torch.Tensor:
    """
    Converts a list of states (ndarray or LazyFrames) to a batched torch.Tensor.
    """
    if len(states) == 1:
        np_states = np.expand_dims(states[0], 0)
    else:
        np_states = np.asarray([np.asarray(s) for s in states])
    return torch.tensor(np_states)


class ProbabilityActionSelector():
    def __call__(self, probs: np.ndarray) -> np.ndarray:
        actions: list[int] = []
        for prob in probs:
            actions.append(np.random.choice(len(prob), p=prob))
        return np.array(actions)


class PolicyAgent:
    def __init__(
        self,
        model: torch.nn.Module,
        action_selector: ProbabilityActionSelector = ProbabilityActionSelector(),
        device: str = "cpu",
        apply_softmax: bool = False,
        preprocessor: callable = states_preprocessor
    ) -> None:
        self.model = model
        self.action_selector = action_selector
        self.device = device
        self.apply_softmax = apply_softmax
        self.preprocessor = preprocessor

    @torch.no_grad()
    def __call__(self,
                states: Sequence[np.ndarray | LazyFrames],
                agent_states: list | None = None
    ) -> tuple[np.ndarray, list]:
        if agent_states is None:
            agent_states = [None] * len(states)
        if self.preprocessor is not None:
            states = self.preprocessor(states)
            if torch.is_tensor(states):
                states = states.to(self.device)
        probs_v = self.model(states)
        if self.apply_softmax:
            probs_v = F.softmax(probs_v, dim=1)
        probs = probs_v.data.cpu().numpy()
        actions = self.action_selector(probs)
        return np.array(actions), agent_states
    