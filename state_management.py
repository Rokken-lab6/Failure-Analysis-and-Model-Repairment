import torch
import torch.nn as nn
import torch.nn.functional as F

def reset_state(state):
    if state is None:
        return
    state = {}
    return state

def detach_state(state):
    if state is None:
        return
    for k in state:
        state[k] = state[k].detach()
    return state
