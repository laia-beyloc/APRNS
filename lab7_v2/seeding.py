"""Llavors i estat dels generadors aleatoris globals (python, numpy, torch, cuda)."""
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Sembra tots els generadors globals; torch.manual_seed també sembra CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def capture_rng_state() -> dict:
    """Retorna una còpia de l'estat de tots els generadors globals."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict) -> None:
    """Restaura l'estat capturat amb capture_rng_state()."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
