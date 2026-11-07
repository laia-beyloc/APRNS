"""Memòria de repetició per a DDPG i TD3 (apartat 2.2.1: mètode off-policy)."""
import numpy as np


class ReplayBuffer:
    """Memòria circular de transicions (s, a, r, s', terminated, truncated).

    Les dades es guarden en arrays numpy reservats al primer put(): mostrejar és
    una indexació vectoritzada i l'estat es desa trivialment al checkpoint.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.size = 0
        self.position = 0
        self.arrays = None  # (obs, action, reward, next_obs, terminated, truncated)

    def _allocate(self, obs, action) -> None:
        obs_dim, act_dim = np.shape(obs), np.shape(action)
        self.arrays = (
            np.empty((self.capacity, *obs_dim), dtype=np.float32),
            np.empty((self.capacity, *act_dim), dtype=np.float32),
            np.empty(self.capacity, dtype=np.float32),
            np.empty((self.capacity, *obs_dim), dtype=np.float32),
            np.empty(self.capacity, dtype=np.float32),
            np.empty(self.capacity, dtype=np.float32),
        )

    def put(self, obs, action, reward, next_obs, terminated, truncated) -> None:
        """Afegeix una transició; quan és plena sobreescriu la més antiga."""
        if self.arrays is None:
            self._allocate(obs, action)
        for array, value in zip(self.arrays, (obs, action, reward, next_obs, terminated, truncated)):
            array[self.position] = value
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get(self, batch_size: int) -> tuple:
        """Mostreja un minibatch uniforme (amb reemplaçament).

        Retorna (s [B,obs], a [B,act], r [B], s' [B,obs], terminated [B], truncated [B]).
        Fa servir el generador global de numpy, que és el que sembren i desen
        seeding.py i el checkpoint.
        """
        idx = np.random.randint(0, self.size, size=batch_size)
        return tuple(array[idx] for array in self.arrays)

    def __len__(self) -> int:
        return self.size

    def state_dict(self) -> dict:
        """Estat serialitzable (només les files ocupades)."""
        arrays = None if self.arrays is None else [a[:self.size].copy() for a in self.arrays]
        return {"capacity": self.capacity, "size": self.size, "position": self.position, "arrays": arrays}

    def load_state_dict(self, state: dict) -> None:
        """Restaura l'estat desat amb state_dict()."""
        self.capacity, self.size, self.position = state["capacity"], state["size"], state["position"]
        self.arrays = None
        if state["arrays"] is not None:
            self.arrays = tuple(np.empty((self.capacity, *a.shape[1:]), dtype=a.dtype) for a in state["arrays"])
            for array, saved in zip(self.arrays, state["arrays"]):
                array[:self.size] = saved
