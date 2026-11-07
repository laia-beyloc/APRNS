"""Configuració experimental del laboratori (apartats 2 i 4 de l'enunciat).

Aquí només hi ha el que és experimental (pressupost, llavors, avaluació, variants...).
Els hiperparàmetres que l'esquelet fixa al codi donat (optimitzadors, τ, xarxes)
continuen on són.
"""
from dataclasses import dataclass, field
from pathlib import Path

ENV_ID = "LunarLander-v3"

# Mateixos valors que el bloc __main__ de l'esquelet (DDPG.py); el constructor
# té batch_size=32 per defecte, per això cal passar-los explícitament.
AGENT_KWARGS = {"replay_size": 1_000_000, "batch_size": 64, "gamma": 0.99}

# Apartat 4: cada variant canvia UN sol mecanisme respecte de TD3 complet.
# El codi de TD3 llegeix aquests interruptors, mai el nom de la variant.
VARIANTS = {
    "full":         {"use_minimum": True,  "target_noise": 0.2, "policy_delay": 2},
    "no_minimum":   {"use_minimum": False, "target_noise": 0.2, "policy_delay": 2},
    "no_smoothing": {"use_minimum": True,  "target_noise": 0.0, "policy_delay": 2},
    "no_delay":     {"use_minimum": True,  "target_noise": 0.2, "policy_delay": 1},
}

# Apartat 3.1.1: retall del soroll de la política objectiu (c = 0.5).
TARGET_NOISE_CLIP = 0.5

# σ de l'exploració gaussiana de TD3: valor de l'article de TD3 [2]; és el mateix
# a totes les variants perquè l'exploració no és un mecanisme ablacionat.
TD3_EXPLORATION_SIGMA = 0.1


@dataclass
class ExperimentConfig:
    """Paràmetres d'un experiment; es desen sencers al config.json de cada execució."""

    total_steps: int = 300_000          # pressupost T en transicions d'entrenament
    seeds: tuple = (0, 1, 2, 3, 4)      # mateixes llavors a totes les variants
    eval_every: int = 1_000             # transicions entre avaluacions
    eval_episodes: int = 5              # es fixarà després del benchmark
    eval_seed_base: int = 10_000        # llavors d'avaluació comunes a totes les execucions
    checkpoint_fraction: float = 1 / 3  # checkpoint a la primera frontera d'episodi ≥ T/3
    bias_min_horizon: int = 300         # passos mínims per davant per fiar-se de G_t si l'episodi es trunca
    summary_last_n: int = 10            # mètrica resum pre-registrada: mitjana de les últimes N avaluacions
    results_dir: Path = field(default_factory=lambda: Path("results"))
    device: str = "cpu"                 # "cpu" força CPU (paral·lelització per processos)
    num_threads: int = 1

    def eval_seeds(self) -> list:
        """Llavors dels episodis d'avaluació (iguals a cada avaluació i a cada execució)."""
        return [self.eval_seed_base + i for i in range(self.eval_episodes)]
