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

# Prova de control FORA DEL PROTOCOL (no forma part de l'estudi d'ablació ni del seu
# anàlisi). L'enunciat fixa els hiperparàmetres de [1] i τ = 0.001; amb ells TD3 complet
# no aprèn a aterrar dins T. Aquesta prova comprova que no és un error d'implementació:
# TD3 complet (mateixos interruptors que "full") amb canvis declarats.
#   td3_paper: hiperparàmetres de la taula 3 de [2] (τ = 0.005, lr 1e-3 a actor i crítics,
#              sense weight decay, batch 100, 10.000 passos inicials d'accions aleatòries).
#              L'arquitectura (amb LayerNorm) és la de l'esquelet, que no es pot canviar.
#   no_weight_decay: només treu el weight decay dels crítics (hipòtesi: la penalització
#              dels pesos del crític contribueix a la infraestimació i a planejar).
SANITY_CONFIGS = {
    "td3_paper": {"agent": {"tau": 0.005, "actor_lr": 1e-3, "critic_lr": 1e-3,
                            "critic_weight_decay": 0.0, "batch_size": 100},
                  "random_start_steps": 10_000},
    "no_weight_decay": {"agent": {"critic_weight_decay": 0.0}, "random_start_steps": 0},
}


# Segona ablació amb la configuració base de TD3 que el professor va recomanar després de
# publicar l'enunciat: τ = 0.005, sense weight decay als crítics, inicialització per
# defecte de PyTorch als crítics i targets dels crítics fora del retard (el de l'actor
# continua dins). La resta (xarxes, lr, batch, exploració, T) no canvia. Les variants
# prof_<nom> apliquen els mateixos interruptors de VARIANTS sobre aquesta base.
PROF_BASE = {"tau": 0.005, "critic_weight_decay": 0.0, "ddpg_critic_init": False,
             "critic_targets_every_step": True}
PROF_PREFIX = "prof_"
PROF_VARIANTS = {PROF_PREFIX + name: switches for name, switches in VARIANTS.items()}


def td3_settings(name: str) -> tuple:
    """(interruptors, arguments de l'agent, passos aleatoris inicials) d'una variant o prova de control."""
    if name in VARIANTS:
        return VARIANTS[name], dict(AGENT_KWARGS), 0
    if name in PROF_VARIANTS:
        return PROF_VARIANTS[name], {**AGENT_KWARGS, **PROF_BASE}, 0
    sanity = SANITY_CONFIGS[name]
    return VARIANTS["full"], {**AGENT_KWARGS, **sanity["agent"]}, sanity["random_start_steps"]


@dataclass
class ExperimentConfig:
    """Paràmetres d'un experiment; es desen sencers al config.json de cada execució."""

    total_steps: int = 300_000          # pressupost T en transicions d'entrenament
    seeds: tuple = tuple(range(10))     # mateixes llavors a totes les variants (fixat pel benchmark)
    eval_every: int = 1_000             # transicions entre avaluacions
    eval_episodes: int = 5              # fixat pel benchmark
    eval_seed_base: int = 10_000        # llavors d'avaluació comunes a totes les execucions
    checkpoint_fraction: float = 1 / 3  # checkpoint a la primera frontera d'episodi ≥ T/3
    checkpoint_every: int = 25_000      # checkpoint periòdic per reprendre després d'un tall
    bias_min_horizon: int = 300         # passos mínims per davant per fiar-se de G_t si l'episodi es trunca
    summary_last_n: int = 10            # mètrica resum pre-registrada: mitjana de les últimes N avaluacions
    random_start_steps: int = 0         # passos inicials amb accions uniformes (sense aprendre); 0 a l'estudi
    final_eval_episodes: int = 100      # avaluació final de la política en acabar (llavors noves)
    final_eval_seed_base: int = 20_000  # disjuntes de les d'avaluació periòdica
    results_dir: Path = field(default_factory=lambda: Path("results"))
    device: str = "cpu"                 # "cpu" força CPU (paral·lelització per processos)
    num_threads: int = 1

    def eval_seeds(self) -> list:
        """Llavors dels episodis d'avaluació (iguals a cada avaluació i a cada execució)."""
        return [self.eval_seed_base + i for i in range(self.eval_episodes)]

    def final_eval_seeds(self) -> list:
        """Llavors de l'avaluació final (una per episodi)."""
        return [self.final_eval_seed_base + i for i in range(self.final_eval_episodes)]
