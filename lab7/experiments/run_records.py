"""Registre en disc d'una execució: config.json i CSV de mètriques (apartat 5).

Una execució = una carpeta. La carpeta no pot existir prèviament: mai es
sobreescriuen mesures.
"""
import csv
import json
from pathlib import Path

EVALUATIONS_CSV = "evaluations.csv"
EPISODES_CSV = "train_episodes.csv"
CONFIG_JSON = "config.json"


class RunRecords:
    """Escriu les mesures d'una execució a mesura que es produeixen."""

    def __init__(self, run_dir: Path, config: dict):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.config = dict(config)
        self._fields = {}  # capçalera de cada CSV, fixada per la primera fila
        self.write_config()

    def write_config(self, **updates) -> None:
        """Desa (o actualitza) la configuració completa de l'execució."""
        self.config.update(updates)
        (self.run_dir / CONFIG_JSON).write_text(json.dumps(self.config, indent=2, default=str), encoding="utf-8")

    def log_evaluation(self, row: dict) -> None:
        """Afegeix una fila a evaluations.csv (una per avaluació)."""
        self._append(EVALUATIONS_CSV, row)

    def log_episode(self, row: dict) -> None:
        """Afegeix una fila a train_episodes.csv (una per episodi d'entrenament)."""
        self._append(EPISODES_CSV, row)

    def _append(self, name: str, row: dict) -> None:
        # S'obre i es tanca a cada fila perquè una execució interrompuda conservi
        # tot el que ja ha mesurat.
        is_new = name not in self._fields
        if is_new:
            self._fields[name] = list(row)
        with (self.run_dir / name).open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=self._fields[name], restval="")
            if is_new:
                writer.writeheader()
            writer.writerow(row)
