"""Registre en disc d'una execució: config.json i CSV de mètriques (apartat 5).

Una execució = una carpeta. Una execució nova no pot reutilitzar una carpeta existent;
només es reobre per continuar-la des del seu propi checkpoint (reopen).
"""
import csv
import json
import os
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

    @classmethod
    def reopen(cls, run_dir: Path) -> "RunRecords":
        """Reobre una execució interrompuda per continuar-la des del seu checkpoint.

        Els CSV s'esborren perquè train() els reescriu des de l'historial del checkpoint:
        les files posteriors al checkpoint es tornaran a generar idènticament.
        """
        records = cls.__new__(cls)
        records.run_dir = Path(run_dir)
        records.config = json.loads((records.run_dir / CONFIG_JSON).read_text(encoding="utf-8"))
        records._fields = {}
        for name in (EVALUATIONS_CSV, EPISODES_CSV):
            (records.run_dir / name).unlink(missing_ok=True)
        return records

    def write_config(self, **updates) -> None:
        """Desa (o actualitza) la configuració completa de l'execució (escriptura atòmica)."""
        self.config.update(updates)
        tmp = self.run_dir / (CONFIG_JSON + ".tmp")
        tmp.write_text(json.dumps(self.config, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, self.run_dir / CONFIG_JSON)

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
