# Execució de l'estudi en una VM Linux (p. ex. Azure)

Protocol fixat a `experiments/config.py`: 1 DDPG (llavor 0) + 4 variants TD3 × 10 llavors,
T = 300.000 transicions, 5 episodis d'avaluació cada 1.000 passos.

## Llançar
Des de `lab7/` a la VM (Ubuntu, CPU; triar una màquina amb ≥ 8 nuclis físics):
```bash
bash cloud/setup_linux.sh --workers 8
```
Instal·la `requirements.txt` (mateixes versions que en local), passa els tests i llança
`python -m experiments.launch` en segon pla amb `nohup`. Seguiment: `tail -f launch.out`.
Estat de cada execució: `python -m experiments.launch --dry-run`.

**Interrupció:** tornar a llançar la mateixa ordre; les execucions completades se salten.
Una carpeta incompleta bloqueja la seva execució (no es sobreescriu mai): moure-la fora de
`results/` i tornar a llançar.

## Portar els resultats al repo
```bash
rsync -a --ignore-existing --exclude '*.pt' --exclude '*.log' vm:APRNS/lab7/results/ lab7/results/
```
Els CSV/JSON de `results/` es lliuren i es versionen; els `.pt` i els logs no.

## Repartir entre màquines (només si cal)
`--seeds` i `--variants` (`ddpg`, `full`, `no_minimum`, `no_smoothing`, `no_delay`) permeten
que cada màquina faci un subconjunt disjunt. `config.json` registra dispositiu i versions;
si es barregen màquines, cal declarar-ho a l'informe.
