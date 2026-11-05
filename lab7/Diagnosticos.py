# Función: generar figuras, tablas e informe Markdown medidos; apartados 4 y 5.
# TODO (4, 5): agregar las cinco semillas y redactar el informe con resultados reales.
#______________________________
#PROPUESTA: agregación descriptiva y generación del informe; sin inferencias de significación.
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from Experimentos import lab_path, write_csv
from TD3 import VARIANTS


LABELS = {'full': 'TD3 completo', 'no_minimum': 'Sin mínimo',
          'no_smoothing': 'Sin smoothing', 'no_delay': 'Sin retraso'}


def read_runs(folders, seeds, budget):
    runs = {}
    for folder in folders:
        for config_path in lab_path(folder).glob('*/configuracion.json'):
            config = json.loads(config_path.read_text(encoding='utf-8'))
            key = (config['algorithm'], config['seed'])
            if key in runs:
                raise ValueError(f'Ejecución duplicada: {key}')
            if not config['completed']:
                raise ValueError(f'Ejecución incompleta: {config_path.parent.name}')
            if config['environment'] != 'LunarLander-v3':
                raise ValueError('El informe solo admite mediciones de LunarLander-v3.')
            if config['training_budget'] != budget or config['actual_transitions'] != budget:
                raise ValueError('Los presupuestos de entrenamiento deben coincidir.')
            if config['evaluation_interval'] != 1000 or config['evaluation_episodes'] != 5:
                raise ValueError('Se requieren evaluaciones de cinco episodios cada 1.000 pasos.')
            with (config_path.parent / 'evaluaciones.csv').open(encoding='utf-8', newline='') as stream:
                rows = list(csv.DictReader(stream))
            expected = list(range(1000, budget + 1, 1000))
            if budget % 1000:
                expected.append(budget)
            if [int(row['transitions']) for row in rows] != expected:
                raise ValueError('La serie contiene huecos o pasos duplicados.')
            runs[key] = (config, rows)
    required = [('ddpg', seeds[0])] + [(variant, seed) for variant in VARIANTS for seed in seeds]
    missing = [f'{variant}/semilla {seed}' for variant, seed in required if (variant, seed) not in runs]
    if missing:
        raise ValueError('Faltan mediciones: ' + ', '.join(missing))
    # Se verifica que ninguna variante haya cambiado hiperparámetros de control.
    shared = ('batch_size', 'replay_capacity', 'gamma', 'tau', 'actor_lr', 'critic_lr',
              'critic_weight_decay', 'architecture')
    td3_shared = ('exploration', 'exploration_sigma', 'target_noise_clip')
    reference = runs[('full', seeds[0])][0]
    for key in required:
        config = runs[key][0]
        if any(config[field] != reference[field] for field in shared):
            raise ValueError('Las arquitecturas o hiperparámetros de control difieren.')
        if key[0] != 'ddpg':
            if config['switches'] != VARIANTS[key[0]] or any(
                    config[field] != reference[field] for field in td3_shared):
                raise ValueError('Una ablación modifica más mecanismos de los permitidos.')
    return {key: runs[key] for key in required}


def generate(folders, seeds, budget=300000):
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError('El estudio solicitado utiliza cinco semillas distintas.')
    runs = read_runs(folders, seeds, budget)
    graphics = lab_path('graficas')
    graphics.mkdir(exist_ok=True)
    seed = seeds[0]
    plt.figure(figsize=(7, 3))
    for variant, label in [('ddpg', 'DDPG'), ('full', 'TD3 completo')]:
        rows = runs[(variant, seed)][1]
        plt.plot([int(row['transitions']) for row in rows],
                 [float(row['mean_return']) for row in rows], label=label)
    plt.xlabel('Transiciones de entrenamiento')
    plt.ylabel('Retorno medio de evaluación')
    plt.legend()
    plt.tight_layout()
    plt.savefig(graphics / 'comparacion.png', dpi=160)
    plt.close()

    summaries = []
    plt.figure(figsize=(7, 3))
    for variant in VARIANTS:
        records = [runs[(variant, seed)] for seed in seeds]
        values = np.array([[float(row['mean_return']) for row in rows] for _, rows in records])
        steps = np.array([int(row['transitions']) for row in records[0][1]])
        mean, std = values.mean(axis=0), values.std(axis=0, ddof=1)
        line, = plt.plot(steps, mean, label=LABELS[variant])
        plt.fill_between(steps, mean - std, mean + std, color=line.get_color(), alpha=0.12)
        # Se conserva una fila por semilla: cada tabla queda respaldada por mediciones.
        for config, rows in records:
            # En una bifurcación, el diagnóstico del mecanismo usa solo su tramo activo.
            active_rows = [row for row in rows if int(row['transitions']) > config['branch_step']]
            summaries.append({
                'variant': variant, 'seed': config['seed'], 'budget': budget,
                'final_return': float(rows[-1]['mean_return']),
                'mean_over_evaluations': float(np.mean([float(row['mean_return']) for row in rows])),
                'critic_updates': config['critic_updates'], 'actor_updates': config['actor_updates'],
                'mean_target_reduction': float(np.mean([float(row['target_min_reduction']) for row in active_rows])),
                'mean_action_perturbation': float(np.mean([float(row['target_action_perturbation']) for row in active_rows])),
                'mean_target_disagreement': float(np.mean([float(row['target_critic_disagreement']) for row in active_rows])),
            })
    plt.xlabel('Transiciones de entrenamiento')
    plt.ylabel('Retorno medio de evaluación')
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(graphics / 'ablaciones.png', dpi=160)
    plt.close()
    write_csv('resumen_mediciones.csv', summaries)

    groups = {variant: [row for row in summaries if row['variant'] == variant] for variant in VARIANTS}
    full_final = np.array([row['final_return'] for row in groups['full']])
    table = ['| Variante | Retorno final, media ± DE | Media temporal | Actor/crítico por ejecución |',
             '|---|---:|---:|---:|']
    for variant, group in groups.items():
        final = np.array([row['final_return'] for row in group])
        temporal = np.mean([row['mean_over_evaluations'] for row in group])
        counts = [f'{row["actor_updates"]}/{row["critic_updates"]}' for row in group]
        actor_counts = [row['actor_updates'] for row in group]
        critic_counts = [row['critic_updates'] for row in group]
        counters = (counts[0] if len(set(counts)) == 1 else
                    f'{min(actor_counts)}–{max(actor_counts)}/{min(critic_counts)}–{max(critic_counts)}')
        table.append(f'| {LABELS[variant]} | {final.mean():.1f} ± {final.std(ddof=1):.1f} | {temporal:.1f} | {counters} |')
    comparisons = []
    for variant in ('no_minimum', 'no_smoothing', 'no_delay'):
        delta = np.array([row['final_return'] for row in groups[variant]]) - full_final
        comparisons.append(f'{LABELS[variant]}: diferencia final emparejada frente a TD3 completo '
                           f'{delta.mean():+.1f} ± {delta.std(ddof=1):.1f} puntos (media ± DE entre semillas).')
    minimum = np.mean([row['mean_target_reduction'] for row in groups['full']])
    smoothing = np.mean([row['mean_action_perturbation'] for row in groups['full']])
    ddpg_final = float(runs[('ddpg', seeds[0])][1][-1]['mean_return'])
    td3_final = float(runs[('full', seeds[0])][1][-1]['mean_return'])
    branched = any(config['branch_step'] > 0 for config, _ in runs.values())
    branch_note = ('Se utilizaron bifurcaciones de checkpoints; el prefijo de cada semilla es compartido '
                   'y las curvas comparan intervenciones posteriores a esa frontera.' if branched else
                   'Cada variante se entrenó desde cero; no se compartieron prefijos de entrenamiento.')
    report = f'''<!-- Función: informe del apartado 5, generado exclusivamente desde CSV y JSON locales. -->
# Laboratorio 7: DDPG y TD3

## Protocolo e implementación

Se fijó T = {budget:,} transiciones de entrenamiento antes de ejecutar. La comparación exploratoria
usa una semilla por algoritmo ({seeds[0]}); las cuatro variantes TD3 usan las mismas cinco semillas
{seeds}. Cada 1.000 transiciones se evalúan cinco episodios completos, sin ruido y en otro entorno.
Las evaluaciones no consumen el presupuesto de entrenamiento ni modifican sus generadores aleatorios.
Las semillas de evaluación siguen `1000000 + 1000*semilla + índice_episodio`.
No se selecciona la mejor semilla ni se detiene por umbral de recompensa. {branch_note}

Se conservan las redes suministradas (400/300, LayerNorm, salida tanh del actor), con inicialización
del crítico mediante su método original. Se usan Adam, lr actor=0,0001, lr críticos=0,001,
weight decay críticos=0,01, gamma=0,99, tau=0,001, batch=64 y replay=1.000.000.
DDPG explora con OU (sigma=0,15, theta=0,2, dt=0,01); TD3 usa ruido gaussiano sigma=0,15
en las cuatro variantes. El ruido objetivo TD3 es sigma=0,2, recortado a ±0,5; las acciones a ±1.
Los detalles de dispositivo y versiones están en cada `configuracion.json`.
El tamaño de evaluación, la exploración gaussiana y la infraestructura son propuestas explícitas del código.

El crítico DDPG minimiza MSE(Q(s,a), r+gamma*(1-terminated)*Q_obj(s',actor_obj(s'))).
Las ecuaciones se contrastaron con Lecture5.pdf, diapositivas 70–73.
TD3 sustituye Q_obj por el mínimo de sus dos críticos y añade smoothing a la acción objetivo.
La truncación por tiempo conserva el bootstrap; la terminación real lo anula.
El actor maximiza Q1, también en las ablaciones. Ambos críticos aprenden siempre;
el actor y todas las redes objetivo se actualizan cada dos iteraciones, o cada una sin retraso.
Los checkpoints completos se guardan en una frontera de episodio próxima a T/3.
El replay conserva la implementación de Lab5; el ejecutor guarda y restaura sus atributos.

## Comparación exploratoria DDPG–TD3

![Retorno frente a transiciones, una semilla por algoritmo.](graficas/comparacion.png)

En T, DDPG obtuvo {ddpg_final:.1f} puntos y TD3 {td3_final:.1f}: diferencia TD3−DDPG
{td3_final-ddpg_final:+.1f}. Es una observación de una ejecución por algoritmo, sin conclusión estadística.

## Ablaciones controladas

![Media de cinco semillas; bandas de una desviación estándar muestral entre semillas.](graficas/ablaciones.png)

{chr(10).join(table)}

La media temporal promedia las evaluaciones cada 1.000 pasos, desde 1.000 hasta T.
Las columnas actor/crítico cuentan pasos de optimización (rango si varía entre semillas);
cada paso de crítico TD3 actualiza ambas redes.

{chr(10).join(comparisons)}

El mínimo redujo Q1 objetivo una media de {minimum:.4f} en TD3 completo; la reducción aplicada
es cero en «sin mínimo», aunque sus dos críticos siguen aprendiendo. El smoothing produjo una
perturbación absoluta media de {smoothing:.4f} por componente de acción; es cero en «sin smoothing».
Son diagnósticos de minibatches de hasta 256 transiciones, posteriores a la bifurcación si la hay;
no prueban que el objetivo sea insesgado
ni que estime con precisión el retorno real. Los contadores de la tabla verifican el cambio efectivo
de frecuencia en «sin retraso», manteniendo tau y tasas de aprendizaje.

Las diferencias describen estas cinco semillas y este presupuesto. Las bandas muestran variabilidad,
no intervalos de confianza. No se atribuye una diferencia de rendimiento a un mecanismo sin considerar
su dispersión. Los retornos por episodio de evaluación, pérdidas y diagnósticos se conservan en los CSV.

## Reproducción y referencias

`python Experimentos.py` genera las 21 ejecuciones; `python Diagnosticos.py` genera este Markdown,
las dos figuras y `resumen_mediciones.csv`. Se adjuntan código, CSV y JSON; los checkpoints quedan locales.
Mantener el informe dentro de cuatro páginas al convertirlo a PDF (referencias excluidas).

1. Lab_7___APRNS_GIA.pdf, apartados 2–5 y figuras 1 y 5.
2. Lecture5.pdf, diapositivas 70–73 (páginas PDF 86–89).
3. Lillicrap et al., Continuous control with deep reinforcement learning, ICLR 2016. Referencia [1] del enunciado.
4. Fujimoto et al., Addressing Function Approximation Error in Actor-Critic Methods, ICML 2018. Referencia [2] del enunciado.
'''
    lab_path('Informe.md').write_text(report, encoding='utf-8')
    print('Generados Informe.md, graficas/ y resumen_mediciones.csv con mediciones verificadas.')


def main():
    parser = argparse.ArgumentParser(description='Informe Markdown con cinco semillas completas por variante.')
    parser.add_argument('--input', nargs='+', default=['resultados'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument('--budget', type=int, default=300000)
    args = parser.parse_args()
    generate(args.input, args.seeds, args.budget)


if __name__ == '__main__':
    main()
#______________________________
# END TODO (4, 5)
