<!-- Función: guía de ejecución y entrega; apartados 2–5 del laboratorio. -->
# Ejecución del laboratorio 7

Todo el trabajo está en `Lab7`. En `DDPG.py` solo se han añadido bloques dentro de los TODO,
entre los separadores solicitados. Los comentarios y las líneas originales se conservan.
`ActorCriticNetworks.py`, `Noise.py`, `helper.py` y `RunDDPG_Colab.ipynb` permanecen intactos.

## Archivos y referencias

| Archivo | Función | Apartado |
|---|---|---|
| `ReplayBuffer.py` | Implementación original de Lab5: lista circular y método get | 2.2.1 |
| `DDPG.py` | Redes, acción, MSBE, pérdida del actor y actualizaciones | 2.2.2–7 |
| `TD3.py` | Adaptación de DDPG, dos críticos y cuatro variantes | 3–4 |
| `Experimentos.py` | Presupuesto fijo, evaluación, CSV/JSON, checkpoints | 2, 4–5 |
| `Diagnosticos.py` | Gráficas y generación de `Informe.md` desde mediciones | 4–5 |
| `Verificar.py` | Pruebas de ecuaciones, interruptores y reanudación | Validación |
| `Ejecutar_Lab7_Colab.ipynb` | Instalación y ejecución en Colab | Ejecución |
| `Informe.md` | Informe en Markdown; comienza como borrador sin resultados | 5 |
| `figura_ddpg.png`, `figura_td3.png` | Pseudocódigos extraídos del PDF, figuras 1 y 5 | Referencia |

Las ecuaciones se basan en las figuras 1 y 5 y los apartados 2–4 del PDF, contrastadas con
`Lecture5.pdf`, diapositivas 70–73 (páginas PDF 86–89).
`#PROPUESTA` identifica decisiones sin pseudocódigo en el material disponible, incluida la organización
de TD3 mediante herencia como copia funcional de DDPG, sin duplicar ni modificar sus líneas originales.
La diapositiva 72 incluye el pseudocódigo DDPG; la 71, sus pérdidas, y la 73, las tres mejoras TD3.
Para el retraso se usa el valor exacto 2 que pide el laboratorio, aunque la diapositiva menciona 2 o 3.
La revisión del pseudocódigo también aplica el reinicio de OU entre episodios a `DDPG.learn`,
dentro de su TODO (7); el ejecutor de experimentos ya reiniciaba ese proceso.

`ReplayBuffer.py` conserva la implementación original de Lab5 que has elegido, sin validaciones,
copias defensivas, alias ni métodos adicionales. DDPG y el ejecutor utilizan directamente `get`.
El guardado y la restauración de `capacity`, `position` y `buffer` para el apartado 4 se realizan
en `Experimentos.py`, conservando la clase original. Cada carga del checkpoint crea una memoria
independiente para las ablaciones. Lab7 no depende de importar archivos de Lab5 durante la ejecución.

## CPU, Intel Iris Xe y Colab

La comprobación local encontró Python 3.13, PyTorch `2.12.1+cpu`, `cuda.is_available() == False`
y Box2D ausente. Con la selección de dispositivo del código suministrado, este equipo ejecuta las
redes en **CPU**. La Intel Iris Xe no se utiliza por la ruta CUDA de este programa. La CPU permite
ejecutarlo si se instala Box2D; no se ha medido aquí cuánto tardaría el estudio completo.

Para los entrenamientos largos, sigue la recomendación del PDF y utiliza **Google Colab con GPU**.
Abre `Ejecutar_Lab7_Colab.ipynb` en Colab, selecciona un entorno de ejecución con GPU y ejecuta
las celdas en orden. La celda de comprobación muestra si hay GPU CUDA realmente asignada.
Sube únicamente los `.py` de `Lab7` cuando lo solicite el cuaderno. Conserva copias de resultados
y checkpoints antes de cerrar el entorno: el almacenamiento de la sesión es temporal.

El programa aprovecha la GPU para las redes y sus minibatches. La simulación Box2D y las
evaluaciones siguen siendo secuenciales dentro de cada ejecución. Para paralelizar el estudio,
distribuye **variantes o semillas independientes** entre recursos de ejecución que tengas disponibles.
Cada proceso necesita una carpeta de salida diferente. No hay que cambiar el algoritmo ni el batch
para paralelizar. Varios procesos sobre una misma GPU comparten sus recursos; eso no asegura acelerar.
No se ha lanzado ninguna sesión remota ni ningún entrenamiento largo desde este entorno.

## Ejecutar el estudio completo

Desde una terminal situada en `Lab7`, con `torch`, `numpy`, `matplotlib` y `gymnasium[box2d]` instalados:

```powershell
# TODO (ejecución): validar y lanzar las mediciones del laboratorio.
#______________________________
python Verificar.py
python Experimentos.py
python Diagnosticos.py
#______________________________
```

El ejecutor usa por defecto **cinco semillas independientes**, `0 1 2 3 4`, por variante TD3.
Son cinco ejecuciones completas, no cinco pasos ni cinco episodios. Cada ejecución tiene
**T = 300.000 transiciones de entrenamiento**, batch 64 y evaluación cada 1.000 pasos.
También se entrena DDPG con la semilla 0; la ejecución TD3 completo/semilla 0 se reutiliza
en la comparación exploratoria. En total son **21 ejecuciones y 6.300.000 transiciones**,
además de las transiciones de evaluación. No se seleccionan semillas ni se para por recompensa.

Si necesitas instalar dependencias en Windows, crea un entorno dentro de `Lab7` para aislarlas:

```powershell
# TODO (preparación local): crear dependencias dentro del directorio del laboratorio.
#______________________________
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --no-cache-dir torch numpy matplotlib swig
.\.venv\Scripts\python.exe -m pip install --no-cache-dir "gymnasium[box2d]"
.\.venv\Scripts\python.exe Verificar.py
.\.venv\Scripts\python.exe Experimentos.py
.\.venv\Scripts\python.exe Diagnosticos.py
#______________________________
```

Si Box2D necesita herramientas de compilación que no tienes, utiliza el cuaderno Colab suministrado
en lugar de modificar los archivos auxiliares. No ejecutes el bloque `__main__` original de `DDPG.py`
para la comparación: contiene una parada por recompensa y un presupuesto de 500.000 pasos.
`Experimentos.py` aplica el protocolo fijo del PDF y evalúa en un entorno distinto.
`TD3.learn()` también usa este bucle común.

## Distribuir ejecuciones

Ejemplo de división por variante, lanzando cada comando en un recurso independiente:

```powershell
# TODO (paralelización): ejecutar cada trabajo en su propio recurso y salida.
#______________________________
python Experimentos.py --variant ddpg --seeds 0 --output resultados_ddpg
python Experimentos.py --variant full --output resultados_full
python Experimentos.py --variant no_minimum --output resultados_sin_minimo
python Experimentos.py --variant no_smoothing --output resultados_sin_smoothing
python Experimentos.py --variant no_delay --output resultados_sin_retraso
#______________________________
```

También puedes distribuir semillas usando `--seeds 0`, `--seeds 1`, etc. Al descargar los resultados,
mantén todas las carpetas dentro de `Lab7`. Agrega las salidas así:

```powershell
# TODO (diagnósticos): reunir las mediciones completas de todos los trabajos.
#______________________________
python Diagnosticos.py --input resultados_ddpg resultados_full resultados_sin_minimo resultados_sin_smoothing resultados_sin_retraso
#______________________________
```

Las opciones `--steps` y `--eval-every` permiten pruebas breves. Esas pruebas no sustituyen al estudio
de 300.000 pasos. El generador de informe exige el presupuesto solicitado, cinco semillas distintas,
evaluaciones completas cada 1.000 pasos y datos de LunarLander; rechaza ejecuciones incompletas.
Si ya existe una ejecución medida, el ejecutor pide otra carpeta de salida mediante un error
en vez de sobrescribir silenciosamente sus CSV.

## Checkpoints y posible ahorro de cómputo

TD3 completo guarda `checkpoint_tercio.pt` en la primera frontera de episodio a partir de T/3,
cuando el aprendizaje ya ha empezado en el presupuesto oficial. Contiene las seis redes,
los tres optimizadores, replay, contadores de actor/críticos, RNG de Python/NumPy/PyTorch/CUDA,
RNG del entorno/espacio de acciones, semilla, índice del próximo episodio e historial de mediciones.
La siguiente observación se reconstruye con `reset(seed=semilla+índice_episodio)`; no se pretende
guardar un mundo Box2D a mitad de episodio. El ruido TD3 es gaussiano y no tiene estado temporal.

Por defecto todas las variantes se entrenan desde cero. Opcionalmente, cada ablación puede partir
de una carga independiente del checkpoint de TD3 completo de su misma semilla:

```powershell
# TODO (opción del apartado 4): bifurcar una semilla en una nueva salida.
#______________________________
python Experimentos.py --variant no_minimum --seeds 0 --resume resultados_full/full_seed0/checkpoint_tercio.pt --output bifurcacion_sin_minimo
#______________________________
```

Repite para las cinco semillas y las tres ablaciones. `--steps 300000` sigue indicando el paso final
total, incluyendo el prefijo compartido. No mezcles ejecuciones desde cero y bifurcadas sin explicarlo.
El informe generado señala si hay prefijos compartidos. La reanudación exacta se ha comprobado
con el entorno mínimo de pruebas; al cambiar de versiones o dispositivo no se promete identidad bit a bit.

## Mediciones y entrega

Cada carpeta de ejecución contiene `configuracion.json`, `evaluaciones.csv` y `episodios.csv`.
Cada evaluación conserva los cinco retornos individuales, su media, dispersión, pérdidas,
contadores, reducción aplicada por el mínimo, perturbación de acción objetivo y desacuerdo entre críticos.
Los diagnósticos usan hasta 256 transiciones sin consumir el RNG de entrenamiento.
No miden el error frente al valor verdadero, por lo que no demuestran ausencia de sesgo.

`Diagnosticos.py` genera `graficas/comparacion.png`, `graficas/ablaciones.png`,
`resumen_mediciones.csv` e `Informe.md` con valores calculados. La banda de ablaciones es la
desviación estándar muestral entre las cinco semillas, no la dispersión entre episodios de una evaluación.
El informe está diseñado como texto breve con dos figuras; comprueba el máximo de cuatro páginas
al convertirlo a PDF según tu formato. No se ha generado ningún PDF.

Entrega un ZIP con los `.py`, el cuaderno, `LEEME.md`, `Informe.md`, las dos gráficas,
`resumen_mediciones.csv` y los CSV/JSON de las 21 ejecuciones. Excluye `.venv`, `__pycache__`
y los `.pt`/replay: el PDF permite conservarlos localmente. El cuaderno incluye una celda para crear
el ZIP después de generar el informe final. El borrador actual declara explícitamente que faltan
los entrenamientos; no contiene cifras inventadas.
