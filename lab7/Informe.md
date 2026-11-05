<!-- Función: borrador del informe del apartado 5; se completará con Diagnosticos.py. -->
# Laboratorio 7: DDPG y TD3

**Estado:** implementación y protocolo preparados y contrastados con `Lecture5.pdf`, diapositivas 70–73.
Falta ejecutar los entrenamientos de LunarLander.
No se presentan resultados experimentales ni se afirma que un algoritmo haya aprendido.

## Implementación

DDPG sigue la figura 1 del enunciado y las diapositivas 70–72. Se inicializan actor y crítico y se copian sus pesos a las
redes objetivo. La acción procede del actor determinista; la exploración OU se añade al interactuar.
El proceso OU se reinicia al terminar cada episodio, siguiendo la diapositiva 72.
El crítico minimiza el error cuadrático de Bellman:

\[
y=r+\gamma(1-\mathrm{terminated})Q^-(s',\mu^-(s')),\qquad
L_Q=\frac{1}{N}\sum_i(Q(s_i,a_i)-y_i)^2.
\]

El actor minimiza \(L_\mu=-\frac1N\sum_i Q(s_i,\mu(s_i))\).
Las redes objetivo siguen \(\theta^-\leftarrow0{,}001\theta+0{,}999\theta^-\).
Una truncación temporal conserva el bootstrap; solo una terminación real lo anula.
El replay almacena ambos indicadores por separado.
Reutiliza la implementación original de `Lab5/ReplayBuffer.py`, sin añadir métodos ni protecciones.
El ejecutor utiliza `get` y se encarga de guardar y restaurar sus atributos para los checkpoints.

TD3 sigue la figura 5, los apartados 3–4 y la diapositiva 73. Dos críticos independientes aprenden hacia el mismo objetivo:

\[
\tilde a'=\operatorname{clip}(\mu^-(s')+
\operatorname{clip}(\epsilon,-0{,}5,0{,}5),-1,1),\quad
\epsilon\sim\mathcal N(0,0{,}2^2I),
\]
\[
y=r+\gamma(1-\mathrm{terminated})\min(Q_1^-(s',\tilde a'),Q_2^-(s',\tilde a')).
\]

El actor maximiza Q1. Ambos críticos se actualizan cada iteración de aprendizaje; el actor y las
tres redes objetivo, cada dos. La exploración TD3 es gaussiana, distinta del ruido objetivo.
Los interruptores de ablación modifican únicamente mínimo, smoothing o retraso.

Se conservan las arquitecturas suministradas, con capas 400/300 y LayerNorm. Se llama al método
original de inicialización del crítico sin alterar su archivo. Adam usa lr actor=0,0001,
lr críticos=0,001 y weight decay críticos=0,01; gamma=0,99, tau=0,001,
batch=64 y capacidad del replay=1.000.000.

## Protocolo fijado antes de entrenar

La comparación exploratoria utilizará una ejecución DDPG y una TD3 completo con semilla 0 y
presupuesto común **T=300.000 transiciones**. El estudio controlado utilizará las semillas
**0, 1, 2, 3 y 4** para cada variante. Se reutilizará TD3 completo/semilla 0 en la comparación:
el total será de 21 ejecuciones, sin parada por recompensa ni selección de la mejor semilla.

| Variante | Objetivo de críticos | Ruido objetivo | Retraso |
|---|---|---|---:|
| TD3 completo | mínimo Q1/Q2 | sigma=0,2; recorte=0,5 | 2 |
| Sin mínimo | Q1; se siguen entrenando ambos críticos | igual | 2 |
| Sin smoothing | mínimo Q1/Q2 | cero | 2 |
| Sin retraso | mínimo Q1/Q2 | igual | 1 |

Cada 1.000 transiciones se evaluarán cinco episodios sin ruido en un entorno independiente.
Las evaluaciones no cuentan en T ni alteran los RNG del entrenamiento.
Se registrarán retornos individuales, pérdidas, actualizaciones, reducción del mínimo,
perturbación del smoothing y desacuerdo entre críticos. TD3 completo guardará un checkpoint
completo en la primera frontera de episodio posterior a T/3.

La infraestructura experimental, el número de episodios de evaluación y sigma=0,15 de exploración
TD3 son propuestas marcadas con `#PROPUESTA`. Las ablaciones mantienen esa exploración fija.
Por defecto se entrenan desde cero; la bifurcación opcional de checkpoints queda registrada.

## Resultados y discusión pendientes

Las catorce pruebas locales han validado la interfaz reutilizada de Lab5, las ecuaciones,
el reinicio de OU entre episodios,
el tratamiento de truncaciones, el recorte
del ruido, los dos críticos, el retraso, los gradientes, el presupuesto fijo y la reanudación exacta
en un entorno mínimo. No son mediciones de rendimiento en LunarLander.
El entorno local no dispone de Box2D ni de una instalación PyTorch con CUDA.

Tras entrenar, `python Diagnosticos.py` sustituirá este borrador por un informe con dos figuras,
tabla de media y desviación estándar entre semillas, diferencias emparejadas y diagnósticos medidos.
El generador rechazará mediciones incompletas. La discusión se basará en esos datos:
un menor objetivo por el mínimo no demuestra que sea insesgado, y una única ejecución por
algoritmo no permite una conclusión de significación estadística.

## Reproducción y referencias

Las instrucciones de CPU, Colab, distribución de ejecuciones y entrega se encuentran en `LEEME.md`.
Se entregarán código, Markdown, gráficas y mediciones; los checkpoints pueden conservarse localmente.
El informe final debe ocupar como máximo cuatro páginas al convertirse a PDF, excluidas referencias.

1. Lab_7___APRNS_GIA.pdf, apartados 2–5 y figuras 1 y 5.
2. Lecture5.pdf, diapositivas 70–73 (páginas PDF 86–89): DDPG y TD3.
3. Lillicrap et al., Continuous control with deep reinforcement learning, ICLR 2016; referencia [1] del enunciado.
4. Fujimoto et al., Addressing Function Approximation Error in Actor-Critic Methods, ICML 2018; referencia [2] del enunciado.
