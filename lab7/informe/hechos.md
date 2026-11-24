# Ficha de hechos

Todo lo que el informe afirma sale de aquí. Cada entrada con su origen:
`archivo:línea`, `conversación` (decisión o motivo hablado con el usuario), `usuario` (respuesta a una duda) o `enunciado`.
Rutas relativas a `lab7/`. Enunciado: `Lab_7___APRNS_GIA.pdf` (recuperado del commit 9db566a^; copia en el scratchpad de la sesión).

## Problema y objetivos
| # | Hecho | Origen |
|---|---|---|
| P1 | Implementar DDPG (4 pt) y TD3 (3 pt) en LunarLander-v3 continuo (2 acciones en [−1, 1]: motor principal y motores laterales) y hacer un estudio de ablación controlado (3 pt). | enunciado §2–4 |
| P2 | Comparación DDPG–TD3: presupuesto común T declarado antes, una semilla por algoritmo, retorno frente a transiciones, sin elegir semilla ni parar antes; es exploratoria, no evidencia estadística. Llegar a la puntuación de la figura 4 no es requisito. | enunciado §2 (p. 4) |
| P3 | Ablación: variantes full, no minimum (target con Q1'), no smoothing (σ_targ = 0), no delay (policy_delay = 1); cambiar solo el mecanismo indicado; 3–5 semillas; T = 300.000; gráfica de la media cada 1.000 pasos y comentario basado en mediciones propias. | enunciado §4 |
| P4 | Informe PDF de máx. 4 páginas sin contar referencias; zip con implementación y script de diagnósticos. | enunciado §5 |
| P5 | Hiperparámetros: los de la sección 7 del artículo de DDPG; el enunciado advierte contra buscarlos por cuenta propia; τ = 0,001 también en TD3. | enunciado §2, §3.1.3 |

## Entorno y datos
| # | Hecho | Origen |
|---|---|---|
| D1 | `gym.make("LunarLander-v3", continuous=True)`; episodios truncados a 1.000 pasos; observación de 8 dimensiones (x, y, vx, vy, ángulo, velocidad angular, contacto de cada pata). | `experiments/config.py:10`, `analysis/hovering.py` |
| D2 | Retorno ≥ 200 se usa como criterio de episodio resuelto (umbral habitual de LunarLander). | `analysis/supplementary.py` (succ) |
| D3 | Versiones: Python 3.12.3, torch 2.15.0+cpu, gymnasium 1.3.0, numpy 2.4.6; CPU, 1 hilo por proceso. | `results/full/seed_0/config.json` |
| D5 | Recompensa por paso: shaping por distancia a la plataforma, velocidad e inclinación; +10 por pata en contacto; −0,3 por paso con motor principal (escalado por potencia) y −0,03 con laterales; −100 al estrellarse o salir (|x| ≥ 1) y +100 al quedar en reposo (terminación). Episodio resuelto con ≥ 200 puntos (`reward_threshold=200`); `max_episode_steps=1000`. | gymnasium 1.3.0 `envs/box2d/lunar_lander.py:110-124, 636-661`; `envs/__init__.py:93-98` |
| D4 | Ejecución en 2 VM Azure Standard_D4s_v4 (Spain Central y Belgium Central); reparto por semillas, no por variantes. | `cloud/README.md`, conversación |

## Método e implementación
| # | Hecho | Origen |
|---|---|---|
| M1 | Redes del profesor sin modificar: actor 400-300 con LayerNorm y tanh; crítico con camino del estado 400-300 (LayerNorm), acción proyectada a 300, suma y ReLU. | `ActorCriticNetworks.py` |
| M2 | Optimizadores (esqueleto): Adam, crítico lr 1e-3 y weight decay 1e-2; actor lr 1e-4. τ = 0,001. γ = 0,99, batch 64, buffer 10^6. | `DDPG.py:59,67-68`, `experiments/config.py:14` |
| M3 | Crítico DDPG: MSE contra y = r + γ(1 − terminated)·Q'(s', μ'(s')); solo la terminación real corta el bootstrap (la truncación por tiempo sí hace bootstrap). | `DDPG.py:170-176` |
| M4 | Actor: minimiza −Q(s, μ(s)); actualización suave de los dos targets tras cada iteración. | `DDPG.py:192,199-225` |
| M5 | Inicialización: `initialize_weights()` del profesor; el constructor del crítico no la llama y se llama explícitamente. Esa función usa `size()[0]` (salidas) como f. | `DDPG.py:46-48`, `ActorCriticNetworks.py:34-45` |
| M6 | Exploración DDPG: OU del esqueleto con θ = 0,2, σ = 0,15, dt = 0,01 (el artículo de DDPG usa θ = 0,15, σ = 0,2); se reinicia en cada episodio. | `Noise.py:47`, `results/ddpg/seed_0/config.json`, `experiments/training.py` |
| M7 | TD3: suavizado ã' = clip(μ'(s') + clip(ε, −0,5, 0,5), −1, 1), ε ~ N(0, 0,2²); dos críticos hacia el mismo target con mínimo; actor maximiza Q1; actor y los tres targets cada 2 actualizaciones del crítico (contadas en iteraciones de crítico). | `TD3.py:184-194, 197-228, 248-277` |
| M8 | Exploración TD3: gaussiana N(0, 0,1) de Noise.py en todas las variantes; sin ruido en evaluación. | `experiments/config.py` (TD3_EXPLORATION_SIGMA), `experiments/evaluation.py` |
| M9 | Interruptores: use_minimum, target_noise, policy_delay; no_minimum sigue entrenando los dos críticos; el ruido del target se sortea siempre (también con σ = 0) para que todas las variantes consuman la misma secuencia aleatoria; con la misma semilla parten de los mismos pesos. | `experiments/config.py:19-24`, `TD3.py:70-72,189-193` |
| M10 | Bucle común con presupuesto fijo T = 300.000 transiciones; aprendizaje desde que el buffer tiene > 64 transiciones (299.936 actualizaciones de crítico por ejecución); sin parada temprana. | `experiments/training.py`, configs (critic_updates) |
| M11 | Evaluación cada 1.000 pasos: 5 episodios deterministas con semillas fijas 10000–10004 (las mismas en todas las evaluaciones y ejecuciones). | `experiments/config.py`, `experiments/evaluation.py` |
| M12 | Diagnóstico de sesgo: Q1(s, μ(s)) − G_t sobre los estados de los episodios de evaluación, G_t descontado con γ; en episodios truncados se excluyen los estados con menos de 300 pasos por delante. | `experiments/diagnostics.py` |
| M13 | Diagnósticos del crítico registrados: critic_disagreement = media|Q1' − Q2'|, min_reduction = media(Q1' − min). | `TD3.py:222-224` |
| M14 | Checkpoint completo de TD3 full en la primera frontera de episodio ≥ T/3 (pasos 100.043–100.851); las variantes se entrenaron desde cero (no se usó la rama). | `results/full/seed_*/config.json` (checkpoint_step), `experiments/run_experiment.py` |
| M15 | Métricas preregistradas: media de las últimas 10 evaluaciones y AUC/T (área bajo la curva de evaluación dividida por T). IC 95 % bootstrap entre semillas (10.000 remuestreos); diferencias aparejadas por semilla respecto de full. | `experiments/config.py` (summary_last_n), `analysis/analysis.py` |
| M16 | Análisis complementario no preregistrado: test de permutación aparejado exacto (2^10 signos) con corrección de Holm (3 comparaciones por métrica), mediana e IQM, fracción de episodios con retorno ≥ 200 y fracción de evaluaciones con los 5 episodios truncados a 1.000 pasos, todas sobre las últimas 10 evaluaciones. | `analysis/supplementary.py` |
| M17 | Prueba de control fuera de protocolo, semillas 0–4: td3_paper (tabla 3 de TD3: τ = 0,005, lr 1e-3 en actor y críticos, sin weight decay, batch 100, 10.000 pasos iniciales de acciones uniformes sin aprender; misma arquitectura con LayerNorm) y no_weight_decay (solo weight decay 0 en críticos). Evaluación final de 100 episodios con semillas 20000–20099 y redes finales guardadas. | `experiments/config.py` (SANITY_CONFIGS), `experiments/run_experiment.py` |
| M18 | Comprobación de comportamiento con las redes finales de la prueba de control: 20 episodios por ejecución (semillas 20000–20019), resumen de los últimos 200 pasos de los episodios truncados. | `analysis/hovering.py` |

## Decisiones y motivos
| # | Decisión | Motivo | Alternativas descartadas | Origen |
|---|---|---|---|---|
| DE1 | 10 semillas por variante (el enunciado pide 3–5). | Suficiencia estadística; varianza alta entre semillas. | 5 semillas | conversación |
| DE2 | Mantener los hiperparámetros del esqueleto/artículo DDPG en todo el estudio. | Lo exige el enunciado. | ajustar hiperparámetros | enunciado, conversación |
| DE3 | σ de exploración TD3 = 0,1. | Valor del artículo de TD3; el enunciado no lo fija. | — | `experiments/config.py`, conversación |
| DE4 | Entrenar cada variante desde cero en lugar de ramificar en T/3. | Rigor: comparación limpia; había cómputo. | rama del checkpoint (opción del enunciado para ahorrar 1/3) | conversación |
| DE5 | Métricas fijadas antes de ver resultados (últimas 10 evaluaciones y AUC/T). | Evitar elegir métrica a posteriori. | — | conversación |
| DE6 | Prueba de control fuera de protocolo y declarada. | Descartar un error de implementación al ver que TD3 full no aterriza. | ampliar semillas, cambiar hiperparámetros del estudio | conversación |
| DE7 | Reparto de ejecuciones entre VM por semillas. | Que una diferencia de máquina no se confunda con una variante. | por variantes | `cloud/README.md` |

## Resultados
Últimas 10 evaluaciones (L10) y AUC/T: media ± desviación estándar entre semillas (ddof = 1); IC 95 % bootstrap de `analysis/tables/metrics_by_variant.csv`.

| # | Métrica / resultado | Valor exacto | Condiciones | Origen |
|---|---|---|---|---|
| R1 | full L10 | −51,3 ± 45,9; IC [−77,7, −22,3] | 10 semillas | facts.py, metrics_by_variant.csv |
| R2 | no_minimum L10 | 55,4 ± 110,1; IC [−8,8, 120,0] | 10 | ídem |
| R3 | no_smoothing L10 | −54,6 ± 95,1; IC [−97,2, 9,6] | 10 | ídem |
| R4 | no_delay L10 | 19,7 ± 110,5; IC [−39,1, 88,0] | 10 | ídem |
| R5 | AUC/T full / no_min / no_smooth / no_delay | −115,2 ± 43,5 / −45,8 ± 67,0 / −95,6 ± 48,4 / −86,8 ± 45,3 | 10 | ídem |
| R6 | Diferencia aparejada L10 vs full | no_min +106,6 [27,5, 177,6], 8/10 semillas mejor, p = 0,029, Holm 0,088; no_smooth −3,4 [−51,1, 55,1], 4/10, p = 0,90; no_delay +70,9 [11,2, 142,7], 8/10, p = 0,076, Holm 0,152 | 10 pares | paired_differences_vs_full.csv, paired_permutation_tests.csv |
| R7 | Diferencia aparejada AUC vs full | no_min +69,4 [17,1, 118,2], p = 0,035, Holm 0,106; no_smooth +19,6 [−13,3, 52,7], p = 0,31; no_delay +28,4 [−6,5, 63,7], p = 0,16 | 10 pares | ídem |
| R8 | Mediana / IQM L10 | full −58,2/−55,8; no_min 52,8/53,6; no_smooth −80,3/−81,3; no_delay −27,7/−14,4 | 10 | robust_summary.csv |
| R9 | Fracción de episodios con retorno ≥ 200 (últimas 10 eval.) | full 2,0 %; no_min 27,6 %; no_smooth 7,0 %; no_delay 17,8 % | 10 semillas × 50 episodios | robust_summary.csv |
| R10 | Fracción de evaluaciones con los 5 episodios truncados a 1.000 pasos | full 65 %; no_min 25 %; no_smooth 70 %; no_delay 57 % | ídem | robust_summary.csv |
| R11 | Semillas con ≥ 50 % de episodios resueltos | full 0; no_min 4; no_smooth 1; no_delay 2 | 10 | robust_summary.csv |
| R12 | Ventaja de no_min uniforme en las 5 condiciones iniciales de evaluación: 21–33 % de episodios resueltos por condición frente a 0–5 % en full | | | conversación (chk2.py) |
| R13 | Sesgo Q − G (últimas 10 eval.) | DDPG +40,2 (1 semilla); full −6,4 ± 7,5; no_min +32,2 ± 18,2; no_smooth −2,2 ± 13,3; no_delay −14,4 ± 11,5 | | facts.py |
| R14 | DDPG vs full semilla 0: L10 −83,6 vs −70,3; sesgo +40,2 vs −10,6; DDPG 1.318 episodios de entrenamiento vs 570,1 ± 140,2 de full (media 10 semillas) | | 1 semilla | facts.py, extra.py |
| R15 | Spearman sesgo final–L10 dentro de cada variante: full −0,42; no_min −0,71; no_smooth −0,24; no_delay −0,72; global (40) −0,18 | | | summary.txt |
| R16 | Caídas > 200 entre evaluaciones consecutivas tras 50.000 pasos: full 17, no_min 73, no_smooth 27, no_delay 22 (suma 10 semillas) | | | summary.txt |
| R17 | Tiempo por ejecución: full/no_min/no_smooth 6,8 h; no_delay 8,0 h; DDPG 5,3 h | | VM D4s_v4 | summary.txt, facts.py |
| R18 | min_reduction y critic_disagreement crecen durante el entrenamiento (≈0,5–0,7 y ≈0,8–1,3 al final) | | | figuras diagnostic_*.png |
| R19 | Prueba de control L10 (semillas 0–4): full −64,4; td3_paper 67,8 ± 126,0 (4/5 mejor, p = 0,125); no_wd −47,7 ± 68,8 (2/5, p = 0,69) | | n = 5, p mínimo 0,0625 | sanity_paired_vs_full.csv, facts.py |
| R20 | Prueba de control AUC/T: full −129,2; td3_paper −31,6 (5/5, p = 0,0625); no_wd −80,2 (5/5, p = 0,0625) | | | ídem |
| R21 | Prueba de control sesgo: full −4,5; td3_paper −12,2 (1/5 mayor); no_wd +26,1 (5/5 mayor, p = 0,0625) | | | ídem |
| R22 | td3_paper semilla 3: L10 274,0; evaluación final 100 episodios 269,7, 98 % resueltos, 1 % truncados. Semilla 1: 87,6 / 34 %. Semillas 0, 2, 4: −21,4 / 9,8 / −24,3 con 100 %, 83 %, 99 % truncados | | | sanity_per_seed.csv |
| R23 | Evaluación final media (5 semillas): td3_paper 64,3, 26,8 % resueltos, 65,2 % truncados; no_wd −53,4, 6,4 %, 79,8 % | | 100 episodios | facts (sanity.py), configs |
| R24 | Evaluación final coherente con L10 por semilla (p. ej. 269,7 vs 274,0; 87,6 vs 102,8) | | | sanity_per_seed.csv |
| R25 | Comportamiento en episodios truncados (redes finales de la prueba de control): 143/200 truncados; últimos 200 pasos: y media 0,38 (88 % con y > 0,05), contacto de las dos patas 1,6 % del tiempo, |vy| 0,021, motor principal encendido 72,7 %, retorno medio −42,4 | | 10 ejecuciones × 20 episodios | `analysis/supplementary/hovering.csv`, `analysis/hovering.py` |
| R26 | no_wd semilla 3: sesgo +91,3 (evaluación final +117,2) | | | sanity_per_seed.csv |

## Segunda ablación: configuración B (indicaciones del profesor, 2026-11)
Configuración A = la del enunciado (estudio principal). Configuración B = A con los cuatro cambios que el profesor recomendó después del enunciado. Resultados en `analysis/prof/` (tablas, figuras, `supplementary/summary.txt`, `final_evaluation_*.csv`, `base_comparison.csv`).

| # | Hecho | Origen |
|---|---|---|
| B1 | Mensaje del profesor (Mario) publicado después del enunciado: 1) τ = 0,005 en lugar de 0,001; 2) la actualización suave de los targets de los críticos y del actor fuera del "if" del retardo ("com a mínim el dels crítics"); 3) sin weight_decay en los críticos de TD3 ("tal com demanen al paper"); 4) sin la inicialización de DDPG en los críticos de TD3 (la de PyTorch por defecto; "la inicialització dels pesos és específica de DDPG"). | conversación (texto pegado por el usuario) |
| B2 | Implementación de B: τ = 0,005, weight decay 0 en los dos críticos, inicialización por defecto de PyTorch en los críticos, targets de los críticos actualizados en cada iteración de crítico; el target del actor sigue dentro del retardo (opción "com a mínim el dels crítics"). Resto igual que A (redes, lr 1e-4/1e-3, batch 64, γ, exploración N(0, 0,1²), T, evaluación). Las variantes prof_* aplican los mismos interruptores. Sense retard en B: actor y su target en cada iteración. | `experiments/config.py` (PROF_BASE), `TD3.py` (critic_targets_every_step, ddpg_critic_init), `results/prof_full/seed_0/config.json` |
| B3 | 40 ejecuciones (4 variantes × semillas 0–9), mismas semillas, T, evaluación y VM (D4s_v4, reparto por semillas: 0–4 y 5–9). Cada ejecución guarda además las redes finales y una evaluación final de 100 episodios (semillas 20000–20099). | `cloud/README.md` §7, configs |
| B4 | L10 (media ± DE, ddof 1; IC 95 %): prof_full 207,6 ± 118,1 [127,2, 268,0]; prof_no_minimum 193,8 ± 117,6 [117,6, 255,9]; prof_no_smoothing 196,2 ± 136,1 [103,2, 264,8]; prof_no_delay 244,7 ± 98,0 [181,3, 278,6]. Medianas 262,1 / 249,6 / 261,8 / 277,2. | `analysis/prof/tables/metrics_by_variant.csv`, `supplementary/robust_summary.csv`, cálculo de sesión |
| B5 | AUC/T: 96,2 ± 101,6 / 105,3 ± 95,1 / 83,4 ± 91,0 / 115,9 ± 83,1. | ídem |
| B6 | Δ L10 vs prof_full [IC 95 %], semillas mejores, p sin corregir, Holm: no_min −13,8 [−101,0, 68,6], 4/10, 0,691, 1,000; no_smooth −11,4 [−107,9, 76,1], 4/10, 0,676, 1,000; no_delay +37,1 [−58,5, 131,9], 6/10, 0,400, 1,000. AUC: +9,0 / −12,8 / +19,7, todos Holm 1,000. | `analysis/prof/tables/paired_differences_vs_full.csv`, `supplementary/paired_permutation_tests.csv` |
| B7 | Episodios resueltos (últimas 10 eval.): 77,6 % / 73,6 % / 77,2 % / 90,0 %. Evaluaciones con los 5 truncados: 13 % / 7 % / 16 % / 10 %. Semillas con ≥ 50 % resueltos: 8 / 8 / 8 / 9. | `supplementary/robust_summary.csv` |
| B8 | Sesgo final (últimas 10 eval., media ± DE): prof_full −6,5 ± 5,1; no_min +36,4 ± 21,7 (positivo en las 10 semillas, 13,4 a 85,7); no_smooth −1,3 ± 5,8; no_delay −9,3 ± 3,0. | `supplementary/summary.txt`, cálculo de sesión |
| B9 | Evaluación final (100 episodios, redes finales): retorno medio 204,8 / 171,0 / 193,6 / 238,8; resueltos 78,0 % / 69,0 % / 76,7 % / 88,2 %; truncados 14,9 % / 19,0 % / 21,2 % / 11,2 %; semillas con ≥ 50 % resueltos 8 / 7 / 8 / 9. | `supplementary/final_evaluation_summary.csv` |
| B10 | Fracasos en B: 8 ejecuciones con < 50 % de episodios resueltos en la evaluación final (full 3, 4; no_min 2, 3, 9; no_smooth 1, 4; no_delay 7); 7 de ellas tienen más del 50 % de episodios truncados (planeo); la excepción es no_min semilla 9 (6 % truncados, 15 % resueltos). | `supplementary/final_evaluation_per_seed.csv` |
| B11 | Comparación B − A por semilla (misma variante): full L10 +258,9, 10/10 semillas mejor, p = 0,002 (mínimo posible con 10 pares); evaluaciones truncadas −52 puntos porcentuales; sesgo −0,1 (p = 0,96). Las otras tres variantes también mejoran (L10 +138,4, +250,8, +225,0; p ≤ 0,041). | `supplementary/base_comparison.csv` |
| B12 | Interacción (Δ de quitar el mínimo en B menos Δ en A, por semilla): L10 −120,4, p = 0,107; suavizado −8,0 (p = 0,91); retardo −33,9 (p = 0,59). | `supplementary/base_comparison.csv` |
| B13 | Caídas > 200 entre evaluaciones consecutivas tras 50.000 pasos (suma 10 semillas): B full 13, no_min 34, no_smooth 18, no_delay 9. Diferencia aparejada no_min − full: B p = 0,016; A (73 vs 17) p = 0,031 (tests exactos sin corregir, exploratorios). | `supplementary/summary.txt` (A y B) |
| B14 | Spearman sesgo final–L10 dentro de cada variante en B: full 0,04; no_min −0,90; no_smooth −0,77; no_delay 0,15. En no_min, las tres semillas con más sesgo (85,7 / 50,8 / 52,2: semillas 2, 9, 3) son las tres que fracasan. | `supplementary/summary.txt`, final_evaluation_per_seed.csv |
| B15 | DDPG vs TD3 B, semilla 0: TD3 B L10 251,2, sesgo −8,4; primera evaluación > 200 a los 248.000 pasos. | cálculo de sesión (results/prof_full/seed_0) |
| B16 | Media entre semillas de las evaluaciones 291–300 frente a 241–250: prof_full 207,6 vs 176,4; no_min 193,8 vs 212,9; no_smooth 196,2 vs 185,6; no_delay 244,7 vs 232,0. | cálculo de sesión |
| B17 | No se puede separar cuál de los cuatro cambios de B produce la mejora (se aplicaron juntos). td3_paper (τ = 0,005 y sin weight decay, pero inicialización de DDPG y targets de los críticos dentro del retardo) resolvió 1 de 5 semillas; sin weight decay, ninguna. | M17, R22, R23 |
| B18 | Sesgo final, diferencia aparejada variante − full y test de permutación exacto sin corregir: A no_min +38,6 (p = 0,004), no_smooth +4,2 (p = 0,46), no_delay −8,0 (p = 0,11); B no_min +42,9 (p = 0,002), no_smooth +5,3 (7/10, p = 0,047), no_delay −2,7 (p = 0,090). | cálculo de sesión (results/*/evaluations.csv, últimas 10 eval.) |

## Problemas encontrados y limitaciones
| # | Hecho | Origen |
|---|---|---|
| L1 | Ninguna variante resuelve el entorno en media dentro de T; las curvas medias siguen subiendo a 300.000 pasos. | ablation_curves.png |
| L2 | Tras Holm ninguna diferencia de la ablación es significativa al 5 %. | paired_permutation_tests.csv |
| L3 | DDPG con una sola semilla y ruido OU frente a gaussiano en TD3: comparación con factor de confusión. | enunciado, M6, M8 |
| L4 | Redes con LayerNorm (el texto del enunciado habla de batch normalization); inicialización con `size()[0]`. | M1, M5 |
| L5 | Weight decay 1e-2 en los críticos de TD3 (heredado de DDPG; el artículo de TD3 no lo usa). | M2 |
| L6 | El crítico aprende el valor de la política suavizada; el sesgo se mide para μ determinista. Sesgo medido sobre la distribución de estados de cada política. | M7, M12 |
| L7 | Evaluación periódica con 5 condiciones iniciales fijas (mitigado por R24). | M11 |
| L8 | No se guardaron las redes finales del estudio principal (solo checkpoints de T/3); la comprobación de comportamiento R25 usa las redes de la prueba de control. | conversación (auditoría) |
| L9 | Con n = 5 el test exacto no puede bajar de p = 0,0625. | summary.txt |

## Referencias consultadas
| Clave en refs.bib | Para qué se cita | Origen (consultada / aportada por el usuario) |
|---|---|---|
| lillicrap2016 | DDPG, hiperparámetros sección 7 | enunciado (ref. [1]) |
| fujimoto2018 | TD3, tabla 3 de hiperparámetros | enunciado (ref. [2]) |
| gymnasium | Recompensa y criterio de resolución de LunarLander-v3 | consultada (código y documentación del paquete instalado) |
