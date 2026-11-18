# Estado del informe

## Fase actual
Fase: 3
Paso: 3 (PDF) terminado: compilado y revisado página a página
Fuente vigente: latex/

## Datos
- Idioma: catalán
- Título: Pràctica 7: mètodes actor-crític off-policy, DDPG i TD3
- Práctica: Lab 7
- Asignatura: APRNS (Aprenentatge per Reforç i No Supervisat)
- Grado: Grau en Intel·ligència Artificial (GIA), UPC
- Autores (nombres completos): Laia Beyloc Guzmán, Ilyas Ouacham Rhailane
- Grupo: 11
- Curso y cuatrimestre: 2026-2027, quadrimestre de tardor
- Logo en la portada: no
- Plantilla de portada del profesor: no consta

## Fuentes
- Enunciado: Lab_7___APRNS_GIA.pdf (git 9db566a^:lab7/Lab_7___APRNS_GIA.pdf)
- Rúbrica: no hay; se sigue el reparto del enunciado (DDPG 4, TD3 3, estudio controlado 3) (confirmado)
- Código: lab7/ (DDPG.py, TD3.py, experiments/, analysis/)
- Resultados: lab7/results/, lab7/analysis/{figures,tables,supplementary}/
- Otras: conversación de esta sesión (decisiones y auditoría)

## Condiciones de la asignatura
- Extensión: máximo 4 páginas de cuerpo; no cuentan portada ni referencias (ni elementos que no suelen contar)
- Apartados obligatorios: comparación exploratoria DDPG–TD3 y estudio controlado de la sección 4 (enunciado §5)
- Formato de referencias: APA 7
- Otras: entrega en zip con implementación y script de diagnósticos; las mediciones entregadas deben respaldar cada figura y tabla. Uso de IA permitido para orquestar experimentos (enunciado §4).

## Extensión medida
- 2026-11-18: compilación de prueba del Markdown (11 pt, A4, márgenes de 2 cm, figuras a ancho completo): el cuerpo (secciones 1–6) ocupa 4 páginas y las referencias empiezan en la 5.ª. Con márgenes de 2,5 cm no cabe. En la fase 2 se usarán márgenes de 2 cm. PDF de prueba: scratchpad/measure/d_2cm.pdf.

## Estilo acordado
Muestra aprobada sin cambios: impersonal con "es va", primera persona del plural para decisiones, cifras en cada afirmación.

## Esquema aprobado
| Apartado | Mensaje principal | Rúbrica | Peso | Estado |
|---|---|---|---|---|
| 1 Implementació de DDPG i TD3 | Implementación fiel al enunciado; decisiones no triviales; hiperparámetros fijados por el enunciado | DDPG 4, TD3 3 | 0,8 pág. | pendiente |
| 2 Protocol experimental | T, 10 semillas, evaluación, métricas preregistradas, sesgo, estadística, ejecución | estudio 3 | 0,5 pág. | pendiente |
| 3 Comparació exploratòria DDPG–TD3 | Una semilla: nivel parecido; DDPG sobreestima, TD3 no; sin conclusión de rendimiento; confusión OU/gaussiano | §2 | 0,4 pág. | pendiente |
| 4 Estudi d'ablació: resultats | Curvas con IC, tabla de métricas, diagnósticos; nada resuelto en media; nada significativo tras Holm | estudio 3 | 1,1 pág. | pendiente |
| 5 Discussió | Planeo como mínimo local medido; efecto de cada mecanismo sin sobreinterpretar; prueba de control; limitaciones | estudio 3 | 1,0 pág. | pendiente |
| 6 Conclusions | Respuesta a objetivos y trabajo futuro | — | 0,2 pág. | pendiente |

## Figuras
| Archivo | Origen (usuario / script) | Aprobada |
|---|---|---|
| img/fig_ddpg_td3.png | figuras/figures.py | sí |
| img/fig_ablacio.png | figuras/figures.py | sí |

## Dudas resueltas
- Idioma → catalán (usuario, 2026-11-18)
- Autores, grupo, grado, asignatura → usuario, 2026-11-18
- Extensión → 4 páginas máx.; portada aparte que no cuenta (usuario)
- Título, curso → confirmados; logo no; rúbrica = reparto del enunciado; sin declaración de IA; informe/ en lab7/ (usuario, 2026-11-18)

## Auditorías
| Ronda | Veredicto | Graves | Importantes | Menores | Qué se hizo con los graves |
|---|---|---|---|---|---|
| 1 | NO APTO | 1 | 4 | 7 | 28,0 % → 28,4 % (sense mínim, llavors 0–4) en tabla y texto. Importantes corregidos (argumento del biaix con mediana y valor atípico; conclusión sin causalidad fuerte; planeig: se dice que se midió con las redes de control; extensión: se verifica al compilar en la fase 2). Menores corregidos salvo el año de gymnasium (no consta; biblatex pondrá s.d.). |
| 2 | APTO | 0 | 0 | 8 | — (menores corregidos: γ y τ nombrados; título 5.1; 88 % en el aire; "hi arriben o el superen"; conclusión limitada a tres variantes; repetición en limitacions; título de gymnasium; "corbes de retorn"). Año de gymnasium sigue sin constar. |

## Registro
- 2026-11-18: con permiso del usuario, PyMuPDF en carpeta aislada del scratchpad; revisión visual de las 6 páginas. Corregido: separadores de la fórmula de clip y de N(0; σ²) (icomma), «s/ d» → «s. d.». Log limpio, fidelidad 95/95, cuerpo en 4 páginas. informe/informe.pdf actualizado.
- 2026-11-18: fase 2 terminada. Paquetes instalados en TinyTeX desde el archivo histórico de TL2025 (microtype, icomma/was, csquotes, biblatex, biber, caption, listings, xstring, logreq, newunicodechar, hyphen-catalan). Fidelidad 95/95 frases; refs sin errores; log sin errores ni cajas desbordadas. Márgenes 2 cm, T1 antes de babel, l·l con \l., signos Unicode con newunicodechar, cabecera agrupada en la Taula 2. PDF: portada + 4 páginas de cuerpo + referencias (p. 6). Copia en informe/informe.pdf.
- 2026-11-18: FALLO en fase 3: no hay pdftoppm/ghostscript/PyMuPDF y el visor del navegador integrado solo da instantáneas estáticas; revisado visualmente solo portada e inicio de la p. 2.
- 2026-11-18: auditoría ronda 2 APTO; menores corregidos; extensión re-medida (cuerpo en 4 páginas a 2 cm).
- 2026-11-18: 1.4 esquema aprobado (usuario).
- 2026-11-18: 1.5 figuras y 1.6 estilo aprobados; 1.7 redactado; 1.8 lint y refs revisados (avisos restantes: rayas = celdas vacías de tabla; frases largas = enumeraciones de hiperparámetros; gymnasium sin año). Extensión estimada ~3,9 páginas a 2,5 cm; se verificará en la fase 2.
- 2026-11-18: fase 0 (estructura creada en APRNS-repo/informe/, movida a lab7/informe/), 1.1 contexto recogido, 1.2 hechos.md rellenado; añadido lab7/analysis/hovering.py (comprobación de comportamiento, sin commit).
