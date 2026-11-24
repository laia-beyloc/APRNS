# Estado del informe

## Fase actual
Fase: 3
Paso: PDF compilado y revisado página a página tras integrar la configuración B (2026-11-24); pendiente el visto bueno del usuario
Fuente vigente: latex/ (sincronizado con informe.md)

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
- Resultados: lab7/results/, lab7/analysis/{figures,tables,supplementary}/ (configuración A) y lab7/analysis/prof/{figures,tables,supplementary}/ (configuración B, variantes prof_*)
- Indicaciones del profesor publicadas después del enunciado (τ = 0,005; targets de los críticos fuera del retardo; sin weight decay ni inicialización de DDPG en los críticos de TD3): hechos.md B1
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
Esquema original aprobado el 2026-11-18; revisado el 2026-11-24 para integrar la configuración B (el usuario pidió "termina el informe" con los nuevos resultados).
| Apartado | Mensaje principal | Rúbrica | Peso | Estado |
|---|---|---|---|---|
| 1 Implementació de DDPG i TD3 | Implementación fiel; configuraciones A (enunciado) y B (indicaciones del profesor); interruptores | DDPG 4, TD3 3 | 0,9 pág. | redactado |
| 2 Protocol experimental | T, 10 semillas × 2 configuraciones, evaluación, métricas preregistradas, sesgo, estadística | estudio 3 | 0,5 pág. | redactado |
| 3 Comparació exploratòria DDPG–TD3 | Una semilla; DDPG sobreestima; TD3 con B aterriza; sin conclusión de rendimiento | §2 | 0,4 pág. | redactado |
| 4 Estudi d'ablació: resultats | Figura y tabla con A y B; A: nada resuelto; B: TD3 resuelve en 8/10; nada significativo tras Holm | estudio 3 | 1,1 pág. | redactado |
| 5 Discussió | Prueba de control; planeo (medido, mucho menos frecuente con B); efecto de cada mecanismo en las dos configuraciones; limitaciones | estudio 3 | 1,0 pág. | redactado |
| 6 Conclusions | Respuesta a objetivos y trabajo futuro | — | 0,1 pág. | redactado |

## Figuras
| Archivo | Origen (usuario / script) | Aprobada |
|---|---|---|
| img/fig_ddpg_td3.png | figuras/figures.py (2026-11-24: añadida TD3 con B, llavor 0) | sí (versión original) |
| img/fig_ablacio.png | figuras/figures.py (2026-11-24: dos filas, A y B) | sí (versión de una fila); la de dos filas sigue el mismo diseño |

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

| 3 (2026-11-24, config. B) | APTO | 0 | 0 | 4 | — (menores: título de gymnasium → «Gymnasium»; frase larga de 5.3 dividida; «8 de 10 llavors» definido en la conclusión con el criterio de >75 % de episodios finales resueltos; terminología «resolts/aterrar» se deja: «aterrar» se usa para llavors y «resolts» para episodios, de forma consistente). Año de gymnasium: 2026 (PyPI, gymnasium 1.3.0 publicado el 2026-04-22; dato del usuario verificado en la API de PyPI). Además, revisión propia: «el suavitzat no té efecte ni en el biaix» corregido (en B +5,3, p = 0,047 sin corregir; hechos B18). |

## Registro
- 2026-11-24: auditoría ronda 3 APTO; menores corregidos. PDF recompilado (log limpio, fidelidad 90/91 + 1 diferencia de conversión), revisado página a página: portada + 4 páginas de cuerpo + referencias (p. 6). Copia en informe/informe.pdf.
- 2026-11-24: fase 1 reabierta. Análisis de la configuración B (analysis.py --study prof, supplementary.py --prefix prof_). hechos.md B1–B17. informe.md reescrito con A y B; Taula 2 (prueba de control) eliminada y resumida en el texto para caber. LaTeX regenerado; para caber en 4 páginas: parámetros de flotantes (topfraction 0,9, textfraction 0,07, floatpagefraction 0,9), parskip 0,3em, tabla en footnotesize, figuras más bajas y recortes de texto. Cuerpo: páginas 2–5; referencias en la 6. Log limpio; fidelidad 89/90 idénticas + 1 diferencia de conversión (	extcite).
- 2026-11-18: con permiso del usuario, PyMuPDF en carpeta aislada del scratchpad; revisión visual de las 6 páginas. Corregido: separadores de la fórmula de clip y de N(0; σ²) (icomma), «s/ d» → «s. d.». Log limpio, fidelidad 95/95, cuerpo en 4 páginas. informe/informe.pdf actualizado.
- 2026-11-18: fase 2 terminada. Paquetes instalados en TinyTeX desde el archivo histórico de TL2025 (microtype, icomma/was, csquotes, biblatex, biber, caption, listings, xstring, logreq, newunicodechar, hyphen-catalan). Fidelidad 95/95 frases; refs sin errores; log sin errores ni cajas desbordadas. Márgenes 2 cm, T1 antes de babel, l·l con \l., signos Unicode con newunicodechar, cabecera agrupada en la Taula 2. PDF: portada + 4 páginas de cuerpo + referencias (p. 6). Copia en informe/informe.pdf.
- 2026-11-18: FALLO en fase 3: no hay pdftoppm/ghostscript/PyMuPDF y el visor del navegador integrado solo da instantáneas estáticas; revisado visualmente solo portada e inicio de la p. 2.
- 2026-11-18: auditoría ronda 2 APTO; menores corregidos; extensión re-medida (cuerpo en 4 páginas a 2 cm).
- 2026-11-18: 1.4 esquema aprobado (usuario).
- 2026-11-18: 1.5 figuras y 1.6 estilo aprobados; 1.7 redactado; 1.8 lint y refs revisados (avisos restantes: rayas = celdas vacías de tabla; frases largas = enumeraciones de hiperparámetros; gymnasium sin año). Extensión estimada ~3,9 páginas a 2,5 cm; se verificará en la fase 2.
- 2026-11-18: fase 0 (estructura creada en APRNS-repo/informe/, movida a lab7/informe/), 1.1 contexto recogido, 1.2 hechos.md rellenado; añadido lab7/analysis/hovering.py (comprobación de comportamiento, sin commit).
