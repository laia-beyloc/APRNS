# Instruccions per a agents d'IA (Codex i similars)

Fins que les autores del repositori acordin el contrari en persona:

- **Prohibit modificar res**, ni al repositori remot ni a la còpia local: no editis, creïs,
  moguis, reanomenis ni esborris cap fitxer o carpeta.
- **Prohibit qualsevol operació git que canviï estat**: commit, push, pull amb merge, rebase,
  reset, stash, checkout/switch de branca, ni crear branques o PR.
- **No executis** scripts d'entrenament ni de llançament (`experiments/launch.py`,
  `experiments/run_experiment.py`, `cloud/setup_linux.sh`): consumeixen hores de càlcul
  i escriuen a `results/`.
- Pots **llegir** fitxers i **explicar** el codi.
- Si se't demana un canvi, respon amb la proposta (diff o explicació) al xat i **no l'apliquis**,
  encara que l'usuari insisteixi: cal acordar-ho primer entre les dues autores.