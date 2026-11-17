# Execució de l'estudi a Azure (2 VM)

Protocol fixat a `experiments/config.py`: 1 DDPG (llavor 0) + 4 variants TD3 × 10 llavors = 41
execucions, T = 300.000 transicions, 5 episodis d'avaluació cada 1.000 passos.
Repartiment per **llavors** (no per variants), perquè una diferència entre màquines no es
confongui amb l'efecte d'una variant:

| VM | Regió | Execucions | Ordre |
|---|---|---|---|
| `vm-es` | Spain Central | 21: DDPG + llavors 0–4 | `bash cloud/setup_linux.sh --workers 4 --seeds 0 1 2 3 4` |
| `vm-be` | Belgium Central | 20: llavors 5–9, sense DDPG | `bash cloud/setup_linux.sh --workers 4 --seeds 5 6 7 8 9 --variants full no_minimum no_smoothing no_delay` |

## 0. Comprovacions (Cloud Shell, Bash; "No storage account required" és suficient)
```bash
az vm list-skus -l spaincentral   --size Standard_D4s_v4 -o table
az vm list-skus -l belgiumcentral --size Standard_D4s_v4 -o table
az vm list-usage -l spaincentral -o table | grep -i "DSv4\|Total Regional"
```
Si una regió no ofereix D4s_v4 (o la subscripció no la permet), feu servir `Standard_D4s_v5`
a les **dues** VM, perquè el maquinari sigui del mateix tipus.

## 1. Crear el grup i les VM
`MYIP` és la IP del **vostre PC** (mireu-la al navegador, p. ex. ifconfig.me); des de Cloud
Shell `curl` donaria la IP d'Azure. `SSHKEY` és el contingut de `~/.ssh/id_ed25519.pub` del PC
(creat amb `ssh-keygen -t ed25519`).
```bash
RG=aprns-lab7
MYIP=X.X.X.X
SSHKEY="ssh-ed25519 AAAA... usuari@pc"
az group create -n $RG -l spaincentral
for spec in "vm-es spaincentral" "vm-be belgiumcentral"; do
  set -- $spec
  az vm create -g $RG -n $1 -l $2 --image Ubuntu2404 --size Standard_D4s_v4 \
    --admin-username azureuser --ssh-key-values "$SSHKEY" \
    --storage-sku StandardSSD_LRS --os-disk-size-gb 30 --priority Regular \
    --public-ip-sku Standard --nsg-rule NONE
  az network nsg rule create -g $RG --nsg-name ${1}NSG -n ssh-my-ip --priority 1000 \
    --source-address-prefixes $MYIP/32 --destination-port-ranges 22 --protocol Tcp --access Allow
done
az vm list-ip-addresses -g $RG -o table
```
Si canvia la IP del PC: `az network nsg rule update -g $RG --nsg-name vm-esNSG -n ssh-my-ip --source-address-prefixes NOVA_IP/32` (i igual per a `vm-beNSG`).

## 2. Llançar (a cada VM)
```bash
ssh azureuser@IP_VM
git clone https://github.com/laia-beyloc/APRNS && cd APRNS/lab7   # repo privat: cal un token
bash cloud/setup_linux.sh <arguments de la taula>
```
`setup_linux.sh` instal·la `requirements.txt` (mateixes versions que en local, torch de CPU),
passa els tests (~2 min) i llança el llançador amb `nohup` (es pot tancar l'SSH).
Abans de l'estudi, opcionalment, mesurar el ritme real de la VM (~15 min):
`. .venv/bin/activate && python -m experiments.benchmark --parallel 4`.

Seguiment: `tail -f launch.out` · estat: `. .venv/bin/activate && python -m experiments.launch --dry-run <mateixos --seeds/--variants>`.

**Tall o reinici de la VM:** tornar a executar la mateixa ordre de la taula. Les execucions
completades se salten i les interrompudes continuen des del seu `checkpoint_latest.pt`
(es desa cada 25.000 passos; es perden com a màxim ~25.000 passos per execució en curs, i la
continuació és exacta: mateixos CSV que sense tall). No llanceu dues vegades alhora.

## 3. Recollir els resultats (des del PC)
```bash
ssh azureuser@IP_VM "cd APRNS/lab7 && tar czf results.tgz --exclude='*.pt' --exclude='*.log' results"
scp azureuser@IP_VM:APRNS/lab7/results.tgz results_vm.tgz
```
Descomprimiu cada fitxer dins `lab7/` (`tar xzf results_vm.tgz`); les carpetes de les dues VM
no col·lisionen. Comproveu amb `python -m experiments.launch --dry-run` que les 41 surten
`done` i genereu l'anàlisi: `python -m analysis.analysis`.
Els CSV/JSON de `results/` es versionen i es lliuren; els `.pt` i els logs, no.

## 4. Aturar la despesa
```bash
az group delete -n aprns-lab7 --yes --no-wait   # esborra VM, discos, IP i xarxa
```
`az vm deallocate` només atura el còmput: el disc i la IP continuen cobrant.

## 5. Pressupost
Portal → Cost Management → Pressupostos: crear-ne un (≈25 US$) amb alertes al 50/80/100 %.
L'alerta no atura res i arriba amb hores de retard; si s'acaba el crèdit d'Azure for Students
la subscripció es desactiva i les execucions s'aturen (es poden continuar després gràcies als
checkpoints, si encara hi ha VM).

## 6. Prova de control fora de protocol (10 execucions)
TD3 complet amb canvis declarats (`SANITY_CONFIGS` a `experiments/config.py`), llavors 0–4
(aparellades amb `full` 0–4 de l'estudi). No forma part de l'ablació ni de `analysis.py`.

| Nom | Canvi respecte de `full` | Pregunta |
|---|---|---|
| `td3_paper` | hiperparàmetres de [2]: τ = 0,005, lr 1e-3, sense weight decay, batch 100, 10.000 passos aleatoris inicials | la implementació aprèn a aterrar amb la configuració de l'article de TD3? |
| `no_weight_decay` | només sense weight decay als crítics | el weight decay explica la infraestimació i el planeig? |

Totes les execucions noves desen `final_model.pt` (xarxes finals, no es versiona) i
`final_evaluation.csv` (100 episodis amb llavors noves; resum a `config.json`).

Amb 2 VM de 4 vCPU (mateixa creació que l'apartat 1; ~14–18 h de paret):
```bash
bash cloud/setup_linux.sh --workers 4 --seeds 0 1 2 --variants td3_paper no_weight_decay   # VM 1: 6 execucions
bash cloud/setup_linux.sh --workers 4 --seeds 3 4   --variants td3_paper no_weight_decay   # VM 2: 4 execucions
```
Amb 1 sola VM: `--seeds 0 1 2 3 4` (3 tandes, ~21–27 h). Estat: `python -m experiments.launch
--dry-run --seeds 0 1 2 3 4 --variants td3_paper no_weight_decay`. Recollida i tancament com als
apartats 3 i 4 (el `tar` ja exclou els `.pt`; si es volen les xarxes finals per gravar vídeos,
feu un segon `tar` de `results/*/seed_*/final_model.pt`).

## A l'informe
Cada `config.json` registra dispositiu i versions. Les dues VM són del mateix tipus, però en
regions diferents (el model de CPU pot variar): cal declarar-ho.
