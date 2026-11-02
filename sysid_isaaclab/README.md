# Note di lavoro — sysid Isaac Lab (25/10/2026)

Stato del lavoro su ottimizzatore OT, scala logaritmica e parametri inerziali, per riprendere da qui.

**Contesto.** `my_fit.py --optimizer ot` usa [ot_optimizer.py](ot_optimizer.py), basato su arXiv:2605.02147 (OT + Sinkhorn), con l'obiettivo "loss media − λ·diversità, con limite sulla loss". Restituisce un set di combinazioni plausibili in `plausible_params.pt`. Rispetto al paper ci sono due modifiche:
- `marginal_relaxation=0.75`: con l'OT bilanciato il set collassava su un solo bacino;
- covarianza delle proposte adattata dal coupling: con proposte isotrope, a 49 parametri il minimo trovato era peggiore di CMA-ES.

Il set copre la regione plausibile, non è una distribuzione: usato in modo uniforme dà un domain randomization robusto. La manopola da scegliere bene è `loss_tolerance`, validando su una traiettoria non usata nel fit.

Tutto è fuori dal submodule: dentro `pace-sim2real` ci sono solo modifiche precedenti. **Testato solo offline**, con un'articolazione finta che riproduce l'API di Isaac Lab 3 e i valori dell'USD di aliengo, **non in Isaac Sim**.

## Scala logaritmica
- **Tolta per i parametri dei giunti** (26/10): armatura, attrito, smorzamento, bias e ritardo sono mappati linearmente sui bounds, come in PACE, e CMA-ES si comporta come l'originale. Con un'armatura nota (per esempio 0.33–0.9 su Pegasus) conviene stringere i bounds per giunto intorno al nominale.
- Resta solo sui fattori di massa e inerzia, dove tiene il nominale al centro dei bounds.

## Massa, CoM e inerzia

Le opzioni hanno gli stessi nomi e lo stesso significato dello script MuJoCo:
```bash
python3 sysid_isaaclab/my_fit.py --optimizer ot --identify-inertia-tensor physics=physx
```
- `--identify-link-mass`: solo la massa.
- `--identify-center-of-mass`: massa e CoM.
- `--identify-inertia-tensor`: massa, CoM e inerzia.

I parametri sono correzioni dei valori nominali dell'USD:
- **Massa:** un fattore di scala, e l'inerzia scala con la massa.
- **CoM:** un offset in metri nel frame del corpo.
- **Inerzia:** tre fattori sui momenti secondi principali della distribuzione di massa. Qualunque valore dentro i bounds dà un corpo fisicamente consistente (vale la disuguaglianza triangolare), quindi l'ottimizzatore non può proporre inerzie impossibili.

Altri dettagli:
- **Gambe condivise:** di default FL/FR/RL/RR condividono i parametri, quindi su aliengo sono 28 parametri in più invece di 112. Gli offset del CoM vengono specchiati dove il nominale è specchiato: la x dell'hip tra davanti e dietro, la y tra sinistra e destra.
- **Corpi:** di default tutti quelli con massa tranne la base fissa; con `--inertial-bodies` li restringi.
- **Risultati:** i valori identificati sono in `body_params.pt` (CMA-ES) oppure sotto `inertial` in `plausible_params.pt` (OT).

Il codice è in [inertial_params.py](inertial_params.py); [my_fit.py](my_fit.py) e il README principale sono aggiornati.

## Test offline
- Con i parametri al nominale il simulatore riceve esattamente i valori originali.
- Su valori casuali dentro i bounds l'inerzia resta sempre consistente.
- In un fit completo, con una massa vera pari a 1.3× il nominale, entrambi gli ottimizzatori recuperano 1.3.

## Differenze rispetto a MuJoCo e limiti
- **Assi principali fissi:** niente "shear", perché PhysX salva l'inerzia come diagonale nel frame principale.
- **Bounds di default 0.5–2.0 per massa e inerzia** (MuJoCo usa 0.5–5.0 e 0.5–1.5). Sono simmetrici in scala logaritmica, così il centro della ricerca è il modello nominale.
- **Newton e CoM:** Isaac Lab avverte che cambiare il CoM a runtime con Newton può essere instabile, e Newton è il default. Con CoM o inerzia usare `physics=physx`.
- **Identificabilità:** a base fissa molte di queste grandezze non sono identificabili separatamente. Per esempio armatura e inerzia del link attorno all'asse del giunto si compensano, e così il piede e il calf. Conviene partire da `--identify-link-mass` o `--identify-center-of-mass`. Con l'OT lo si vede direttamente dai range nel set plausibile.

## Bug corretto nell'OT
Per a2, go2 e pegasus il bias ha bounds [0, 0]. Quelle 12 dimensioni non hanno nessun effetto fisico, ma venivano contate come diversità. Ora i parametri fissi sono esclusi da diversità e campionamento.

## Bug corretto nel fit MuJoCo (gambe condivise)
Con `--tie-quadruped-inertias` (attivo di default) il parametro condiviso, che contiene valori *assoluti* di FL, veniva copiato così com'era su FR/RL/RR. Con CoM o inerzia le altre gambe ricevevano il CoM e i prodotti d'inerzia di FL con i segni sbagliati, già al valore nominale: circa 4.5 cm di errore sul CoM dell'hip di aliengo e del thigh di go2. Per la sola massa non cambiava nulla, perché le quattro gambe hanno la stessa massa.

Ora il parametro contiene ancora i valori di FL, ma ogni altra gamba riceve la stessa correzione dei *propri* valori nominali ([sysid_mujoco/common.py](../sysid_mujoco/common.py), `shared_body_inertia_maps`):
- la massa è scalata dello stesso fattore;
- l'offset del CoM è specchiato con la stessa regola di Isaac Lab;
- nella pseudo-inerzia log-massa e stretch sono condivisi, shear e CoM specchiati (una riflessione S trasforma il fattore di Cholesky U in S U S).

Verificato offline su aliengo e go2: al nominale ogni gamba riproduce il modello originale, e con un valore perturbato FR/RL/RR sono lo specchio di FL. **I fit MuJoCo con CoM o inerzia e gambe condivise fatti prima di questa correzione vanno rifatti.**

## PACE v0.2.0 e più traiettorie (02/11/2026)
Il submodule è passato a PACE v0.2.0. Adattato tutto fuori dal submodule:
- **Delay sul comando:** [tasks/pace_actuator.py](tasks/pace_actuator.py) ritarda solo i target (posizione, velocità, effort) e il PD agisce sulle misure attuali, come il default di v0.2.0 e il delay del fit MuJoCo. L'opzione che ritarda la coppia è stata tolta. `basic-locomotion-isaaclab` ritarda ancora la coppia: per usare lì il delay identificato va allineato.
- **Frizione statica/dinamica separata:** `separate_dynamic_friction=True` nel `PaceCfg` del task aggiunge 12 rapporti dinamica/statica dopo il delay. Supportato da CMA-ES e OT, solo con PhysX (Newton ha un solo valore di attrito secco). Non ancora provato in Isaac Sim.
- **CMA-ES con parametri inerziali:** v0.2.0 controlla che i bounds siano esattamente 49 (o 61). `BackendAgnosticCMAESOptimizer` costruisce l'ottimizzatore sui parametri PACE e poi lo ricostruisce con quelli inerziali in coda.

**Più traiettorie** (`--dataset`, oppure `data_dir` del task che può essere un file, una cartella di `.pt` o una lista; go2 ora usa tutta `datasets/go2`):
- gli env sono divisi in K gruppi, uno per traiettoria: l'env `k * P + i` riproduce la traiettoria k con il candidato i, quindi i candidati per iterazione sono `num_envs // K`;
- il punteggio è la media dell'errore su tutti i passi registrati di tutte le traiettorie (con K=1 è identico a PACE); le traiettorie più corte tengono l'ultimo target e i passi dopo la fine non contano;
- `best_trajectory.pt` e il `config.pt` dell'ottimizzatore contengono solo la prima traiettoria; in `config.pt` c'è anche `data_files` con l'elenco completo;
- kp/kd devono essere uguali in tutti i dataset, altrimenti errore.

Verificato in Isaac Sim (isaaclab3_env, Newton, CMA-ES):
- stesso dataset una volta con 32 env e due volte con 64 env: punteggi identici per 4 iterazioni (0.20388, 0.18030, 0.17776, 0.17137), quindi aggregazione e normalizzazione sono corrette;
- go2 con i 5 dataset (5 × 12 = 60 env, `trajectory_1` più lunga di 143 passi): gira, circa 11 s per iterazione contro 10 s con una sola traiettoria.

In Isaac Lab 3 si lancia senza `--headless` e senza il `PYTHONPATH` dell'env pixi:
```bash
env -u PYTHONPATH ~/miniforge3/envs/isaaclab3_env/bin/python sysid_isaaclab/my_fit.py  # CMA-ES di default, --optimizer ot per l'OT
```

## Prossimi passi
1. **Primo run in Isaac Sim:** lanciare `--identify-link-mass` con bounds stretti, per esempio `--link-mass-scale-bounds 0.99 1.01`, e controllare che la loss coincida con un run senza identificazione. Se non coincide, il problema è nella scrittura delle proprietà nel simulatore.
2. **Confronto OT vs CMA-ES su aliengo** e validazione del set plausibile su una traiettoria non usata nel fit (per esempio `traj_1`).
