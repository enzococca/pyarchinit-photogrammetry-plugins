# Paesaggio nel tempo (Landlab)

*[English](README.md)*

**[Guida d'uso](https://enzococca.github.io/pyarchinit-photogrammetry/it/plugins/landscape/)**: il DEM da cui partire, lo scenario passo per passo, come leggere i risultati.

Plugin di [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry)
(1.3.4 o successiva): come cambia negli anni la superficie rilevata per il **creep del suolo**
(diffusione del versante) e per l'**acqua che scorre** (erosione e deposito), simulati con
[Landlab](https://landlab.csdms.io) (CSDMS, licenza MIT) sul DEM del chunk. Come si consuma un tell o
un tumulo, dove si raccoglie il colluvio al piede di un pendio o in una conca, quanto se ne va di una
superficie di scavo esposta in dieci o cento anni.

**Scenari, non previsioni.** Il risultato dipende dai parametri, che nel migliore dei casi si
conoscono entro un ordine di grandezza: va letto come "che cosa farebbero questi processi, a queste
velocità, a questa superficie". Confrontalo con il DEM di una fase successiva per tarare i
parametri. Il tempo va solo in avanti: una superficie più antica non si ricostruisce facendo girare il
modello all'indietro (la diffusione all'indietro non è fisica, amplifica ogni errore) e il plugin non
lo propone. Non simulati: sollevamento tettonico, piogge che cambiano nel tempo, vegetazione, vento,
spostamento laterale dei canali, il mare, quello che sta sotto la superficie.

## I processi

| Processo | Componente di Landlab | Parametri |
|---|---|---|
| **Creep del suolo** (diffusione del versante, Culling 1963) | `LinearDiffuser`: il flusso di suolo è `D` per la pendenza | `D` (m²/anno) |
| **Acqua che scorre** (SPACE, Shobe et al. 2017) | `SpaceLargeScaleEroder` (poi `Space`, poi `ErosionDeposition` se manca): erosione di uno strato di suolo sciolto e del materiale sottostante, trasporto e deposito del sedimento | `K_sed`, `K_br`, `F_f`, `phi`, `H_star`, `v_s`, `m_sp`, `n_sp`, ruscellamento, spessore del suolo |
| Percorso dell'acqua | `PriorityFloodFlowRouter` (richdem) quando i dati arrivano al bordo della griglia, altrimenti `FlowAccumulator` + `DepressionFinderAndRouter`: D8, depressioni riempite; portata = area drenata x ruscellamento | ruscellamento (m/anno) |

Si può spegnere uno dei due processi (non entrambi). Le celle senza valore sono chiuse. Il bordo dei
dati è **aperto** (acqua e sedimento escono, come su un pendio che continua oltre il rilievo) o
**chiuso** (non esce niente; con l'acqua che scorre la cella di bordo più bassa accanto a una cella
interna resta aperta come unico sbocco).

## Scenari predefiniti

I valori sono punti di partenza difendibili scelti per il plugin, nella parte alta degli intervalli
pubblicati per materiale sciolto e senza vegetazione. Sono **indicativi**: tarali con un DEM
successivo quando ce l'hai.

| Scenario | Anni | D (m²/anno) | K_sed | K_br | F_f | phi | H* (m) | v_s (m/anno) | ruscellamento (m/anno) | suolo (m) |
|---|---|---|---|---|---|---|---|---|---|---|
| *Tell in mattoni crudi* | 1000 | 0,01 | 5e-4 | 1e-4 | 0,5 | 0,3 | 0,5 | 1 | 0,2 | 0,5 |
| *Versante in terra* | 1000 | 0,005 | 1e-3 | 2e-4 | 0,2 | 0,3 | 0,5 | 1 | 0,3 | 0,5 |
| *Scavo esposto* | 10 | 0,02 | 2e-3 | 5e-4 | 0,3 | 0,3 | 0,2 | 1 | 0,3 | 0,2 |
| *Personalizzato* | qualsiasi valore; cambiando un valore di uno scenario diventa personalizzato ||||||||||

`m_sp` = 0,5 e `n_sp` = 1 in tutti (gli esponenti consueti della legge di erosione).

### Da dove vengono i valori

* **Diffusività del versante D.** Il modello a diffusione del creep è di Culling (1963, *Journal of
  Geology* 71). I valori misurati su versanti con copertura di suolo stanno per lo più fra 1e-4 e
  1e-2 m²/anno: Roering, Kirchner e Dietrich (1999, *Water Resources Research* 35) hanno trovato circa
  3e-3 m²/anno nella Oregon Coast Range; il degrado delle scarpate di faglia e delle antiche linee di
  riva dà circa 1e-3 m²/anno (Hanks et al. 1984). Il materiale nudo e sciolto si muove più in fretta
  del suolo con vegetazione: gli scenari mettono un tell in mattoni crudi a 1e-2 e uno scavo fresco e
  non protetto a 2e-2, apposta sopra l'intervallo naturale; un versante in terra inerbito a 5e-3.
* **Erodibilità K_sed e K_br.** In SPACE (Shobe, Tucker e Barnhart 2017, *Geoscientific Model
  Development* 10) il suolo è eroso a `K_sed Q^m S^n (1 - exp(-H/H*))` e il materiale sotto a
  `K_br Q^m S^n exp(-H/H*)`. Le loro unità dipendono da `m`, `n` e dalla portata (qui in m³/anno),
  quindi un valore è confrontabile solo con gli stessi esponenti e le stesse unità; il suolo sciolto
  è più erodibile del materiale sotto (`K_sed` maggiore di `K_br`, di 5 volte negli scenari). Non
  esiste una tabella generale: gli esempi che accompagnano SPACE in Landlab usano valori da 1e-5 a 1
  (quelli predefiniti del componente sono 0,02), pensati per bacini che evolvono in migliaia di anni.
  Gli scenari restano bassi (5e-4 - 2e-3) perché i pendii di un tell sono corti e i mattoni crudi
  sono coesivi. Sono i due valori meno noti: tarali.
* **F_f**, la frazione del materiale eroso sotto il suolo così fine da andare via in sospensione senza
  depositarsi (0-1): alta per mattoni crudi e riempimenti limosi (0,5), più bassa per terra più
  grossolana.
* **phi**, la porosità del sedimento depositato: 0,3 è un valore comune per sabbie e ghiaie (0,2-0,5
  per i sedimenti naturali).
* **H\***, lo spessore di suolo oltre il quale l'acqua non raggiunge più il materiale sotto
  (`exp(-H/H*)`): 0,5 m negli scenari, 0,2 m per il piano di uno scavo dove c'è poco suolo sciolto.
* **v_s**, la velocità efficace di deposito (m/anno): con v_s molto maggiore del ruscellamento il
  sedimento si deposita vicino a dove è stato eroso (limitato dal trasporto), molto minore viene
  portato via (limitato dal distacco); Shobe et al. (2017) discutono questo passaggio. 1 m/anno con
  0,2-0,3 m/anno di ruscellamento sta fra i due.
* **Ruscellamento**, la pioggia che scorre in superficie in un anno, uniforme (m/anno): 0,2-0,3 m è un
  clima da semiarido a mediterraneo, tolta l'infiltrazione.

## Il risultato sul DEM

La simulazione gira su una griglia sua: il DEM mediato su una cella più grande (in modo predefinito
la sua risoluzione ingrandita fino ad avere al massimo 500 celle sul lato lungo; oltre 1000 celle per
lato il pannello avverte). Il **cambiamento** fatto dalla simulazione - dz, e ogni istantanea meno la
partenza - viene interpolato sulla griglia del DEM (bilineare) e **sommato al DEM**: il dettaglio che
la simulazione più grossolana non poteva vedere (pietre, muri, i margini delle trincee) resta nella
superficie finale. Le celle del DEM che l'interpolazione non raggiunge, lungo il margine più esterno
dei dati, hanno dz 0. L'area drenata torna con il vicino più prossimo. Ogni strato è un GeoTIFF
float32 con NaN come nessun valore, sulla griglia, nel sistema e con il CRS del DEM; `dz.tif` è
scritto come differenza delle due superfici float32, così `finale - DEM = dz` vale anche nei file.

## I volumi

`balance.json` di ogni simulazione, e il pannello:

* volumi **apparenti** da dz: `eroded_m3`, `deposited_m3`, `net_m3` - quello che mostra la mappa,
  vuoti compresi;
* volumi **solidi**, per cella `(1 - phi) dH + d(substrato)`: `eroded_solid_m3`, `deposited_solid_m3`
  (il suolo è poroso, il materiale sotto no; con il solo creep solido = apparente);
* quello che esce: `bedload_out_m3` e `wash_load_m3` (i fini F_f), la loro somma `fluvial_out_m3`, e
  `diffusion_out_m3` / `diffusion_out_solid_m3` (il creep attraverso i bordi aperti; quello solido
  conta anche i vuoti che si formano quando il creep trasforma in suolo il materiale sotto);
* `closure_m3` = solido eroso - solido depositato - tutto quello che esce, e `closure_percent` del
  solido eroso: circa 0 quando i conti tornano (lo controllano i test, in ogni combinazione di bordi,
  F_f e phi). Vuoto con `Space` o `ErosionDeposition`, che non dicono quanto esce.

Eroso e depositato apparenti differiscono quando il sedimento si deposita (i vuoti) o esce:
confronta quelli solidi.

## Quanto dura

Prima di una simulazione il pannello mostra una stima: i sottopassi della diffusione (esatti: nessun
passo più lungo di 0,14 dx²/D), un limite superiore dei passi dell'acqua (ogni passo più corto di un
limite di tipo CFL, dall'incisione più rapida possibile: tutta la griglia che drena in una cella) e
i secondi, dai costi misurati su un Apple M4 Pro proporzionati alle celle. Oltre 10 minuti avverte,
oltre 2 ore rifiuta (`--force` dalla riga di comando). La stima è un tetto: i passi veri seguono la
portata vera e di solito sono molti meno. Una cella più grande accorcia molto una simulazione (i
passi della diffusione vanno con il quadrato della cella).

## Dove

* il pannello **Paesaggio nel tempo**, a destra accanto a *PyArchInit* (*Vista > Pannelli*): il DEM
  da cui partire (quello attivo o una versione conservata), lo scenario, gli anni, le istantanee, la
  cella, il bordo, il passo, i due processi con i loro parametri (quelli avanzati dell'acqua dietro
  una casella), la stima, *Avvia la simulazione*; le simulazioni del chunk in una tabella, i loro
  strati sopra la vista 2D (superficie finale, erosione e deposito con una scala divergente, area
  drenata, istantanee, differenze), opacità, *Aggiungi come versione del DEM*, *Confronta con un altro
  chunk...*, *Esporta...*, *Rimuovi*;
* le simulazioni nell'albero dell'area di lavoro, nel progetto QGIS del programma, nel calcolatore
  raster (programma 1.4.0 o successivo); *File > Esporta > Paesaggio nel tempo (GeoTIFF)*;
* *Plugin > Paesaggio nel tempo (Landlab)* nel menu;
* la riga di comando:

```bash
ppg plugin landscape status
ppg plugin landscape install                        # l'ambiente del plugin (uv)
ppg plugin landscape estimate scavo.ppg --preset mudbrick_tell --years 1000
ppg plugin landscape run scavo.ppg --preset mudbrick_tell --years 1000 --snapshots 100,500 --name "Tell, 1000 anni"
ppg plugin landscape run scavo.ppg --preset custom --years 50 --D 0.02 --no-water --boundary closed
ppg plugin landscape run scavo.ppg --preset earth_slope --cell 2 --K-sed 0.002 --runoff 0.4
ppg plugin landscape list scavo.ppg
ppg plugin landscape show scavo.ppg r20261030-101500
ppg plugin landscape version scavo.ppg r20261030-101500       # la superficie finale fra le versioni del DEM
ppg plugin landscape compare scavo.ppg r20261030-101500 --with "Fase 2"
ppg plugin landscape export scavo.ppg r20261030-101500 paesaggio.tif            # tutti gli strati, una banda ciascuno
ppg plugin landscape export scavo.ppg r20261030-101500 dz.tif --layer dz
ppg plugin landscape remove scavo.ppg r20261030-101500
```

`--chunk ETICHETTA` lavora su un chunk diverso da quello attivo; `--source` parte da una versione
conservata del DEM; `--dt` fissa il passo più lungo; `--json` stampa tutta la voce. I parametri dello
scenario: `--D --K-sed --K-br --runoff --soil-depth --F-f --phi --H-star --v-s --m-sp --n-sp`,
`--no-diffusion`, `--no-water`. Gli scenari si scrivono con le chiavi inglesi: `mudbrick_tell`,
`earth_slope`, `exposed_excavation`, `custom`.

Il DEM attivo non viene mai sostituito: *Aggiungi come versione del DEM* mette la superficie finale
fra le versioni conservate (*Paesaggio dopo 1.000 anni (...)*), e curve di livello, sezioni, volumi
e rilievo la usano quando la rendi attiva. *Confronta* dà la superficie finale meno il DEM di un
altro chunk (una fase successiva) - mediana, NMAD e 95° percentile - accanto agli stessi numeri per
il DEM di partenza: i parametri si avvicinano quando la differenza si riduce.

## Come gira Landlab

Landlab e le sue dipendenze compilate (richdem, netCDF4) non vengono mai importati dal programma:
ogni simulazione gira in un processo a sé (`ppg_landscape/worker.py`, la simulazione in
`ppg_landscape/engine.py`). Il plugin gli scrive il DEM alla cella della simulazione, il worker
scrive gli array e i due si parlano con righe JSON (avanzamento, risultato, errore); *Annulla* lo
ferma subito e non lascia niente. Il worker gira con:

* il Python del programma, se Landlab è installato lì (`pip install landlab`);
* altrimenti un ambiente che il plugin installa su richiesta con
  [uv](https://github.com/astral-sh/uv) - Python 3.12 e Landlab 2.x da PyPI, circa 500 MB, nella
  cartella del plugin (`PPG_LANDSCAPE_ENV` per metterlo altrove): è così che funziona
  l'applicazione impacchettata, che non ha pip. Il pannello lo propone la prima volta.

Landlab ha versioni per macOS (Apple Silicon; Intel con macOS 15 o successivo), Linux (x86_64 e
aarch64) e Windows x64; non per Windows su ARM, dove il plugin dice che non è disponibile.

## Installazione

* Nel programma: *Strumenti > Plugin...*, `landscape`, *Installa / aggiorna*, *Riavvia ora*.
* Con pip, nell'ambiente Python del programma:

  ```bash
  pip install "ppg-landscape @ git+https://github.com/enzococca/pyarchinit-photogrammetry-plugins#subdirectory=plugins/landscape"
  pip install landlab        # facoltativo: altrimenti il plugin installa un ambiente suo
  ```

Test: `pip install -e . pytest` nell'ambiente del programma, poi `pytest tests`. Le simulazioni vere
e proprie girano dove Landlab si importa (`<env>/bin/python -m pytest tests` con un ambiente che ha
Landlab, numpy e pytest), e i test del worker vero usano il Python indicato da `PPG_TEST_LANDLAB`;
gli altri usano un worker che parla lo stesso protocollo senza Landlab.

## Bibliografia

* Culling, W. E. H. (1963). Soil creep and the development of hillside slopes. *The Journal of
  Geology* 71(2), 127-161.
* Hanks, T. C., Bucknam, R. C., Lajoie, K. R., Wallace, R. E. (1984). Modification of wave-cut and
  faulting-controlled landforms. *Journal of Geophysical Research* 89(B7), 5771-5790.
* Roering, J. J., Kirchner, J. W., Dietrich, W. E. (1999). Evidence for nonlinear, diffusive
  sediment transport on hillslopes and implications for landscape morphology. *Water Resources
  Research* 35(3), 853-870.
* Shobe, C. M., Tucker, G. E., Barnhart, K. R. (2017). The SPACE 1.0 model: a Landlab component for
  2-D calculation of sediment transport, bedrock erosion, and landscape evolution. *Geoscientific
  Model Development* 10, 4577-4604.
* Barnhart, K. R., et al. (2020). Short communication: Landlab v2.0: a software package for Earth
  surface dynamics. *Earth Surface Dynamics* 8, 379-397.

Licenza: **GPL-3.0-or-later** (richdem, che Landlab usa per il percorso dell'acqua, è GPL-3; il
programma e la maggior parte dei plugin di questo repository sono GPL-2.0-or-later, che lo permette).
Landlab ha licenza MIT. Autore: Enzo Cocca.
