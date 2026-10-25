# Samgeo

*[English](README.md)*

Un plugin di [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry)
(1.5.1 o successivo): quello che scrivi - *pietre*, *mattoni*, *tegole e coppi*, o una frase
qualsiasi in inglese - trovato nell'ortofoto del chunk, o in un raster qualsiasi che il calcolatore
raster legge. Il motore predefinito segue la ricetta di `LangSAM` di
[segment-geospatial](https://github.com/opengeos/segment-geospatial) (Qiusheng Wu):
[Grounding DINO](https://huggingface.co/IDEA-Research/grounding-dino-base) (IDEA Research) trova i
riquadri di quello che è scritto, [SAM 2.1](https://github.com/facebookresearch/sam2) (Meta) disegna
la maschera in ogni riquadro. Girano entrambi sul tuo computer, con pesi aperti e senza account.

Quello che i modelli trovano è un **layer di proposte** nella vista 2D, da controllare prima che
diventi altro:

| Da una ricerca | Che cosa ottieni |
|---|---|
| **Il layer** `samgeo: <nome>` | Gli oggetti trovati, ciascuno con il suo numero, il testo, la frase che l'ha trovato, il punteggio (la confidenza del motore selezionato) e l'area; disegnato di nuovo su ogni raster che la vista mostra; una proposta eliminata nella vista (*Seleziona* e Canc) esce anche dalla ricerca |
| **US** | Le proposte (quelle selezionate nella vista, quelle non ancora US, o tutte) diventano unità stratigrafiche del chunk, del tipo scelto (US, USM...), con il testo e il punteggio come attributi (`samgeo_prompt`, `samgeo_score`): nel collegamento dal vivo con QGIS, numerate da *Proponi US e rapporti*, nelle schede US di PyArchInit; una sola voce di *Modifica > Annulla* per tutte. Le proposte già US restano nel layer con il solo contorno; quando lo sono tutte, il layer viene nascosto |
| **Classi della nuvola densa** | I punti dentro le proposte (in pianta) prendono la classe `samgeo: <nome>` (codici da 150); i punti classificati a mano tengono la loro; rifatta, la classe segue le proposte rimaste |
| **GeoPackage** | Le ricerche scelte, un layer ciascuna (`MULTIPOLYGON`: `n`, `prompt`, `label`, `score`, `quality`, `area_m2`) |
| **La maschera** | Gli oggetti numerati in un GeoTIFF che il calcolatore raster trova da solo (`samgeo_<ricerca>_mask`) |

Una ricerca chiede:

* **cosa cercare**: una voce pronta (*pietre*, *muri*, *mattoni*, *tegole e coppi*, *frammenti
  ceramici*, *ossa*, *tessere di mosaico*, *alberi*, *cespugli*, *persone*, *attrezzi e secchi*,
  *teli e coperture*), data ai modelli in inglese, o un testo tuo - in inglese funziona meglio; più
  cose separate da virgole (`stone, brick`);
* **il raster**: l'ortofoto, o un raster qualsiasi del chunk (una visualizzazione del rilievo del DEM,
  per tumuli e fossati: in via sperimentale - i modelli sono stati addestrati su fotografie);
* **dove**: tutto il raster, un poligono che disegni (con lo strumento di disegno del programma), o la
  parte visibile nella vista 2D;
* la **confidenza minima** (0,30), l'**area minima** e **massima** in m², la **risoluzione** (quella
  del raster, non più fine di 1 cm/px - basta per le pietre; più grossolana è più veloce), i
  **modelli**: normali (Grounding DINO base, SAM 2.1 small), veloci (tiny, tiny), precisi (base, large).

## SAM 3: licenza, accesso e token personale

Dalla versione 0.2.0 il campo **Motore** permette di scegliere **SAM 3**. Il codice del plugin è
GPL-2.0-or-later; **SAM 3 e i suoi pesi hanno la distinta [licenza SAM di Meta](https://github.com/facebookresearch/sam3/blob/main/LICENSE), non GPL**.
Il plugin non include né redistribuisce questi pesi. Installare il plugin o i suoi pacchetti non
concede accesso al modello e non equivale ad accettarne i termini per conto dell'utente.

1. Accedi a Hugging Face e apri [facebook/sam3](https://huggingface.co/facebook/sam3).
2. Leggi le condizioni, richiedi l'accesso a Meta e attendi l'approvazione.
3. Crea un **[token personale Hugging Face](https://huggingface.co/settings/tokens)** con permesso di lettura del modello autorizzato.
   Il token viene da Hugging Face: **avere un token non sostituisce l'approvazione di Meta**.
4. Nel pannello scegli **SAM 3**, installa i pacchetti del motore e inserisci il token nel campo
   protetto. Il plugin lo mantiene soltanto in memoria per la sessione; non lo salva nel progetto,
   nei parametri delle ricerche o nei log. Lasciando il campo vuoto usa `HF_TOKEN` oppure il login
   Hugging Face già presente sul computer.
5. Prova prima una piccola area. Al primo uso i pesi vengono scaricati nella cache Hugging Face;
   poi possono essere riutilizzati localmente. Restano soggetti alla licenza SAM.

Per la riga di comando, usa `hf auth login` nell'ambiente dei modelli oppure configura `HF_TOKEN`
nel tuo ambiente locale. Non inserire il token come argomento di `ppg` o in un file di progetto.

```bash
ppg plugin samgeo access
ppg plugin samgeo install --sam3
ppg plugin samgeo status --engine sam3
ppg plugin samgeo segment scavo.ppg "stone wall" --engine sam3 --device cuda
```

I profili *Normali / Veloci / Precisi* valgono per Grounding DINO + SAM 2.1. Per SAM 3 viene usato
`facebook/sam3` con backend `transformers`; modello, backend, versioni dei pacchetti e dispositivo
sono registrati nella ricerca. I punteggi dei due motori non sono direttamente confrontabili.
Tutte le uscite restano proposte da verificare prima della conversione in US.

## Come funziona una ricerca

1. L'area viene letta alla risoluzione della ricerca, in tasselli di 1024 px sovrapposti di 128 px.
   Tre bande a 8 bit si leggono come sono; ogni altro raster (16 bit, riflettanze, una banda) viene
   stirato tra il 2° e il 98° percentile dell'area, una banda in grigio.
2. Ogni tassello va ai modelli; ogni maschera diventa un poligono nelle coordinate del chunk.
3. Un oggetto tagliato dal bordo di un tassello viene unito alla sua altra parte del tassello vicino
   (le due coincidono nella striscia che i tasselli hanno in comune); due riquadri sullo stesso
   oggetto lasciano una sola proposta; quando l'area ha più di un tassello, un riquadro grande quanto
   un tassello intero viene scartato (Grounding DINO spesso risponde "tutta l'immagine" accanto agli
   oggetti). Poi i filtri di punteggio e di area e, con un poligono, solo quello che vi cade dentro.
4. La ricerca resta nel progetto, i suoi file nella cartella del chunk: `samgeo/<ricerca>.gpkg` e
   `samgeo/<ricerca>_mask.tif`.

## Dove

* il pannello **Samgeo** (accanto all'area di lavoro; *Vista > Pannelli* lo riapre): cosa cercare,
  *Trova*, le ricerche del chunk attivo (la casella mostra o nasconde un layer, il doppio clic lo
  inquadra), *US...*, *Classi della nuvola*, *GeoPackage...*, *Inquadra*, *Rimuovi*, *Installa i
  modelli...*;
* *Plugin > Samgeo* nel menu; le ricerche sotto ogni chunk nell'area di lavoro;
* la riga di comando:

```bash
ppg plugin samgeo status
ppg plugin samgeo install                                  # PyTorch e i modelli (uv)
ppg plugin samgeo segment scavo.ppg pietre
ppg plugin samgeo segment scavo.ppg "roof tile" --area 2480000,4500000,2480004,4500003 --units
ppg plugin samgeo segment scavo.ppg muri --resolution 0.02 --min-area 0.5 --models precise --json
ppg plugin samgeo segment scavo.ppg mound --raster dem --threshold 0.25
ppg plugin samgeo list scavo.ppg
ppg plugin samgeo export scavo.ppg samgeo.gpkg --run pietre
ppg plugin samgeo remove scavo.ppg pietre
```

`--chunk ETICHETTA` lavora su un chunk diverso da quello attivo; `--raster` prende il nome di un
raster nel calcolatore raster.

## Come girano i modelli

PyTorch non gira mai nel processo del programma (il suo runtime OpenMP e quello di pycolmap fanno
chiudere il processo quando partono entrambi): i modelli girano in un processo loro
(`ppg_samgeo/worker.py`), avviato una volta per ricerca e nutrito un tassello alla volta; annullare
lo ferma subito. Il worker gira con:

* l'ambiente dei plugin AI, installato dal plugin a richiesta con [uv](https://github.com/astral-sh/uv):
  Python 3.12, PyTorch (la versione per la tua scheda grafica, scelta da `uv --torch-backend auto`:
  CUDA con una scheda NVIDIA, Metal su Apple Silicon, altrimenti il processore), transformers e
  sam2, circa 2 GB, in `<dati dell'applicazione>/pyarchinit-photogrammetry/geoai-env`
  (`PPG_GEOAI_ENV` per metterlo altrove). Il plugin geoai usa lo stesso ambiente, così PyTorch c'è
  una volta sola. L'applicazione pacchettizzata funziona così;
* il Python del programma, quando vi sono installati PyTorch, transformers e sam2.

I pesi dei modelli (circa 1 GB: Grounding DINO base e SAM 2.1 small) si scaricano da Hugging Face
alla prima ricerca e restano nella sua cache (`HF_HOME`). Sul processore un tassello richiede
decine di secondi; una GPU è molto più veloce.

## Limiti

* I testi in inglese funzionano meglio; le voci pronte le traduce il plugin.
* I modelli trovano oggetti - pietre, mattoni, tegole, alberi, persone, attrezzi - molto meglio del
  contrasto tra due terreni: quello è compito di geoai, che impara dai tuoi disegni.
* Con un'area di un solo tassello (un poligono piccolo) Grounding DINO può rispondere con l'area
  intera: elimina quella proposta, o imposta un'area massima.
* SAM 3 usa il backend Transformers di SamGeo3. CUDA è il percorso da preferire; CPU e Metal
  richiedono verifica sul proprio computer, e Metal è sperimentale. Il plugin non cambia motore
  silenziosamente in caso di errore.

## Installazione

* Nel programma: *Strumenti > Plugin...*, `samgeo`, *Installa / aggiorna*, *Riavvia ora*; poi
  *Installa i modelli...* nel suo pannello.
* Con pip, nell'ambiente Python del programma:

  ```bash
  pip install "ppg-samgeo @ git+https://github.com/enzococca/pyarchinit-photogrammetry-plugins#subdirectory=plugins/samgeo"
  ```

Test: `pip install -e . pytest` nell'ambiente del programma, poi `pytest tests`; usano un motore
finto del worker (le macchie chiare di un'ortofoto sintetica sono le pietre), senza PyTorch.

Licenza del plugin: GPL-2.0-or-later. Le dipendenze mantengono le proprie licenze; SAM 3 e i suoi
pesi sono soggetti alla licenza SAM di Meta, distinta dalla GPL (vedi sopra). Autore: Enzo Cocca.

## Identificativi, gruppi e ricognizione (app 1.5.1)

Samgeo 0.3 integra l'anteprima di numerazione e nomi dell'app 1.5.1: le proposte selezionate o tutte
quelle non ancora convertite possono ricevere numeri progressivi, nomi progressivi oppure un nome
comune di gruppo, mantenendo identificativi e geometrie distinti. I numeri già occupati vengono saltati.
Scegli UT per la ricognizione e indica il progetto: la TOC usa “Unità topografiche”, con il gruppo
quando assegnato. Per le unità già create usa Layer > Identifica / raggruppa...; dal menu Seleziona
attiva la selezione a mano libera (lazo). Maiusc/Cmd aggiunge, Alt sottrae. La modifica in blocco
si annulla con una sola voce di Modifica > Annulla. Le UT usano ut_table e non sono pubblicate come US.
Sulle app precedenti resta il dialogo originale; il nuovo flusso richiede l'aggiornamento dell'app.
