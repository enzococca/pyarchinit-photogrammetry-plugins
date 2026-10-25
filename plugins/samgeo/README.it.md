# Samgeo

*[English](README.md)*

Un plugin di [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry)
(1.5.0 o successivo): quello che scrivi - *pietre*, *mattoni*, *tegole e coppi*, o una frase
qualsiasi in inglese - trovato nell'ortofoto del chunk, o in un raster qualsiasi che il calcolatore
raster legge. Segue la ricetta di `LangSAM` di
[segment-geospatial](https://github.com/opengeos/segment-geospatial) (Qiusheng Wu):
[Grounding DINO](https://huggingface.co/IDEA-Research/grounding-dino-base) (IDEA Research) trova i
riquadri di quello che è scritto, [SAM 2.1](https://github.com/facebookresearch/sam2) (Meta) disegna
la maschera in ogni riquadro. Girano entrambi sul tuo computer, con pesi aperti e senza account.

Quello che i modelli trovano è un **layer di proposte** nella vista 2D, da controllare prima che
diventi altro:

| Da una ricerca | Che cosa ottieni |
|---|---|
| **Il layer** `samgeo: <nome>` | Gli oggetti trovati, ciascuno con il suo numero, il testo, la frase che l'ha trovato, il punteggio (la confidenza di Grounding DINO) e l'area; disegnato di nuovo su ogni raster che la vista mostra; una proposta eliminata nella vista (*Seleziona* e Canc) esce anche dalla ricerca |
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
* SAM 3 (un solo modello che trova ogni esemplare di un concetto) arriverà in una prossima versione,
  per chi ha accesso ai suoi pesi.

## Installazione

* Nel programma: *Strumenti > Plugin...*, `samgeo`, *Installa / aggiorna*, *Riavvia ora*; poi
  *Installa i modelli...* nel suo pannello.
* Con pip, nell'ambiente Python del programma:

  ```bash
  pip install "ppg-samgeo @ git+https://github.com/enzococca/pyarchinit-photogrammetry-plugins#subdirectory=plugins/samgeo"
  ```

Test: `pip install -e . pytest` nell'ambiente del programma, poi `pytest tests`; usano un motore
finto del worker (le macchie chiare di un'ortofoto sintetica sono le pietre), senza PyTorch.

Licenza: GPL-2.0-or-later. Grounding DINO, SAM 2.1, PyTorch e transformers (Apache-2.0 e BSD)
girano nel loro ambiente. Autore: Enzo Cocca.
