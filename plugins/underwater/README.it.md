# Subacqueo

**[Guida d'uso](https://enzococca.github.io/pyarchinit-photogrammetry/it/plugins/underwater/)**: le foto e i
fotogrammi da usare, i colori, le profondità. *[English](README.md)*

Plugin di [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry) (1.3.4 o
successiva) per i rilievi sott'acqua - relitti, strutture sommerse, reperti sul fondale - con le foto di un
subacqueo o le immagini di un ROV. Non chiede niente oltre al programma (numpy, SciPy, OpenCV, Pillow, pycolmap).

| Dove | Che cosa |
|---|---|
| **1. Foto e fotogrammi** | |
| *Fotogrammi da un video...* | Video di una GoPro o di un ROV: l'estrazione del programma (un fotogramma ogni volta che la vista si è spostata, il più nitido dei momenti dopo), con l'ora di ogni fotogramma (l'ora di creazione scritta nell'MP4/MOV, o una data, più il tempo nel video) |
| *Fotogrammi da una sequenza...* | Una cartella di immagini a 10-30 al secondo (un ROV): ogni fotogramma abbinato al precedente (punti ORB, una similitudine con RANSAC), i fotogrammi di un tratto concatenati; il fotogramma tenuto dopo è il più nitido fra quelli che si sono spostati da metà a tutto `1 - sovrapposizione` della larghezza dall'ultimo, esclusi quelli molto meno nitidi dei vicini; un ROV fermo non dà fotogrammi nuovi, un salto apre un nuovo tratto. I fotogrammi restano dove sono e diventano foto con la loro ora (una tabella dei tempi - nome o numero, ora - oppure i nomi dei file, oppure l'EXIF) |
| *Calibrazione (Kalibr)...* | La calibrazione della camera di un ROV da Kalibr (radtan: OPENCV, equidistant: OPENCV_FISHEYE), il punto principale spostato di mezzo pixel alla convenzione di COLMAP, tenuta fissa nell'allineamento |
| *Valuta le foto* | Tre punteggi per foto: **qualità** (la nitidezza del programma rispetto a quella delle foto più nitide del chunk: foto mosse o sfocate), **contrasto** (estensione della luminosità, 2°-98° percentile: acqua lattiginosa, foto troppo scure) e **velo** (la mediana del canale scuro di verde e blu sulla luce massima: torbidità, retrodiffusione). Soglie nel pannello (0,5, 0,12, 0,6), le foto segnate in una tabella con il perché; *Escludi le segnate* le disattiva (allineamento e prodotti le lasciano fuori), *Riattiva le escluse* le riporta; una tabella CSV |
| **2. Colori** | |
| *Ripristino fisico (Sea-thru)* | Con la **distanza di ogni pixel** dalle mappe di profondità del programma (*Flusso di lavoro > Mappe di profondità*; le mappe sono delle foto senza distorsione: ogni pixel della foto è portato al suo raggio con il modello della camera e alla mappa, `distanza = z * sqrt(1 + x² + y²)`, in metri quando lo è il chunk) il modello di formazione dell'immagine di Sea-thru (Akkaynak e Treibitz, CVPR 2019), scritto qui a partire dall'articolo: la **retrodiffusione** di ogni canale stimata sui pixel più scuri di ogni fascia di distanza, `B(z) = B∞ (1 - e^(-βB z))`, e tolta; l'**illuminante** - la media locale fra i pixel a distanza simile - descritta da `log E = w + e_foto - βD(z) z`, con `βD(z)` le due esponenziali dell'articolo; i colori `J = D e^(βD(z) z)`. I coefficienti **una volta per il rilievo** (le foto di un'immersione hanno la stessa correzione: niente cuciture di colore in ortofoto e texture; l'esposizione e l'intensità della retrodiffusione di ogni foto restano sue) o **per ogni foto** |
| *Correzione rapida* | Senza distanze: prima dell'allineamento, o senza mappe di profondità. Rosso (e, in acqua verde, blu) compensato con il verde (Ancuti et al., IEEE TIP 2018), bilanciamento del bianco a mondo grigio, estensione della luminosità, CLAHE facoltativo (contrasto locale: può rendere possibile l'allineamento di foto scure e torbide). Medie e guadagni da un campione del rilievo insieme |
| *Anteprima*, *Correggi le foto* | Prima e dopo di una foto; poi le copie di tutte, stessi nomi, dimensioni e formato, l'EXIF conservato, in `<dati del chunk>/underwater/colour/` (gli originali non vengono mai toccati) |
| *Usa le foto corrette*, *Usa le originali* | Le camere puntate alle copie o di nuovo agli originali (quello che fa *Ricollega le foto* del programma, nei due sensi): ortofoto, texture, mappe di profondità e un nuovo allineamento leggono le foto dove puntano le camere |
| **3. Profondità invece del GPS** | |
| *Importa il registro...* | Tabelle CSV e di testo (tempo Unix in s/ms/us/ns, date ISO, ore con una colonna della data, secondi dall'inizio; profondità in m o ft, oppure pressione in bar, mbar, dbar, kPa, Pa, psi trasformata in profondità con la densità del mare o dell'acqua dolce), esportazioni dei computer subacquei (Subsurface: date, time, sample time, sample depth), UDDF, il file delle profondità di AQUALOC. Associate alle foto per ora (l'ora dei fotogrammi, i nomi dei file, l'EXIF con frazioni di secondo e fuso) più uno **scarto di tempo** |
| *Stima lo scarto* | Lo scarto di tempo con cui le profondità si accordano meglio con le camere allineate (la messa in bolla risolta per ogni scarto provato, prima grossolana poi fine) |
| *Metti in bolla e in scala* | `v · c + b = -profondità` per ogni centro di camera `c` (sistema interno): lineare, risolto con pesi di Huber; `|v|` è la scala, `v/|v|` la verticale, `b` la quota. Il chunk in un sistema locale in metri, **Z = -profondità** (la superficie a 0), X, Y e la direzione come erano; lo scarto di ogni camera, quelle fuori misura, l'incertezza della verticale. Quando la profondità cambia troppo poco per dare la scala (un rilievo a profondità costante) la danno le barre di scala di controllo, o *Scala: dalle barre di scala* |
| *Posizioni note (USBL)...* | X, Y di alcune camere per nome della foto o per ora (interpolate): dopo la messa in bolla una similitudine orizzontale dà la direzione e la posizione, nel loro sistema di riferimento |

I riferimenti delle camere del programma sono posizioni in tre coordinate con una sola accuratezza per tutte e
tre: un riferimento con la sola Z non ci sta, quindi la messa in bolla la fa il plugin e la tiene in
`chunk.meta["underwater"]["levelling"]`. Un nuovo allineamento azzera la trasformazione del chunk: il pannello e
l'area di lavoro dicono *Bolla da rifare*, un clic la rifà.

Dalla riga di comando:

```bash
ppg plugin underwater frames-sequence scavo.ppg raw_data/images_sequence_6 --times raw_data/img_sequence_6.csv --overlap 0.9
ppg plugin underwater calibration scavo.ppg archaeo_camera_calib.yaml
ppg plugin underwater score scavo.ppg --quality 0.5 --contrast 0.12 --veil 0.6 --exclude
ppg plugin underwater quick scavo.ppg --clahe --alpha 0 --no-balance --use     # prima dell'allineamento
ppg plugin underwater restore scavo.ppg --per survey --use                       # dopo le mappe di profondità
ppg plugin underwater use scavo.ppg original
ppg plugin underwater depth scavo.ppg depth_sequence_6.csv --offset 0
ppg plugin underwater offset scavo.ppg --span 900
ppg plugin underwater level scavo.ppg [--scale bars] [--positions usbl.csv --crs 32633]
ppg plugin underwater status scavo.ppg
ppg plugin underwater csv scavo.ppg foto.csv
```

## Provato su dati veri

**AQUALOC**, sito archeologico, sequenza 6 (M. Ferrera, V. Creuze, J. Moras, P. Trouvé-Peloux, *AQUALOC: An
Underwater Dataset for Visual-Inertial-Pressure Localization*, IJRR 2019): 3385 fotogrammi monocromatici
968 x 608 a 20 al secondo su un relitto di anfore a 387-391 m, la profondità del sensore di pressione del ROV,
una traiettoria di COLMAP come riferimento. 139 fotogrammi scelti in 8,5 s (sovrapposizione 0,9); allineati dal
programma con la calibrazione Kalibr fissa: 63 su 139 con i fotogrammi originali, **136 su 139** con la
correzione rapida (CLAHE) usata per l'allineamento (tratti di sabbia con poca tessitura). In bolla e in scala con
le profondità: scarto delle profondità delle camere **2,3 cm** (scarto quadratico; massimo 6,1 cm, su 3,9 m di
dislivello), errore standard della scala 0,5 %, verticale entro **0,18°** da quella che le profondità danno al
riferimento; dopo una similitudine le posizioni delle camere stanno a **5,8 cm** (scarto quadratico) dal
riferimento su 38 m. Il riferimento stesso chiede una scala di 1,065 per accordarsi con le stesse profondità (la
nostra scala è quella delle profondità). Un orologio della camera spostato di 7,3 s: lo scarto trovato è 7,33 s.

## Riferimenti

- D. Akkaynak, T. Treibitz, *Sea-thru: A Method for Removing Water From Underwater Images*, CVPR 2019 (il modello
  del ripristino; il codice qui è scritto a partire dall'articolo).
- C. O. Ancuti, C. Ancuti, C. De Vleeschouwer, P. Bekaert, *Color Balance and Fusion for Underwater Image and
  Video Enhancement*, IEEE Transactions on Image Processing 27(1), 2018 (la compensazione del rosso).
- P. Drews Jr., E. do Nascimento, F. Moraes, S. Botelho, M. Campos, *Transmission Estimation in Underwater Single
  Images*, ICCV Workshops 2013 (il canale scuro subacqueo del punteggio del velo).
- AQUALOC (sopra): usato per le prove su questo computer; non è distribuito con il plugin.

Licenza: GPL-2.0-or-later, come il programma in cui gira il plugin.
