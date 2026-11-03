# Relitto

*[English](README.md)*

**[Guida d'uso](https://enzococca.github.io/pyarchinit-photogrammetry/it/plugins/shipwreck/)**: dal modello 3D di uno scafo scavato ai suoi disegni, passo per passo.

Plugin di [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry) (1.7.0 o successiva): il
disegno tecnico di uno scafo scavato dal suo modello 3D. Dove corre l'asse della nave, la forma dello scafo a ogni
ordinata, una pianta, un elenco numerato dei legni con le loro misure. Misura quello che c'è: non ricostruisce niente e
non sposta una linea dal modello per carenarla.

| Passo | Che cosa fa |
|---|---|
| **Sistema della nave** | x lungo la chiglia da un punto di partenza (0 alla prua o a una stazione scelta), y a babordo, z in alto: da una linea tracciata sul raster o da due marker. L'alto è la verticale del mondo, la linea di chiglia è messa in piano e la sua pendenza resta un numero. Quale estremo è la prua lo dici tu (*La prua è l'altro capo*, che scambia babordo e tribordo): dal modello non si vede. *x dall'altro capo* gira soltanto x: prua, y e lati restano. |
| **Stazioni** | regolari, da una tabella `nome, x` con i nomi del cantiere, o dalle ordinate del catalogo. Separata da virgole una tabella vuole il punto decimale; da punto e virgola o tabulazione anche la virgola. Una riga con troppe celle viene rifiutata. |
| **Sezioni** | sezioni trasversali alle stazioni, longitudinali a y date, orizzontali a z date, tagliate dalla mesh in doppia precisione, di traverso fin dove arriva il modello se non dici altro; `<dati del chunk>/shipwreck/sections.json`. Sezioni e disegni superati vengono detti: cambio di sistema della nave, modello, georeferenziazione o sistema del chunk. |
| **Acqua** | un livello e un indice (1,34): sotto il livello un punto si disegna a `livello - (livello - z) * n`. Spenta di partenza. Prima approssimazione per una vista nadirale: raggi obliqui e superficie in movimento non sono modellati. La linea grezza e quella corretta restano entrambe e ogni disegno dice quale mostra. |
| **Disegni** | piano di costruzione (visto da poppa, babordo a sinistra), piano delle linee d'acqua, profilo, pianta con le stazioni sull'ortofoto: scala automatica o scelta, barra di scala, nord, cartiglio; SVG, PDF, PNG e DXF con un layer per tipo. Le piante non sono mai speculari, in qualunque verso corra x. |
| **Legni** | chiglia, paramezzale, ordinate, fasciame, fasciame interno, serrette, elementi di fissaggio: disegnati sul raster o da un CSV; lunghezza lungo il legno sul modello, larghezza, quote sopra un datum, maglia delle ordinate; una tabella, un layer della vista raster, CSV, XLSX e GeoPackage. |
| **Fasce (clinker)** | *Rileva i corsi*: i gradini (dove una fascia si sovrappone a quella sotto) sulla sezione grezza di ogni stazione e lato, dalla battura all'orlo, uniti in linee lungo lo scafo; le stazioni con un altro numero di gradini (una fascia che si interrompe, un danno) o con battura o torello che saltano sono elencate come incerte; le linee nei disegni e nel layer "Fasce" della vista raster, DXF e GeoPackage con Z. *Modello delle fasce*: una fascia fra due linee di gradini, la sua larghezza a ogni stazione con l'incertezza, lunghezza, spessore (il gradino mediano) e sovrapposizione, una mesh per fascia (OBJ, PLY nelle coordinate del chunk), CSV e XLSX; *Nel catalogo* aggiunge una tavola per fascia. *Per Clinker Boat Tool* scrive i file per lo strumento Houdini di Fröhlich e Nielsen (Y in alto); *Fasce da Houdini* rilegge le sue tavole con la distanza dalla scansione. I gradini sono misurati; le linee e il modello delle fasce sono un'ipotesi, e ogni file lo dice. |

Limiti. Un chunk in gradi (WGS 84) viene rifiutato: un sistema della nave ha bisogno di metri (*GNSS > Sistema e quote* porta il
chunk in un sistema proiettato; l'import da Metashape lo fa da sé). Niente ricostruzione dello scafo, lati specchiati,
superfici lofted, idrostatica, sviluppo del fasciame, riconoscimento automatico dei legni, unione di più giornate. I clic
nella vista 3D non sono offerti direttamente: metti lì due marker e usa *Da due marker*. Le quote sul modello valgono circa
un lato della mesh.

Limiti delle fasce. Su sezioni di una mesh con un rumore oltre circa 0,5 mm le larghezze valgono a circa un centimetro; a
1 mm di rumore circa un lato su quindici può avere un numero di gradini sbagliato (segnalato come stazione incerta). Un
salto di 2-2,5 cm della battura a una stazione d'estremità, tre o più stazioni vicine spostate insieme o un gradino
graduale non vengono segnalati; un gradino permanente della chiglia (uno scafo rotto o spostato, una palella) segnala
ogni stazione da un lato; oltre circa 3 mm di dispersione sulla battura una stazione può essere segnalata a torto. Una V
di fasciame più ripida di 80° entro la metà larghezza della chiglia, senza suola, è letta come chiglia finché non si
apre; un fianco della chiglia allineato al torello è un lato non letto (`keel_unclear`). Il modello delle fasce vuole
almeno tre stazioni con una larghezza su ogni lato con gradini. Uno scafo visto dall'interno (fasciame sotto le ordinate,
chiglia nascosta dal paramezzale) dà stazioni incerte: la prossima versione traccerà i gradini sul rilievo dello scafo.
I salti della battura e del torello si cercano con cinque stazioni o più su un lato: con meno un salto vero non viene
preso. Un nuovo rilevamento rinomina `.old` i file del modello delle fasce precedente e toglie le sue fasce dal catalogo
dei legni; salva le tavole esportate dal Clinker Boat Tool con un nome tuo (`tavole.obj`), mai `strakes.obj` o
`lands.obj` nella cartella dell'esportazione.

Il Clinker Boat Tool è un Houdini Digital Asset di Fred Fröhlich e Sigurd Solhaug Nielsen (MIT,
[10.5281/zenodo.17488549](https://doi.org/10.5281/zenodo.17488549)), descritto in *From Photogrammetry to Construction
Knowledge: 3D Documentation and Parametric Reconstruction of Historic Clinker Boats* (CC BY 4.0,
[10.5281/zenodo.23422398](https://doi.org/10.5281/zenodo.23422398)). Il plugin scrive i file che i passi dello strumento
leggono e rilegge le tavole che esporta; niente dello strumento è copiato o usato.

La trasformazione, il modello e i prodotti del chunk non vengono mai cambiati. Tutto si scrive in
`<dati del chunk>/shipwreck/`, salvo quello che esporti.

## Riga di comando

```bash
ppg plugin shipwreck frame PROGETTO.ppg --start X,Y[,Z] --end X,Y[,Z] [--bow start|end] [--z0 H] [--x0 X]
ppg plugin shipwreck frame PROGETTO.ppg --markers --start A --end B
ppg plugin shipwreck frame PROGETTO.ppg --flip        # x dall'altro capo; i lati restano
ppg plugin shipwreck frame PROGETTO.ppg --other-bow   # la prua è l'altro capo; babordo e tribordo si scambiano
ppg plugin shipwreck stations PROGETTO.ppg --spacing 1 --count 11 --first 1     # oppure --table FILE, --from-catalogue
ppg plugin shipwreck sections PROGETTO.ppg --longitudinals "0;1;-1" --horizontals "-0.2;-0.5" --half-width 4.5
ppg plugin shipwreck water PROGETTO.ppg --level -0.1 --n 1.34 --on
ppg plugin shipwreck drawings PROGETTO.ppg [--scale 20] [--out CARTELLA]
ppg plugin shipwreck timbers PROGETTO.ppg [--table FILE]
ppg plugin shipwreck export PROGETTO.ppg --out legni.xlsx                       # .csv, .xlsx, .gpkg
ppg plugin shipwreck status PROGETTO.ppg
ppg plugin shipwreck clinker detect PROGETTO.ppg [--min-step MM] [--min-strake M]   # i gradini e le loro linee
ppg plugin shipwreck clinker strakes PROGETTO.ppg [--lap MM] [--thickness MM] [--catalogue]
ppg plugin shipwreck clinker export PROGETTO.ppg --out CARTELLA                     # file per il Clinker Boat Tool
ppg plugin shipwreck clinker import PROGETTO.ppg --obj TAVOLE.obj [--side-from y|name] [--no-deviation]
ppg plugin shipwreck clinker status PROGETTO.ppg                                    # dice anche se i gradini sono superati
```

Una tabella dei legni ha l'intestazione: `tipo`, `lato`, `nome`, `stazione`, `note` e `x, y` (un punto) oppure `x1, y1, x2, y2`
(una linea) in coordinate della nave, o `wkt` nelle coordinate del chunk.

## Licenza

GPL-2.0-or-later.
