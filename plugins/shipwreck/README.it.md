# Relitto

*[English](README.md)*

**[Guida d'uso](https://enzococca.github.io/pyarchinit-photogrammetry/it/plugins/shipwreck/)**: dal modello 3D di uno scafo scavato ai suoi disegni, passo per passo.

Plugin di [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry) (1.7.1 o successiva): il
disegno tecnico di uno scafo scavato dal suo modello 3D. Dove corre l'asse della nave, la forma dello scafo a ogni
ordinata, una pianta, un elenco numerato dei legni con le loro misure, la pianta del relitto con ogni legno contornato e
nominato, e ogni legno un oggetto 3D. Misura quello che c'è e non sposta una linea dal modello per carenarla; ciò che
aggiunge (una chiglia di dimensioni date, il modello delle fasce dalle sezioni, la ricostruzione di ciò che manca) è
un'ipotesi dichiarata, campita e a parte.

| Passo | Che cosa fa |
|---|---|
| **Sistema della nave** | x lungo la chiglia da un punto di partenza (0 alla prua o a una stazione scelta), y a babordo, z in alto: da una linea tracciata sul raster o da due marker. L'alto è la verticale del mondo, la linea di chiglia è messa in piano e la sua pendenza resta un numero. Quale estremo è la prua lo dici tu (*La prua è l'altro capo*, che scambia babordo e tribordo): dal modello non si vede. *x dall'altro capo* gira soltanto x: prua, y e lati restano. |
| **Stazioni** | regolari, da una tabella `nome, x` con i nomi del cantiere, o dalle ordinate del catalogo. Separata da virgole una tabella vuole il punto decimale; da punto e virgola o tabulazione anche la virgola. Una riga con troppe celle viene rifiutata. |
| **Sezioni** | sezioni trasversali alle stazioni, longitudinali a y date, orizzontali a z date, tagliate dalla mesh in doppia precisione, di traverso fin dove arriva il modello se non dici altro; `<dati del chunk>/shipwreck/sections.json`. Sezioni e disegni superati vengono detti: cambio di sistema della nave, modello, georeferenziazione o sistema del chunk. |
| **Acqua** | un livello e un indice (1,34): sotto il livello un punto si disegna a `livello - (livello - z) * n`. Spenta di partenza. Con il programma 1.7.7 o successivo il livello dato al chunk nel programma (*Acqua*) è il valore proposto qui finché non imposti quello del plugin. Prima approssimazione per una vista nadirale: raggi obliqui e superficie in movimento non sono modellati. La linea grezza e quella corretta restano entrambe e ogni disegno dice quale mostra. |
| **Disegni** | piano di costruzione (visto da poppa, babordo a sinistra), piano delle linee d'acqua, profilo, pianta con le stazioni sull'ortofoto: scala automatica o scelta, barra di scala, nord, cartiglio; SVG, PDF, PNG e DXF con un layer per tipo. Le piante non sono mai speculari, in qualunque verso corra x. |
| **Legni** | chiglia, paramezzale, ordinate, fasciame, fasciame interno, serrette, elementi di fissaggio: disegnati sul raster o da un CSV; lunghezza lungo il legno sul modello, larghezza, quote sopra un datum, maglia delle ordinate; una tabella, un layer della vista raster, CSV, XLSX e GeoPackage. |
| **Fasce (clinker)** | *Rileva i corsi*: i gradini (dove una fascia si sovrappone a quella sotto) sulla sezione grezza di ogni stazione e lato, dalla battura all'orlo, uniti in linee lungo lo scafo; le stazioni con un altro numero di gradini (una fascia che si interrompe, un danno) o con battura o torello che saltano sono elencate come incerte; le linee nei disegni e nel layer "Fasce" della vista raster, DXF e GeoPackage con Z. *Modello delle fasce*: una fascia fra due linee di gradini, la sua larghezza a ogni stazione con l'incertezza, lunghezza, spessore (il gradino mediano) e sovrapposizione, una mesh per fascia (OBJ, PLY nelle coordinate del chunk), CSV e XLSX; *Nel catalogo* aggiunge una tavola per fascia. *Per Clinker Boat Tool* scrive i file per lo strumento Houdini di Fröhlich e Nielsen (Y in alto); *Fasce da Houdini* rilegge le sue tavole con la distanza dalla scansione. I gradini sono misurati; le linee e il modello delle fasce sono un'ipotesi, e ogni file lo dice. |
| **Rilievo e ricalco** | *Rilievo dello scafo*: lo scafo visto dall'alto nel sistema della nave (un prospetto del programma, 5 mm di base, u = x, v = y alla sinistra di x) con ombreggiatura e rilievo locale, su cui disegnare. *Proponi*: comenti, ordinate, paramezzale, serrette e fasce letti sul rilievo, proposte da *Accettare*, *Scartare*, *Correggere* o *Aggiungere* a mano (con lo snap), nominate da una tabella, segnate *Sciolto*; ogni modifica un passo di Annulla. *Chiglia (dimensioni)*: una chiglia da larghezza e altezza date sotto il paramezzale, un'ipotesi. *Legni 3D*: ogni poligono ricalcato un oggetto chiuso (una fascia con il suo spessore, un'ordinata o un legno longitudinale un solido dal fasciame alla cima), un PLY ciascuno. *Esporta 3D*: OBJ, GLB o IFC4 (un elemento per legno con `Pset_PyArchInit_Timber`) nelle coordinate della nave o del chunk, con le tabelle. *Pianta del relitto*: ogni legno contornato, colorato per tipo e nominato sopra l'ombreggiatura o l'ortofoto, le ipotesi campite; SVG, PDF, PNG, DXF. |
| **Disegno classificato** (0.5.0, programma 1.7.7 o successivo) | *Importa un disegno classificato...* al passo 8: un tuo ricalco (un layer di disegni liberi o un file GeoPackage, GeoJSON o DXF) ordinato dal programma lungo la chiglia, nel sistema della nave: coppie di linee di traverso alla chiglia sono **ordinate**, linee lungo la chiglia **comenti**, coppie lungo la chiglia **fasce**; le linee di traverso singole e le altre direzioni restano fuori. Prima la tavola di controllo (con i nomi del plugin: `ordinata N` dalla prua, `comento B1`, `fascia T2`), poi i tipi spuntati entrano nel catalogo come legni misurati, un passo di Annulla; importando di nuovo lo stesso disegno i legni già nel catalogo vengono saltati e contati. Ordinate disegnate come linee sciolte e vicine fra loro vogliono *Isolamento delle coppie* 1, oppure disegnale come poligoni: un poligono è sempre una coppia. Nascosto con un programma più vecchio. |
| **Ricostruzione (ipotesi)** | *Prua e poppa*: fin dove la ricostruzione può andare oltre la lunghezza rilevata. *Ricostruisci*: il lato conservato riflesso dove l'altro è perso (così come il relitto giace; dove entrambi sono misurati, il loro disaccordo elencato), le fasce prolungate lungo i comenti entro la lunghezza rilevata e fino alle estremità date, le ordinate mancanti rimesse al passo misurato, la carenatura per le sezioni alle stazioni dove il relitto non mostra niente (aperta a un'estremità non data), anche sugli altri chunk dello stesso relitto (allineati, o con il sistema della nave da marker degli stessi nomi; *Usa il sistema nave di*); ogni costruzione un passo di Annulla. *Tabella*: misurato e ricostruito, legno per legno (CSV, XLSX). *Esporta 3D* e *Pianta con ricostruzione*: la ricostruzione trasparente (`_ric`) e campita, una campitura per modo. |
| **Sito web** (0.6.0, programma 1.10.0 o successivo) | Quando il programma costruisce il sito web di un rilievo (*File > Esporta > Sito web*, `ppg web build`), ogni chunk con un catalogo aggiunge il livello *Legni del relitto* sulla pianta (ogni legno col contorno disegnato, colorato per tipo, una spunta per tipo, nome, tipo, lato, stazione, stato e lunghezza al clic) e la scheda *Catalogo dei legni* (una cartolina per legno con le misure: x lungo la chiglia, y di traverso, lunghezza, larghezza, quote dall'inizio della linea di chiglia, anche corrette dall'acqua quando la correzione è attiva). Il programma porta i contorni nel sistema del sito (locale di default: traslati, nessun sistema) e ripulisce ogni testo; le misure sono nel sistema della nave e non dicono nulla del luogo, le quote mai assolute (quelle del sito dove l'aggancio del sito del programma ne dice il sistema, `site_hook` 2, con le parole nelle due lingue). Un sistema della nave impostato prima che cambiasse il sistema del chunk dà la scheda senza livello e senza misure. Niente con un programma più vecchio. |

Limiti. Un chunk in gradi (WGS 84) viene rifiutato: un sistema della nave ha bisogno di metri (*GNSS > Sistema e quote* porta il
chunk in un sistema proiettato; l'import da Metashape lo fa da sé). Niente idrostatica né sviluppo del fasciame; i legni trovati sul rilievo sono proposte,
mai accettate senza di te. I clic
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
chiglia nascosta dal paramezzale) dà stazioni incerte: ricalca invece i comenti sul rilievo dello scafo.
I salti della battura e del torello si cercano con cinque stazioni o più su un lato: con meno un salto vero non viene
preso. Un nuovo rilevamento rinomina `.old` i file del modello delle fasce precedente e toglie le sue fasce dal catalogo
dei legni; salva le tavole esportate dal Clinker Boat Tool con un nome tuo (`tavole.obj`), mai `strakes.obj` o
`lands.obj` nella cartella dell'esportazione.

Limiti del rilievo e del ricalco. Tutto ciò che si legge sul rilievo è una proposta: un legno sciolto di cui si legge un
solo bordo dà un falso comento; un legno più stretto di 5 cm accanto a un comento può sfuggire; un modello che mostra
l'esterno dello scafo vuole la vista "esterno"; un legno lungo a filo di quasi tutto un comento rinomina i comenti oltre
di lui; un avanzo in una fascia di larghezza normale è proposto come `comento ?` (accettarlo divide la fascia); un comento
che manca lungo tutto lo scafo dà una banda sola su due fasce; un rumore oltre 2 mm spezza i comenti e può aggiungere una
fascia; pezzi spuri a prua possono prendere il rango 1; su un relitto vero ordinate vicine sono spesso proposte come un
elemento largo dubbio (`ordinata ?`) e il sedimento piatto come paramezzale; le fasce sono bande fra i comenti letti,
rettangoli dove mancano comenti (una banda molto più larga delle vicine o sopra i legni longitudinali è una dubbia
`fascia B?`): correggile a mano. Il rilievo è un campo di altezze: ciò che dall'alto nel sistema della nave non si vede
(sottosquadri, la faccia di sotto di un legno, un legno sotto un altro, la faccia esterna del fasciame di un relitto
visto dall'interno) non è né nelle proposte né nei solidi. Ricalca tutti i legni che toccano
quello che costruisci in 3D; a 5 mm uno scafo di 12 m (1,9 M di celle) richiede circa 0,75 GB per i legni 3D,
PLY 160 MB, IFC 20 s e 290 MB, e uno scafo di 20 m (16 M di celle) diversi GB di memoria e un IFC di qualche GB in
minuti: per le esportazioni 3D di uno scafo lungo costruisci il rilievo a 1 cm (la decimazione è prevista per una versione successiva); una costruzione
interrotta o superata viene detta e l'esportazione avvisa. I nomi possono affollarsi dove fasce strette o più serrette finiscono insieme.

Limiti della ricostruzione. È un'ipotesi costruita su ciò che è ricalcato: un legno non ricalcato sul fasciame entra
nelle sezioni e la carenatura gli passa sopra; un relitto torto è riflesso torto; oltre l'ultima stazione la carenatura
è un'ipotesi della sola larghezza, con l'altezza dell'ultima sezione; un comento senza linee dei gradini è seguito per un
metro al massimo oltre i suoi punti, poi tenuto dritto (`uncertain`); una fascia prolungata è tagliata dove si allarga
oltre 1,5 volte la sua larghezza misurata o esce dalla mezza larghezza dello scafo (`uncertain`); con più chunk le ordinate devono avere i numeri del
cantiere, e si scrivono solo i file del chunk attivo; senza la simmetria, dove un lato è perso e l'altro conservato è
ricostruita solo la carenatura. La ricostruzione è disegnata ed esportata solo quando la si chiede (una casella in
*Disegni*, *Pianta del relitto* ed *Esporta 3D*, `--reconstruction`); quando cambia ciò da cui è stata costruita viene
detta da rifare, e in un altro sistema della nave l'esportazione 3D la rifiuta.

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
ppg plugin shipwreck drawings PROGETTO.ppg [--scale 20] [--out CARTELLA] [--reconstruction]
ppg plugin shipwreck timbers PROGETTO.ppg [--table FILE]
ppg plugin shipwreck export PROGETTO.ppg --out legni.xlsx                       # .csv, .xlsx, .gpkg
ppg plugin shipwreck status PROGETTO.ppg
ppg plugin shipwreck clinker detect PROGETTO.ppg [--min-step MM] [--min-strake M]   # i gradini e le loro linee
ppg plugin shipwreck clinker strakes PROGETTO.ppg [--lap MM] [--thickness MM] [--catalogue]
ppg plugin shipwreck clinker export PROGETTO.ppg --out CARTELLA                     # file per il Clinker Boat Tool
ppg plugin shipwreck clinker import PROGETTO.ppg --obj TAVOLE.obj [--side-from y|name] [--no-deviation]
ppg plugin shipwreck clinker status PROGETTO.ppg                                    # dice anche se i gradini sono superati
ppg plugin shipwreck relief build PROGETTO.ppg [--cell MM]                          # il rilievo dello scafo dall'alto
ppg plugin shipwreck relief export PROGETTO.ppg --kind hillshade --out FILE.tif [--world]
ppg plugin shipwreck propose PROGETTO.ppg [--view inside|outside] [--min-step MM] [--opening M] [--top-hat MM]
ppg plugin shipwreck trace list PROGETTO.ppg
ppg plugin shipwreck trace accept PROGETTO.ppg ID... [--name NOME]                  # oppure --kind frame [--min-score 0.5]
ppg plugin shipwreck trace reject PROGETTO.ppg ID...
ppg plugin shipwreck trace names PROGETTO.ppg --table FILE
ppg plugin shipwreck keel PROGETTO.ppg --siding MM --moulding MM [--from X --to X]  # oppure --none
ppg plugin shipwreck solids PROGETTO.ppg [--thickness MM]
ppg plugin shipwreck export3d PROGETTO.ppg --out FILE.obj|.glb|.ifc [--world] [--reconstruction]
ppg plugin shipwreck plan PROGETTO.ppg [--background relief|ortho|none] [--scale N] [--out CARTELLA] [--reconstruction]
ppg plugin shipwreck ends PROGETTO.ppg [--stem X] [--stern X]                      # oppure --none
ppg plugin shipwreck frame-of PROGETTO.ppg CHUNK [--chunk QUESTO]                  # il sistema della nave di un altro chunk
ppg plugin shipwreck chunks PROGETTO.ppg                                           # i chunk della ricostruzione
ppg plugin shipwreck reconstruct PROGETTO.ppg [--no-mirror] [--no-extension] [--no-frames] [--no-loft]
ppg plugin shipwreck reconstruct table PROGETTO.ppg [--out CARTELLA]
ppg plugin shipwreck classified PROGETTO.ppg --layer NOME | --file FILE [--pair-isolation 1] [--param NOME=VALORE] [--plate FILE.png|.svg] [--accept [ID ...]] [--kinds frame,seam,plank]   # 0.5.0, programma 1.7.7
```

Una tabella dei legni ha l'intestazione: `tipo`, `lato`, `nome`, `stazione`, `note` e `x, y` (un punto) oppure `x1, y1, x2, y2`
(una linea) in coordinate della nave, o `wkt` nelle coordinate del chunk.

## Licenza

GPL-2.0-or-later.
