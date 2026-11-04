<!-- Sezioni da aggiungere alla guida del programma docs/plugins/shipwreck.it.md per la versione 0.3.0 del plugin
(serve pyarchinit-photogrammetry 1.7.1 o successiva). Posto suggerito: dopo "Fasce a clinker", prima di "Riga di
comando"; le righe di comando vanno in "Riga di comando"; le figure di img/ accanto alle immagini della guida. -->

## Rilievo, ricalco, legni 3D e pianta del relitto

Il passo 8 del pannello, **Rilievo e ricalco**, porta il modello 3D di uno scafo scavato al disegno che fa
l'archeologo sul campo: ogni legno contornato, nominato e tipizzato in una pianta, e ogni legno un oggetto 3D. Serve il
sistema della nave (passo 1); stazioni e sezioni non servono.

| Pulsante | Che cosa fa |
|---|---|
| *Rilievo dello scafo...* | Lo scafo visto dall'alto nel sistema della nave: un prospetto del programma su un piano orizzontale (u = x, v = segno × y: y alla sinistra di x, qualunque sia il capo della prua), le sue altezze, l'ombreggiatura e il rilievo locale. |
| *Mostra* | Il rilievo nella vista raster, con le altezze, l'ombreggiatura o il rilievo locale in grigio sopra. |
| *Proponi...* | Comenti, ordinate, paramezzale, serrette e fasce letti sul rilievo: proposte, tratteggiate, un colore per tipo. |
| *Accetta* / *Scarta* | Le proposte selezionate sulla vista nel catalogo dei legni / buttate per sempre (anche *Canc* sulla vista). |
| *Correggi...* / *Aggiungi...* | Un legno ridisegnato, o uno nuovo disegnato a mano, sul rilievo, con gli strumenti di disegno e lo snap del programma. |
| *Nomi da tabella...* | Una tabella `nome, x`: le ordinate prendono i nomi del cantiere. |
| *Sciolto* | I legni selezionati segnati come fuori posto. |
| *Chiglia (dimensioni)...* | Una chiglia da larghezza e altezza date: un'ipotesi. |
| *Legni 3D...* | Ogni poligono ricalcato un oggetto 3D chiuso. |
| *Esporta 3D...* | OBJ, GLB o IFC, un oggetto per legno, con le tabelle. |
| *Pianta del relitto...* | La pianta con ogni legno contornato, colorato per tipo e nominato: SVG, PDF, PNG, DXF. |

Ogni modifica del catalogo, delle proposte o della chiglia è un passo di *Modifica > Annulla* (proposte e legni
eliminati insieme sulla vista sono un passo solo).

### Il rilievo dello scafo

Il modello è letto dall'alto nel sistema della nave: per ogni cella la superficie più alta del modello (l'interno di uno
scafo adagiato sul fondo), un GeoTIFF in metri della nave con il sistema nei tag, e una copia orientata a nord nel sistema
del chunk per il GIS. L'ombreggiatura da quattro direzioni e il rilievo locale si vedono in grigio sopra, come il
programma fa per un DEM. La cella viene dalla densità del modello; 5 mm va bene per un modello ravvicinato di uno scafo
(un modello di 15 milioni di facce richiede circa un minuto). I disegni fatti sul rilievo restano con lui quando lo
rifai, anche dopo un cambio del sistema della nave.

Il rilievo è un campo di altezze: una quota per cella, la più alta. Ciò che dall'alto, nel sistema della nave, non si
vede non c'è, e quindi non c'è nelle proposte né nei legni 3D: un sottosquadro, un fianco più ripido della verticale, la
faccia di sotto di un legno, un legno sotto un altro, la faccia esterna del fasciame. Per un relitto visto dall'interno
significa: il fasciame è la sua faccia interna (lo spessore viene dai gradini dei comenti, dal modello delle fasce o dal
valore che scrivi), la chiglia non si vede (per questo la chiglia di dimensioni date), le ordinate sono la loro parte
visibile sopra il fasciame e ciò che sta sotto le ordinate è interpolato. Uno scafo adagiato su un fianco vuole un
sistema della nave suo, perché la parte che interessa si veda dall'alto.

### Le proposte

![Il rilievo dello scafo sintetico con le proposte](img/shipwreck-relief-proposals.png)

Il plugin legge sul raster delle altezze:

- i **comenti** (i gradini del fasciame a clinker): salti di traverso allo scafo, seguiti lungo x, interrotti nelle
  lacune, una linea ciascuno; numerati dalla chiglia (`comento B1`, `B2`, ... a babordo, `T1`, ... a tribordo); un pezzo
  che cade fra due comenti già presenti è un dubbio `comento B?` con punteggio 0,1;
- le **ordinate**, il **paramezzale** e le **serrette**: ciò che sporge dal fasciame (l'altezza sopra un'apertura lungo x,
  tolta prima la salita del fondo lungo x, così che un'ordinata tenga tutto il suo contorno dove lo scafo sale verso
  prua e poppa), come poligoni, classificati per lunghezza e direzione; le ordinate nominate dalla prua, con la loro
  larghezza e il passo per lato. Un elemento di traverso che manca solo la prova dell'allungamento (lungo fra 1,8 e 3
  volte la larghezza: quasi sempre un'ordinata unita alla vicina o a fasciame rialzato) è un dubbio `ordinata ?` con
  punteggio 0,1: accettalo e correggine il contorno;
- le **fasce**: le bande fra comenti consecutivi, un poligono ciascuna, `fascia B1`, ...; la banda 1 va dalla linea di
  chiglia (spostata di metà larghezza della chiglia) al comento 1, e ogni banda comincia e finisce dove cominciano e
  finiscono i pezzi dei suoi comenti, così dove i comenti interni non si leggono le bande sono rettangoli. Una banda
  larga più di 1,5 volte le altre bande fra comenti del suo lato, o stesa per lo più sopra paramezzale e serrette
  proposti, è una dubbia `fascia B?` con punteggio 0,1.

| Parametro | Predefinito | Quando cambiarlo |
|---|---|---|
| Il modello mostra | l'interno | *L'esterno* per uno scafo sollevato: decide il verso del gradino di un comento. |
| Gradino minimo dei comenti (mm) | 8 | Più basso per un fasciame sottile o consumato; più alto se il rumore del modello dà falsi comenti. |
| Apertura (m) | 0,3 | Più lunga della larghezza fra due ordinate, più corta del fasciame fra due ordinate. |
| Altezza minima di un legno sopra il fasciame (mm) | 20 | Più bassa per ordinate consumate, più alta se vengono proposti grumi di sedimento. |

Tutto è una **proposta**: accettala, correggila o scartala a mano. Una nuova lettura non tocca il catalogo e non
ripropone ciò che è stato accettato o scartato. I limiti, detti chiaramente:

- un legno sciolto di cui si legge un solo bordo dà un comento che non lo è;
- un legno più stretto di 5 cm accanto a un comento può sfuggire;
- un modello che mostra l'esterno dello scafo vuole *L'esterno*, altrimenti ogni comento è dubbio;
- un legno lungo appoggiato a filo lungo quasi tutto un comento rinomina i comenti oltre di lui;
- un avanzo dentro una fascia di larghezza normale è proposto come `comento ?`: accettarlo divide la fascia in due;
- un comento che manca lungo tutto lo scafo dà una banda sola su due fasce;
- un rumore del modello oltre 2 mm spezza i comenti e può aggiungere una fascia;
- pezzi spuri a prua possono prendere il rango 1;
- su un relitto vero ordinate vicine, o con il fasciame fra di loro, sono lette come un elemento largo, proposto come
  `ordinata ?` (o *altro* se ancora più corto): correggine il contorno o ricalca le ordinate a mano con *Aggiungi...*;
  separare le ordinate unite è previsto per la versione 0.4.0; zone piatte di sedimento possono essere proposte come
  paramezzale;
- le fasce sono bande fra i comenti letti, non i contorni delle tavole: dove mancano comenti sono rettangoli sopra
  qualsiasi cosa ci sia (sedimento, legni longitudinali); correggile o ricalcale a mano.

### I legni 3D

![I legni dello scafo sintetico in 3D](img/shipwreck-timbers-3d.png)

*Legni 3D...* fa di ogni legno ricalcato come poligono un oggetto chiuso nel sistema della nave, un PLY per legno in
`<dati del chunk>/shipwreck/timbers/`:

- una **fascia** è la superficie del modello dentro il suo contorno con il suo spessore (il suo, quello del modello delle
  fasce dalle sezioni o il gradino dei suoi comenti; il valore che scrivi vale solo per le fasce senza). Dove un'ordinata,
  il paramezzale o una serretta la coprono, la superficie è interpolata dalla fascia ai due lati e segnata nascosta;
- un'**ordinata**, il **paramezzale**, una **serretta**, il serrettame e ogni altro poligono è un solido dal fasciame sotto
  di lui (interpolato da un anello di celle intorno) fino alla sua cima (la superficie del modello).

Limiti: ricalca tutti i legni che toccano quello che costruisci (uno non ricalcato finisce nell'anello da cui si legge il
fasciame sotto); i file sono grandi sugli scafi lunghi (vedi le dimensioni sotto); un tratto nascosto
con celle da un lato solo è tenuto costante; un legno ricalcato dove sullo scafo non c'è niente dà una mesh di spessore
nullo. Di traverso a un legno lungo lo scafo sotto di lui è una curva per le quote e le pendenze del fasciame ai due
lati (una corda dove le pendenze sono quelle del fianco di un altro legno). Sullo scafo sintetico tutte le 23 ordinate,
da prua a poppa, stanno entro il 2 % del loro volume vero, il paramezzale entro l'1 %, le fasce entro l'1,4 % di quello
sulla superficie a clinker (il torello, una striscia larga 2 cm vista dall'alto, entro il 10 %). Una costruzione
interrotta è detta nel pannello e nell'albero (*interrotto*): i suoi file sono parziali. Quando catalogo, chiglia o
rilievo cambiano dopo la costruzione il pannello dice *da rifare*, e un'esportazione avvisa che i file dicono ciò che il
catalogo diceva allora.

### La chiglia: un'ipotesi

Il modello di uno scafo scavato ne mostra l'interno: la chiglia non si vede. *Chiglia (dimensioni)...* ne fa una da
larghezza e altezza date: un parallelepipedo lungo la linea di chiglia, sotto il fondo del paramezzale (altrimenti sotto
lo scafo a y = 0), lungo tutto il paramezzale (altrimenti tutto il rilievo) o fra due x date. Il suo stato è
*ricostruito: dimensioni date*: è campita nei disegni, più chiara e trasparente (`_ric`) nei file 3D, una riga sua nelle
tabelle; l'intervallo di x resta quello del momento in cui la imposti. Un legno di chiglia ricalcato su un chunk che
mostra l'esterno la sostituisce. *Nessuna chiglia* la toglie.

### Esporta 3D

Un oggetto per legno con il suo nome, nel sistema della nave o nelle coordinate del chunk:

- **OBJ**: `o <nome>` per legno, un materiale per tipo nel `.mtl`;
- **GLB**: un nodo per legno, un colore per tipo;
- **IFC4**: un elemento per legno (IfcPlate per fasciame e serrettame, IfcMember per gli altri) con
  `Pset_PyArchInit_Timber` (nome, tipo, lato, chunk, stato, lunghezza, larghezza, quote, volume, spessore in metri). Un
  chunk in un sistema locale o geografico non ha la IfcMapConversion, e il file lo dice. L'IFC di uno scafo lungo
  richiede da decine di secondi a minuti (sotto);
- le tabelle `timbers_3d.csv` e `.xlsx` accanto (spessore in millimetri).

Dimensioni a 5 mm, misurate su uno scafo sintetico di 12 m (1,9 milioni di celle): proposte 1 s e circa 0,45 GB di
memoria, legni 3D circa 0,75 GB durante la costruzione (le due fasce sui bordi 1,5 milioni di facce ciascuna), file PLY
160 MB, IFC circa 20 s e 290 MB. Uno scafo di 20 m a 5 mm ha circa 16 milioni di celle: conta su diversi GB di memoria
per le proposte (circa 3,5 GB) e i legni 3D, un IFC di qualche GB che richiede minuti e una decina di minuti per il
rilievo di un modello fotogrammetrico. Per le esportazioni 3D di uno scafo lungo costruisci il rilievo a 1 cm (quattro
volte meno celle); la decimazione dei solidi è prevista per la versione 0.4.0.

Le ipotesi finiscono in `_ric`, più chiare e semitrasparenti, con il loro stato nel Pset.

### La pianta del relitto

![La pianta del relitto dello scafo sintetico](img/shipwreck-wreck-plan.png)

*Pianta del relitto...* disegna ogni legno del catalogo sopra l'ombreggiatura del rilievo, l'ortofoto o niente:
contornato e campito leggero nel colore del suo tipo, il fasciame più chiaro perché il rilievo sotto resti leggibile,
nominato lungo la sua lunghezza (un'ordinata di traverso, vicino all'estremità lontana dalla chiglia); le parti di una
fascia sotto un'ordinata, il paramezzale o una serretta a trattini; le ricostruzioni (la chiglia data, le fasce dalle
sezioni, le tavole importate dal Clinker Boat Tool) campite a righe oblique (`////`). Legenda per tipo e stato, scala
grafica, nord, cartiglio; SVG, PDF, PNG e DXF con un layer per tipo (`LEGNI_<TIPO>`, `RICOSTRUZIONE_<TIPO>`) in
`<dati del chunk>/shipwreck/drawings`. Con legni ricalcati il piano di costruzione segna le fasce a ogni stazione e il
profilo mostra paramezzale e chiglia. Se il rilievo è da rifare il pannello propone prima di rifarlo.

Il modello delle fasce dalle sezioni (versione 0.2.0) resta disponibile come livello d'ipotesi: una fascia delle sezioni
non è disegnata dove una fascia ricalcata ha il suo nome.

Limiti dei disegni: fasce strette o più serrette che finiscono alla stessa x possono affollare i nomi; un'ordinata che
attraversa la linea di chiglia è nominata all'estremità alta; un legno di chiglia misurato non ha il riquadro sul
profilo; i nomi delle stazioni del piano di costruzione possono sovrapporsi.

<!-- Righe da aggiungere a "Riga di comando": -->

```bash
ppg plugin shipwreck relief build PROGETTO.ppg [--cell MM]
ppg plugin shipwreck relief export PROGETTO.ppg --kind height|hillshade|slrm --out FILE.tif [--world]
ppg plugin shipwreck propose PROGETTO.ppg [--view inside|outside] [--min-step MM] [--opening M] [--top-hat MM]
ppg plugin shipwreck trace list PROGETTO.ppg
ppg plugin shipwreck trace accept PROGETTO.ppg ID... [--name NOME]     # oppure --kind frame [--min-score 0.5]
ppg plugin shipwreck trace reject PROGETTO.ppg ID...
ppg plugin shipwreck trace names PROGETTO.ppg --table FILE             # nome, x: i nomi delle ordinate
ppg plugin shipwreck keel PROGETTO.ppg --siding MM --moulding MM [--from X --to X]   # oppure --none
ppg plugin shipwreck solids PROGETTO.ppg [--thickness MM]
ppg plugin shipwreck export3d PROGETTO.ppg --out FILE.obj|.glb|.ifc [--world] [--no-tables]
ppg plugin shipwreck plan PROGETTO.ppg [--background relief|ortho|none] [--scale N] [--out CARTELLA]
ppg plugin shipwreck status PROGETTO.ppg                               # con i conteggi dei legni 3D
```
