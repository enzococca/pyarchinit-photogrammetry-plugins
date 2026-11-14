# Murature

Un plugin di [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry) (1.7.0 o successiva):
muri, edifici e torri in pietra rilevati per facce.

* **Facce**: un muro (un edificio, una torre) si rileva per facce, ognuna un prospetto del programma - sul piano
  verticale per due punti tracciati in pianta (`v` è la quota), su un cilindro per una torre tonda, un pozzo o
  un'abside (tre punti in pianta, o il cerchio adattato a una sezione orizzontale del modello; visto da fuori o da
  dentro), un piano per lato di una torre poligonale, oppure un prospetto già fatto dal programma (*Prospetto
  esistente...*, `--existing`), preso così com'è con le sue patch e i suoi disegni, anche sviluppato sull'asse della
  torre. Le facce sono elencate con dimensioni, risoluzione e disegni.
* **Forma**: sulla superficie del modello nella finestra della faccia, il piano (o il cilindro) adattato ai minimi
  quadrati con il peso di Huber; lo **strapiombo** (gradi, cm/m, azimut verso cui va la sommità); la carta dello
  **spanciamento** (scarto dalla superficie adattata, verso chi guarda, in cm) sopra il prospetto con una scala
  divergente e la legenda; **sezioni orizzontali** a quote scelte in un DXF e un SVG con gli scarti fra loro a stazioni
  scelte; l'**asse di una torre tonda** dai cerchi a più quote (inclinazione, direzione, fuori tondo); **due rilievi**
  dello stesso muro confrontati sulla faccia: la differenza come raster, mediana e 90° percentile.

* **Pietre e filari**: sulle pietre disegnate sulla faccia (lo strumento Pietre del programma sul prospetto), per
  ogni pietra larghezza, altezza, area, ellisse equivalente (allungamento, orientamento) e rettangolarità; la quota di
  malta (1 - unione delle pietre / area del muro: i limiti della faccia o un poligono disegnato; una pietra disegnata
  due volte conta una volta); i filari (ogni pietra unita alla vicina a destra con cui condivide almeno metà
  dell'altezza: anche i filari inclinati restano filari; quota media, numero, spessore, pendenza); tabelle CSV ed
  Excel, GeoPackage, misure aggiunte all'esportazione della faccia. Le pietre più piccole di quattro pixel della
  faccia sono lasciate fuori e nominate.
* **Corsi e giunti da un disegno classificato** (0.1.3, col programma 1.7.7 o successivo; con un programma più vecchio
  il pulsante è nascosto): le linee disegnate sulla faccia (o un GeoPackage, GeoJSON o DXF nelle sue `(u, v)`) sono
  classificate dal programma lungo l'orizzontale della faccia - le linee lungo l'asse sono i **letti**, una coppia di
  linee lungo l'asse (anche una fascia tracciata come poligono) un **filare** di spessore noto, le linee di traverso i
  **giunti**. Un filare sta fra un letto e il letto subito sopra; le sue pietre sono le celle divise dai giunti che ne
  coprono almeno metà dell'altezza. Una tavola di controllo mostra filari e pietre proposte; solo dopo le pietre si
  aggiungono a quelle della faccia (o le sostituiscono), saltando quelle che si sovrappongono per più di metà a una
  pietra già presente - ripetere lo stesso disegno non aggiunge nulla. Un passo di *Annulla*. I contorni delle USM
  della faccia restano fuori, salvo richiesta.
* **Tipo suggerito**, mai un verdetto: *a filari* se almeno il 70 % delle pietre sta in filari di 3 o più pietre
  inclinati meno di 2°; *regolare* se lo spessore dei filari varia poco (deviazione mediana assoluta / mediana < 20 %);
  *a conci squadrati* se la rettangolarità mediana è sopra 0,8, *allungati* se l'allungamento mediano è 2 o più.
* **USM**: le unità disegnate sulla faccia; rapporti proposti dalla geometria - *si appoggia a* (un'unità sopra
  un'altra che la tocca per almeno il 20 % della sua larghezza), *taglia* (un'unità che interrompe un filare di
  un'altra), *copre* (un'unità sopra un'altra per tutta la sommità e la larghezza), *riempie* (un'unità racchiusa da
  un'altra: un'apertura tamponata) - ognuno con la sua evidenza, scritti solo se spuntati; le regole leggono forme e
  non distinguono un taglio da un appoggio a giunto dritto; *Salva in PyArchInit*: schede US, rapporti nei due
  sensi, disegni della faccia come media.

* **Uscite**: per faccia una tavola in scala (SVG e PDF: l'ortofoto alla scala normalizzata più grande che sta in
  un A3, pietre, filari, USM, sezioni e curve dello spanciamento, barra di scala e cartiglio con le misure), un DXF
  con i layer PIETRE, CORSI, USM, SEZIONI, BORDO, QUOTE, i GeoPackage, le tabelle e i raster; per il muro un IFC4
  `IfcWall` - la lastra fra due facce rilevate (in versi opposti entro 15°, a meno di 3 m), o dietro ogni altra
  faccia per uno spessore dato (ogni lato di una torre poligonale) - con
  `Pset_PyArchInit_Masonry` e le USM come `IfcBuildingElementPart`; la faccia aperta in QGIS, spiegata.
  Dalla 0.1.2 il muro e le sue USM tengono i loro GlobalId da un'esportazione all'altra, con la regola che il
  plugin conservazione condivide, così i suoi degradi aderiscono allo stesso muro.

La prova sulla catena del programma (un muro sintetico fotografato, allineato, densificato, con il modello; pietre
trovate da SAM 2.1): `proof/make_proof.py`.

Le misure dicono come sta un muro, non lo giudicano. Il plugin scrive solo in `<dati del chunk>/masonry/`.

Rimuovendo un muro o una faccia si possono rimuovere anche i prospetti delle facce: con il programma 1.10.0 i disegni
tenuti con quei prospetti (le pietre della faccia, le sue US, quote e punti) vanno via con loro, e la domanda dice prima
quanti sono (`remove --elevations` lo dice dopo).

Riga di comando: `ppg plugin masonry face|shape|stones|classified|usm|plate|export|ifc|list|remove`.

Licenza: GPL-2.0-or-later.
