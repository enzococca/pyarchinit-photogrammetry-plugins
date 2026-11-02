# Murature

Un plugin di [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry) (1.7.0 o successiva):
muri, edifici e torri in pietra rilevati per facce.

* **Facce**: un muro (un edificio, una torre) si rileva per facce, ognuna un prospetto del programma - sul piano
  verticale per due punti tracciati in pianta (`v` è la quota), su un cilindro per una torre tonda, un pozzo o
  un'abside (tre punti in pianta, o il cerchio adattato a una sezione orizzontale del modello; visto da fuori o da
  dentro), o un piano per lato di una torre poligonale. Le facce sono elencate con dimensioni, risoluzione e disegni.
* **Forma**: sulla superficie del modello nella finestra della faccia, il piano (o il cilindro) adattato ai minimi
  quadrati con il peso di Huber; lo **strapiombo** (gradi, cm/m, azimut verso cui va la sommità); la carta dello
  **spanciamento** (scarto dalla superficie adattata, verso chi guarda, in cm) sopra il prospetto con una scala
  divergente e la legenda; **sezioni orizzontali** a quote scelte in un DXF e un SVG con gli scarti fra loro a stazioni
  scelte; l'**asse di una torre tonda** dai cerchi a più quote (inclinazione, direzione, fuori tondo); **due rilievi**
  dello stesso muro confrontati sulla faccia: la differenza come raster, mediana e 90° percentile.

Le misure dicono come sta un muro, non lo giudicano. Il plugin scrive solo in `<dati del chunk>/masonry/`.

Riga di comando: `ppg plugin masonry face|shape|list|remove`.

Licenza: GPL-2.0-or-later.
