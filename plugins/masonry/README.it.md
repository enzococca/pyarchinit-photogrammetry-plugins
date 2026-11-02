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

* **Pietre e filari**: sulle pietre disegnate sulla faccia (lo strumento Pietre del programma sul prospetto), per
  ogni pietra larghezza, altezza, area, ellisse equivalente (allungamento, orientamento) e rettangolarità; la quota di
  malta (1 - area delle pietre / area del muro: i limiti della faccia o un poligono disegnato); i filari (centri delle
  pietre separati dove distano in quota più di metà dell'altezza mediana; quota media, numero, spessore, pendenza);
  tabelle CSV ed Excel, GeoPackage, misure aggiunte all'esportazione della faccia. Una faccia con un prospetto più
  grossolano di un quarto della pietra più piccola viene rifiutata.
* **Tipo suggerito**, mai un verdetto: *a filari* se almeno il 70 % delle pietre sta in filari di 3 o più pietre
  inclinati meno di 2°; *regolare* se lo spessore dei filari varia poco (deviazione mediana assoluta / mediana < 20 %);
  *a conci squadrati* se l'allungamento mediano è sotto 2 e la rettangolarità mediana sopra 0,8.
* **USM**: le unità disegnate sulla faccia; rapporti proposti dalla geometria - *si appoggia a* (un'unità sopra
  un'altra che la tocca per almeno il 20 % della sua larghezza), *taglia* (un'unità che interrompe un filare di
  un'altra), *copre* (un'unità che racchiude la sommità di un'altra) - ognuno con la sua evidenza, scritti solo se
  spuntati; *Salva in PyArchInit*: schede US, rapporti nei due sensi, disegni della faccia come media.

Le misure dicono come sta un muro, non lo giudicano. Il plugin scrive solo in `<dati del chunk>/masonry/`.

Riga di comando: `ppg plugin masonry face|shape|stones|usm|list|remove`.

Licenza: GPL-2.0-or-later.
