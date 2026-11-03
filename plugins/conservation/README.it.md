# Conservazione

Un plugin di [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry) (1.7.0 o successivo):
la mappatura del degrado sui prospetti del programma.

* **Lessico**: i termini del glossario ICOMOS-ISCS *Illustrated glossary on stone deterioration patterns* (2008) sono
  i codici (`ISCS.CRACK`, `ISCS.CRACK.HAIR_CRACK`, `ISCS.ALVEOLIZATION`, ...); i 27 termini della UNI 11182:2006 ne
  sono il volto italiano, ognuno collegato al termine o ai termini ICOMOS-ISCS corrispondenti (il collegamento è del
  plugin: non esiste una tabella ufficiale); i sintomi delle murature e strutturali che nessuno dei due glossari ha
  (fuori piombo, spanciamento, perdita di malta, pietre smosse, giunto aperto) sono termini locali, segnalati come
  tali; gli interventi seguono le famiglie della EN 15898. I nomi sono quelli dei glossari, le definizioni di una riga
  sono parafrasi del plugin con la fonte. Un progetto può aggiungere i suoi termini.
* **Legenda**: colori e retini sono quelli predefiniti del plugin - nessuna norma li prescrive - e si cambiano per
  progetto; vengono scritti anche come stili di QGIS.
* **Campagne**: un rilievo di una faccia (un prospetto del programma) in una data, con il rilevatore, il metodo e le
  classi EN 16096 di condizione e di urgenza scelte dal rilevatore; la classe di raccomandazione dell'intero oggetto una
  volta per data.
* **Elementi**: aree di degrado, lesioni come linee con la loro apertura, interventi, note - disegnati sul prospetto
  nei suoi metri (u, v), con estensione, intensità, le foto che li vedono e le unità stratigrafiche su cui stanno; nel
  mondo vengono adagiati sulla superficie attraverso il rilievo del prospetto.

Il plugin scrive solo `chunk.meta["conservation"]` e i file sotto `<dati del chunk>/conservation/<prospetto>/`.

Licenza: GPL-2.0-or-later.
