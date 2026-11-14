# Conservazione

Un plugin di [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry) (1.7.1 o successivo):
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

* **Confronto**: due campagne della stessa faccia - nello stesso chunk, o un rilievo successivo in un altro chunk il
  cui prospetto è costruito sullo stesso piano (`compare.ensure_elevation`) - per termine l'area guadagnata e persa,
  le lesioni nuove, più lunghe o più larghe, le aree trattate che mostrano di nuovo degrado, il cambiamento di colore
  dell'ortofoto in CIELAB (`change.tif`) e, con due rilievi, il cambiamento del rilievo (`relief_change.tif`, cm);
  tabelle e una pagina PDF in `<prospetto>/compare_<A>_<B>/`. I due chunk devono essere già allineati dal programma
  (*Allinea chunk*): il plugin lo verifica (marker sulle due facce entro 1 cm, altrimenti uno spostamento delle
  ortofoto per correlazione di fase sotto 2 px) e altrimenti rifiuta, senza scrivere niente.

* **Sito web** (0.2.0, programma 1.10.0 o successivo): quando il programma costruisce il sito web di un rilievo
  (*File > Esporta > Sito web*, `ppg web build`), il livello *Degrado* sulla pianta (le aree di degrado con i colori
  della legenda del progetto, una spunta per termine; su un muro il tratto che ciascuna copre lungo la traccia della
  faccia, su un pavimento l'area stessa) e le schede *Degrado: riepilogo* (per campagna e termine: elementi,
  estensione, quota della faccia) e *Campagne di conservazione* (data, metodo, quota della faccia con degrado, lesione
  più aperta, classi EN 16096; il rilevatore non è pubblicato). Il programma li porta nel sistema del sito (locale di
  default) e ripulisce ogni testo. Dove l'aggancio del sito del programma dice il sistema del sito (`site_hook` 2) le
  aree vanno sui disegni dei loro prospetti, nei loro (u, v), con le quote in quelle del sito, e le parole nelle due
  lingue. Niente con un programma più vecchio.

La guida: [Conservazione](https://enzococca.github.io/pyarchinit-photogrammetry/it/plugins/conservation/) (il sorgente
è in `docs/` qui finché non passa nella documentazione del programma; la tabella della legenda la genera
`scripts/legend_table.py`). `proof/run_proof.py` fa girare i rilevatori, una campagna, gli esiti e l'IFC su un
prospetto vero e scrive un `LEGGIMI.md` con i numeri e i tempi; copia prima il progetto e non scrive mai
l'originale.

Il plugin scrive solo `chunk.meta["conservation"]` e i file sotto `<dati del chunk>/conservation/<prospetto>/`.

Licenza: GPL-2.0-or-later.
