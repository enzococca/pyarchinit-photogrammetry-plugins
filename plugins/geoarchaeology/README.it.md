# Geoarcheologia

*[English](README.md)*

**[Guida d'uso](https://enzococca.github.io/pyarchinit-photogrammetry/it/plugins/geoarchaeology/)**: i dati da raccogliere, come compilare le tabelle CSV, i passi nel programma, come leggere i risultati.

Plugin di [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry)
(1.3.3 o successiva): i log di trivellate manuali, carotaggi meccanici e sezioni di scavo
trasformati in superfici e in un modello 3D dei depositi, sullo stesso terreno del rilievo
fotogrammetrico (il DEM e l'ortofoto del chunk).

* **Sondaggi** da CSV: una tabella dei collari (id, x, y nel sistema di riferimento del chunk, quota
  della bocca, profondità) e una degli intervalli (id, da e a in metri dalla bocca, l'unità - un
  numero di US o un'unità litologica -, descrizione, colore, litologia), oppure una tabella sola con
  entrambi. Nomi delle colonne in italiano o in inglese come si scrivono sul campo
  (`Sigla sondaggio`, `Quota s.l.m. (m)`, `Codice US`, `US n.`, `Da (cm)`... oppure `Hole ID`,
  `Elevation (m a.s.l.)`, `From (cm)`, `Context No.`...), unità di misura convertite; un nome che il
  campo lo indovina soltanto (`codice`, `h`, `tetto`/`letto` con dentro delle quote, distinte dalle
  profondità in base ai valori) viene preso quando non c'è niente di più chiaro, e viene segnalato.
  Separatore virgola, punto e virgola o tabulazione; numeri letti nel contesto del file (punto o
  virgola decimale, separatori delle migliaia); un file separato da virgole con virgole decimali
  senza virgolette viene ricomposto dove si può fare in un modo solo. Gli intervalli si possono dare
  anche in quote (`quota_tetto`, `quota_letto`). Colori come notazioni Munsell (anche `2,5Y 5/2`),
  `#rrggbb`, `r,g,b` o a parole in italiano o in inglese (`bruno rossastro`,
  `light yellowish brown`). Codici delle unità confrontati in una forma sola (`US 1`, `US1`, `us 1`
  e, quando un solo tipo ha quel numero, `1`), mostrati come sono scritti più spesso. I collari senza
  quota la prendono dal DEM; quelli che l'hanno vengono confrontati con esso. Sovrapposizioni, vuoti,
  unità fuori ordine, righe e colori lasciati fuori vengono segnalati, con le righe dei file.
* **Ordine delle unità** dai log (un'unità sopra un'altra in un sondaggio sta sopra di essa, i
  disaccordi vanno alla maggioranza), oppure dato a mano.
* **Colonne stratigrafiche**: un sondaggio con le sue descrizioni, oppure un pannello di più
  sondaggi su un asse comune delle quote con i tetti della stessa unità collegati; le unità nel
  loro colore (il colore Munsell del log quando c'è) con il retino della litologia nominata
  (argilla, limo, sabbia, ghiaia, pietre, mattone crudo, cenere, torba, organico: prima la colonna
  della litologia, poi la descrizione, e decide il nome principale - `ghiaia sabbiosa` è ghiaia,
  `sandy gravel` anche); SVG e PNG con matplotlib, un pannello in un file suo (`colonne_S1-S2-S3`).
* **Superfici di contatto per kriging**, scritto in NumPy: per il tetto di ogni unità il
  semivariogramma sperimentale, un modello sferico, esponenziale o gaussiano con effetto pepita
  adattato ai minimi quadrati pesati (pesi di Cressie), kriging ordinario o kriging universale con
  una deriva lineare (un contatto che pende attraverso il sito). Con `auto` un modello o una
  deriva non vincono per un guadagno trascurabile dell'errore leave-one-out: fra quelli entro il 5 %
  dell'errore minore si prende il più stabile (sferico, esponenziale, senza deriva prima che con,
  gaussiano per ultimo), e ogni modello provato viene riportato con il suo errore. La varianza di
  kriging è la seconda banda; inverso della distanza sotto i 5 sondaggi. Un sondaggio che finisce
  sopra un contatto tiene il contatto sotto il proprio fondo. Sulla griglia del DEM (o con una
  cella data), NaN fuori dall'inviluppo dei sondaggi più un margine.
* **Ogni contatto nel suo modo**: per *quota* (un contatto con una forma sua: il suolo naturale
  sotto un tell, un piano sepolto) o per *profondità* dal terreno, appesa al DEM (un contatto che
  segue il terreno: la base dell'humus, gli strati drappeggiati su un rilievo). Il modo `auto`,
  predefinito, prende la profondità quando le profondità del contatto variano fra i sondaggi meno
  della metà delle sue quote (la dispersione attorno a un piano), la quota altrimenti; il modo si
  può fissare per alcune unità (`--mode auto,US2=depth,US5=elevation`, il campo *Per unità* del
  dialogo, `modes={"US 2": "depth"}`). Il registro dice il modo di ogni contatto e perché.
* **Pila coerente**: dove due contatti si incrociano cede quello con la varianza di kriging
  maggiore (in ogni colonna la regressione pesata sulle varianze che tiene i contatti in ordine:
  un suolo naturale ben determinato non viene spinto giù da un contatto incerto sopra di esso);
  nessuno sale sopra il terreno (il DEM). Un'unità che manca in un log manca anche intorno: i
  sondaggi votano la sua presenza con l'inverso della distanza, e dove vincono quelli senza e i
  due contatti non distinguono uno spessore da nessuno (sotto due deviazioni standard del loro
  kriging) il suo tetto e la sua base sono una sola superficie; niente unità sottili spurie fra i
  sondaggi. Le celle spostate vengono contate (abbassate al terreno, abbassate da un contatto
  sopra, alzate da uno sotto) e la correzione è la terza banda (positiva abbassata, negativa
  alzata). **Spessori** (isopache) fra contatti consecutivi, con il volume sommato sulle celle col
  DEM (quelle dove il terreno è la superficie dei collari sono riportate a parte) e dove l'unità
  manca.
* **Sezioni** lungo una traccia disegnata sull'ortofoto o sul DEM (il disegno del programma,
  prestato al plugin: due o più vertici, doppio clic o Invio per finire), digitata, o per i sondaggi
  scelti: il profilo del terreno e di ogni
  contatto, le unità colorate fra di essi, i sondaggi entro una distanza proiettati sulla traccia
  con i loro log e la distanza dalla traccia; in scala (scelta per un foglio A0 quando non data,
  come le sezioni del programma) con un'esagerazione verticale; SVG, PNG, DXF (coordinate di
  sezione, unità come campiture del loro colore) e CSV del profilo. Dove matplotlib non sa scrivere
  SVG o PDF (l'applicazione desktop fino alla 1.3.0 ha solo il backend del PNG), sezioni e colonne
  vengono scritte solo in PNG, e il plugin lo dice.
* **Modello 3D delle unità**: un solido chiuso per unità fra il suo tetto e il tetto dell'unità
  sotto (l'ultima fino a un fondo: il fondo più basso dei sondaggi, tenuto un margine - il 5 %
  della loro altezza - sotto il tetto più basso dell'ultima unità, perché il solido non abbia
  buchi), costruito sulla griglia come il programma costruisce i proxy delle unità stratigrafiche;
  un'unità il cui codice corrisponde a US disegnate nel chunk viene ritagliata sui loro poligoni.
  Un PLY per unità (doppia precisione, colorato), un OBJ con un oggetto e un materiale per unità, i
  volumi per unità in un CSV (sulle celle col DEM, le altre nelle loro colonne). Alla fine le unità
  si vedono insieme nella vista 3D del programma, ognuna nel suo colore (`unita_vista.ply`, se si
  vuole separate in verticale); il pannello le fa rivedere, apre la cartella del modello o fa delle
  unità il modello di un nuovo chunk del progetto (*Apri come modello di un nuovo chunk*), dove il
  pulsante *Taglia* della vista 3D del programma le taglia e chiude ogni unità sul taglio nel suo
  colore.
* Il pannello *Geoarcheologia* (con il programma 1.3.4 un pannello a sé accanto all'area di lavoro, da *Vista > Pannelli*; prima una scheda del dock del progetto): sondaggi, superfici e sezioni in tre
  tabelle, i sondaggi come layer di punti della vista (id, quota, profondità, unità), una superficie
  (la quota, la varianza o la correzione - questa su una rampa divergente attorno a zero, celle
  abbassate e alzate distinte) o uno spessore sopra l'ortofoto o il DEM con palette e opacità, una
  sezione per i sondaggi nell'ordine in cui sono stati scelti, le tracce delle sezioni come layer
  (sondaggi e tracce ridisegnati su ogni raster che la vista mostra),
  le esportazioni, il progetto QGIS (superfici, sondaggi e sezioni; anche per un chunk in un sistema
  locale). *File > Esporta > Sondaggi e superfici geoarcheologiche (GeoPackage)*.
* `ppg plugin geoarchaeology import|list|order|surfaces|section|model|model-chunk|logs|export|remove PROGETTO ...`;
  `ppg plugin geoarchaeology help` stampa le opzioni di ogni comando.

```bash
pip install "ppg-geoarchaeology @ git+https://github.com/enzococca/pyarchinit-photogrammetry-plugins#subdirectory=plugins/geoarchaeology"
ppg plugin geoarchaeology import scavo.ppg collari.csv intervalli.csv
ppg plugin geoarchaeology surfaces scavo.ppg --model auto --trend auto --mode auto
ppg plugin geoarchaeology surfaces scavo.ppg --mode "auto,US2=depth,US5=elevation"
ppg plugin geoarchaeology section scavo.ppg --through S1,S4,S7 --exaggeration 5
ppg plugin geoarchaeology section scavo.ppg --coords "500100,4500800;500180,4500860"
ppg plugin geoarchaeology logs scavo.ppg S1 S2 S3 S4
ppg plugin geoarchaeology model scavo.ppg
ppg plugin geoarchaeology model-chunk scavo.ppg
ppg plugin geoarchaeology export scavo.ppg geoarcheologia.gpkg
```

Una tabella dei collari e una degli intervalli:

```text
sondaggio;x;y;quota;profondita
S1;500102,5;4500803,1;101,25;4,2
S2;500130,0;4500811,7;;3,6

sondaggio;da;a;us;descrizione;colore
S1;0;0,3;1;humus;10YR 3/2
S1;0,3;1,6;2;limo sabbioso;10YR 5/4
S1;1,6;4,2;3;argilla grigia;2.5Y 5/2
```

Tutto sta in `chunk.meta["geoarchaeology"]` e in `<dati del chunk>/geoarchaeology/` (`sources/`
le tabelle lette, `sondaggi.gpkg` sondaggi, intervalli e sezioni, `surfaces/`, `sections/`,
`logs/`, `model/`). Nessuna dipendenza oltre a quelle del programma (NumPy, SciPy, rasterio,
shapely, ezdxf, matplotlib con l'interfaccia; colour-science per i colori Munsell quando c'è).

Licenza: GPL-2.0-or-later.
