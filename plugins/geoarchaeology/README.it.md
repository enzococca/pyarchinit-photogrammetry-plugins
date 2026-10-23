# Geoarcheologia

*[English](README.md)*

Plugin di [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry)
(1.2.2 o successiva): i log di trivellate manuali, carotaggi meccanici e sezioni di scavo
trasformati in superfici e in un modello 3D dei depositi, sullo stesso terreno del rilievo
fotogrammetrico (il DEM e l'ortofoto del chunk).

* **Sondaggi** da CSV: una tabella dei collari (id, x, y nel sistema di riferimento del chunk,
  quota della bocca, profondità) e una degli intervalli (id, da e a in metri dalla bocca,
  l'unità - un numero di US o un'unità litologica -, descrizione, colore Munsell o RGB,
  litologia), oppure una tabella sola con entrambi. Nomi delle colonne in italiano o in inglese
  (`sondaggio`, `da`, `a`, `us`, `descrizione`, `colore`... oppure `hole_id`, `from`, `to`, `unit`,
  `description`, `colour`...), separatore virgola, punto e virgola o tabulazione, decimali col punto
  o con la virgola; gli intervalli si possono dare anche in quote (`quota_tetto`, `quota_letto`). I
  collari senza quota la prendono dal DEM; quelli che l'hanno vengono confrontati con esso.
  Sovrapposizioni, vuoti e unità fuori ordine vengono segnalati.
* **Ordine delle unità** dai log (un'unità sopra un'altra in un sondaggio sta sopra di essa, i
  disaccordi vanno alla maggioranza), oppure dato a mano.
* **Colonne stratigrafiche**: un sondaggio con le sue descrizioni, oppure un pannello di più
  sondaggi su un asse comune delle quote con i tetti della stessa unità collegati; le unità nel
  loro colore (il colore Munsell del log quando c'è) con un retino per le litologie nominate
  (argilla, limo, sabbia, ghiaia, pietre, cenere, torba, organico); SVG e PNG con matplotlib.
* **Superfici di contatto per kriging**, scritto in NumPy: per il tetto di ogni unità il
  semivariogramma sperimentale, un modello sferico, esponenziale o gaussiano con effetto pepita
  adattato ai minimi quadrati pesati (pesi di Cressie), kriging ordinario o kriging universale con
  una deriva lineare (un contatto che pende attraverso il sito); il modello e la deriva con
  l'errore leave-one-out minore quando richiesto (`auto`), l'errore riportato; la varianza di
  kriging come seconda banda; inverso della distanza sotto i 5 sondaggi. Dove un'unità manca in un
  log il suo tetto è il contatto fra le unità sopra e sotto (spessore nullo); un sondaggio che
  finisce sopra un contatto tiene il contatto sotto il proprio fondo. In modo `depth` si
  interpolano le profondità dal terreno, poi appese al DEM. Sulla griglia del DEM (o con una cella
  data), NaN fuori dall'inviluppo dei sondaggi più un margine.
* **Pila coerente**: nessuna superficie sopra il terreno (il DEM) o sopra quella dell'unità
  superiore; le celle abbassate vengono contate (dal terreno, da un contatto superiore) e la
  correzione è la terza banda. **Spessori** (isopache) fra contatti consecutivi, con il volume.
* **Sezioni** lungo una traccia disegnata sull'ortofoto o sul DEM (lo strumento linea del
  programma, due clic), digitata, o per i sondaggi scelti: il profilo del terreno e di ogni
  contatto, le unità colorate fra di essi, i sondaggi entro una distanza proiettati sulla traccia
  con i loro log e la distanza dalla traccia; in scala (scelta per un foglio A0 quando non data,
  come le sezioni del programma) con un'esagerazione verticale; SVG, PNG, DXF (coordinate di
  sezione, unità come campiture del loro colore) e CSV del profilo.
* **Modello 3D delle unità**: un solido chiuso per unità fra il suo tetto e il tetto dell'unità
  sotto (l'ultima fino a un fondo), costruito sulla griglia come il programma costruisce i proxy
  delle unità stratigrafiche; un'unità il cui codice corrisponde a US disegnate nel chunk viene
  ritagliata sui loro poligoni. Un PLY per unità (doppia precisione, colorato), un OBJ con un
  oggetto e un materiale per unità, i volumi per unità in un CSV.
* Il pannello *Geoarcheologia* del dock del progetto: sondaggi, superfici e sezioni in tre
  tabelle, i sondaggi come layer di punti della vista (id, quota, profondità, unità), una superficie
  (la quota, la varianza o la correzione) o uno spessore sopra l'ortofoto o il DEM con palette e
  opacità, le tracce delle sezioni come layer, le esportazioni, il progetto QGIS (superfici,
  sondaggi e sezioni). *File > Esporta > Sondaggi e superfici geoarcheologiche (GeoPackage)*.
* `ppg plugin geoarchaeology import|list|order|surfaces|section|model|logs|export|remove PROGETTO ...`;
  `ppg plugin geoarchaeology help` stampa le opzioni di ogni comando.

```bash
pip install "ppg-geoarchaeology @ git+https://github.com/enzococca/pyarchinit-photogrammetry-plugins#subdirectory=plugins/geoarchaeology"
ppg plugin geoarchaeology import scavo.ppg collari.csv intervalli.csv
ppg plugin geoarchaeology surfaces scavo.ppg --model auto --trend auto
ppg plugin geoarchaeology section scavo.ppg --through S1,S4,S7 --exaggeration 5
ppg plugin geoarchaeology section scavo.ppg --coords "438100,4553800;438180,4553860"
ppg plugin geoarchaeology logs scavo.ppg S1 S2 S3 S4
ppg plugin geoarchaeology model scavo.ppg
ppg plugin geoarchaeology export scavo.ppg geoarcheologia.gpkg
```

Una tabella dei collari e una degli intervalli:

```text
sondaggio;x;y;quota;profondita
S1;438102,5;4553803,1;101,25;4,2
S2;438130,0;4553811,7;;3,6

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
