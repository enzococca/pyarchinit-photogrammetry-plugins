# Linee elettriche

Un plugin di [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry)
(1.1 o successiva) che trova le **linee elettriche aeree** nella nuvola densa di un chunk: ogni
conduttore come catenaria con lunghezza, freccia, franco dal suolo e distanza dall'ostacolo più
vicino (vegetazione, edifici), i pali dove i conduttori terminano, i punti dei cavi e dei pali
classificati (classi ASPRS 14 e 15). Niente da installare oltre il programma: numpy, SciPy,
shapely ed ezdxf arrivano con lui.

| Dove | Cosa |
|---|---|
| *Plugin > Linee elettriche > Rileva linee elettriche...* | Il rilevamento sulla nuvola densa del chunk attivo, in background, con i parametri qui sotto. I conduttori (layer *Linee elettriche*) e i pali (layer *Pali*) vengono disegnati sulle viste dell'ortofoto e del DEM con i loro attributi; un riepilogo dà conduttori, linee, lunghezza totale, franco minimo e ostacolo più vicino. I punti dei cavi prendono la classe 14, quelli dei pali la 15 (vista 3D: colori per classe). |
| *... > Mostra le linee elettriche* | Di nuovo i layer, per esempio dopo aver riaperto il progetto. |
| *... > Elenco delle linee elettriche...* | Una tabella: id, linea, lunghezza, freccia, franco, ostacolo, azimut, punti; i pali con altezza e conduttori. |
| *... > Dimentica le linee elettriche* | Toglie il risultato e le classi 14 e 15 date dal plugin (le etichette date a mano restano). |
| *File > Esporta > Linee elettriche (GeoPackage)* | Tabelle `linee_elettriche` (LINESTRING Z, un vertice ogni metro, gli attributi della tabella) e `pali` (POINT Z a terra), nel CRS del chunk. |
| *File > Esporta > Linee elettriche (DXF)* | Polilinee 3D sul layer `LINEE_ELETTRICHE`, i pali come linee verticali su `PALI`, etichette `L1`, `P1`... su `ETICHETTE`; metri, coordinate mondo. |
| *File > Esporta > Linee elettriche (CSV)* | Una riga per conduttore con attributi ed estremi; i pali in `<nome>_pali.csv` accanto. |

Dalla riga di comando:

```bash
ppg plugin powerlines detect scavo.ppg [--chunk NOME] [--min-height 4] [--radius 1] [--linearity 0.85] \
    [--gap 2] [--min-length 5] [--min-points 10] [--merge-gap 15] [--no-classify] [--export linee.gpkg]
ppg plugin powerlines list scavo.ppg
ppg plugin powerlines export scavo.ppg linee.dxf      # .gpkg, .dxf o .csv
ppg plugin powerlines clear scavo.ppg
```

## Installazione

* Nel programma: *Strumenti > Plugin...*, `powerlines`, *Installa / aggiorna*, *Riavvia ora*.
* Con pip, nell'ambiente Python del programma:

  ```bash
  pip install "ppg-powerlines @ git+https://github.com/enzococca/pyarchinit-photogrammetry-plugins#subdirectory=plugins/powerlines"
  ```

## Come funziona

Il rilevamento lavora sui punti della nuvola densa, nel sistema mondo del chunk (metri, z verso
l'alto: un rilievo georeferenziato o livellato).

1. **Altezza dal suolo** di ogni punto: dai punti del terreno se la nuvola è classificata
   (classe 2 su almeno il 2 % dei punti, per esempio con la classificazione del terreno del
   plugin terrio), altrimenti dal filtro morfologico progressivo del programma (celle di 1 m,
   finestre fino a 40 m).
2. **Candidati**: punti sopra l'altezza minima i cui vicini (entro il raggio) stanno lungo una
   retta - la direzione del punto - senza nulla di fianco alla retta (un tetto ha punti di
   fianco a ogni punto, un cavo nulla fino al conduttore vicino) e senza nulla nell'aria
   0,3-3 m sotto (il bordo di un tetto ha il muro sotto, un ramo il tronco, un cavo è sospeso).
   La direzione non può essere più ripida di 37 gradi.
3. **Cavi**: candidati collegati quando ciascuno sta entro la distanza massima ed entro 25 cm
   dalla retta dell'altro, così i conduttori paralleli non si collegano mai; una sequenza di
   punti che scavalca un palo viene tagliata dove comincia la campata successiva (entrambe le
   campate pendono via dal palo: due parabole aderiscono molto meglio di una); restano i gruppi
   con abbastanza punti e lunghezza.
4. **Catenaria** `z = z0 + a (cosh((s - s0) / a) - 1)` lungo la direzione del cavo in pianta,
   stimata ai minimi quadrati robusti (il punto più basso può stare fuori dal tratto visto).
   I tratti di uno stesso cavo separati da un buco della ricostruzione vengono uniti quando
   stanno sulla stessa retta, il buco non supera il massimo e la catenaria di uno raggiunge
   l'altro con la stessa pendenza.
5. **Pali** dove gli estremi di più cavi si incontrano entro 3 m, lontano dal bordo del rilievo
   (dove la nuvola semplicemente finisce), confermati da una struttura verticale sottile di
   punti quando la nuvola ce l'ha. I pali sottili spesso non lasciano punti in fotogrammetria:
   un palo senza punti resta solo dove terminano insieme almeno due conduttori, e viene messo
   dai loro estremi.
6. **Franco e ostacoli**: lungo ogni cavo (un campione ogni metro) l'altezza minima dal suolo e
   la distanza dal punto più vicino della nuvola che non sia cavo né palo e stia ad almeno
   0,5 m dal suolo.

| Parametro | Predefinito | |
|---|---|---|
| Altezza minima dal suolo | 4 m | Sotto non si cercano cavi: abbassarla per linee basse, alzarla per escludere recinzioni, muri e veicoli. |
| Raggio dell'intorno | 1 m | I punti vicini danno la direzione: tenerlo sotto la distanza fra conduttori paralleli. |
| Linearità minima | 0,85 | `(l1 - l2) / l1` dei punti lungo la direzione: 1 è una retta perfetta; abbassarla con nuvole rumorose. |
| Distanza massima fra punti di un cavo | 2 m | Alzarla con ricostruzioni rade. |
| Lunghezza e punti minimi di un cavo | 5 m, 10 | Gruppi più corti o più poveri non sono cavi. |
| Buco massimo da scavalcare | 15 m | Tratti dello stesso cavo separati da un buco della ricostruzione vengono uniti fino a questo buco. |
| Classi 14 e 15 nella nuvola | sì | Punti dei cavi e dei pali classificati; un passaggio precedente del plugin viene prima azzerato. |

Il risultato resta nel progetto (`chunk.meta["powerlines"]`): ogni cavo con la sua catenaria
(`origin`, `u`, `a`, `s0`, `z0`, `s_min`, `s_max`), i vertici ogni metro, la lunghezza (dell'arco),
la freccia (massima distanza sotto la corda), il franco, la distanza dall'ostacolo, l'azimut e il
numero di punti; ogni palo con posizione, quota del terreno, altezza (cima dei suoi punti, o
dell'attacco più alto), conduttori, linee e se la nuvola contiene la sua struttura. Le "linee"
sono gruppi di conduttori paralleli affiancati sulla stessa campata.

## Limiti

* Il rilevamento lavora sui punti: i cavi devono stare nella nuvola densa. In fotogrammetria i
  cavi sottili lontani dalla camera non vengono ricostruiti: volare vicino (voli di ispezione a
  20-30 m) e costruire le mappe di profondità in qualità alta. Un rilevamento sulle foto stesse
  è un possibile passo successivo.
* I conduttori più vicini di mezzo metro (fasci) non vengono separati.
* Un cavo ha bisogno di aria libera 0,3-3 m sotto di sé: cavi appoggiati a un tetto o lungo un
  muro non vengono trovati.
* Le quote sono quelle della nuvola: senza punti di controllo il franco porta con sé l'errore
  del rilievo.

## Test

Un rilievo sintetico (terreno in pendenza con alberi, un edificio, sei pali, una linea di tre
conduttori paralleli su tre campate - una con un buco nella ricostruzione - e una linea più
bassa che la incrocia): cavi, catenarie, pali, franco, ostacoli, classi, le esportazioni, i layer
della finestra, la riga di comando e il plugin nel programma.

```bash
pip install -e plugins/powerlines pytest
python -m pytest -q plugins/powerlines/tests
```

## Licenza

GPL-2.0-or-later, come il programma.
