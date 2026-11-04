# PhotoCAD

Un plugin di [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry) (1.7.1 o
successivo) che invia l'**ortofoto**, il **DEM** (come ombreggiatura) o l'**ortofoto di un prospetto** a
[PhotoCAD](https://photocad.ai) per vettorializzarli e riporta il disegno **al suo posto**: nelle coordinate del
chunk sulla pianta, nei metri (u, v) del prospetto (e, se richiesto, in 3D sulla sua superficie). Il disegno diventa
un layer della vista 2D, un DXF georeferenziato, un GeoPackage e un SVG; i suoi contorni chiusi possono diventare US o
USM.

PhotoCAD è un **servizio esterno a pagamento**: l'immagine esce dal computer e ogni conversione usa un credito (circa
2 $; i primi 10 $ sono gratuiti). Prima di ogni invio il plugin lo dice, con i crediti e le conversioni gratuite rimasti e la dimensione
dell'immagine, e aspetta un sì. Niente da installare oltre al programma: la libreria standard parla con l'API; ezdxf,
rasterio, Pillow e shapely sono già nel programma.

| Dove | Cosa |
|---|---|
| *Plugin > PhotoCAD > Vettorializza l'ortofoto...* | Tutta l'ortofoto, la parte visibile nella vista o un'area disegnata. Opzioni: disegno dettagliato o semplificato, togli lo sfondo. |
| *... > Vettorializza il DEM...* | Il DEM come ombreggiatura (luce da nord-ovest), con le stesse scelte. |
| *... > Vettorializza il prospetto mostrato...* | L'ortofoto del prospetto o della sezione mostrata nella vista 2D, nelle sue (u, v); a scelta anche in 3D sulla sua superficie. |
| *... > Converti in US / USM...* | I poligoni PhotoCAD selezionati nella vista (o un intero layer PhotoCAD) diventano unità stratigrafiche del chunk attivo, sulla pianta o nei disegni del prospetto mostrato. Tipo (US, USM, ...) e un primo numero facoltativo; le polilinee chiuse valgono come poligoni, le linee aperte si saltano e si contano. Una sola voce di *Modifica > Annulla*. |
| *(automatico)* Pietre | PhotoCAD disegna le pietre come linee aperte: le estremità entro 2 px si uniscono, le linee si nodano e si poligonizzano; le facce sotto i 50 px o soprattutto sul bianco (senza dati) si scartano. Le pietre vanno in `polygons.gpkg` e in un secondo layer **«PhotoCAD: \<nome\> (pietre)»**; *Converti in US / USM* le usa per prime (`--from lines` per i contorni chiusi di PhotoCAD). Di nuovo con altri valori: `ppg plugin photocad polygonize PROGETTO ID --snap-px 3 --min-area-px 80`. |
| *... > Importa come disegni liberi...* | (programma 1.7.2) Gli elementi di una conversione come disegni liberi 3D nel layer *PhotoCAD*: sulla pianta alla quota del DEM, su un prospetto sulla sua superficie. |
| *... > Recupera una conversione...* | Una conversione già pagata scaricata e georeferenziata di nuovo, senza un nuovo invio: una finita mentre la rete o lo scaricamento si sono interrotti, o annullata quando PhotoCAD aveva già finito. |
| *... > Esporta una conversione...* / *Rimuovi una conversione...* | Una copia in DXF, GeoPackage o SVG; una conversione tolta dal progetto con i suoi file (le US fatte dai suoi poligoni restano). |
| *... > Chiave API PhotoCAD...* | La chiave nel portachiavi del sistema: *Salva nel portachiavi*, *Dimentica*. |
| *... > Credito PhotoCAD* | I crediti e le conversioni gratuite rimasti sull'account. |

Ogni conversione è un layer **«PhotoCAD: \<nome\>»** della vista 2D (poligoni e linee, selezionabili, nel pannello
Layer), sulla pianta o sul prospetto su cui è stata fatta, ridisegnato ogni volta che il raster viene ricaricato.

## La chiave API

Dalla variabile d'ambiente `PHOTOCAD_API_KEY` se c'è, altrimenti dal **portachiavi del sistema** (servizio
`pyarchinit-photogrammetry`, utente `photocad`: la stessa famiglia di voci in cui l'assistente del programma tiene le
chiavi di Claude e OpenAI). Non viene mai scritta nelle impostazioni, nel progetto, nei log, in `conversion.json` o in
un messaggio d'errore (mascherata come `pcad_live_…abcd`). Senza portachiavi (la libreria `keyring` arriva con
l'extra `assistant` del programma) funziona solo la variabile d'ambiente.

## Come il disegno torna al suo posto

L'immagine è tagliata in **tessere alla risoluzione nativa** del raster: al massimo 2400 px sul lato lungo, circa il
10 % di sovrapposizione, una conversione ciascuna (una striscia sotto i 600 px è ingrandita fino a 4 volte; più
sottile non viene inviata: PhotoCAD fallisce sulle strisce sottili). Prima dell'invio la conferma dice quante
conversioni servono e rifiuta se l'account ha meno crediti e conversioni gratuite; `--max-tiles` (6) evita che
un'immagine lunga ne prenda troppe.

Il disegno di PhotoCAD non arriva nel riferimento dell'immagine: la sua pagina è il riquadro del **soggetto** che ha
trovato, in millimetri a una scala naturale (qualunque cosa chieda `units`). Quindi il disegno di ogni tessera si
scarica in mm e si colloca per **registrazione**: la pagina sul riquadro del contenuto dell'immagine (una sola
scala, proporzioni entro il 2 %), poi spostamento e scala affinati (±24 px, ±3 %) portando le linee sui bordi
dell'immagine (l'intensità del gradiente sotto di esse, massimizzata fino a un quarantesimo di pixel). Lo scarto (distanza
mediana delle linee dai bordi dell'immagine, in pixel) resta in `conversion.json`; oltre 2 px, o con meno del 60 % delle
linee vicino a un bordo, il disegno è rifiutato e non viene messo nulla. Le tessere poi si uniscono: da ciascuna ciò
che è più vicino al suo centro (le copie nelle sovrapposizioni una volta sola). Su un prospetto ogni vertice può
andare anche nel mondo, alla profondità letta dal rilievo del prospetto.

File, in `<dati del chunk>/photocad/<id>/`: per tessera `t0/`, `t1/`...: `image.jpg` (ciò che è stato inviato),
`photocad_mm.dxf` (come l'ha dato PhotoCAD), `geometry.json`, `drawing.svg` con `drawing.svg.json` (dove si trova),
`drawing.dxf` (la tessera collocata); per l'insieme: `drawing.dxf` (georeferenziato, con i layer di PhotoCAD),
`drawing.gpkg` (tabella `photocad`: poligoni e linee con il layer PhotoCAD, lunghezza e area; nel sistema del chunk, o
senza sistema per un prospetto; `photocad_3d` con la Z per un prospetto in 3D), `drawing_3d.dxf`, `conversion.json`
(sorgente, tessere con i loro id PhotoCAD, chiavi di idempotenza e registrazioni, data, crediti usati). Ogni invio porta
una `Idempotency-Key` salvata prima di inviarlo: un invio ripetuto (un 429, una connessione persa, `fetch` dopo un
arresto) non viene mai addebitato due volte.

## Riga di comando

```bash
ppg plugin photocad key set            # digitata senza eco, nel portachiavi
ppg plugin photocad key status|forget
ppg plugin photocad credit
ppg plugin photocad list scavo.ppg     # elenca anche i prospetti: elevation:<id>
ppg plugin photocad convert scavo.ppg --source ortho --out pianta.dxf
ppg plugin photocad convert scavo.ppg --source elevation:3236beab --3d --out prospetto.gpkg
ppg plugin photocad convert scavo.ppg --source dem --bbox 500010,4000020,500060,4000070 --simplified --yes
ppg plugin photocad fetch scavo.ppg --id <id o id PhotoCAD> [--out disegno.dxf]
ppg plugin photocad units scavo.ppg <id> --tipo USM --start 101
ppg plugin photocad export scavo.ppg <id> disegno.svg
ppg plugin photocad remove scavo.ppg <id>
```

`convert` chiede conferma al prompt se non c'è `--yes` (o se al progetto è stato detto di non chiedere più); opzioni
`--chunk NOME`, `--remove-background`, `--pdf`, `--name`, `--max-tiles N`.

## Limiti

- La collocazione presuppone che l'immagine abbia dei bordi dove il disegno ha delle linee (pietre, muri, tagli: ciò
  che PhotoCAD disegna); il disegno di un'immagine piatta e senza dettagli può essere rifiutato.
- Una grande ortofoto alla risoluzione nativa richiede molte tessere (molte conversioni): si invia una parte (area
  visibile o disegnata), o si alza `--max-tiles` sapendolo.
- 20 conversioni al minuto; una conversione rifiutata o fallita viene spiegata a parole (una fallita viene rimborsata
  da PhotoCAD).

Licenza: GPL-2.0-or-later.
