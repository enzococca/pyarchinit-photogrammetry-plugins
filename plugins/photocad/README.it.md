# PhotoCAD

Un plugin di [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry) (1.7.1 o
successivo) che invia l'**ortofoto**, il **DEM** (come ombreggiatura) o l'**ortofoto di un prospetto** a
[PhotoCAD](https://photocad.ai) per vettorializzarli e riporta il disegno **al suo posto**: nelle coordinate del
chunk sulla pianta, nei metri (u, v) del prospetto (e, se richiesto, in 3D sulla sua superficie). Il disegno diventa
un layer della vista 2D, un DXF georeferenziato, un GeoPackage e un SVG; i suoi contorni chiusi possono diventare US o
USM.

PhotoCAD è un **servizio esterno a pagamento**: l'immagine esce dal computer e ogni conversione usa un credito (circa
2 $; i primi 10 $ sono gratuiti). Prima di ogni invio il plugin lo dice, con i crediti rimasti e la dimensione
dell'immagine, e aspetta un sì. Niente da installare oltre al programma: la libreria standard parla con l'API; ezdxf,
rasterio, Pillow e shapely sono già nel programma.

| Dove | Cosa |
|---|---|
| *Plugin > PhotoCAD > Vettorializza l'ortofoto...* | Tutta l'ortofoto, la parte visibile nella vista o un'area disegnata. Opzioni: disegno dettagliato o semplificato, togli lo sfondo. |
| *... > Vettorializza il DEM...* | Il DEM come ombreggiatura (luce da nord-ovest), con le stesse scelte. |
| *... > Vettorializza il prospetto mostrato...* | L'ortofoto del prospetto o della sezione mostrata nella vista 2D, nelle sue (u, v); a scelta anche in 3D sulla sua superficie. |
| *... > Converti in US / USM...* | I poligoni PhotoCAD selezionati nella vista (o un intero layer PhotoCAD) diventano unità stratigrafiche del chunk attivo, sulla pianta o nei disegni del prospetto mostrato. Tipo (US, USM, ...) e un primo numero facoltativo; le polilinee chiuse valgono come poligoni, le linee aperte si saltano e si contano. Una sola voce di *Modifica > Annulla*. |
| *... > Importa come disegni liberi...* | (programma 1.7.2) Gli elementi di una conversione come disegni liberi 3D nel layer *PhotoCAD*: sulla pianta alla quota del DEM, su un prospetto sulla sua superficie. |
| *... > Esporta una conversione...* / *Rimuovi una conversione...* | Una copia in DXF, GeoPackage o SVG; una conversione tolta dal progetto con i suoi file (le US fatte dai suoi poligoni restano). |
| *... > Chiave API PhotoCAD...* | La chiave nel portachiavi del sistema: *Salva nel portachiavi*, *Dimentica*. |
| *... > Credito PhotoCAD* | I crediti rimasti sull'account. |

Ogni conversione è un layer **«PhotoCAD: \<nome\>»** della vista 2D (poligoni e linee, selezionabili, nel pannello
Layer), sulla pianta o sul prospetto su cui è stata fatta, ridisegnato ogni volta che il raster viene ricaricato.

## La chiave API

Dalla variabile d'ambiente `PHOTOCAD_API_KEY` se c'è, altrimenti dal **portachiavi del sistema** (servizio
`pyarchinit-photogrammetry`, utente `photocad`: la stessa famiglia di voci in cui l'assistente del programma tiene le
chiavi di Claude e OpenAI). Non viene mai scritta nelle impostazioni, nel progetto, nei log, in `conversion.json` o in
un messaggio d'errore (mascherata come `pcad_live_…abcd`). Senza portachiavi (la libreria `keyring` arriva con
l'extra `assistant` del programma) funziona solo la variabile d'ambiente.

## Come il disegno torna al suo posto

L'immagine copre un blocco di celle intere del raster, ridotto di un fattore intero perché il lato lungo sia al
massimo 2400 px (PhotoCAD riduce comunque a quella misura): ogni pixel dell'immagine è un quadrato noto del rilievo e
il suo angolo in basso a sinistra un punto noto. Il DXF si chiede in metri (`units=m`), senza margine (`margin=0`), con
il lato lungo dell'immagine in metri (`longest`): PhotoCAD disegna dall'angolo in basso a sinistra dell'immagine, Y in
alto, quindi spostare il disegno di quell'angolo lo mette nelle coordinate del chunk (o nelle (u, v) del prospetto).
Su un prospetto ogni vertice può andare anche nel mondo, alla profondità letta dal rilievo del prospetto.

File, in `<dati del chunk>/photocad/<id>/`: `image.jpg` (ciò che è stato inviato), `photocad.dxf` (come l'ha dato
PhotoCAD), `drawing.dxf` (georeferenziato, con i layer di PhotoCAD), `drawing.gpkg` (tabella `photocad`: poligoni e
linee con il layer PhotoCAD, lunghezza e area; nel sistema del chunk, o senza sistema per un prospetto; `photocad_3d`
con la Z per un prospetto in 3D), `drawing.svg` con `drawing.svg.json` (dove si trova: l'SVG tiene il riferimento di
PhotoCAD, Y in basso), `drawing_3d.dxf`, `drawing.pdf` se richiesto, `conversion.json` (sorgente, limiti, pixel, data,
crediti usati).

## Riga di comando

```bash
ppg plugin photocad key set            # digitata senza eco, nel portachiavi
ppg plugin photocad key status|forget
ppg plugin photocad credit
ppg plugin photocad list scavo.ppg     # elenca anche i prospetti: elevation:<id>
ppg plugin photocad convert scavo.ppg --source ortho --out pianta.dxf
ppg plugin photocad convert scavo.ppg --source elevation:3236beab --3d --out prospetto.gpkg
ppg plugin photocad convert scavo.ppg --source dem --bbox 500010,4000020,500060,4000070 --simplified --yes
ppg plugin photocad units scavo.ppg <id> --tipo USM --start 101
ppg plugin photocad export scavo.ppg <id> disegno.svg
ppg plugin photocad remove scavo.ppg <id>
```

`convert` chiede conferma al prompt se non c'è `--yes` (o se al progetto è stato detto di non chiedere più); opzioni
`--chunk NOME`, `--remove-background`, `--pdf`, `--name`.

## Limiti

- Il riferimento di PhotoCAD (origine, Y in alto, `longest` applicato all'immagine, nessun margine) segue la sua
  documentazione; il plugin è stato provato con un server finto che si comporta come documentato, non con photocad.ai.
- Il disegno è buono quanto l'immagine: un'ortofoto di qualche migliaio di pixel ridotta a 2400 px perde dettaglio;
  per il dettaglio si invia una parte (area visibile o disegnata).
- 20 conversioni al minuto; una conversione rifiutata o fallita viene spiegata a parole (una fallita viene
  rimborsata da PhotoCAD).

Licenza: GPL-2.0-or-later.
