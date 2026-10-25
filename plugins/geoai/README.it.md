# GeoAI

GeoAI 0.1 impara classi di pixel dai **disegni US verificati** e applica il modello ad altre ortofoto.
Produce proposte da controllare: elimina i poligoni errati nella vista, poi trasforma quelli rimasti
o selezionati in US oppure esportali in GeoPackage. Non deduce i rapporti stratigrafici.

Richiede pyarchinit-photogrammetry 1.5.0. GeoAI si installa autonomamente; quando è presente anche
SamGeo, i due plugin condividono `geoai-env` preservando le funzionalità già installate.
PyTorch e GeoAI girano in un processo separato dal programma.
`Plugin > GeoAI > Installa GeoAI...` installa le dipendenze dei modelli.
È consigliata una GPU; sul processore sono adatte solo piccole prove. CUDA e Metal dipendono dalle
versioni installate di PyTorch e GeoAI. Non servono i pesi SAM 3 né il relativo token.

## Nel programma

1. Apri `Plugin > GeoAI > Pannello GeoAI`, con ortofoto e disegni US controllati.
2. Scegli il campo delle US. Nella tabella assegna ogni valore a una classe e a un tipo US (`US`,
   `USM`...). Lo stesso nome raggruppa più valori; una classe vuota esclude quel valore.
3. **Disegna area verificata**: delimita soltanto l'area completamente annotata. Il terreno senza
   US dentro quell'area diventa sfondo. Fuori dall'area, nelle US escluse, nelle sovrapposizioni con
   classi contraddittorie e nei pixel nodata, i pixel restano sconosciuti e sono ignorati.
4. **Prepara dataset...** crea una nuova cartella. Servono le prime tre bande RGB uint8 dell'ortofoto,
   con pixel quadrati e griglia in metri orientata a nord. I tasselli sono di 512 px. L'area deve
   coprire almeno quattro colonne di tasselli: la parte destra è riservata alla validazione e una
   colonna intera separa i due insiemi. Ogni classe e lo sfondo devono comparire in entrambi.
5. **Addestra modello...** sceglie il dataset e una nuova cartella, per esempio `muri.ppgseg`. La
   U-Net/ResNet-34 di GeoAI parte dall'encoder ImageNet; i pixel sono normalizzati con uint8/255.
   Alla fine sono disponibili IoU per classe e matrice di confusione. Si conserva il modello con
   la migliore IoU media sulla validazione: quella misura serve a scegliere il modello, non è una
   prova finale indipendente.
6. **Applica modello...** usa il modello sull'ortofoto del chunk attivo, alla stessa risoluzione e
   con la stessa codifica RGB del dataset. Se serve, ricampiona prima. Produce un GeoTIFF delle
   classi e un GeoPackage di proposte, visibili in azzurro.
7. Controlla le proposte con Seleziona/Canc. **US** converte quelle selezionate o tutte quelle
   rimaste se non c'è selezione; le conversioni già presenti non sono duplicate e Annulla permette
   di ripeterle. **GeoPackage...** esporta le proposte corrette. Il raster mantiene la classificazione
   originale del modello anche quando elimini proposte vettoriali.

La cartella `.ppgseg` contiene `model.json`, `weights.pt` e `history.json`: trasferiscila intera.
Le classi, i tipi US, la risoluzione, la normalizzazione, la divisione spaziale, le versioni dei
pacchetti e le metriche accompagnano il modello. Un dataset o addestramento fallito non riceve il
manifest pronto; ripeti in una nuova cartella. Annullare ferma il processo, ma può lasciare file
parziali da ispezionare.

## Riga di comando

```sh
ppg plugin geoai install
ppg plugin geoai status
ppg plugin geoai prepare scavo.ppg classi.json dataset
ppg plugin geoai train dataset muri.ppgseg --epochs 20 --batch-size 2 --device cuda
ppg plugin geoai apply altro.ppg muri.ppgseg --min-area 0.01
ppg plugin geoai export altro.ppg ID_RICERCA proposte.gpkg
```

Esempio di `classi.json`, con l'area da sostituire con quella verificata nelle coordinate del chunk:

```json
{
  "field": "tipo",
  "mapping": {"USM": "muratura", "US": "terreno"},
  "kinds": {"muratura": "USM", "terreno": "US"},
  "reviewed_area": "POLYGON ((0 0, 50 0, 50 50, 0 50, 0 0))",
  "tile_size": 512,
  "validation_fraction": 0.2
}
```

## Ambito della prima versione

Un chunk per dataset, sole bande RGB. DEM, dataset da più chunk e confronto fra date sono passi
successivi. La separazione spaziale limita la contaminazione fra addestramento e validazione;
verifica comunque il modello su un rilievo indipendente. Una classe descrive un'ipotesi di materiale
o aspetto, non l'interpretazione stratigrafica.

I test sintetici verificano etichette, aree sconosciute, nodata, separazione spaziale, processo
isolato, addestramento/applicazione simulati, poligoni alle giunzioni e attributi delle US. Il motore
`PPG_GEOAI_FAKE=1` serve esclusivamente ai test e i suoi modelli sono rifiutati nell'uso normale.
Addestramento GPU reale e qualità archeologica devono essere verificati con dati reali.

[API GeoAI](https://opengeoai.org/train/). Codice del plugin GPL-2.0-or-later; le dipendenze e i pesi
preaddestrati conservano le rispettive licenze.
