# Piani di volo

Un plugin di [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry) (1.7.4 o successivo):
i piani di volo del drone, preparati prima di andare sul campo.

In lavorazione (la 0.1.0 non è ancora pubblicata). Il pianificatore - griglie a serpentina, anelli, percorsi
punto per punto, tratte che stanno in una batteria, i profili delle camere - viene dal pianificatore di Photo3D
dello stesso autore.

## Riga di comando

```
ppg plugin flightplan cameras
ppg plugin flightplan design PROGETTO.ppg --bbox O,S,E,N --goal orto --value 2 [--terrain] [--takeoff LAT,LON]
ppg plugin flightplan show PROGETTO.ppg
ppg plugin flightplan coverage PROGETTO.ppg [--cell M] [--low N]
ppg plugin flightplan export PROGETTO.ppg --format kmz-zip --out volo.zip
ppg plugin flightplan djifly PROGETTO.ppg --zip pronte.zip --missions UUID,UUID
ppg plugin flightplan design --coords "lat,lon;lat,lon;lat,lon" --plan piano.json   # senza progetto
```

I piani di un progetto restano nel suo chunk (`list`, `use ID`, `remove ID`); la copertura prevista (foto per
cella e GSD sul DEM del chunk) si scrive nella cartella del chunk. Ogni esportazione ricalcola prima le quote dei
blocchi che seguono il terreno e sono cambiati nel frattempo.

Licenza: GPL-2.0-or-later.
