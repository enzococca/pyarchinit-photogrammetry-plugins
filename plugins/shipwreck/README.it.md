# Relitto

*[English](README.md)*

**[Guida d'uso](https://enzococca.github.io/pyarchinit-photogrammetry/it/plugins/shipwreck/)**: dal modello 3D di uno scafo scavato ai suoi disegni, passo per passo.

Plugin di [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry) (1.3.4 o successiva): il
disegno tecnico di uno scafo scavato dal suo modello 3D. Dove corre l'asse della nave, la forma dello scafo a ogni
ordinata, una pianta, un elenco numerato dei legni con le loro misure. Misura quello che c'è: non ricostruisce niente e
non sposta una linea dal modello per carenarla.

| Passo | Che cosa fa |
|---|---|
| **Sistema della nave** | x lungo la chiglia da un punto di partenza (0 alla prua o a una stazione scelta), y a babordo, z in alto: da una linea tracciata sul raster o da due marker. L'alto è la verticale del mondo, la linea di chiglia è messa in piano e la sua pendenza resta un numero. Quale estremo è la prua lo dici tu: dal modello non si vede, e decide quale lato è babordo. |
| **Stazioni** | regolari, da una tabella `nome, x` con i nomi del cantiere, o dalle ordinate del catalogo. |
| **Sezioni** | sezioni trasversali alle stazioni, longitudinali a y date, orizzontali a z date, tagliate dalla mesh in doppia precisione; `<dati del chunk>/shipwreck/sections.json`. |
| **Acqua** | un livello e un indice (1,34): sotto il livello un punto si disegna a `livello - (livello - z) * n`. Spenta di partenza. Prima approssimazione per una vista nadirale: raggi obliqui e superficie in movimento non sono modellati. La linea grezza e quella corretta restano entrambe e ogni disegno dice quale mostra. |
| **Disegni** | piano di costruzione (visto da poppa, babordo a sinistra), piano delle linee d'acqua, profilo, pianta con le stazioni sull'ortofoto: scala automatica o scelta, barra di scala, nord, cartiglio; SVG, PDF, PNG e DXF con un layer per tipo. Le piante non sono mai speculari, in qualunque verso corra x. |
| **Legni** | chiglia, paramezzale, ordinate, fasciame, fasciame interno, serrette, elementi di fissaggio: disegnati sul raster o da un CSV; lunghezza lungo il legno sul modello, larghezza, quote sopra un datum, maglia delle ordinate; una tabella, un layer della vista raster, CSV, XLSX e GeoPackage. |

Limiti. Un chunk in gradi (WGS 84) viene rifiutato: un sistema della nave ha bisogno di metri (*GNSS > Sistema e quote* porta il
chunk in un sistema proiettato; l'import da Metashape lo fa da sé). Niente ricostruzione dello scafo, lati specchiati,
superfici lofted, idrostatica, sviluppo del fasciame, riconoscimento automatico dei legni, unione di più giornate. I clic
nella vista 3D non sono offerti direttamente: metti lì due marker e usa *Da due marker*. Le quote sul modello valgono circa
un lato della mesh.

La trasformazione, il modello e i prodotti del chunk non vengono mai cambiati. Tutto si scrive in
`<dati del chunk>/shipwreck/`, salvo quello che esporti.

## Riga di comando

```bash
ppg plugin shipwreck frame PROGETTO.ppg --start X,Y[,Z] --end X,Y[,Z] [--bow start|end] [--z0 H] [--x0 X]
ppg plugin shipwreck frame PROGETTO.ppg --markers --start A --end B
ppg plugin shipwreck frame PROGETTO.ppg --flip
ppg plugin shipwreck stations PROGETTO.ppg --spacing 1 --count 11 --first 1     # oppure --table FILE, --from-catalogue
ppg plugin shipwreck sections PROGETTO.ppg --longitudinals "0;1;-1" --horizontals "-0.2;-0.5" --half-width 4.5
ppg plugin shipwreck water PROGETTO.ppg --level -0.1 --n 1.34 --on
ppg plugin shipwreck drawings PROGETTO.ppg [--scale 20] [--out CARTELLA]
ppg plugin shipwreck timbers PROGETTO.ppg [--table FILE]
ppg plugin shipwreck export PROGETTO.ppg --out legni.xlsx                       # .csv, .xlsx, .gpkg
ppg plugin shipwreck status PROGETTO.ppg
```

Una tabella dei legni ha l'intestazione: `tipo`, `lato`, `nome`, `stazione`, `note` e `x, y` (un punto) oppure `x1, y1, x2, y2`
(una linea) in coordinate della nave, o `wkt` nelle coordinate del chunk.

## Licenza

GPL-3.0-or-later.
