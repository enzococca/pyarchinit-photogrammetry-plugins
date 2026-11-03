<!-- Sezioni da aggiungere alla guida del programma docs/plugins/shipwreck.it.md per la versione 0.2.0 del plugin
(serve pyarchinit-photogrammetry 1.7.0 o successiva). Posto suggerito: dopo "I legni", prima di "Riga di comando"; le
righe di comando vanno in "Riga di comando". -->

## Fasce a clinker

Il passo 7 del pannello, **Fasce (clinker)**, legge il fasciame di uno scafo a clinker sulle sezioni: dove ogni fascia
si sovrappone a quella sotto, e da lì le fasce. Servono il sistema della nave e le sezioni alle stazioni (passi da 1 a
3); una stazione ogni metro o meno è un buon inizio.

| Pulsante | Che cosa fa |
|---|---|
| *Rileva i corsi...* | I gradini sulla sezione di ogni stazione, le loro linee lungo lo scafo, il layer "Fasce" nella vista raster e nei disegni, `lands.dxf` e `lands.gpkg`. |
| *Modello delle fasce...* | Una fascia fra due linee di gradini: larghezze, lunghezza, spessore e sovrapposizione, una mesh per fascia, le tabelle. |
| *Nel catalogo* | Una scheda `plank` del catalogo dei legni per fascia, con il suo contorno in pianta e la nota *modello delle fasce (ipotesi)*. |
| *Per Clinker Boat Tool...* | I file per lo strumento Houdini di Fröhlich e Nielsen, in una cartella che scegli. |
| *Fasce da Houdini...* | Le tavole che lo strumento esporta, di nuovo nel catalogo con la loro distanza dalla scansione. |
| *Apri la cartella* | `<dati del chunk>/shipwreck/clinker`. |

### Come si trovano i gradini

A ogni stazione il plugin prende la sezione **grezza** (mai quella corretta per l'acqua) e la divide alla chiglia:
babordo (y > 0) e tribordo. Ogni lato si legge dalla battura, la cima del fianco della chiglia, fino all'orlo, e finisce
dove il profilo gira attorno all'orlo o piega nel sedimento, in una trincea o nel fondale. Una media mobile su 12 cm del
profilo è lo scafo "senza tavole"; lo scostamento del profilo grezzo da essa, su uno scafo a clinker, è un dente di sega
che sale piano dentro una fascia e scende del gradino a ogni sovrapposizione. Un gradino è dove quello scostamento cambia
di più del **gradino minimo** entro 3 cm; due gradini distano almeno la **larghezza minima di fascia**. Ogni gradino ha la
sua posizione lungo la sezione dalla battura e il suo salto in millimetri.

Fra le stazioni, il numero di gradini trovato più spesso è quello dello scafo su quel lato. Una stazione con un altro
numero (una fascia che si interrompe, un orlo eroso, un danno) è **incerta**, e così una stazione la cui battura o il cui
torello stanno a più di 2 cm dalla curva per le stazioni vicine. I gradini di una stazione incerta si assegnano alle
linee per la loro posizione lungo la sezione; un gradino fuori dalle linee viene elencato. Niente viene carenato o
riempito: un lato con un buco, un pezzo staccato o una chiglia illeggibile non dà gradini e viene elencato con il motivo.

| Parametro | Predefinito | Quando cambiarlo |
|---|---|---|
| Gradino minimo (mm) | 8 | Più basso per un fasciame sottile o consumato, più alto se il rumore del modello dà gradini falsi. |
| Larghezza minima di fascia (m) | 0,08 | Più bassa per fasce molto strette. |
| Metà larghezza della chiglia (m) | 0,10 | Fin dove dal centro della chiglia può arrivare il suo fianco. |
| Larghezza minima della suola (m) | 0,02 | Abbassala se una chiglia stretta e consumata lascia lati non letti a ogni stazione. |

Il pannello dice quante fasce ha ogni lato, quali stazioni sono incerte e perché, i gradini fuori dalle linee e i lati
non letti. Se le sezioni vengono tagliate di nuovo dopo il rilevamento, il pannello dice di rilevare di nuovo i gradini e
le linee nella vista raster si disegnano tratteggiate.

### Il modello delle fasce: un'ipotesi

I gradini sono misurati. Le linee che li uniscono, e le fasce fra di esse, sono **un'ipotesi dalle sezioni, non una
misura**: ogni file lo dice (*modello delle fasce: ipotesi dalle sezioni, non una misura*) e il modello non si mescola mai
con le sezioni.

Su ogni lato la fascia 1 è il torello, fra la battura e la prima linea di gradini; l'ultima fascia arriva all'orlo. Per
fascia: la larghezza a ogni stazione (la lunghezza lungo la sezione fra le sue due linee) con un'incertezza in
millimetri, niente dove la stazione è incerta o non letta; la lunghezza lungo la linea superiore; lo spessore (il gradino
mediano dove si sovrappone a quella sotto, o il valore che dai); la sovrapposizione che dai (20 mm di partenza). Le fasce si
chiamano `B1, B2, ...` a babordo e `T1, T2, ...` a tribordo. File in `<dati del chunk>/shipwreck/clinker/`: `strakes.obj`
(coordinate della nave, un oggetto per fascia), `strakes.ply` (coordinate del chunk, doppia precisione), `strakes.csv` e
`strakes.xlsx`.

Il modello viene rifiutato quando un lato con gradini ha meno di tre stazioni con una larghezza: sarebbe tutto
interpolato. Quando i gradini si rilevano di nuovo, i file del modello precedente (e le loro copie nell'ultima cartella
esportata per il Clinker Boat Tool) prendono il suffisso `.old` e il pannello li elenca.

### Il Clinker Boat Tool

Il Clinker Boat Tool di Fred Fröhlich e Sigurd Solhaug Nielsen è un Houdini Digital Asset che costruisce una barca a
clinker tavola per tavola da una scansione. *Per Clinker Boat Tool...* scrive `clinker_boat_tool/` nella cartella che
scegli, nel sistema della nave girato con Y in alto (l'asse verticale di Houdini) e la chiglia lungo X: `hull.obj`,
`sections.obj`, `stations.json`, `frame.json`, `lands.obj` e `strakes.obj` con le tabelle delle fasce quando ci sono, e un
`README.txt`. In Houdini, con i sei passi dello strumento:

1. **Importa la scansione**: `hull.obj`; `sections.obj`, `lands.obj` e `strakes.obj` come riferimenti.
2. **Calibra e allinea**: già fatto, in metri con la chiglia lungo X e Y in alto.
3. **Definisci le sezioni**: le x delle stazioni sono in `stations.json`; `sections.obj` mostra lì i tagli.
4. **Genera la disposizione delle tavole**: larghezze, spessore e sovrapposizione per fascia da `lands.obj` e
   `strakes.xlsx`.
5. **Piega e allinea** le tavole sulla scansione.
6. **Preset ed esportazione**: esporta le tavole in OBJ, un oggetto o un gruppo per tavola.

*Fasce da Houdini...* rilegge quell'OBJ: ogni tavola diventa una scheda `plank` del catalogo (il contorno in pianta, il
lato dalla y media o da B/T nel nome), con la distanza dei suoi vertici dalla scansione, mediana e 90° percentile in
millimetri, in `<file>_scostamenti.csv` e `.xlsx`. Importare di nuovo lo stesso file ne sostituisce le tavole, mai le tue.
Un gruppo che contiene tutti gli altri (come *all*) non è una tavola e viene lasciato fuori, che contenga o no anche le
facce fuori da ogni gruppo.

Il plugin scrive i file che i passi dello strumento leggono e rilegge le tavole che esporta: niente dello strumento è
copiato o usato. Da citare:

- Fröhlich, F., Nielsen, S. S. *From Photogrammetry to Construction Knowledge: 3D Documentation and Parametric
  Reconstruction of Historic Clinker Boats*. CC BY 4.0. [10.5281/zenodo.23422398](https://doi.org/10.5281/zenodo.23422398)
- Fröhlich, F., Nielsen, S. S. *Clinker Boat Tool* (Houdini Digital Asset). Licenza MIT.
  [10.5281/zenodo.17488549](https://doi.org/10.5281/zenodo.17488549)

### Limiti delle fasce

- Su sezioni di una mesh con un rumore oltre circa 0,5 mm le larghezze valgono a circa un centimetro. A 1 mm di rumore
  circa un lato su quindici può avere un numero di gradini sbagliato; viene segnalato come stazione incerta.
- Un salto di 2-2,5 cm della battura a una stazione d'estremità non viene preso. Tre o più stazioni vicine spostate
  insieme, o un gradino graduale, non vengono segnalati.
- Un gradino permanente della chiglia (uno scafo rotto o spostato, una palella) segnala ogni stazione da un lato. Oltre
  circa 3 mm di dispersione sulla battura una stazione può essere segnalata a torto.
- Senza suola, una V di fasciame più ripida di 80° entro la metà larghezza della chiglia è letta come chiglia finché non
  si apre. Un fianco della chiglia allineato al torello è un lato non letto (`keel_unclear`).
- Uno scafo visto dall'interno (il fasciame sotto le ordinate, come in molti relitti scavati) mostra la faccia interna:
  la chiglia è nascosta dal paramezzale e la lettura per stazione dà stazioni incerte. La prossima versione traccerà i
  gradini sul rilievo dello scafo.

<!-- Righe da aggiungere a "Riga di comando": -->

```bash
ppg plugin shipwreck clinker detect PROGETTO.ppg [--min-step MM] [--min-strake M]   # stampa le stazioni incerte
ppg plugin shipwreck clinker strakes PROGETTO.ppg [--lap MM] [--thickness MM] [--catalogue]
ppg plugin shipwreck clinker export PROGETTO.ppg --out CARTELLA
ppg plugin shipwreck clinker import PROGETTO.ppg --obj TAVOLE.obj [--side-from y|name] [--no-deviation]
ppg plugin shipwreck clinker status PROGETTO.ppg                                    # dice se i gradini sono superati
```
