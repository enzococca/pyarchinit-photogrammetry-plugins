# Elementi architettonici

Un plugin di [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry) (1.3.4 o successivo):
elementi architettonici parametrici ricavati dal modello fotogrammetrico, per un modello HBIM.

* **Colonna per rivoluzione**: si disegna la zona in pianta (e due quote); asse e profilo si misurano sul modello, il
  solido è una spline per K nodi adattata al profilo misurato, o il profilo come misurato. Base, fusto e capitello
  possono essere elementi separati sullo stesso asse.
* **Modanatura, cornice, cornice marcapiano**: si traccia il percorso sulla pianta, su un prospetto o una sezione del
  progetto, o fra due marker; la sezione si taglia dal modello con il piano normale al percorso e si chiude contro il
  muro; il solido è la sezione portata lungo il percorso con gli angoli a quartabuono (un'estrusione se il percorso è
  dritto).
* **Arco e volta a botte**: su una sezione del progetto (il punto nell'apertura si clicca su di essa) o su un piano
  verticale per due punti; l'intradosso si adatta a un cerchio (a tutto sesto o ribassato), a un arco a sesto acuto, a
  un'ellisse o resta come misurato, il tipo scelto dallo scarto; spessore e profondità si misurano o si danno.

Ogni elemento si scrive come **STEP** (AP214 in metri, da CadQuery e OpenCascade), **STL** (il formato non ha unità:
i suoi numeri sono metri), solido **IFC4** parametrico
(`IfcColumn` con `IfcRevolvedAreaSolid`, `IfcMember` con `IfcExtrudedAreaSolid` o `IfcFixedReferenceSweptAreaSolid`,
`IfcBuildingElementProxy` "Arch"/"Vault" con gli archi come `IfcArcIndex`, `Pset_PyArchInit_Element`, la mesh del
rilievo accanto a richiesta, `IfcMapConversion` sull'origine del chunk) e **DXF** del profilo o della sezione. Porta
sempre il suo **scarto**: la mediana, il 90° percentile e il massimo della distanza dei punti del modello dal solido.
Il solido è un'interpretazione del modello misurato; lo scarto dice quanto se ne discosta.

Le coordinate dei file sono quelle del sistema del chunk meno un'origine tonda, una per chunk: uno STEP esportato ha
accanto un `<nome>.origin.json` con quell'origine. Se il chunk viene georeferenziato o portato in un altro sistema dopo
aver fatto gli elementi, il plugin rifiuta di esportarli o modificarli finché *Ricostruisci* non li rifà nel sistema
nuovo.

## La prova

Su due scene sintetiche fotografate, allineate, densificate e trasformate in mesh dal programma (script
`proof/make_proof.py`): una colonna alta 2 m con entasi è uscita con l'asse entro 0,001 gradi e 0,1 mm, i nodi entro
1,6 mm, il volume +0,5 %; un arco a sesto acuto (luce 1,6 m) è stato riconosciuto come tale, luce -4 mm, freccia +4 mm,
spessore della ghiera +4 mm, dove il modello del programma dista lui stesso 2 mm (mediana) dalla superficie vera. La
guida del programma ha i dettagli.

## L'ambiente

CadQuery gira in un ambiente a parte, mai nel processo del programma. *Installa CadQuery* nel pannello (o
`ppg plugin architecture install`) lo crea con `uv` nella cartella del plugin: Python 3.12, CadQuery, OpenCascade (OCP)
e il VTK a cui OpenCascade è collegato, senza casadi, numba, scipy, matplotlib e trame. **Circa 900 MB**, alcuni minuti;
la prima elaborazione dopo l'installazione richiede circa un minuto. Un Python che ha già CadQuery si usa così com'è
(`pip install cadquery`), e `PPG_ARCHITECTURE_ENV` mette l'ambiente altrove.

OpenCascade ha versioni per macOS (Apple Silicon e Intel), Linux (x86-64 e aarch64) e Windows x64, **non per Windows
ARM**: lì il pannello dice che gli elementi non sono disponibili.

## Riga di comando

```
ppg plugin architecture install | status
ppg plugin architecture column   PROGETTO --polygon "POLYGON((x y, ...))" --zmin 0.05 --zmax 1.95 --knots 12
ppg plugin architecture moulding PROGETTO --path "LINESTRING(x y, x y)" --height 3.35 [--elevation ID] [--markers A,B]
                                 [--window u0,v0,u1,v1]   (di base: trovata dal taglio)
ppg plugin architecture arch     PROGETTO --section ID --pick u,v [--type auto|pointed|...] [--thickness T] [--vault]
ppg plugin architecture rebuild  PROGETTO     (dopo che la trasformazione o il sistema del chunk sono cambiati)
ppg plugin architecture list | show ID | remove ID PROGETTO
ppg plugin architecture export   PROGETTO --out elementi.ifc [--mesh] | --out cartella/ | --out arco.dxf --ids Arco
```

## Licenze

Il plugin: GPL-2.0-or-later. Nel suo ambiente: CadQuery e OCP (Apache-2.0), OpenCascade (LGPL-2.1 con eccezione), VTK
(BSD-3-Clause), ezdxf (MIT), nlopt (MIT/LGPL), numpy (BSD).
