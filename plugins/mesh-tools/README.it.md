# Mesh (MeshLab)

*[English](README.md)*

Plugin di [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry)
(1.3.3 o successiva): i filtri di [MeshLab](https://www.meshlab.net) sul modello del chunk, con
[pymeshlab](https://github.com/cnr-isti-vclab/PyMeshLab) (Visual Computing Lab, ISTI-CNR). Ogni
risultato è una **nuova versione del modello**: quella di prima resta tra le versioni del chunk
(*Versioni* nell'area di lavoro), per confrontarle o per tornare indietro.

| Operazione | Cosa fa MeshLab | La texture |
|---|---|---|
| **Decima** | Collasso degli spigoli con le quadriche fino a un numero o a una frazione delle facce; bordi e topologia conservati in modo predefinito; semplificazione planare a richiesta, soglia di qualità dei triangoli | resta: la variante di MeshLab che porta con sé le coordinate della texture degli angoli delle facce (anche con più pagine di texture) |
| **Rimaglia** | Rimaglia isotropa esplicita: triangoli regolari di un solo lato (in modo predefinito il lato medio del modello), spigoli vivi conservati oltre un angolo, adattiva a richiesta, una distanza massima dalla superficie di partenza; topologia riparata prima e dopo (la rimaglia riunisce i vertici che una riparazione aveva diviso) | da ricostruire |
| **Leviga** | Taubin (λ 0,5, μ -0,53, 10 passi: non restringe) o HC laplaciana (1 passo); i bordi restano fermi (a richiesta si muovono anche loro) | resta |
| **Chiudi i buchi** | Buchi fino a un numero di lati del bordo chiusi con nuove facce (senza autointersezioni, raffinamento a un lato a richiesta); il contorno esterno di ogni parte separata - il suo bordo più lungo - resta aperto se non chiedi altrimenti (il fondo di un oggetto): MeshLab da solo coprirebbe con un tappo un'intera superficie di scavo | da ricostruire, se dei buchi sono stati chiusi |
| **Ripara la topologia** | Vertici e facce doppi, facce senza area, spigoli non-manifold (facce tolte o vertici divisi), vertici non-manifold (divisi), vertici non usati | resta |
| **Misura** | Area della superficie, volume di un modello chiuso (watertight), 2-manifold o no, parti separate, anelli di bordo, spigoli di bordo e non-manifold, vertici non-manifold e non usati, dimensioni, lato medio | - |

Lunghezze e misure sono in metri del rilievo (in unità del modello se il chunk non ha una scala).
Prima che MeshLab veda il modello, i vertici che un modello con texture ripete lungo le cuciture
della texture vengono contati una volta sola: altrimenti MeshLab prende ogni cucitura per un bordo,
la levigatura la apre e le misure la contano come un buco (le copie che si toccano solo in un
punto, come un vertice non-manifold diviso dalla riparazione, restano separate). Il resoconto di
un'operazione dice i vertici e gli spigoli non-manifold che MeshLab ha lasciato, se ce ne sono. Le
proprietà che MeshLab non può portare (la confidenza dei vertici, qualsiasi altra) tornano dal
vertice del modello più vicino a ogni nuovo vertice; i colori sono quelli di MeshLab, interpolati
sui vertici nuovi. Se MeshLab non cambia niente (nessun buco così piccolo, niente da riparare) il
modello resta com'è e non si aggiunge nessuna versione. La nuova versione prende il nome da ciò che
ha fatto MeshLab (`MeshLab: decimazione a 200.000 facce (texture conservata)`) o quello che le dai;
il progetto viene salvato con essa, come fa il programma dopo le sue elaborazioni.

Quello che il programma fa già resta del programma: la mesh dalla nuvola densa o dalle mappe di
profondità (Poisson), la pulizia del modello per confidenza, parti staccate e triangoli stirati, il
taglio con i piani, la texture.

## Dove

* il pannello **Mesh (MeshLab)** del dock del progetto: il modello attivo, le operazioni (i loro
  parametri in una finestra), le misure, *Installa MeshLab...*; il risultato si vede nella vista 3D;
* *Plugin > Mesh (MeshLab)* nel menu, le stesse operazioni;
* la riga di comando:

```bash
ppg plugin mesh-tools status
ppg plugin mesh-tools install                       # l'ambiente del plugin (uv)
ppg plugin mesh-tools decimate scavo.ppg --faces 200000
ppg plugin mesh-tools decimate scavo.ppg --fraction 0.25 --planar --name "Per il web"
ppg plugin mesh-tools remesh scavo.ppg --edge 0.01
ppg plugin mesh-tools smooth scavo.ppg --method hc --steps 2
ppg plugin mesh-tools close-holes scavo.ppg --max-edges 100
ppg plugin mesh-tools close-holes vaso.ppg --max-edges 5000 --close-outline   # anche il fondo di un oggetto
ppg plugin mesh-tools repair scavo.ppg --split
ppg plugin mesh-tools measure scavo.ppg --json
```

`--chunk ETICHETTA` lavora su un chunk diverso da quello attivo, `--overwrite` non conserva il
modello di prima come versione, `--json` stampa tutto il resoconto.

## Come gira MeshLab

pymeshlab porta con sé le sue librerie native (anche Qt), e alcuni filtri si chiudono di colpo su
un modello che non accettano (la rimaglia di un modello non-manifold è un segmentation fault): il
programma non lo importa mai. Ogni operazione gira in un processo a sé (`ppg_mesh_tools/worker.py`):
il plugin gli scrive il modello in una cartella di lavoro accanto al modello, il worker applica i
filtri di MeshLab e scrive il risultato, e i due si parlano con righe JSON (avanzamento, risultato,
errore); annullare lo ferma subito, un crash lascia il modello com'era e lo dice. Il worker gira con:

* il Python del programma, se pymeshlab è installato lì (`pip install pymeshlab`);
* altrimenti un ambiente che il plugin installa su richiesta con
  [uv](https://github.com/astral-sh/uv) - Python 3.12 e pymeshlab da PyPI, circa 300 MB, nella
  cartella del plugin (`PPG_MESH_TOOLS_ENV` per metterlo altrove): è così che funziona
  l'applicazione impacchettata, che non ha pip. Il pannello lo propone la prima volta.

pymeshlab ha versioni per macOS (Apple Silicon e Intel), Linux (x86_64 e aarch64) e Windows x64;
non per Windows su ARM, dove il plugin dice che non è disponibile.

Su Linux pymeshlab ha bisogno di librerie di sistema che le sue wheel non portano con sé,
soprattutto `libOpenGL.so.0`: senza, si importa ma non legge nessun file. Installale una volta:

```bash
sudo apt install libopengl0 libgl1                 # Debian, Ubuntu
sudo dnf install libglvnd-opengl libglvnd-glx      # Fedora
```

Prima della prima operazione di una sessione il plugin prova MeshLab su un triangolo (`ppg plugin
mesh-tools status` fa lo stesso) e, se manca qualcosa, dice quale libreria e quale pacchetto la
contiene.

## Installazione

* Nel programma: *Strumenti > Plugin...*, `mesh-tools`, *Installa / aggiorna*, *Riavvia ora*.
* Con pip, nell'ambiente Python del programma:

  ```bash
  pip install "ppg-mesh-tools @ git+https://github.com/enzococca/pyarchinit-photogrammetry-plugins#subdirectory=plugins/mesh-tools"
  pip install pymeshlab        # facoltativo: altrimenti il plugin installa un ambiente suo
  ```

Test: `pip install -e . pytest` nell'ambiente del programma, poi `pytest tests`; i test di MeshLab
vero e proprio (`tests/test_meshlab.py`) girano se pymeshlab è installato per quel Python, gli
altri usano un worker che parla lo stesso protocollo senza di esso.

Licenza: **GPL-3.0-or-later** (pymeshlab è GPL-3; il programma e gli altri plugin di questo
repository sono GPL-2.0-or-later, che lo permette). Autore: Enzo Cocca.
