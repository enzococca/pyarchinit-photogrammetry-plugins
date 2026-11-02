# Elementi architettonici

Un plugin di [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry): elementi
architettonici parametrici ricavati dal modello fotogrammetrico.

* **Colonna per rivoluzione**: si disegna la zona in pianta; asse e profilo si misurano sul modello (il raggio mediano
  per quota e la sua dispersione), il solido è una spline per K nodi adattata al profilo, o il profilo come misurato.
  Base, fusto e capitello possono essere elementi separati sullo stesso asse.
* **Modanatura, cornice, cornice marcapiano**: si traccia il percorso; la sezione si taglia dal modello con il piano
  normale al percorso e si chiude contro il muro; il solido è la sezione portata lungo il percorso con gli angoli a
  quartabuono.
* **Arco e volta a botte**: su una sezione del progetto o su un piano verticale per due punti; l'intradosso si adatta a
  un cerchio (a tutto sesto o ribassato), a un arco a sesto acuto, a un'ellisse o resta come misurato, il tipo scelto
  dallo scarto.

Ogni elemento si scrive come STEP (CadQuery, OpenCascade) e come solido IFC parametrico, e porta sempre il suo
**scarto**: la mediana, il 90° percentile e il massimo della distanza dei punti del modello dal solido. Il solido è
un'interpretazione del modello misurato; lo scarto dice quanto se ne discosta.

CadQuery gira in un ambiente a parte che il plugin installa su richiesta (circa 900 MB; non su Windows ARM), mai nel
processo del programma. Riga di comando: `ppg plugin architecture column|moulding|arch|list|show|export|remove|install|status`.

Licenza: GPL-2.0-or-later. CadQuery e OCP sono Apache-2.0, OpenCascade LGPL-2.1 con eccezione, VTK BSD.
