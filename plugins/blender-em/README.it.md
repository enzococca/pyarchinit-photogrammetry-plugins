# Blender ed EMtools

*[English](README.md)*

**[Guida d'uso](https://enzococca.github.io/pyarchinit-photogrammetry/it/plugins/blender-em/)**: che cosa deve contenere il progetto, l'em.json, il progetto EM, Blender dal vivo, la sincronizzazione con EMtools.

Plugin di [pyarchinit-photogrammetry](https://github.com/enzococca/pyarchinit-photogrammetry)
(1.3.4 o successiva): i **proxy 3D delle unità stratigrafiche** e il **rilievo fotogrammetrico** in
[Blender](https://www.blender.org) e nell'Extended Matrix di
[EMtools](https://github.com/zalmoxes-laran/EM-blender-tools), aggiornati mentre il lavoro va avanti,
come il collegamento con QGIS tiene aggiornati i disegni.

| | Che cosa | Come |
|---|---|---|
| **La matrice** | l'em.json del sito, scritto da PyArchInit (*Extended Matrix → Esporta sito in em.json…*, o *Esporta progetto EM per EMStudio…* con tutto l'albero `EM/`), EMStudio o EMtools 1.6 | letto, mai scritto: le unità, le loro epoche e i colori, lo spostamento del progetto |
| **L'abbinamento** | ogni proxy del chunk (*Strumenti > Proxy 3D delle US*) con la sua unità | con il `node_uuid` della scheda US (un database PyArchInit collegato), altrimenti con area, tipo e numero, altrimenti con il nome; le ambiguità sono dette, non indovinate |
| **Il progetto EM** | `EM/proxies/<unità>.glb`, `RB/<chunk>/<chunk>.glb`, `ppg_scene.json` | l'albero standard di s3dgraphy e di EMtools 1.6; ogni proxy con il nome del suo nodo (`1.US12`), il nome con cui EMtools abbina un oggetto proxy |
| **Blender, dal vivo** | Blender aperto sul progetto EM con uno script del plugin | gli oggetti del manifest importati e seguiti: un proxy ricostruito è sostituito entro un secondo, tenendo il suo oggetto e il materiale che EMtools gli ha dato |
| **EMtools, dal vivo** | la selezione condivisa con EMtools 1.6 | il programma si collega al WebSocket della sincronizzazione di EMtools (porta 8788, wire 2, come EMStudio): un proxy scelto in Blender seleziona la sua unità nel pannello, un'unità scelta nel pannello seleziona e inquadra il suo proxy in Blender |

## Il progetto EM

```
<progetto>/EM/<sito>.em.json        la matrice (di PyArchInit, di EMStudio)
<progetto>/EM/proxies/<unità>.glb   un proxy per unità, con il nome del suo nodo
<progetto>/RB/<chunk>/<chunk>.glb   il rilievo (reality-based), con i suoi colori o la texture
<progetto>/ppg_scene.json           il manifest che lo script in Blender segue
```

Le cartelle vengono dall'em.json (il suo progetto, o accanto a lui come fa EMtools), oppure da una
cartella scelta quando non c'è un em.json collegato; allora i proxy tengono i nomi del programma
(`US12`, `USM3`). glTF ha la Y in alto e Blender lo riporta con la Z in alto. Le coordinate sono
**spostate di un numero intero di metri**, perché un float32 tiene un quarto di metro a quattromila
chilometri: dello spostamento dell'em.json (il suo `geo_position`) quando c'è, altrimenti dell'angolo
più basso di ciò che si scrive, arrotondato per difetto. Una volta scelto lo spostamento resta, così
un aggiornamento non sposta mai ciò che Blender mostra; va nella scena (`ppg_shift`, e i
`BL_x_shift`... che EMtools legge da 3D Survey Collection, quando non sono impostati).

Un file è riscritto solo quando cambia ciò da cui viene, e **mai sopra un file che un altro
programma ha scritto o cambiato** nella stessa cartella (EMtools vi esporta i proxy modellati in
Blender): lo si dice e lo si lascia com'è. Il file di un proxy tolto dal chunk è cancellato solo se è
ancora quello scritto qui.

## Blender

*Apri in Blender* scrive il progetto EM e avvia Blender (il `.blend` più recente del progetto quando
c'è, altrimenti una scena nuova) con `blender_live.py`, che:

* importa i proxy nella collezione `PPG Proxy` e il rilievo in `PPG Rilievo`, con unità, area, tipo,
  epoca, id del nodo e volume come proprietà di ogni oggetto;
* rilegge il manifest ogni secondo: di un proxy ricostruito cambia solo la mesh (il suo oggetto, il
  materiale di EMtools, il suo posto negli elenchi di EMtools restano), un proxy tolto è rimosso;
* **non sovrascrive mai ciò che è stato cambiato in Blender**: un oggetto i cui vertici non sono quelli
  importati è *staccato*, e il pannello (*PPG* nella barra laterale della vista 3D) lo elenca;
* carica l'em.json in EMtools 1.6 (*Carica l'em.json in EMtools*, e da solo in una scena nuova), che
  abbina i proxy alle unità per nome.

Con *Aggiornamento dal vivo* il programma riscrive il progetto EM, in background, ogni volta che
cambiano i proxy, il modello o l'em.json. Blender è cercato dove è installato, o dove dicono *Scegli il
programma Blender...* o `PPG_BLENDER`.

## EMtools dal vivo

EMtools 1.6 ospita la sincronizzazione dell'Extended Matrix: avviala in Blender (pannello EM, *Sync*),
poi spunta *Sincronizza con EMtools* nel pannello. Il programma si collega a `127.0.0.1:8788` (un'altra
porta nelle impostazioni del plugin, `port`), dice quale matrice ha aperto (`client_info`), chiede il
grafo di EMtools (`snapshot`: senza un em.json collegato il pannello e i nomi dei proxy seguono
quello) e scambia la selezione (`select`). Non manda mai modifiche al grafo né comandi: la matrice è di
EMtools e di EMStudio. Si ricollega da solo quando Blender riparte. EMtools 1.5 non ha la
sincronizzazione: i file e lo script dal vivo funzionano lo stesso.

## Uso

* *Plugin > Blender ed EMtools*: il pannello, *Collega l'em.json del sito...*, *Esporta nel progetto
  EM*, *Apri in Blender (aggiornato dal vivo)*, *Scegli il programma Blender...*; la voce *Progetto EM
  (Blender)* sotto il chunk nell'area di lavoro.
* Riga di comando:

  ```bash
  ppg plugin blender-em status scavo.ppg --emjson Scavo/EM/Scavo.em.json
  ppg plugin blender-em export scavo.ppg --emjson Scavo/EM/Scavo.em.json
  ppg plugin blender-em export scavo.ppg --out Scavo --no-survey
  ppg plugin blender-em open scavo.ppg            # l'em.json collegato al chunk
  ppg plugin blender-em blender                   # i Blender trovati
  ```

Niente oltre ciò che il programma porta con sé: l'em.json è letto come JSON, il WebSocket parlato con
la libreria standard, lo script dal vivo gira nel Python di Blender.

## Test

```bash
pip install -e .
pytest -q tests
PPG_TEST_BLENDER=/percorso/di/blender pytest -q tests/test_blender_live.py   # o un Python con bpy
```

Licenza: GPL-2.0-or-later, come il programma in cui il plugin gira. Autore: Enzo Cocca.
