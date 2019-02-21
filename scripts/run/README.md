# Anwendung der Detektoren und Deskriptoren.
## Übersicht
`./scripts/run` enthält die Python Skriptre um die verschiedenen Detektoren und
Deskriptoren zu starten. Der Ordner ist wie folgt aufgebaut:
- config_run_detectors.py: Einstellungen für die Verwendungen von Detektoren.
- config_run_descriptors.py: Einstellungen für die Verwendung von Deskriptoren.
- run_detectors.py: Python Skript um die Detektoren zu starten.
- run_descriptors.py: Python Skript um die Deskriptoren zu staten.

#### Anmerkung:
Damit die Deskriptoren durchlaufen können, müssen zuerst einmal die Detektoren
ausgeführt worden sein.

## Starten der Detektoren
Vom root Verzeichnis dieses Projekts tippe folgendes:

```python
python ./scripts/run/run_detectors.py --root_dir $(pwd)
```

Dies startet alle Detektoren (sift, lift, tilde, superpoint) über alle
Bilderkolletionen (eisert, webcam, example).

Darüber hinaus bietet das Interface auch optionale Paramter. Eine Hilfe dazu
kann aufgerufen werden mit:

```python
python ./scripts/run/run_detectors.py -h
```

Alternativ können die Optionen auch in der File `config_run_detectors.py` betrachtet werden.

Die Keypunkte für die Evaluierung der Detektor-Modell wurden mit folgendem
Befehl erstellt:
```python
 python scripts/run/run_detectors.py --root_dir $(pwd) --max_size 1200 --collection_names example --max_num_keypoints 1000 --detectors sift
 ```

#### Anmerkung:
Der Detektor TILDE wird in einem Docker Container ausgeführt, die Ergebnisse
wieder in den Ausgabeordner des Host-Systems zurückschreibt. Der Docker Container
gilt danach als Besitzer des Ausgabeordners und somit müssen die Rechte für
diesen Ordner neu gesetzt werden. Das Skript macht dies automatisch indem es
den Befehl ausfürht:

```bash
sudo chown -R $USER
```
Dazu muss der Benutzer sein Root-Kennwort einmalig eingeben, andernfalls schlägt
dieser Befehl fehl.

## Start der Deskriptoren
Das Aufrufen der Deskriptoren verhält sich sehr ähnlich zu dem Aufruf für die
Detektoren:

```python
python ./scripts/run/run_descriptors.py --root_dir $(pwd)
```

Dies startet alle standardmäßig eingestellten Deskriptoren (sift, tfeat, doap).
Für jeden Deskriptor werden dann die Keypunkte der eingestellten Detektoren
(standardmäßig sift, tilde, lift, superpoint) geladen und die entsprechenden
Deskriptoren errechnet.

Hilfe zu den Einstellungen findet man in der Datei `config_run_descriptors.py`
oder mittels
```python
python ./scripts/run/run_descriptors.py -h
```

## Ausgabewerte von Destektoren und Deskriptoren
Eine genaue Beschreibung, wie die Ausgabewerte der Detektoren und Deskriptoren
lautet findet sich [hier](../../outputs/README.md).
