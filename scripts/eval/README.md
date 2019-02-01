# Evaluierung von Keypunkt Detektoren
Ziel ist die Evaluierung von der Sensibilität und Wiederholbarkeit der Keypunkt Detektoren.
Voraussetzung ist, dass die zu evaluierenden Detektoren bereits über Bildersets gelaufen sind und somit passende Keypunkte Dateien im `outputs` Verzeichnis gibt.

### Generell
Für jeden Detektor (sift, lift, tilde, superpoint) werden die Ergebnisse in einer Pickle-Datei (sift.pkl, lift.pkl, ...) abgespeichert. Weleche Werte gespeichert werden, hängt von den aktivierten Parameter ab. Die Pickle-Dateien landen standardmäßig im `output_evaluation` Ordner, der sich im Root Verzeichnis dieses Projektes befindet.

### Evalueriungsziel
Für die Evaluierung wird die Bilderkollektion `Webcam` gewählt, die die Sets ``
Zwei Metriken sollen hier evaluiert werden
- 1) Die Sensibilität der Detektoren
- 2) Die Wiederholbarkeit der gefundenen Keypunkte

Die Sensibilität wird durch die Anzahl der gefundenen Keypunkte gemessen.

### Aufbau der Pickle Datei
Die Ergebnisse der Evaluierung landen in einer Pickle Datei. Wie diese aufgebaugt ist erfährt man [hier](../../evaluation/detectors/README.md)

Das Evaluierungsskript sollte von vom Root Verzeichnis aus gestartet werden mittels

```python
python ./sripts/eval/eval_detectors.py [PARAMETER]
```

Hierbei ist wichtig, dass der Parameter `--root_dir` gesetzt ist und auf das Root Verzeichnis dieses Projektes zeigt.

### Hilfe

Eine Auflistung der Parameter erhält man durch:

```python
python ./sripts/eval/eval_detectors.py -h
```

Alle Parameter können in der Datei `config_eval_detectors.py` gefunden werden.

### Beispielaufruf
```python
python ./scripts/eval/eval_detectors.py --root_dir $(pwd) --max_size 1200  --collection_names webcam
```
