# Git Repo für die Diplomarbeit "Evaluierung erlernter Bildmerkmals-Detektoren und Deskriptoren in ausgewählten Deep Learning Netzwerken"

## Voraussetzungen für Linux
- [docker](https://www.docker.com/)
- [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
- python 3.6
- pip >= 18.0
- wget
- tar
- [Matlab](https://de.mathworks.com/products/matlab.html) with Image Processing Toolbox
- imagemagick

## Projektstruktur
- root
  - Diese **README**
  - **data**: Enthält Bildersets für die Auswertung.
  - **scripts**: Enthält alles Skripte, die benutzt und benötigt werden.
  - **desc_doap**: Enhält alle benötigten Files für den DOAP Deskriptor.
  - **det_tilde**: Enhält alle benötigten Files für den TILDE Detektor.
  - **exmaples**: Enthält Beispielcode für die einzelnen Detektoren und Deskriptoren.
  - **extern**: Enthält 3rd-party code der für das Setup des Projektes benötigt wird.

## Installation
Tippe aus dem Projekt-Root-Verzeichnis (in dem sich diese Readme befindet) in das Terminal:

    $ ./scripts/setup_project.sh $(pwd)

## Probleme
### SuperPoint
> cuda runtime error (30) : unknown error at /pytorch/aten/src/THC/THCGeneral.cpp:74

Der CUDA Treiber hat aufgehangen. Tippe folgendes in die Konsole:

        $ sudo rmmod nvidia_uvm
        $ sudo rmmod nvidia_drm
        $ sudo rmmod nvidia_modeset
        $ sudo rmmod nvidia
        $ sudo modprobe nvidia
        $ sudo modprobe nvidia_modeset
        $ sudo modprobe nvidia_drm
        $ sudo modprobe nvidia_uvm


### Bildgrößen
Manche Modell können nur Bilder bis zur einer bestimmten Größe handhaben, bevor sie abstürzen. Die folgende Liste fast diese Größen zusammen. Bei der Benutzung der Modell mit größeren Bilder muss daran gedacht werden, die Bilder entsprechend herunter zu skalieren.

| MODEL             | max Size (px) |
|:------------------|:--------------|
| SIFT              | no Limit      |
| LIFT              | 1300          |
| SuperPoint        | 1700          |
| TILDE             | no Limit      |
| TConvDet          | ?             |
| DOAP              | no Limit      |
| TFeat             | 1400          |

### Bearbeitungsdauer
Hier sieht man die Zeitdauer, die ein Model benötigt, um X Bilder zu bearbeiten.
Die Spalte `#Images` gibt dabei die Anzahl der bearbeiteten Bilder an.
Der Wert `native` in der Spalte `Size` bedeutet, dass die Bilder nicht skaliert
wurden. Standardmäßig werden die Modell für jedes einzelne Bild aufgerufen.
Im Falle von DOAP muss jedoch berücksichtigt werden, dass für jedes Bild Matlab
neu gestartet wird, was auf jeden Fall negativ die Gesamtdauer beeinflusst, da
Matlab einen recht langen Startvorgang besitzt.
| MODEL             | Size          | #Images   | Time      |
|:------------------|:--------------|:----------|:----------|
| SIFT              | native        | 32        | 00:06:42  |
| LIFT              | 1300          |           |           |
| SuperPoint        | 1700          |           |           |
| TILDE             | native        | 32        | 00:22:56  |
| TConvDet          | ?             |           |           |
| DOAP              | native        | 32        | 5:21:44   |
| TFeat             | 1400          |           |           |


### Starten der Netwzwerke:
```python
python ./scripts/eval/run_models.py --root_dir $(pwd)
```

#### Parameter
Siehe die File `./scripts/eval/config_eval.py` für alle Paramter oder tippe

    $ python ./scripts/eval/config_eval.py -h

- --root_dir ROOT: Der absolute Pfad zu diesem Repo-Ordner, sodass alle anderen Netzwerke Unterordner von ROOT sind.

#### Beispiel:
```python
python ./scripts/eval/run_models.py --root_dir $(pwd)
```

- --size SIZE: Maximale Dimensionsgröße für Bilder. Die Bilder werden dann so skaliert, dass ihre größte Dimension (Höhe, Breite) diesem Wert entspricht.

#### Beispiel:
```python
python ./scripts/eval/run_models.py --root_dir $(pwd) --size 800
```

- --networks NETWORK [NETWORK, ...]: Standardmäßig werden alle Netzwerke gestartet. Sollen nur bestimmte Netzwerke gestartet werden, gibt man hier ihre Namen hintereinander an. Folgende Werte sind erlaubt: sift, superpoint, tfeat, doap, lift und tilde.

#### Beispiel: Startet nur das TILDE Netzwerk
```python
python ./scripts/eval/run_models.py --root_dir $(pwd) --networks tilde
```

- --max_num_images NUM: Die maximale Anzahl an Bildern, die verarbeitet werden soll. Wenn der Wert nicht gesetzt ist, werden alle Bilder verarbeitetn, ansonsten nur die ersten NUM Bilder.

#### Beispiel: Verarbeite nur die ersten 2 Bilder
```python
python ./scripts/eval/run_models.py --root_dir $(pwd) --max_num_images 2
```

- --skip_first_n N: Überspringt die ersten N Bilder, die es in im data_dir findet.

#### Beispiel: Überspringe die ersten 5 Bilder
```python
python ./scripts/eval/run_models.py --root_dir $(pwd) --skip_first_n 5
```

