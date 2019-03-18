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
  - **scripts**: Enthält alles Skripte für die Anwendung der Module und ihrer Auswertung.
    - desc_doap: Setup Dateien für DOAP.
    - desc_tfeat: Setup Dateien für TFeat.
    - det_tilde: Setup Dateien für TILDE.
    - eval: Konfiguration und Startskripte für die Evaluation von Detektoren und Deskriptoren.
    - misc: Hilfs- und Genereal-Purpose Skripte.
    - pipe_lift: Setup Dateien für LIFT.
    - pipe_sift: Setup Dateien für SIFT.
    - pipe_superpoint: Setup Dateien für SuperPoint.
    - run: Konfiguration und Startskripte für Detektoren und Deskriptoren.
  - **evaluation**: Modul zur Evaluierung der Detektor und Deskriptoren.
    - detectors
    - descriptors
  - **examples**: Enthält Beispielcode für die einzelnen Detektoren und Deskriptoren.
  - **extern**: Enthält 3rd-party code der für das Setup des Projektes benötigt wird.
  - **desc_doap**: Modul für DOAP Deskriptor.
  - **desc_tfeat**: Modul für TFeat Deskriptor.
  - **det_tilde**: Modul für TILDE Detektor.
  - **pipe_lift**: Modul für LIFT Detektor und Deskriptor
  - **pipe_sift**: Modul für SIFT Detektor und Deskriptor
  - **pipe_superpoint**: Modul für SuperPoint Detektor und Deskriptor
  - **outputs**: Standard Ausgabeordner für Detektoren und Deskriptoren.
  - **output_evaluation**: Standard Ausgabeordner für Pickle Dateine bei der Evaluierung der Detetkoren und Deskriptoren.
    - detectors
    - descriptors

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
| LIFT              | 1200          |
| SuperPoint        | 1700          |
| TILDE             | no Limit      |
| TConvDet          | no Limit      | (reduces image to 1024x768 internally)
| DOAP              | no Limit      |
| TFeat             | 1400          |


### Patchgrößen für Deskriptoren
Dies sind die Größen, die die Patches haben müssen, um mit dem jeweiligen
Deskriptor zu funktionieren.

| MODEL             | Patch Size (px) |
|:------------------|:----------------|
| SIFT              | no Limit        |
| LIFT              | 128x128         |
| SuperPoint        | ?               |
| DOAP              | 32x32           |
| DOAP ST           | 42x42           | (in diesem Repo)
| TFeat             | 32x32           |



### Starten der Detektoren und Deskriptoren
Siehe dazu [hier](./scripts/run/README.md)

### Ausgabe der Detektoren und Deskriptoren
Siehe dazu [hier](./outputs/README.md)

### Starten der Detektor Evalueriung
Siehe dazu [hier](./scripts/eval/README.md)

### Ausgabe der Detetkor Evalueriung
Siehe dazu [hier](./evaluation/detectors/README.md)

### Informationen zu den Bilderkollektionen
Siehe dazu [hier](./data/README.md)

