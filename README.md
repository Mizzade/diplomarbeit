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
  - **extern**: Enthält 3rd-party code der für das Setup des Projektes benötigt wird.
  - **desc_doap**: Modul für DOAP Deskriptor.
  - **desc_tfeat**: Modul für TFeat Deskriptor.
  - **det_tilde**: Modul für TILDE Detektor.
  - **pipe_lift**: Modul für LIFT Detektor und Deskriptor
  - **pipe_sift**: Modul für SIFT Detektor und Deskriptor
  - **pipe_superpoint**: Modul für SuperPoint Detektor und Deskriptor
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

