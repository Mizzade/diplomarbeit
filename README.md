# Git Repo für die Diplomarbeit "Evaluierung erlernter Bildmerkmals-Detektoren und Deskriptoren in ausgewählten Deep Learning Netzwerken"

## Voraussetzungen für Linux
- [docker](https://www.docker.com/)
- [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
- python 3.6
- pip >= 18.0
- wget
- tar
- [Matlab](https://de.mathworks.com/products/matlab.html)
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
