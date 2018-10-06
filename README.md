# Git Repo für die Diplomarbeit "Evaluierung erlernter Bildmerkmals-Detektoren und Deskriptoren in ausgewählten Deep Learning Netzwerken"

## Voraussetzungen
- docker
- nvidia-docker
- python 3.6
- pip >= 18.0

## Projektstruktur
- root
  - Diese **README**
  - **data**: Enthält Bildersets für die Auswertung.
  - **scripts**: Enthält alles Skripte, die benutzt und benötigt werden.
  - **docker**: Enhält Dockerfiles die zum Bauen der Umgebungen für einzelnen Detektoren und Deskriptoren.
  - **config**: Eventuelle Konfigurationsdateien befinden sich hier drin.

## Installation
Tippe aus dem Projekt-Root-Verzeichnis (in dem sich diese Readme befindet) in das Terminal:
    $ ./scripts/setup_project.sh
