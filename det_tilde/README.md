# TILDE
Aufgrund der speziellen Kombination von Treibern und Bibliotheken läuft TILDE nur in einem Docker Container. Die Bauanleitung dieses Containers findet man in der `Dockerfile`.

### Das Docker Image erstellen.
Führe von dem Root Verzeichnis des Repos das Shell-Skript `setup_docker_image_for_tilde.sh` aus:

    $ ./scripts/det_tilde/setup_docker_image_for_tile.she $(pwd)

Dieser Projektordner beinhaltet mehrere Dateien mit dem Namen `use_tilde` aber unterschiedlichen Endungen. Hier wird aufgelistet, was welche Datei leistet:
- use_tilde.sh: Startet den Docker Container und  use_tilde.py aus
- use_tilde.py: Nimmt die Parameter aus use_tilde.sh und konvertiert sie, sodass sie in use_tilde.cpp verwendet werden können. Erstellt Ausgabe Verzeichnisse, falls sie nicht existieren.
- use_tilde.cpp: Führt den TILDE Detektor aus und speichert das Ergebnis als .csv Datei.


### use_tilde.sh
Siehe Datei für Paramter.

### Wie gelange ich in den Docker Container?
    docker run -it --rm <IMAGE_ID> /bin/bash

### Wie kompiliere ich `use_tile.cpp` im Docker Container?
Wenn man in den Docker Container geht landet man in dem Ordner `/home/tilde/TILDE/c++/build`. Von dort gibt man folgendes ein:
```bash
g++ -std=c++11 -o use_tilde use_tilde.cpp /home/tilde/TILDE/c++/build/Lib/libTILDE_static.a -L/usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -lopencv_features2d
```

Sollte man die Dockerfile ändern, dass OpenCV 2.x verwendet wird, muss man
`lopencv_imgcodecs` durch `-lopencv_highgui` ersetzen. Der obere Befehl beinhaltet beides. In diesem Fall muss einf `-lopencv_imgcodecs` entfernt werden.

### Wie benutzt man use_tilde.cpp?
Aus dem Docker Container aus kann die kompilierte `use_tilde` Datei wie folgt benutzt werden:
```bash
./use_tilde --imageDir /home/tilde/TILDE/data --outputDir output --fileName testImage.png --filterPath /home/tilde/TILDE/c++/Lib/filters --filterName Mexico.txt
```

### Rechtemanagment:
Docker und TILDE schreiben in den `output` Ordner als `root` User mit der `root` Gruppe. Damit man später noch die Dateien löschen oder neue Dateien in dem Ordner erzeugen kann, müssen die Rechte für den `output` Ordner neu gesetzt werden:

```bash
sudo chown -R $USER outputs
```
