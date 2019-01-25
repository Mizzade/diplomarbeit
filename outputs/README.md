# Ausgabewerte von Destektoren und Deskriptoren
Dieser Ordner enhält die Ausgaben der einzelnen Deskriptoren und Detektoren.
Für jede Bilderkollektion wird ein Ordner angelegt. Innerhalb dieses Ordners
wernde für jedes Set innerhalb der Kollektion weitere Für jedes Bilderset wird
ein eigenen Ordner mit dem Name des Sets angelegt. Innerhalb dieses Ordner
befinden sich vier Unterordner:
- keypoints: In diesem Ordner werden die Ausgaben von Detektoren als .csv gespeichert.
- descriptors: Dies sind die Ausgaben der Deskriptoren als .csv Datei.
- keypoint_images: Gefundene Keypunkte werden in das Eingabebild eingezeichnet und hier als .png gespeichert.
- heatmap_images: Sollte ein Modell darüberhinaus auch Heatmaps erzeugen, werden diese hier als .png gespeichert.


### Namensschame
Die Benennung einzelner Dateien erfolgt mittels folgender Bausteine:
- TYPE: keypoints | heatmap_images | keypoint_images
- COLLECTION_NAME: eisert | webcam | example
- SET_NAME: v_set_01 | ... | v_set_13 | chamonix | courbevoie | frankfurt | mexico | panorama | stlouis
- DESCRIPTOR_NAME doap | tfeat | sift | superpoint
- DETEKTOR_NAME: sift | tilde | tconvdet | superpoint | lift
- IMAGE_NAME
- SIZE: <Number> | ''
- EXTENSION: z.B. .png, .csv etc.

`[_<SIZE>]` bedeutet, dass der Eintrag an der Stelle optional ist, je nachdem,
ob der ensprechende Detektor/Deskriptor mit einem `max_size` Parameter gestartet
wurde oder nicht.

### Namensschema für Ausgabe von Detektoren

- Keypunkte, Keypunkt Bilder, Heatmaps:

        <COLLECTION_NAME>/<SET_NAME>/<TYPE>/<DETECTOR_NAME>/<IMAGE_NAME>[_<SIZE>].<EXTENSION>

### Namensschema für Ausgabe von Deskriptoren

- Deskriptoren

        <COLLECTION_NAME>/<SET_NAME>/descriptors/<DESCRIPTOR_NAME>/<DETECTOR_NAME>/<IMAGE_NAME>[_<SIZE>].<EXTENSION>


### Beispiel
> eisert/v_set_03/descriptors/doap/sift/1_1300.csv

Dies ist das Bild `1.png` aus dem Set `v_set_03` in der Kollektion `eisert` und
beinhaltet die Deskriptoren des Deskritpors `DOAP` die mithilfe der Keypunkte
des Detektors `SIFT` erzeugt wurden.


#### Keypoints Schema
Eine .csv Datei, die Keypunkte enhält, ist wie folgt aufgebaut:
- die ersten drei Zeilen sind Metainformationen
- Alle folgenden Zeilen sind Keypunkte.

Beispiel:
```
# height, width, number of rows, number of columns
# 800, 600, 243, 7
# x, y, size, angle, response, octave, class_id
2.620826911926269531e+01,4.635076904296875000e+02,2.174523162841796875e+01,4.229751586914062500e+01,2.082937955856323242e-02,1.323878600000000000e+07,-1.000000000000000000e+00
```
Die erste Zeile ist die Beschreibung der zweiten Zeile.
Sie gibt an, wie die Dimensionen des Bildes waren, auf dem die Keypunkte erstellt wurden, gefolgt von der Anzahl von Keypunkten und die Anzahl der Spalten pro Reihe.

Im obigen Beispiel hatte das Bild eine Höhe von 800 Pixel und eine Breite von 600 Pixel. Es wurden 243 Keypunkte gefunden.

Die ersten beiden Zahlen in jeder Reihe der Keypunkte beschreibt die Position im Bild. Die fünf weiteren Spaltenwerte werden nur benötigt, falls man aus dem Keypunkt eine Instanz von openCVs KeyPoint Klasse erzeugen möchte.


#### Descriptor Schema
Die erste Zeile einer .csv Datei, die Deskriptorwerte beinhaltet, beinhaltet Metainformationen, gefolgt von alle Deskriptoren.

Die erste Zahl gibt die Anzahl der Deskriptoren an, die zweite die Dimensionalität des Deskriptors. Im unteren Beispiel sind in dieser Datei also 243 Deskriptoren mit jeweils 128 Dimensionen gespeichert.

Beispiel:

```
# 243, 128
0.000000000000000000e+00,...
...
```
#### Scores Schema
TILDE und SuperPoint liefert außerdem eine .png Datei pro Bild vom Typ `heatmap_images`. Darin befinden sich die Wahrscheinlichkeitswerte für jeden Pixel ein Keypunkt zu sein.

Der Farbwert des Graystufen-Bildes gibt dabei die Wahrscheinlichkeit des Pixels an,
ein interessanter Punkt zu sein. Da die meisten Pixel jedoch keine interessanten
Punkte sind, ist der Großteil der meisten dieser Bilder einfach schwarz, was
einer Wahrscheinlichkeit von 0% entspricht.
