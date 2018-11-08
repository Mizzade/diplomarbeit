### Output
Dieser Ordner enhält die Ausgaben der einzelnen Deskriptoren und Detektoren. Für jedes Bilderset wird ein eigenen Ordner mit dem Name des Sets angelegt. Innerhalb dieses Ordner befinden sich vier Unterordner:
- keypoints: In diesem Ordner werden die Ausgaben von Detektoren als .csv gespeichert.
- descriptors: Dies sind die Ausgaben der Deskriptoren als .csv Datei.
- keypoint_images: Gefundene Keypunkte werden in das Eingabebild eingezeichnet und hier als .png gespeichert.
- heatmap_images: Sollte ein Modell darüberhinaus auch Heatmaps erzeugen, werden diese hier als .png gespeichert.

#### Namensschema
Die Benennung einzelner Dateien erfolgt mittels folgender Bausteine:
- TYPE: desc | kpts | heatmap
- PROJEKT_NAME: doap | tfeat| tilde | sift | superpoint | lift | tconvdet
- IMAGE_NAME
- DETEKTOR_NAME: sift | tilde | tconvdet | superpoint | lift
- DESCRIPTOR_NAME: sift | tfeat | doap | superpoint | lift | ''
- EXTENSION: z.B. .png, .csv etc.

Der Name wird wir folge zusammengesetzt:

        <TYPE>_<PROJECT_NAME>__<IMAGE_NAME>_<DETECTOR_NAME>_<DESCRIPTOR_NAME>.<EXTENSION>


Dateien die mit `ktps` beginnen, haben keinen `<DESCRIPTOR_NAMEN>` im Namen.

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
