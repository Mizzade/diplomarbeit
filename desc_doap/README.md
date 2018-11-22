# local Descriptors Optimized for Average Precision (DOAP)
## Requirments
- Matlab mit einer funktionierenden Lizenz oder zumindest eine laufende Trial-Version.

## Installation des DOAP Ordners
Vom $root$ Verzeichnis dieses Repos führe folgenden Code im Terminal aus:
```bash
  ./scripts/desc_doap/setup_vlfeat.sh $(pwd) && \
  ./scripts/desc_doap/setup_matconvnet.sh $(pwd) && \
  ./scripts/misc/copy_io_utils.sh $(pwd) desc_doap
```

oder führe das komplette Setup aus mittels:
```bash
  ./scripts/setup_project.sh $(pwd)
```

## Ausführen von DOAP
DOAP läuft in Matlab und kann direkt vom Terminal oder in Matlab aufgerufen werden. Es stehen zwei Funktionen zur Verfügung
- use_doap_with_file.m
- use_doap_with_csv.m

## Aufruf für eine einzige File
Um eine einzige File, die Bild-Patches enhält mittels DOAP zu verarbeiten bentutzt man `use_doap_with_file.m`. Die File muss eine `.csv` Datei sein und die Patches um die gefundenen Keypunkte beinhalten. Der Aufruf erfolgt mittels

```bash
matlab -nosplash -r "use_doap_with_csv(vlfeat_root_dir, matconvnet_root_dir, path_to_doap_model, path_to_layers_dir, path_to_input_file, path_to_output_file);quit"
```

Dabei gilt:
- vlfeat_root_dir: Absoluter Pfad um Installationsverzeichnis von VlFeat. Wird standardmäßig mit in diesen Projektordner installiert, wenn man wie oben das Setup ausführt.
- matconvnet_root_dir: Absoluter Pfad zum Installationsverzeichnis von MatConvNet. Wird standardmäßig mit in diesen Projektordner installiert, wenn man wie oben das Setup ausführt.
- path_to_doap_model: Absoluter Pfad zu vortrainierten DOAP Model.
- path_to_layers_dir: DOAP benötigt die File `PaddedBilinear.m`. Der Layers Ordner ist der Ordner, der diese File beinhaltet.
- path_to_input_file: Absoluter Pfad der `.csv` Datei, die die 42x42 Patches der Keypunkte beinhaltet.
- path_to_output_file: Absoluter Pfad und Name der File, in der die Deskriptoren gespeichert werden sollen.

### Beispiel für eine File
```bash
matlab -nosplash -r "use_doap_with_csv('vlfeat-0.9.21', 'matconvnet-1.0-beta25', 'HPatches_ST_LM_128d.mat', '.', 'test_in/patches.csv', 'test_out/descriptors.csv');quit"]
```

## Aufruf für einen Ordner, der Unterdner enthält, in denen die Patches als csv gespeichert sind
Für mehrere Files sollte man `use_doap_with_csv.m` wählen. Anstelle eines Eingabe- und Ausgabepfades werden hier Eingabe- und Ausgabeordner als Parameter mitegegeben, ansonsten sind die Paramter die gleichen wie oben beschrieben.


```bash
matlab -nosplash -r "use_doap_with_csv(vlfeat_root_dir, matconvnet_root_dir, path_to_doap_model, path_to_layers_dir, path_to_input_dir, path_to_output_dir);quit"
```

Der Eingabeordner muss folgende Struktur haben:
- input_dir
  - image_set_01
    - image_01
    - image_02
    - ...
    - image_N
  - image_set_02
    - image_01
    - ...
    - image_M
  - ...
  - image_set_L
    - image_01
    - ...
    - image_K

### Beispiel
```bash
matlab -nosplash -r "use_doap_with_csv('vlfeat-0.9.21', 'matconvnet-1.0-beta25', 'HPatches_ST_LM_128d.mat', '.', 'test_in', 'test_out');quit"]
```

## Sonstiges
DOAP benötigt Image-Patches der Form 42x42 Pixel, um daraus die Deskriptoren zu erzeugen.
