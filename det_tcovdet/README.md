### Training- und Test-Patches erzeugen.

```python
python get_training_pair.py [OPTIONS]
```

### Optionale Paramter
- --no_test: Überspringe die Generierung von Test-Patches.
- --no_train: ÜBerspringe die Generierung von Training-Patches.

Erzeugt Trainings- und Test-Patches im Ordner `./memmaps/train` und `./memmaps/test`. Jeder Order enthält die folgenden drei Dateien:

- im: Orginal Patches (Nx32x32 Pixel).
- im_warped: Transformierte Patches (Nx32x32 Pixel).
- transform_matrix: Feature(Transform)-Matrix (Nx2)

Die original Patches befinden sich im Ordner `./data/patch_set/standard_patch`. Die Datei `mexico_tilde_p24_Mexico_patches.mat` enthält dabei 5879 Patches der Größe (3x51x51). Wichtig ist, hier die Dimensionen zu beachten: (num_channels, height, width).

#### Wichtig
Nach der Ausführung von `get_training_pair.py` haben die Patches die Dimension
(height, width, num_channel). Außerdem wurden die Werte innerhalb der Patches schon durch 255 geteilt, um die Werte der Pixel auf das Interval [0, 1] zu bringen. Anschließend werden Durchschnitt und Standardverteilung ermittelt und die Patches normalisiert.

### Ordnerstruktur
- data:
  - patch_set:
    - standard_patch:
      - mexico_tilde_p24_Mexico_patches.mat: Beinhaltet 5879 Patches der Dimension (3x51x51). Die Patches wurden mit dem TILDE_24 Detektor und Mexico Datensatz aus `Webcam` erzeugt.
- memmaps: Enhält auf Test und Trainingsdaten, jeweils in drei Dateien aufgeteilt nämlich `im`, `warped_im`, `transform_im`. Daten sind als Numpy Memory Maps gespeichert.
  - test:
    - im: 128 Patches der Form (32x32x3). float32
    - warped_im: 128 Patches der Form (32x32x3). float32
    - transform_matrix: 128x2. float32
  - train:
    - im: 256000 Patches der Form (32x32x3). float32
    - warped_im: 256000 Patches der Form (32x32x3). float32
    - transform_matrix: 256000x2. float32

### Ausführen des Detektors
```python
python patch_network_point_test.py --train_name $network_name --stats_name $stats_name --dataset_name $dataset_name --save_feature $conv_feature_name
```

### Punkte extrahieren
```bash
dataset_name='webcam'
conv_feature_name='covariant_point_tilde'
feature_name='feature_point_tilde'
network_name='mexico_tilde_p24_Mexico_train_point_translation_iter_20'
stats_name='mexico_tilde_p24_Mexico_train_point'
point_number='1000'

$matlab -r "point_extractor('$dataset_name','$conv_feature_name','$feature_name',$point_number);  exit(0);";
```
```python
 python ./scripts/run/run_detectors.py --root_dir $(pwd) --max_size 1200 --detectors tcovdet --collection_names webcam eisert --set_names frankfurt v_set_01 --max_num_images 4

```
