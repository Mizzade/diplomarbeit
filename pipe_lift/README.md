### Voraussetzungen:
- Python & Pip installiert
- [Pyenv](https://github.com/pyenv/pyenv) installiert.
- [Lift-TF Repository](https://github.com/cvlab-epfl/tf-lift) heruntergeladen
- Vortrainierte Netzwerke heruntergeladen
- - [Model ohne Rotationsaugmentierung](http://webhome.cs.uvic.ca/~kyi/files/2018/tflift/release-no-aug.tar.gz)
- - [Model mit Rotationsaugmentierung](http://webhome.cs.uvic.ca/~kyi/files/2018/tflift/release-aug.tar.gz)


### Setup innerhalb dieses Ordners
- Installiere Anaconda3-5.2.0 über Pyenv:

        $ pyenv install anaconda3-5.2.0

- Setze die lokale Python Version auf Anaconda3-5.2.0:

        $ pyenv local anaconda3-5.2.0

- Installiere weitere Abhängigkeiten mittels *requirements.txt*.:

        $ pip install -r requirements.txt

- Im Lift-TF Repo erstelle die Ordner `outputs`, `inputs` und `pretrained_networks`.

        $ mkdir outputs
        $ mkdir inputs
        $ pretrained_networks

- Kopiere die zu bearbeiteende Bilder nach `inputs`
- Kopiere die vortrainierten Netzwerek `*.tar.bz` in den `pretrained_networks` Ordner und entpacke sie.

        $ tar -xzf <NAME>.tar.bz


### Ausführung
Die Aufrufe müssen aus dem Order *tf-lift* ausgeführt werden.
In den unten aufgeführten Befehlen heißt das Beispielbild `COCO_test2014_000000000016.jpg`.

#### Keypunkte
Dies ruft den Keypunkt Detektor mit dem vortrainierten Netzerk auf, das für Rotation augmentiert wurde.

    $ python main.py \
    --task=test \
    --subtask=kp \
    --logdir=../pretrained_models/release-aug \
    --test_img_file=../inputs/COCO_test2014_000000000016.jpg \
    --test_out_file=../outputs/COCO_test2014_000000000016_kp_aug.txt \
    --use_batch_norm=False \
    --mean_std_type=hardcoded

Das Resultat sind zwei Datein im *outputs* Ordner:
- COCO_test2014_000000000016_kp_aug.txt
- COCO_test2014_000000000016_kp_aug.txt.jpg

Die `.txt` Datei beinhaltet die Keypunkte. Die ersten geben die Anzahl der Element pro Zeile an (13) und die Anzahl der Zeilen (1000).

Die ersten beiden Elemente pro Zeile sind die (x,y)-Koordinaten des Keypunktes.

Die `.jpg` ist das Eingabebild, in dem alle 1000 Keypunkte eingezeichnet sind.

Keypunkte ohne Rotationsaugmentation sollten wie folgt aufgerufen werden, schlagen jedoch fehl:

    $ python main.py \
    --task=test \
    --subtask=kp \
    --logdir=../pretrained_models/release-no-aug \
    --test_img_file=../inputs/COCO_test2014_000000000016.jpg \
    --test_out_file=../outputs/COCO_test2014_000000000016_kp_no-aug.txt \
    --use_batch_norm=False \
    --mean_std_type=dataset

#### Orientation

    $ python main.py \
    --task=test \
    --subtask=ori \
    --logdir=../pretrained_models/release-aug \
    --test_img_file=../inputs/COCO_test2014_000000000016.jpg \
    --test_out_file=../outputs/COCO_test2014_000000000016_ori_aug.txt \
    --test_kp_file=../outputs/COCO_test2014_000000000016_kp_aug.txt \
    --use_batch_norm=False \
    --mean_std_type=hardcoded

Als Ausgabe erhält man eine weiter `.txt` Datei, die die Orientierungen der Keypunkte beschreibt. Die ersten zwei Zeilen sind wieder Meta-Informationen und beschreiben die Element pro Zeile und die Anzahl an Zeilen.

#### Deskriptoren

    $ python main.py \
    --task=test \
    --subtask=desc \
    --logdir=../pretrained_models/release-aug \
    --test_img_file=../inputs/COCO_test2014_000000000016.jpg \
    --test_out_file=../outputs/COCO_test2014_000000000016_desc_aug.h5 \
    --test_kp_file=../outputs/COCO_test2014_000000000016_ori_aug.txt \
    --use_batch_norm=False \
    --mean_std_type=hardcoded

Die resultiert in einer `.h5` Datei, die mit dem Python Modul `h5py` geöffnet werden und zu numpy konvertiert werden kann. Hier ein Beispiel:

```python
import h5py
import numpy as np

filename = 'COCO_test2014_000000000016_desc_aug.h5'
f = h5py.File(filename, 'r')
print(list(f.keys()))
>>> ['descriptors', 'keypoints']

print(list(f['descriptors']))
>>> array([[ 2.92005644e+02,  2.66805315e+02,  2.33158064e+00, ...,]], dtype=float32)

desc = np.array(list(f['descriptors']))
keyp = np.array(list(f['keypoints']))

print(desc.shape)
>>> (860, 128)

print(keyp.shape)
>>> (860, 13)
```

## Problems

### Theano benötigt libgpuarray.
#### Lösung:
Die Anleitung wie libgpuarray installiert werden kann findet sich [hier](http://deeplearning.net/software/libgpuarray/installation.html#step-by-step-install).

### Problem: --enable-shared is missing in Python version von Pyenv:
> relocation R_X86_64_PC32 against symbol `_Py_NoneStruct' can not be used when making a shared object; recompile with -fPIC
> /usr/bin/ld: final link failed: bad value
> collect2: error: ld returned 1 exit status

#### Lösung:
Siehe folgenden [SO](https://stackoverflow.com/questions/42582712/relocation-r-x86-64-32s-against-py-notimplementedstruct-can-not-be-used-when) Thread:cl

> The solution was to reinstall the pyenv-provided Python with the flag set like this:

```bash
    $ PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install --force 3.5.5
```

### RuntimeWarning
> RuntimeWarning: compiletime version 3.5 of module 'tensorflow.python.framework.fast_tensor_util' does not match runtime version 3.6

#### Lösung:
Anaconda3-5.2.0 wurde mit Python 3.6 kompiliert, Tensorflow 1.4 jedoch mit Python 3.5. Dies erzeugt die Laufzeit Warnung. Sie kann jedoch getrost ignoriert werden, das Programm läuft trotzdem.

### bad_alloc
> terminate called after throwing an instance of 'std::bad_alloc'

#### Lösung:
Dieser Fehler tritt anscheinend auf tritt auf, wenn das Eingabebild zu groß war. Es werden dann keine Keypunkte erzeugt und folglich kann das Programm die Folgeschritte nicht mehr aufführen.
Es lohnt sich daher, das Bild eventuell vor Gebrauch zu verkleinern.


### Update:  11.11.2018
Changed to config files to start the pipeline with minimal parameters:
#### Keypoints:
- subtask
- img_file
- out_file

```python
python main.py \
    --subtask=kp \
    --test_img_file=../../data/v_churchill/1.ppm \
    --test_out_file=../inter/kpts_lift__1_LIFT.txt
```

#### Orientation:
- subtask
- img_file
- out_file
- kp_file

```python
python main.py \
    --subtask=ori \
    --test_img_file=../../data/v_churchill/1.ppm \
    --test_out_file=../inter/ori_lift__1_LIFT.txt \
    --test_kp_file=../inter/kpts_lift__1_LIFT.txt
```

#### Descriptors:
- subtask
- img_file
- out_file
- kp_file

```python
python main.py \
    --subtask=desc \
    --test_img_file=../../data/v_churchill/1.ppm \
    --test_out_file=../inter/desc_lift__1_LIFT_LIFT.h5 \
    --test_kp_file=../inter/ori_lift__1_LIFT.txt
```

