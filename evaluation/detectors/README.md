# Evaluierung von Keypunkt Detektoren

# Wiederholbarkeitsevaluierung
Für jedes Model (sift, lift, tilde, superpoint) werdne die Ergebnisse in einer Pickle-Datei (repeatability_sift.pkl, repeatability_tilde.pkl, ...) abgespeichert. Weleche Werte gespeichert werden, hängt von den aktivierten Parameter ab, siehe dazu `Evalutionsanmerkungen`. Die Pickle-Dateien landen standardmäßig im `outputs` Ordner, der sich im selben Ordner befindet wie diese README.

### Hilfe
Für eine Auflistung von Paramter benutze das Kommando

     $ python evalution.py -h

Alle Parameter können in der Datei `config_repeatability.py` gefunden wernden.

### Parameteranmerkungen
- MODEL_NAMES: sift, lift, tilde, superpoint
- MAX_SIZE: Je nachdem, mit welchem `max_size` Paramter Keypunkte und Deskriptoren erzeugt wurde, ändert sich der Name der Ausgabedatei. Wenn die File für die Keypunkte also mit 1300.csv endet, so muss `1300` als MAX_SIZE Wert eingetragen werden.
- allowed_extensions: Besteht immer aus einem Punkt und dem Extensionnamen, z.B. `.png` oder `.jpg`.
- COLLECTION_NAMES: Beinhaltet bis jetzt nur die Kollektion `webcam`.
- SET_NAMES: chamonix, courbevoie, frankfurt, mexico, panorama, stlouis

### Evalutionsanmerkungen
- eval_set__num_kpts_per_image: Zählt die Anzahl an Keypunkten, die das Model in jedem Bild inerhalb eines Sets gefunden hat. Typ: np.array[int].
- eval_set__num_kpts_per_image_avg: Gibt die durchschnittliche Anzahl an Keypunkten in allen Bildern in einem Set für ein Model wieder. Typ: float.
- eval_set__num_kpts_per_image_std: Gibt die Standardabweichung der durchschnittlichen Anzahl an Keypunkten in allen Bildern in einem Set wieder. Typ: float.
- eval_set__image_names: Gibt die Namen der Bilder innerhalb eines Sets wieder. Typ: List[str].
- eval_set__num_repeatable_kpts: Gibt die Anzahl der wiederholbaren Keypunkte aus dem ersten Bild im Set wieder, die in allen anderen Bildern des Sets ebenfalls gefunden wurden. Typ: int.
- eval_set__idx_repeatable_kpts: Gibt die Indizes der oben gefundenen Keypunkte wieder. Typ: array[int].
- eval_set__cum_repeatable_kpts: Startend mit dem ersten Bild in dem Set (sortiert nach Namen), zähle die Anzahl der Keypunkte aus dem ersten Bild, die bis zum i-ten Bild wiedergefunden wurden. Typ: np.array[int].
- eval_set__repeatable_kpts_image_pairs: Vergleicht jede Bilderpaar-Kombination innerhalb eines Sets und speichert die Anzahl der gleichen Keypunkte in einer 2-dimensionalen Matrix. Typ: array[array[int]].
