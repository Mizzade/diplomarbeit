### Aufbau der Pickle Datei

- [collection_name]
  - num_kpts
    - Typ: pd.DataFrame
    - Index: Name aller Bilder in der Kollektion als absoluter Pfad
    - Column: 0
    - Beschreibung: Index sind die Namen von jedem Bild in der Kollektion. Das Dataframe hat nur eine Spalte und die enthält die Anzahl der gefundenen Keypunkte für das jeweilige Bild.

  - avg_num_kpts
    - Typ: float
    - Beschreibung: Die durchschnittliche Anzahl an gefundenen Keypunkten in jedem Set dieser Kollektion.

  - std_num_kpts
    -Typ: float
    - Beschreibung: Die Standardabweichung der durchschnittlichen Anzahl der gefundenen Keypunkte in jedem Set dieser Kollektion

  - avg_num_matching_kpts_for_e_X
    - X: Ein Integer. Stammt vom config Wert: `epsilons`.
    - Typ: float
    - Beschreibung: Gewichteter Durchschnitt der durchschnittlichen Anzahl an gematchten Keypunkten der Sets.

  - std_num_matching_kpts_for_e_X
    - X: Ein Integer. Stammt vom config Wert: `epsilons`.
    - Typ: float
    - Beschreibung: Gewichteter Durchschnitt der durchschnittlichen Anzahl an gematchten Keypunkten der Sets.

  - avg_perc_matching_kpts_for_e_X
    - X: Ein Integer. Stammt vom config Wert: `epsilons`.
    - Typ: float
    - Beschreibung: Durchschnittlicher Prozentsatz an gematchten Keypunkten über alle Set der Kollektion.

  - std_perc_matching_kpts_for_e_X
    - X: Ein Integer. Stammt vom config Wert: `epsilons`.
    - Typ: float
    - Beschreibung: Standardabweichung des durchschnittlichen Prozentsatzes an gematchten Keypunkten über alle Set der Kollektion.

  - [set_name]
    - max_num_matching_kpts
      - Typ: pd.DataFrame
      - Index: Name der Bilder im Set als absoluter Pfad
      - Column: Name der Bilder im Set als absoluter Pfad
      - Beschreibung: Die maximal Anzahl an Keypunkten, die in den beiden Bildern gematcht werden können. Dies ist entspricht dem Minimum der Anzahl der Keypunkte in beiden Bildern.

    - num_matching_kpts_for_e_X
      - X: Ein Integer. Stammt vom config Wert: `epsilons`.
      - Typ: pd.DataFrame
      - Index: Name der Bilder im Set als absoluter Pfad
      - Column: Name der Bilder im Set als absoluter Pfad
      - Beschreibung: Tatsächliche Anzahl an Keypunkten, die einem Bilderpaar gematcht werden konnten. Der Wert kann nicht größer sein als `max_num_matching_kpts` für das entsprechende Bilderpaar.

    - perc_matching_kpts_for_e_X
      - X: Ein Integer. Stammt vom config Wert: `epsilons`.
      - Typ: pd.DataFrame
      - Index: Name der Bilder im Set als absoluter Pfad
      - Column: Name der Bilder im Set als absoluter Pfad
      - Beschreibung: Verhältnis der Anzahl von Keypunkten, die tatsächlich im Bilderpaar gematcht werden konnten zu der Anzahl maximal möglicher Matches. Werte liegen zwischen 0 und 1.

    - avg_num_kpts
      - Typ: float
      - Beschreibung: Durchschnittliche Anzahl gefundener Keypunkte in allen Bildern eines Sets.

    - std_num_kpts
      - Typ: float
      - Beschreibung: Standardabweichung der durchschnittlichen Anzahl der Keypunkte in allen Bildern des Sets.

    - avg_num_matching_kpts_for_e_X
      - X: Ein Integer. Stammt vom config Wert: `epsilons`.
      - Typ: float
      - Beschreibung: Die durchschnittliche Anzahl gematchter Keypunkte für jedes Bilderpaar in einem Set.

    - std_num_matching_kpts_for_e_X
      - X: Ein Integer. Stammt vom config Wert: `epsilons`.
      - Typ: float
      - Beschreibung: Die Standardabweichung der durchschnittlichen Anzahl gematchter Keypunkte für jedes Bilderpaar in einem Set.

    - avg_max_num_matching_kpts
      - Typ: float
      - Beschreibung: Durchschnittliche Anzahl an maximal möglichen Matches für jedes Bilderpaar im Set.

    - std_max_num_matching_kpts
      - Typ: float
      - Beschreibung: Standardabweichung der durchschnittlichen Anzahl an maximal möglichen Matches für jedes Bilderpaar im Set.

    - avg_perc_matching_kpts_for_e_X
      - X: Ein Integer. Stammt vom config Wert: `epsilons`.
      - Typ: float:
      - Beschreibung: Durchschnittliche Ratio von gematchten Keypunkten zu maximal matchbaren Keypunkten für jedes Bilderpaar in einem Set.




