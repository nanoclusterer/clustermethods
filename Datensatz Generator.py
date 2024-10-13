# =============================================================================
# Datensatz Generator 
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs

# Parameter für die Cluster
num_samples = 1000  # Anzahl der Datenpunkte
num_clusters = 5    # Anzahl der Cluster
num_dimensions = 6  # Anzahl der Dimensionen (Features)
random_state = 42   # Zufallszustand für Reproduzierbarkeit
cluster_std = 0.8   # Standardabweichung der Punkte um Clusterzentren (kleinere Werte = dichtere Cluster)

# Cluster erstellen mit der angegebenen Anzahl an Dimensionen und Standardabweichung
X, y = make_blobs(n_samples=num_samples, centers=num_clusters, n_features=num_dimensions,
                  cluster_std=cluster_std, random_state=random_state)

# DataFrame mit dynamischer Anzahl von Dimensionen erstellen
column_names = [f'dim_{i+1}' for i in range(num_dimensions)]  # Erzeugt Spaltennamen wie 'dim_1', 'dim_2', etc.
df = pd.DataFrame(np.round(X, 1), columns=column_names)  # Runde auf eine Dezimalstelle

# CSV-Datei speichern
csv_file_path = "C:/Users/gannu/Downloads/clustered_dataset.csv"  # Pfad anpassen
df.to_csv(csv_file_path, index=False, sep=',', decimal='.')

print(f"Cluster-Datensatz mit {num_dimensions} Dimensionen erfolgreich gespeichert unter: {csv_file_path}")


