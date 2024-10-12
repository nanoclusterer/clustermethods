import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA
import os

# =============================================================================
# Funktion zur Durchführung von PCA, wenn der Datensatz höherdimensional ist
# =============================================================================
def apply_pca_if_needed(X, n_features):
    if n_features > 2:
        print(f"Datensatz hat {n_features} Dimensionen. Reduziere mit PCA auf 2 Dimensionen.")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        return X_pca
    else:
        print(f"Datensatz hat nur {n_features} Dimensionen. PCA wird nicht angewendet.")
        return X  # Kein PCA notwendig, benutze die originalen 2D-Daten

# =============================================================================
# Datensatz aus Datei laden und nicht-numerische Spalten entfernen
# =============================================================================

# Beispiel: CSV-Datei laden (Dateipfad anpassen)
file_path = 'Silver_nanoparticle_dataset_v3.csv'  # Ersetze dies durch den Pfad zu deiner Datei

if os.path.exists(file_path):
    # Lade den Datensatz aus einer CSV-Datei
    df = pd.read_csv(file_path)
    
    # Entferne nicht-numerische Spalten
    df_numeric = df.select_dtypes(include=[np.number])
    
    if df_numeric.shape[1] == 0:
        print("Keine numerischen Spalten im Datensatz gefunden.")
    else:
        print(f"Verwende {df_numeric.shape[1]} numerische Spalten für das Clustering.")
    
    # Überprüfen, ob der Datensatz eine Zielspalte (Labels) enthält
    if 'Label' in df.columns:
        X = df_numeric.values  # Verwende nur numerische Features
        y = df['Label'].values  # Zielspalte als y speichern
    else:
        X = df_numeric.values  # Wenn keine Label-Spalte vorhanden ist, nimm nur die Features
    
    # Anzahl der Features (Dimensionen)
    n_features = X.shape[1]

    # Daten skalieren (wichtig für PCA und DBSCAN)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Überprüfe die Dimensionen und wende PCA an, wenn der Datensatz mehr als 2 Dimensionen hat
    X_prepared = apply_pca_if_needed(X_scaled, n_features)

else:
    print(f"Datei {file_path} nicht gefunden!")


# =============================================================================
# k-means Algorithmus implementieren
# =============================================================================

kmeans = KMeans(n_clusters=5, random_state=42)  # Anzahl der Cluster kannst du hier anpassen
kmeans.fit(X_prepared)
y_kmeans = kmeans.predict(X_prepared)

# Visualisierung des k-means Clustering Ergebnisses
plt.scatter(X_prepared[:,0], X_prepared[:,1], c=y_kmeans, s=35, cmap="plasma")
centers = kmeans.cluster_centers_  # Zentroiden der Cluster
plt.scatter(centers[:,0], centers[:,1], c='red', s=200, alpha=0.75, marker="X")
plt.title("k-means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# =============================================================================
# DBSCAN Algorithmus implementieren
# =============================================================================

dbscan = DBSCAN(eps=0.5, min_samples=5)
y_dbscan = dbscan.fit_predict(X_prepared)

# Visualisierung des DBSCAN Clustering Ergebnisses
plt.scatter(X_prepared[:,0], X_prepared[:,1], c=y_dbscan, s=35, cmap="plasma")
plt.title("DBSCAN Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# =============================================================================
# Silhouettenkoeffizient berechnen
# =============================================================================

silhouette_avg1 = silhouette_score(X_prepared, y_kmeans)
silhouette_avg2 = silhouette_score(X_prepared, y_dbscan)
print(f"Durchschnittlicher Silhouettenwert k-means: {silhouette_avg1}")
print(f"Durchschnittlicher Silhouettenwert DBSCAN: {silhouette_avg2}")

# =============================================================================
# Elbow-Methode für k-means
# =============================================================================

wcss = [] 
for k in range(1, 11):     
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_prepared)
    wcss.append(kmeans.inertia_)
  
plt.plot(range(1, 11), wcss, marker="o")
plt.title("Elbow-Methode")
plt.xlabel("Anzahl der Cluster")
plt.ylabel("Within-Cluster Sum of Square (WCSS)")
plt.show()

# =============================================================================
# ILS-Methode auf den Daten (nach evtl. angewendeter PCA)
# =============================================================================

labeled_indices = [0, 50, 100]  
unlabeled_indices = list(set(range(X_prepared.shape[0])) - set(labeled_indices))

# Labels initialisieren
labels = -np.ones(X_prepared.shape[0], dtype=int) 
labels[labeled_indices] = y_kmeans[labeled_indices]

# Paarweise Distanzen berechnen für ILS
dist_matrix = pairwise_distances(X_prepared)

r_min_values = []
order_of_labeling = []  

# Iterative Label-Zuordnung für ILS
while np.any(labels == -1):  
    for i in unlabeled_indices:
        nearest_labeled_index = np.argmin([dist_matrix[i, j] for j in labeled_indices])
        nearest_label = labels[labeled_indices[nearest_labeled_index]]
        r_min = dist_matrix[i, labeled_indices[nearest_labeled_index]]  

        # Label zuweisen
        labels[i] = nearest_label
        labeled_indices.append(i)
        unlabeled_indices.remove(i)

        # R_min und Reihenfolge speichern
        r_min_values.append(r_min)
        order_of_labeling.append(i)

# Zwei Plots: ILS Clustering und R_min Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))  

# ILS Clustering Plot
ax1.scatter(X_prepared[:, 0], X_prepared[:, 1], c=labels, cmap='viridis')
ax1.set_title("ILS Clustering")
ax1.set_xlabel("Feature 1")
ax1.set_ylabel("Feature 2")

# R_min Plot
ax2.plot(range(len(r_min_values)), r_min_values, marker='o', linestyle='-')
ax2.set_title("R_min Plot")
ax2.set_xlabel("Reihenfolge der Labelung")
ax2.set_ylabel("R_min (minimale Distanz)")

plt.tight_layout()
plt.show()


