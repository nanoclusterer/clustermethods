# Importiere die notwendigen Bibliotheken
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA  # Für PCA zur Dimensionsreduktion
from scipy.signal import find_peaks  # Für die Erkennung von Peaks
import time

# ILS Funktion
def ILS(df, labelColumn, featureColumns, outColumn='LS', iterative=True):
    indexNames = list(df.index.names)
    oldIndex = df.index
    df = df.reset_index(drop=False)

    # separate labelled and unlabelled points
    labelled = [group for group in df.groupby(df[labelColumn] != 0)][True][1].fillna(0)
    unlabelled = [group for group in df.groupby(df[labelColumn] != 0)][False][1]

    outD = []
    outID = []
    closeID = []

    while len(unlabelled) > 0:
        D = pairwise_distances(labelled[featureColumns].values, unlabelled[featureColumns].values)
        (posL, posUnL) = np.unravel_index(D.argmin(), D.shape)
        idUnL = unlabelled.iloc[posUnL].name
        idL = labelled.iloc[posL].name

        # Update label
        unlabelled.loc[idUnL, labelColumn] = labelled.loc[idL, labelColumn]
        # Füge den neu gelabelten Punkt zur labelled-Liste hinzu
        labelled = pd.concat([labelled, unlabelled.loc[[idUnL]]])
        # Entferne den gelabelten Punkt von der unlabelled-Liste
        unlabelled.drop(idUnL, inplace=True)

        outD.append(D.min())
        outID.append(idUnL)
        closeID.append(idL)

        if len(labelled) + len(unlabelled) != len(df):
            raise Exception('Mismatch in labelled and unlabelled points count.')

    newIndex = oldIndex[outID]
    orderLabelled = pd.Series(data=outD, index=newIndex, name='minR')
    closest = pd.Series(data=closeID, index=newIndex, name='IDclosestLabel')
    labelled = labelled.rename(columns={labelColumn: outColumn})
    newLabels = labelled.set_index(indexNames)[outColumn]

    return newLabels, pd.concat([orderLabelled, closest], axis=1)

# Funktion zum Plotten der Rmin-Werte
def plot_Rmin(minR):
    fig, ax = plt.subplots()
    ax.plot(range(len(minR)), minR, color='blue', label='Rmin')
    ax.set_xlabel('Data Points')
    ax.set_ylabel('Rmin')
    ax.set_title('Rmin Plot (Minimal Distances)')
    plt.show()

# Automatische Bestimmung der optimalen Anzahl von Clustern basierend auf den Spikes im Rmin-Plot
def determine_optimal_clusters(minR):
    # Finde Spikes im Rmin-Plot
    peaks, _ = find_peaks(minR, distance=10, prominence=0.15)  # Distance und prominence für Spikes anpassen
    num_clusters = len(peaks) + 1  # Anzahl der Spikes plus 1 ist die Cluster-Anzahl
    print(f"Erkannte Spikes im Rmin-Plot: {len(peaks)}. Empfohlene Cluster-Anzahl: {num_clusters}")
    return num_clusters

# KMeans Erfolgstest mit ILS und PCA für mehrdimensionale Datensätze
def kMeans_success(df, featureColumns):
    df['label'] = 0
    
    # PCA, um die Daten auf 2 Dimensionen zu reduzieren, falls mehr als 2 Dimensionen vorhanden sind
    if len(featureColumns) > 2:
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(df[featureColumns])
        df_pca = pd.DataFrame(reduced_features, columns=['PCA1', 'PCA2'], index=df.index)
        featureColumns = ['PCA1', 'PCA2']
    else:
        df_pca = df[featureColumns].copy()

    # Führe KMeans mit nur einem Cluster aus
    model = KMeans(n_clusters=1, random_state=0, n_init=10).fit(df_pca[featureColumns])
    df_pca['kMean'] = model.labels_ + 1

    # Berechne ILS und minR Werte für das initiale Clustering
    for label, group in df_pca.groupby(by='kMean'):
        group = group.copy()
        group['label'] = 0
        centroid = model.cluster_centers_[label-1]
        group.loc[min_toCentroid(group[featureColumns], centroid), 'label'] = label

        newL, orderedL = ILS(group, 'label', featureColumns)
        # Plot Rmin-Werte
        plot_Rmin(orderedL['minR'].values)
        
        # Bestimme die optimale Anzahl an Clustern
        num_clusters = determine_optimal_clusters(orderedL['minR'].values)

    # Führe KMeans mit der ermittelten optimalen Cluster-Anzahl durch
    model_optimal = KMeans(n_clusters=num_clusters, random_state=0, n_init=10).fit(df_pca[featureColumns])
    df_pca['kMean_optimal'] = model_optimal.labels_ + 1

    # Plot für das Clustering-Ergebnis
    fig = plt.figure(figsize=(3, 3))
    ax1 = plt.subplot(1, 1, 1)
    plt.xticks(()); plt.yticks(())
    ax1.scatter(df_pca[featureColumns[0]].values, df_pca[featureColumns[1]].values, s=4, color=colors[df_pca['kMean_optimal'].values])
    plt.title(f'ILS Clustering with {num_clusters} Clusters')
    plt.show()

# Finde den Punkt, der dem Schwerpunkt am nächsten liegt
def min_toCentroid(df, centroid=None, features=None):
    if type(features) == type(None):
        features = df.columns

    if type(centroid) == type(None):
        centroid = df[features].mean()

    dist = df.apply(lambda row: sum([(row[j] - centroid[i])**2 for i, j in enumerate(features)]), axis=1)
    return dist.idxmin()

# Funktion zum Laden einer CSV-Datei und automatischer Erkennung numerischer Spalten
def load_and_process_csv(file_path):
    df = pd.read_csv(file_path)
    # Erkennen aller numerischen Spalten
    featureColumns = df.select_dtypes(include=[np.number]).columns.tolist()
    # Standardisieren der numerischen Daten
    scaler = StandardScaler()
    df[featureColumns] = scaler.fit_transform(df[featureColumns])
    df.index.name = 'ID'
    return df, featureColumns

# Set consistant coloring for plotting
# The cycling is only needed if many clusters are identified
from itertools import cycle, islice
colors = np.array(list(islice(cycle(
        ['#837E7C','#377eb8', '#ff7f00',
         '#4daf4a','#f781bf', '#a65628',
         '#984ea3','#999999', '#e41a1c', '#dede00']
        ),int(10))))

# Beispiel für das Laden einer CSV-Datei und Anwenden der Methode
file_path = '3 Dim 1000 Num 3 Clus.csv'  # Ersetze durch den tatsächlichen Pfad

df, feature_columns = load_and_process_csv(file_path)
kMeans_success(df, feature_columns)
