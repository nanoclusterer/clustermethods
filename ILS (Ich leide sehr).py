# Importiere die notwendigen Bibliotheken
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import StandardScaler
import time

# ILS Funktion
def ILS(df, labelColumn, outColumn='LS', iterative=True):
    featureColumns = [i for i in df.columns if i != labelColumn]
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

# Plot ILS Distances Funktion
def plot_ILSdistances(df, minR, centroid, label):
    fig = plt.figure(figsize=(6, 3))
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.001, top=0.96, wspace=0.05, hspace=0.01)

    ax = plt.subplot(1, 2, 1)
    plt.ylim(0, 1)
    plt.xticks(()); plt.yticks(())
    ax.plot(range(len(minR)), minR, color=colors[label])

    ax = plt.subplot(1, 2, 2)
    plt.xticks(()); plt.yticks(())
    plt.xlim(-3, 3); plt.ylim(-3, 3)
    ax.scatter(df['x'].values, df['y'].values, s=4, color=colors[0])
    ax.scatter(centroid[0], centroid[1], s=3, color=colors[label], marker='x', linewidth=20)

# KMeans Erfolgstest mit ILS
def kMeans_success(df, k):
    df['label'] = 0
    features = ['x', 'y']  # Definieren der Features
    model = KMeans(n_clusters=k, random_state=0, n_init=10).fit(df[features])
    df['kMean'] = model.labels_ + 1

    fig = plt.figure(figsize=(3, 3))
    ax1 = plt.subplot(1, 1, 1)
    plt.xticks(()); plt.yticks(())
    ax1.scatter(df['x'].values, df['y'].values, s=4, color=colors[df['kMean'].values])

    for label, group in df.groupby(by='kMean'):
        group = group.copy()
        group['label'] = 0
        centroid = model.cluster_centers_[label-1]
        group.loc[min_toCentroid(group[features], centroid), 'label'] = label

        ti = time.time()
        newL, orderedL = ILS(group, 'label')
        tf = time.time()
        print(f"Iterative label spreading took {tf - ti:.1f}s to label {len(group)} points")

        plot_ILSdistances(group, orderedL['minR'].values, centroid, label)

# Finde den Punkt, der dem Schwerpunkt am nächsten liegt
def min_toCentroid(df, centroid=None, features=None):
    if type(features) == type(None):
        features = df.columns

    if type(centroid) == type(None):
        centroid = df[features].mean()

    dist = df.apply(lambda row: sum([(row[j] - centroid[i])**2 for i, j in enumerate(features)]), axis=1)
    return dist.idxmin()

# Funktion zum Laden einer CSV-Datei
def load_and_process_csv(file_path):
    df = pd.read_csv(file_path)
    # Standardisieren der Daten (z.B. Skalierung)
    scaler = StandardScaler()
    df[['x', 'y']] = scaler.fit_transform(df[['x', 'y']])
    df.index.name = 'ID'
    return df

# Set consistant coloring for plotting
# The cycling is only needed if many clusters are identified
from itertools import cycle, islice
colors = np.array(list(islice(cycle(
        ['#837E7C','#377eb8', '#ff7f00',
         '#4daf4a','#f781bf', '#a65628',
         '#984ea3','#999999', '#e41a1c', '#dede00']
        ),int(10))))

# Beispiel für das Laden einer CSV-Datei und Anwenden der Methode
file_path = 'datensatz1.csv'  # Ersetze durch den tatsächlichen Pfad
df = load_and_process_csv(file_path)
kMeans_success(df, 4)  # z.B. für 3 Cluster
plt.show()
