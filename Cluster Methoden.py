# Import notwendiger Bibliotheken
from tkinter import filedialog, StringVar  # Dateidialoge und GUI-Variablen
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # Matplotlib-Integration in Tkinter
from sklearn.cluster import KMeans  # KMeans-Clustering
from sklearn.cluster import DBSCAN  # DBSCAN-Clustering
from sklearn.metrics.pairwise import pairwise_distances  # Paarweise Distanzberechnungen
from sklearn.preprocessing import MinMaxScaler  # Normalisierung der Daten
from sklearn.decomposition import PCA  # Hauptkomponentenanalyse (PCA)
from scipy.signal import find_peaks  # Finden von Spitzen (Peaks)
import customtkinter as ctk  # Erweiterte Tkinter-Bibliothek für modernes Design
import matplotlib.pyplot as plt  # Matplotlib für Datenvisualisierung
import pandas as pd  # Datenanalyse
import numpy as np  # Numerische Berechnungen
import seaborn as sns  # Erweiterte Datenvisualisierung
import os  # Betriebssystemfunktionen


# Globale Seaborn-Parameter setzen
sns.set_context("notebook")  # Skalierung und Darstellung der Elemente
sns.set_style("darkgrid")  # Hintergrundstil für Diagramme


# Matplotlib-Parameter werden gesetzt, um das Design der Plots zu verbessern
plt.rcParams.update({
    'legend.frameon': False,  # Keine Rahmen für Legenden
    'legend.numpoints': 1,  # Nur ein Punkt pro Eintrag in der Legende
    'legend.scatterpoints': 1,  # Nur ein Punkt pro Eintrag in Scatterplot-Legende
    'xtick.direction': 'out',  # X-Achsen-Ticks nach außen zeigen
    'ytick.direction': 'out',  # Y-Achsen-Ticks nach außen zeigen
    'axes.axisbelow': True,  # Achsenlinien im Hintergrund
    'font.family': 'sans-serif',  # Schriftfamilie
    'grid.linestyle': '-',  # Gitterlinien-Stil
    'lines.solid_capstyle': 'round',  # Linienkappung abgerundet
    'axes.grid': True,  # Gitter anzeigen
    'axes.linewidth': 0,  # Breite der Achsenrahmenlinien; 0 entfernt die Rahmen komplett
    'xtick.major.size': 0,  # Größe der X-Hauptticks
    'ytick.major.size': 0,  # Größe der Y-Hauptticks
    'xtick.minor.size': 0,  # Größe der X-Nebenticks
    'ytick.minor.size': 0,  # Größe der Y-Nebenticks
    'text.color': '0.9',  # Textfarbe
    'axes.labelcolor': '0.9',  # Achsenbeschriftungsfarbe
    'xtick.color': '0.9',  # X-Achsenfarbe
    'ytick.color': '0.9',   # Y-Achsenfarbe
    'grid.color': '#2A3459',   # Gitterfarbe
    'font.sans-serif': ['Overpass', 'Helvetica', 'Helvetica Neue', 'Arial',
                        'Liberation Sans', 'DejaVu Sans', 'Bitstream Vera Sans', 'sans-serif'],
    'axes.prop_cycle': plt.cycler(color=['#18c0c4', '#f62196', '#A267F5', '#f3907e', '#ffe46b', '#fefeff']),
    'image.cmap': 'RdPu',  # Standardfarbkarte für Bilder
    'figure.facecolor': '#2A3459',  # Hintergrundfarbe der Figur
    'axes.facecolor': '#212946',  # Hintergrundfarbe der Achsen
    'savefig.facecolor': '#2A3459'})  # Hintergrundfarbe beim Speichern von Plots


# Farbdefinitionen für das GUI-Design
root_color = "#1D243D"  # Hintergrundfarbe des Hauptfensters
frame_color = "#212946"  # Hintergrundfarbe der Rahmen
button_color = "#2A3459"  # Farbe der Buttons
entry_color = "#2A3459"  # Farbe der Eingabefelder


# =============================================================================
# Hauptfenster wird erstellt (für GUI)
# =============================================================================

root = ctk.CTk(fg_color=root_color)
root.title("ClusterMethods")


# ▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉
# Fenster Funktionen (root)
# ▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉

def set_minimum_window_size(root): 
    """
    Stellt sicher, dass das Fenster eine Mindestgröße hat, basierend auf dem aktuellen Inhalt.
    
    Args:
    - root: Das Hauptfenster der Anwendung.
    """
    root.update()  # Layout aktualisieren

    # Mindestgröße basierend auf dem Inhalt festlegen
    required_width = root.winfo_reqwidth()
    required_height = root.winfo_reqheight()
    root.minsize(required_width, required_height)  # Mindestgröße setzen
    

def resize_window_to_screen(root, scale):
    """
    Passt die Fenstergröße basierend auf der Bildschirmauflösung an.
    
    Args:
    - root: Die Hauptanwendung (CTk-Fenster).
    - scale: Skalierungsfaktor. Bestimmt, wie viel Prozent der Bildschirmgröße genutzt werden.
    """
    # Bildschirmauflösung ermitteln
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Fenstergröße basierend auf dem Skalierungsfaktor festlegen
    window_width = int(screen_width * scale)
    window_height = int(screen_height * scale)

    # Fenstergröße setzen und das Fenster mittig auf dem Bildschirm platzieren
    root.geometry(
    f"{window_width}x{window_height}+"
    f"{int((screen_width - window_width) / 2)}+"
    f"{int((screen_height - window_height) / 2)}"
    )

        
def adjust_text_and_plot_size():
    """
    Passt die Text- und Plot-Größe basierend auf der Bildschirmauflösung an.
    
    Returns:
    - headline_text_size: Größe der Hauptüberschrift.
    - label_text_size: Größe der Beschriftungen.
    - entry_text_size: Größe der Eingabefelder.
    - plot_width: Breite der Plots.
    - plot_height: Höhe der Plots.
    """  
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    
    # Berechnung der Textgröße basierend auf der Bildschirmauflösung
    headline_text_size = int(min(screen_width, screen_height) / 70)
    label_text_size = int(min(screen_width, screen_height) / 70)
    entry_text_size = int(min(screen_width, screen_height) / 70)

    # Plot-Größe basierend auf einem 5:3 Verhältnis
    plot_width = screen_width * 0.30  # 30% der Bildschirmbreite
    plot_height = plot_width * 3 / 5  # Höhe basierend auf 5:3 Verhältnis
    
    return headline_text_size, label_text_size, entry_text_size, plot_width, plot_height


# Anpassung der Text- und Plot-Größe
headline_text_size, label_text_size, entry_text_size, plot_width, plot_height = adjust_text_and_plot_size()


# ▉▉▉▉▉▉ Custom Fonts ▉▉▉▉▉▉

# Schriftarten und -größen für verschiedene GUI-Elemente definieren
headline_font = ctk.CTkFont(family="Segoe UI", size=headline_text_size, weight='bold')
label_font = ctk.CTkFont(family="Segoe UI", size=label_text_size)
entry_font = ctk.CTkFont(family="Segoe UI", size=entry_text_size)


# ▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉
# Data Input Funktionen
# ▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉

# Globale Variable für die Matplotlib-Figur
figData, axData = plt.subplots(figsize=(plot_width / 100, plot_height / 100))
figData.subplots_adjust(bottom=0.18, left=0.18)


# Funktion zum Anzeigen eines leeren Plots, wenn keine Daten geladen sind
def show_empty_plot_data():
    axData.clear()  # Leeren der Figur für neuen Plot
    axData.axis("off")
    axData.text(0.45, 0.45, "No Data loaded", horizontalalignment='center',
                verticalalignment='center', fontsize=12, transform=axData.transAxes)
    canvasData.draw()


# Funktion zum Öffnen eines Dateidialogs und Laden der Datei
def open_file_dialog():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt")])
    entry_var.set(file_path)  # Den ausgewählten Dateipfad ins Entry setzen
    plot_data_points()  # Daten für den ersten Plot automatisch laden


# Funktion, die ausgeführt wird, wenn sich der Text im Entry ändert
def on_entry_change(*args):
    plot_data_points()  # Immer, wenn sich der Dateipfad ändert, wird der Plot aktualisiert


# ▉▉▉▉▉▉ Raw Data Plot ▉▉▉▉▉▉

# Funktion zum plotten der Datenpunkte aus der Datei
def plot_data_points():
    file_path = entry_var.get()  # Den Text aus dem Entry-Feld holen
    if os.path.isfile(file_path):  # Überprüfen, ob der Dateipfad existiert
        try:
            # Datei laden (CSV oder TXT)
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path, delimiter=',', header=0)  # Erster Header als Überschrift
            elif file_path.endswith('.txt'):
                data = pd.read_csv(file_path, delimiter=',', header=0)  # Erster Header als Überschrift
            else:
                raise ValueError("Bitte eine .csv oder .txt Datei auswählen.")
            
            # Daten extrahieren
            X = data.values  # Den gesamten Inhalt als Array nehmen

            # Plot aktualisieren
            axData.clear()
            axData.scatter(X[:, 0], X[:, 1], c='turquoise', alpha=0.6, s=6, label='Data Points')
            axData.set_title("Datenpunkte")
            axData.set_xlabel("X-axis")
            axData.set_ylabel("Y-axis")
            axData.legend(handletextpad=0.25)
            canvasData.draw()
        except Exception as e:
            print(f"Fehler beim Laden der Datei: {e}")
            show_empty_plot_data()  # Bei einem Fehler wird der leere Plot angezeigt
    else:
        show_empty_plot_data()  # Wenn der Dateipfad ungültig ist, leeren Plot anzeigen


# ▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉
# K-Means Funktionen
# ▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉

# Globale Variable für die Matplotlib-Figur
figKmeans, axKmeans = plt.subplots(figsize=(plot_width / 100, plot_height / 100))
figKmeans.subplots_adjust(bottom=0.18, left=0.16)


# Funktion zum Anzeigen des leeren Plots
def show_empty_plot_kmeans():
    axKmeans.clear()  # Leeren der Figur für neuen Plot
    axKmeans.axis("off")
    axKmeans.text(0.45, 0.45, "KMEANS", horizontalalignment='center',
                  verticalalignment='center', fontsize=12, transform=axData.transAxes)
    canvasKmeans.draw()


# Funktion, um die Eingabe für k auf eine dreistellige Zahl zu beschränken
def validate_input(new_value):
    if new_value == "":  # Erlaubt leere Eingaben (falls der User löscht)
        return True
    if new_value.isdigit() and len(new_value) <= 3 and int(new_value) > 1:  # Nur Ziffern, max. 3 Stellen, keine 1
        return True
    return False

# Validierung der Eingabe für das k-Entry
k_var = StringVar()
k_var.trace_add("write", lambda *args: validate_input(k_var.get()))


# ▉▉▉▉▉▉K-Means▉▉▉▉▉▉

# Funktion zum Ausführen des K-Means-Algorithmus und Plotten der Ergebnisse
def kmeans():
    file_path = data_input_entry.get()  # Pfad zur Datei aus Entry
    try:
        # Den Wert von k aus dem Entry lesen
        k = int(k_var.get())
        
        # Datei laden (CSV oder TXT)
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path, delimiter=',', header=0)
        elif file_path.endswith('.txt'):
            data = pd.read_csv(file_path, delimiter='\t', header=0)
        else:
            raise ValueError("Bitte eine .csv oder .txt Datei auswählen.")
     
        # Nur numerische Spalten auswählen
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            raise ValueError("Die Datei enthält keine numerischen Spalten.")  
     
        # Daten skalieren
        X = MinMaxScaler().fit_transform(numeric_data.values)
        
        # PCA für mehrdimensionale Daten
        if X.shape[1] > 2:
            pca = PCA(n_components=2)
            X = pca.fit_transform(X)
            x_label = "PCA1"
            y_label = "PCA2"
        else:
            # Wenn keine PCA notwendig ist, nutze Spaltennamen aus der Datei
            x_label = numeric_data.columns[0] if numeric_data.shape[1] > 0 else "Feature 1"
            y_label = numeric_data.columns[1] if numeric_data.shape[1] > 1 else "Feature 2"
            
       # K-Means Clustering anwenden
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        y_kmeans = kmeans.predict(X)

        # Plot aktualisieren
        axKmeans.clear()
        axKmeans.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=6, alpha=0.6, cmap='viridis')
        axKmeans.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                         s=20, c='red', marker='x', label='Centroids')
        axKmeans.set_title(f"K-Means Clustering with k={k}")
        axKmeans.legend(handletextpad=0.25)
        
        # Dynamische Achsenbeschriftungen hinzufügen
        axKmeans.set_xlabel(x_label)
        axKmeans.set_ylabel(y_label)
        
        canvasKmeans.draw()
        
    except Exception as e:
        print(f"Fehler beim Laden der Datei: {e}")

        
# ▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉
# Elbow Funktionen
# ▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉

# Globale Variable für die Matplotlib-Figur
figElbow, axElbow = plt.subplots(figsize=(plot_width / 100, plot_height / 100))
figElbow.subplots_adjust(bottom=0.18, left=0.16)


# Funktion zum Anzeigen des leeren Plots
def show_empty_plot_elbow():
    '''
    Zeigt einen leeren Plot mit dem Hinweistext "Elbow" an, wenn keine Datengeladen sind.
    '''
    axElbow.clear()  # Leeren der Figur für neuen Plot
    axElbow.axis("off")
    axElbow.text(0.45, 0.45, "Elbow", horizontalalignment='center',
                 verticalalignment='center', fontsize=12, transform=axData.transAxes)
    canvasElbow.draw()


# Funktion, um die Eingabe für epsilon und min_samples auf gültige Werte zu beschränken
def validate_elbow(new_value):
    '''
    Validiert die Benutzereingabe für die maximale Anzahl von Clustern.
    - Nur numerische Werte (Zahlen) größer als 1 und maximal 3 Stellen sind erlaubt.
    '''
    if new_value == "":  # Erlaubt leere Eingaben (falls der User löscht)
        return True
    if new_value.isdigit() and len(new_value) <= 3 and int(new_value) > 1:  # Nur Ziffern, max. 3 Stellen, grösser als 4
        return True
    return False

# Validierung der Eingabe für das k-Entry
elbow_var = StringVar()
elbow_var.trace_add("write", lambda *args: validate_input(elbow_var.get()))


# ▉▉▉▉▉▉Elbow Methode▉▉▉▉▉▉

def elbow_method():
    """
    Führt die Elbow-Methode durch, um die optimale Anzahl von Clustern zu bestimmen.
    - Plottet die Summe der quadratischen Abstände (WCSS) für verschiedene k-Werte.
    """
    # Den Wert von max_k aus dem Entry lesen
    max_k = int(elbow_var.get())
    file_path = data_input_entry.get()  # Pfad zur Datei aus Entry
    try:
        # Datei laden (CSV oder TXT)
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path, delimiter=',', header=0)  # Komma als Trennzeichen
        elif file_path.endswith('.txt'):
            data = pd.read_csv(file_path, delimiter='\t', header=0)  # Tab als Trennzeichen
        else:
            raise ValueError("Bitte eine .csv oder .txt Datei auswählen.")
        
        # Nur numerische Spalten auswählen
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            raise ValueError("Die Datei enthält keine numerischen Spalten.")
        
        # Daten skalieren
        X = MinMaxScaler().fit_transform(numeric_data.values)

        wcss = []  # Liste für die WCSS-Werte (Within-Cluster-Sum of Squares)
    
        # Teste KMeans für k = 1 bis max_k
        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)  # Die Summe der quadratischen Abstände (Inertia)
        
        # Plot aktualisieren
        axElbow.clear()
        axElbow.plot(range(1, max_k + 1), wcss, 'bo-', markersize=8)
        axElbow.set_title('Elbow Method for Optimal k')
        axElbow.set_xlabel('Number of clusters (k)')
        axElbow.set_ylabel('WCSS (Inertia)')
        axElbow.grid(True)
        canvasElbow.draw()
        
    except Exception as e:
        print(f"Fehler beim Laden der Datei: {e}")


# ▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉
# DBSCAN Funktionen
# ▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉

# Globale Variable für die Matplotlib-Figur
figDBSCAN, axDBSCAN = plt.subplots(figsize=(plot_width / 100, plot_height / 100))
figDBSCAN.subplots_adjust(bottom=0.18, left=0.16)

# Funktion zum Anzeigen des leeren Plots
def show_empty_plot_dbscan():
    '''
    Zeigt einen leeren Plot mit dem Hinweistext "DBSCAN" an, wenn keine Daten geladen sind.
    '''
    axDBSCAN.clear()  # Leeren der Figur für neuen Plot
    axDBSCAN.axis("off")
    axDBSCAN.text(0.45, 0.45, "DBSCAN", horizontalalignment='center',
                  verticalalignment='center', fontsize=12, transform=axData.transAxes)
    canvasDBSCAN.draw()


# Funktion, um die Eingabe für epsilon auf gültige Werte zu beschränken
def validate_input_epsilon(new_value):
    """
    Validiert die Benutzereingabe für epsilon (Radius für den DBSCAN-Algorithmus).
    - Erlaubt nur Zahlen im Bereich 0 bis 10000 mit maximal 6 Stellen.
    """
    if new_value == "":  # Leere Eingaben zulassen (z. B. beim Löschen)
        return True
    try:
        value = float(new_value)
        if 0 <= value < 10000 and len(new_value) <= 6:
            return True
        return False
    except ValueError:
        return False


# Funktion, um die Eingabe für epsilon auf gültige Werte zu beschränken
def validate_input_min_samples(new_value):
    """
    Validiert die Benutzereingabe für min_samples (Mindestanzahl von Punkten für ein Cluster).
    - Erlaubt nur ganze Zahlen größer 0 und kleiner 10000.
    """
    if new_value == "" or (new_value.isdigit() and 0 < int(new_value) < 10000): 
        return True
    return False


# Validierung für epsilon und min_samples
epsilon_var = StringVar()
epsilon_var.trace_add("write", lambda *args: validate_input_epsilon(epsilon_var.get()))

min_samples_var = StringVar()
min_samples_var.trace_add("write", lambda *args: validate_input_min_samples(min_samples_var.get()))

    
# ▉▉▉▉▉▉ DBSCAN ▉▉▉▉▉▉

def dbscan():
    """
    Führt den DBSCAN-Algorithmus durch und plottet die Ergebnisse.
    """
    file_path = data_input_entry.get()  # Pfad zur Datei aus Entry
    try:
        # Den Wert von Epsilon und min_samples aus dem Entry lesen
        epsilon = float(epsilon_entry.get())
        min_samples = int(samples_entry.get())

        # Datei laden (CSV oder TXT)
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path, delimiter=',', header=0)
        elif file_path.endswith('.txt'):
            data = pd.read_csv(file_path, delimiter='\t', header=0)
        else:
            raise ValueError("Bitte eine .csv oder .txt Datei auswählen.")
            
        # Nur numerische Spalten auswählen
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            raise ValueError("Die Datei enthält keine numerischen Spalten.")
            
        # Daten skalieren
        X = MinMaxScaler().fit_transform(numeric_data.values)
        
        # PCA für mehrdimensionale Daten
        if X.shape[1] > 2:
            pca = PCA(n_components=2)
            X = pca.fit_transform(X)
            x_label = "PCA1"
            y_label = "PCA2"
        else:
            # Wenn keine PCA notwendig ist, nutze Spaltennamen aus der Datei
            x_label = numeric_data.columns[0] if numeric_data.shape[1] > 0 else "Feature 1"
            y_label = numeric_data.columns[1] if numeric_data.shape[1] > 1 else "Feature 2"

        # DBSCAN Algorithmus ausführen
        dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
        labels = dbscan.fit_predict(X)

        # Plot aktualisieren
        axDBSCAN.clear()
        unique_labels = set(labels)  

        # Farben für Cluster erstellen
        for label in unique_labels:
            label_mask = labels == label  # Mask für Punkte des aktuellen Labels
            if label == -1:
                axDBSCAN.scatter(X[label_mask, 0], X[label_mask, 1], 
                                 s=6, alpha=0.6, c='gray', label='Noise')
            else:
                axDBSCAN.scatter(X[label_mask, 0], X[label_mask, 1], 
                                 s=6, alpha=0.6, label=None)  # Kein Label für Cluster
        
        # Titel und dynamische Achsenbeschriftungen
        axDBSCAN.set_title("DBSCAN Clustering")
        axDBSCAN.set_xlabel(x_label)
        axDBSCAN.set_ylabel(y_label)
        axDBSCAN.legend(loc='best', handletextpad=0.25)  # Legende hinzufügen

        canvasDBSCAN.draw()
        
    except Exception as e:
        print(f"Fehler beim Laden der Datei: {e}")


# ▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉
# ILS Funktionen
# ▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉

# ▉▉▉▉▉▉ R-min ▉▉▉▉▉▉

# Globale Variable für die Matplotlib-Figur
figRmin, axRmin = plt.subplots(figsize=(plot_width / 100, plot_height / 100))
figRmin.subplots_adjust(bottom=0.18, left=0.16)


# Funktion zum Anzeigen eines leeren Plots für R-min
def show_empty_plot_rmin():
    """
    Zeigt einen leeren Plot mit dem Hinweistext "R-min" an, wenn keine Daten geladen sind.
    """
    axRmin.clear() 
    axRmin.axis("off")
    axRmin.text(0.45, 0.45, "R-min", horizontalalignment='center', verticalalignment='center',
                fontsize=12, transform=axData.transAxes)
    canvasRmin.draw()
    
    
# ▉▉▉▉▉▉ ILS ▉▉▉▉▉▉

# Globale Variable für die Matplotlib-Figur
figILS, axILS = plt.subplots(figsize=(plot_width / 100, plot_height / 100))
figILS.subplots_adjust(bottom=0.18, left=0.16)


# Funktion zum Anzeigen eines leeren Plots für ILS
def show_empty_plot_ils():
    """
    Zeigt einen leeren Plot mit dem Hinweistext "ILS" an, wenn keine Daten geladen sind.
    """
    axILS.clear()  
    axILS.axis("off")
    axILS.text(0.45, 0.45, "ILS", horizontalalignment='center', verticalalignment='center',
               fontsize=12, transform=axData.transAxes)
    canvasILS.draw()


# ILS-Algorithmus für iteratives Labeling
def ILS(df, labelColumn, featureColumns, outColumn='LS', iterative=True):
    """
    Führt den Iterative Labelling Spread (ILS) Algorithmus aus, um unlabelled Datenpunkte
    zu klassifizieren.
    Args:
    - df: DataFrame mit den Daten
    - labelColumn: Spalte mit den Labels
    - featureColumns: Liste der Features für die Distanzberechnung
    - outColumn: Name der Ausgabespalte für neue Labels
    - iterative: Gibt an, ob der Algorithmus iterativ ausgeführt werden soll
    Returns:
    - newLabels: DataFrame mit neuen Labels
    - orderedL: DataFrame mit Distanzen und nächsten Nachbarn
    """
    indexNames = list(df.index.names)
    oldIndex = df.index
    df = df.reset_index(drop=False)

    # Trennen von gelabelten und ungelabelten Datenpunkten
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

        # Update Label des nächsten unlabelled Punktes
        unlabelled.loc[idUnL, labelColumn] = labelled.loc[idL, labelColumn]
        labelled = pd.concat([labelled, unlabelled.loc[[idUnL]]])  # Hinzufügen zu labelled
        unlabelled.drop(idUnL, inplace=True)  # Entfernen aus unlabelled

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


# Funktion zum Plotten von Rmin-Werten in dem GUI
def plot_Rmin_gui(minR, canvas):
    """
    Plottet die Rmin-Werte (minimaler Abstand) in einem GUI-Canvas.
    Args:
    - minR: Liste der minimalen Abstände
    - canvas: GUI-Canvas zum Rendern des Plots
    """
    axRmin.clear()  
    axRmin.plot(range(len(minR)), minR, color='blue', label='Rmin')
    axRmin.set_xlabel('Data Points')
    axRmin.set_ylabel('Rmin')
    axRmin.set_title('Rmin Plot (Minimal Distances)')
    canvas.draw()  


# Automatische Bestimmung der optimalen Anzahl von Clustern basierend auf den Spikes im Rmin-Plot
def determine_optimal_clusters(minR):
    """
    Bestimmt die optimale Anzahl von Clustern anhand von Peaks im Rmin-Plot.
    Args:
    - minR: Liste der minimalen Abstände
    Returns:
    - num_clusters: Optimale Anzahl von Clustern
    """
    peaks, _ = find_peaks(minR, distance=10, prominence=0.15)  # Distance und prominence für Spikes 
    num_clusters = len(peaks) + 1  # Anzahl der Peaks plus 1 ergibt die Cluster-Anzahl
    return num_clusters


# Führe KMeans-Clustering aus und visualisiere die Ergebnisse
def kMeans_success(df, featureColumns, canvasILS, canvasRmin):
    """
    Führt das KMeans-Clustering durch und visualisiert die Ergebnisse.
    Args:
    - df: DataFrame mit den Daten
    - featureColumns: Liste der zu verwendenden Features
    - canvasILS: Canvas für die ILS-Visualisierung
    - canvasRmin: Canvas für die Rmin-Visualisierung
    """
    df['label'] = 0
    
    if len(featureColumns) > 2:
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(df[featureColumns])
        df_pca = pd.DataFrame(reduced_features, columns=['PCA1', 'PCA2'], index=df.index)
        featureColumns = ['PCA1', 'PCA2']
    else:
        df_pca = df[featureColumns].copy()

    # Führe KMeans mit einem Cluster aus
    model = KMeans(n_clusters=1, random_state=0, n_init=10).fit(df_pca[featureColumns])
    df_pca['kMean'] = model.labels_ + 1

    for label, group in df_pca.groupby(by='kMean'):
        group = group.copy()
        group['label'] = 0
        centroid = model.cluster_centers_[label-1]
        group.loc[min_toCentroid(group[featureColumns], centroid), 'label'] = label

        newL, orderedL = ILS(group, 'label', featureColumns)

        # Plot Rmin im GUI
        plot_Rmin_gui(orderedL['minR'].values, canvasRmin)

        num_clusters = determine_optimal_clusters(orderedL['minR'].values)

    # Führe KMeans mit der optimalen Cluster-Anzahl durch
    model_optimal = KMeans(n_clusters=num_clusters, random_state=0, n_init=10).fit(df_pca[featureColumns])
    df_pca['kMean_optimal'] = model_optimal.labels_ + 1

    # Zeichne den KMeans-Plot
    axILS.clear()  # Lösche den vorherigen Plot
    axILS.scatter(df_pca[featureColumns[0]].values, df_pca[featureColumns[1]].values, 
                               s=6, alpha=0.6, c=df_pca['kMean_optimal'].values, cmap='viridis')
    axILS.set_title(f'ILS Clustering with {num_clusters} Clusters')
    axILS.set_xlabel(featureColumns[0])
    axILS.set_ylabel(featureColumns[1])
    
    # Zeichne Zentroiden
    centroids = model_optimal.cluster_centers_
    axILS.scatter(centroids[:, 0], centroids[:, 1], s=20, c='red', marker='x', label='Centroids')
    axILS.legend(handletextpad=0.25)
    
    # Aktualisiere den Canvas
    canvasILS.draw()


# Bestimmt den Punkt, der dem Schwerpunkt am nächsten liegt
def min_toCentroid(df, centroid=None, features=None):
    """
    Findet den Punkt, der dem gegebenen Schwerpunkt (Centroid) am nächsten liegt.
    Args:
    - df: DataFrame mit den Punkten
    - centroid: Koordinaten des Schwerpunkts
    - features: Liste der zu verwendenden Features
    Returns:
    - Index des nächsten Punktes
    """
    if type(features) == type(None):
        features = df.columns

    if type(centroid) == type(None):
        centroid = df[features].mean()

    dist = df.apply(lambda row: sum([(row[j] - centroid[i])**2 for i, j in enumerate(features)]), axis=1)
    return dist.idxmin()


# Funktion zum Laden und Vorverarbeiten einer CSV-Datei
def load_and_process_csv(file_path):
    """
    Lädt eine CSV-Datei und skaliert die numerischen Daten.
    Args:
    - file_path: Pfad zur CSV-Datei
    Returns:
    - df: DataFrame mit den Daten
    - featureColumns: Liste der numerischen Features
    """
    df = pd.read_csv(file_path)
    featureColumns = df.select_dtypes(include=[np.number]).columns.tolist()
    scaler = MinMaxScaler()
    df[featureColumns] = scaler.fit_transform(df[featureColumns])
    df.index.name = 'ID'
    return df, featureColumns


# Compute-Button-Funktion zur Ausführung von ILS und Rmin
def compute_ils_rmin():
    """
    Führt die ILS- und Rmin-Berechnung durch, basierend auf den hochgeladenen Daten.
    """
    file_path = data_input_entry.get()
    df, feature_columns = load_and_process_csv(file_path)
    kMeans_success(df, feature_columns, canvasILS, canvasRmin)


# ▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉
# GUI Elemente
# ▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉

# =============================================================================
# Data Input Frame
# =============================================================================

data_input_frame = ctk.CTkFrame(root, fg_color=frame_color)  # Frame für Dateieingabe
data_input_frame.grid(row=0, column=0, padx=6, pady=6, sticky = 'nsew')

entry_var = StringVar()  # StringVar für das Entry, um Änderungen nachzuverfolgen
entry_var.trace_add("write", on_entry_change)  # Überwache Änderungen im Entry-Feld

# Label für die Eingabe
data_input_label = ctk.CTkLabel(data_input_frame, text = "Data", font=headline_font)
data_input_label.grid(row=0, column=0, padx=10, pady=10, sticky = 'w')

# Eingabefeld für den Dateipfad
data_input_entry = ctk.CTkEntry(data_input_frame, textvariable=entry_var, placeholder_text = "filepath",
                                font=entry_font, fg_color=entry_color, border_width=0)
data_input_entry.grid(row=1, column=0, padx=10, pady=10, sticky = "ew")

# Button zum Öffnen des Dateidialogs
data_input_button = ctk.CTkButton(data_input_frame, text="📂", width=30, command=open_file_dialog,
                                  font=label_font, fg_color=button_color)
data_input_button.grid(row=1, column=1, padx=5, pady=5, sticky = 'e')

# Matplotlib Canvas in das Tkinter-Fenster einfügen
canvasData = FigureCanvasTkAgg(figData, data_input_frame)
canvasData.get_tk_widget().grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

# Leeren Plot anzeigen beim Start
show_empty_plot_data()


# =============================================================================
# K-Means Input Frame
# =============================================================================

# Frame für K-Means-Eingabe
kmeans_frame = ctk.CTkFrame(root, fg_color=frame_color) 
kmeans_frame.grid(row=0, column=1, padx=6, pady=6, sticky = 'nsew')

# Label für K-Means
kmeans_label = ctk.CTkLabel(kmeans_frame, text = "K-Means", font=headline_font)
kmeans_label.grid(row=0, column=0, padx=10, pady=10, sticky = 'w')

# Button zum Berechnen des K-Means-Clusters
compute_button = ctk.CTkButton(kmeans_frame, text="Compute", width=30, command=kmeans, font=label_font,
                               fg_color=button_color)
compute_button.grid(row=3, column=2, padx=10, pady=10, sticky = 'w')

# Button zum Zurücksetzen des Plots
reset_button = ctk.CTkButton(kmeans_frame, text="Reset", width=30, command=show_empty_plot_kmeans, font=label_font,
                             fg_color=button_color)
reset_button.grid(row=3, column=3, padx=10, pady=10, sticky = 'w')

# Label und Eingabe für die Anzahl der Cluster
k_parameter_label = ctk.CTkLabel(kmeans_frame, text = "cluster quantity k", font=label_font)
k_parameter_label.grid(row=1, column=2, padx=5, pady=5, sticky = 'w')

# Eingabefeld für k
vcmd = (root.register(validate_input), "%P")
k_entry = ctk.CTkEntry(kmeans_frame, textvariable=k_var, width=40, height=30,
                       validate="key", validatecommand=vcmd,
                       justify="center", font=entry_font, fg_color=entry_color, border_width=0)
k_entry.grid(row=1, column=3, padx=10, pady=10)

# Matplotlib Canvas in das Tkinter-Fenster einfügen
canvasKmeans = FigureCanvasTkAgg(figKmeans, kmeans_frame)
canvasKmeans.get_tk_widget().grid(row=2, column=2, columnspan=2, padx=10, pady=10, sticky="nsew")

# Leeren Plot anzeigen beim Start
show_empty_plot_kmeans()


# ======Elbow Method======

elbow_button = ctk.CTkButton(kmeans_frame, text="Compute", width=30, command=elbow_method, font=label_font,
                             fg_color=button_color)
elbow_button.grid(row=3, column=0, padx=10, pady=10, sticky = 'w')

elbow_reset_button = ctk.CTkButton(kmeans_frame, text="Reset", width=30, command=show_empty_plot_elbow,
                                   font=label_font, fg_color=button_color)
elbow_reset_button.grid(row=3, column=1, padx=10, pady=10, sticky = 'w')

# Label und Eingabe für max k
k_parameter_label = ctk.CTkLabel(kmeans_frame, text = "max. k", font=label_font)
k_parameter_label.grid(row=1, column=0, padx=10, pady=10, sticky = 'w')

# Eingabefeld für max k
vcmd_elbow = (root.register(validate_elbow), "%P")
k_entry = ctk.CTkEntry(kmeans_frame, textvariable=elbow_var, width=40, height=30, validate="key",
                       validatecommand=vcmd_elbow, justify="center", font=entry_font,
                       fg_color=entry_color, border_width=0)
k_entry.grid(row=1, column=1, padx=10, pady=10)

# Matplotlib Canvas in das Tkinter-Fenster einfügen
canvasElbow = FigureCanvasTkAgg(figElbow, kmeans_frame)
canvasElbow.get_tk_widget().grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

# Leeren Plot anzeigen beim Start
show_empty_plot_elbow()

# =============================================================================
# DBSCAN Input Frame
# =============================================================================

# Frame für DBSCAN-Eingabe
dbscan_frame = ctk.CTkFrame(root, fg_color=frame_color)
dbscan_frame.grid(row=1, column=0, padx=6, pady=6, sticky = 'nsew')

# Label für DBSCAN
dbscan_label = ctk.CTkLabel(dbscan_frame, text = "DBSCAN", font=headline_font)
dbscan_label.grid(row=0, column=0, padx=10, pady=10, sticky = 'w')

# Eingabefelder für Epsilon und Min-Samples
epsilon_parameter_label = ctk.CTkLabel(dbscan_frame, text = "min. Points in r", font=label_font)
epsilon_parameter_label.grid(row=1, column=0, padx=10, pady=10, sticky = 'w')

vcmd_epsilon = (root.register(validate_input_epsilon), "%P")
epsilon_entry = ctk.CTkEntry(dbscan_frame, textvariable=epsilon_var, width=40, height=30,
                             validate="key", validatecommand=vcmd_epsilon, justify="center",
                             font=entry_font, fg_color=entry_color, border_width=0)
epsilon_entry.grid(row=1, column=1, padx=10, pady=10, sticky = 'e')

samples_parameter_label = ctk.CTkLabel(dbscan_frame, text = "radius ε", font=label_font)
samples_parameter_label.grid(row=1, column=1, padx=10, pady=10, sticky = 'w')

vcmd_samples = (root.register(validate_input_min_samples), "%P")
samples_entry = ctk.CTkEntry(dbscan_frame, textvariable=min_samples_var, width=40, height=30,
                             validate="key", validatecommand=vcmd_samples, justify="center",
                             font=entry_font, fg_color=entry_color, border_width=0)
samples_entry.grid(row=1, column=0, padx=10, pady=10, sticky = 'e')

# Matplotlib Canvas in das Tkinter-Fenster einfügen
canvasDBSCAN = FigureCanvasTkAgg(figDBSCAN, dbscan_frame)
canvasDBSCAN.get_tk_widget().grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

# Buttons für DBSCAN
compute_button = ctk.CTkButton(dbscan_frame, text="Compute", width=30, command=dbscan,
                               font=label_font, fg_color=button_color)
compute_button.grid(row=4, column=0, padx=10, pady=(1 ,60), sticky = 'w')
reset_button = ctk.CTkButton(dbscan_frame, text="Reset", width=30, command=show_empty_plot_dbscan,
                             font=label_font, fg_color=button_color)
reset_button.grid(row=4, column=1, padx=5, pady=(1, 60), sticky = 'w')

# Leeren Plot anzeigen beim Start
show_empty_plot_dbscan()


# =============================================================================
# ILS Input Frame
# =============================================================================

# Frame für ILS-Eingabe
ils_frame = ctk.CTkFrame(root, fg_color=frame_color)
ils_frame.grid(row=1, column=1, padx=6, pady=6, sticky = 'nsew')

# Label für ILS
ils_label = ctk.CTkLabel(ils_frame, text = "ILS", font=headline_font)
ils_label.grid(row=0, column=0, padx=10, pady=10, sticky = 'w')

# Buttons für ILS
compute_button = ctk.CTkButton(ils_frame, text="Compute", width=30, command=compute_ils_rmin,
                               font=label_font, fg_color=button_color)
compute_button.grid(row=2, column=2, padx=10, pady=10, sticky = 'w')
reset_button = ctk.CTkButton(ils_frame, text="Reset", width=30, command=show_empty_plot_ils,
                             font=label_font, fg_color=button_color)
reset_button.grid(row=2, column=3, padx=10, pady=10, sticky = 'w')


# Canvas für Rmin-Plot
canvasRmin = FigureCanvasTkAgg(figRmin, ils_frame)
canvasRmin.get_tk_widget().grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
show_empty_plot_rmin()

# Canvas für ILS-Plot
canvasILS = FigureCanvasTkAgg(figILS, ils_frame)
canvasILS.get_tk_widget().grid(row=1, column=2, columnspan=2, padx=10, pady=10, sticky="nsew")
show_empty_plot_ils()


# =============================================================================
# Layout und Fensterkonfiguration
# =============================================================================

# Layout-Konfiguration für flexibles Größenanpassen
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=2)
root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)


# Fenstergröße an Bildschirmauflösung anpassen
resize_window_to_screen(root, scale=0.6)
                                                        

# Nach dem Hinzufügen aller Widgets die minimale Fenstergröße festlegen
set_minimum_window_size(root)


# Hauptloop der root starten
root.mainloop()
