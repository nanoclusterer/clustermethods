import customtkinter as ctk
from tkinter import filedialog, StringVar
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import os
import seaborn as sns

# Globale Seaborn-Parameter setzen
sns.set_context("notebook")
sns.set_style("darkgrid")

# Definieren der Parameter
plt.rcParams.update({
    'legend.frameon': False,
    'legend.numpoints': 1,
    'legend.scatterpoints': 1,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'axes.axisbelow': True,
    'font.family': 'sans-serif',
    'grid.linestyle': '-',
    'lines.solid_capstyle': 'round',
    'axes.grid': True,
    'axes.edgecolor': 'white',
    'axes.linewidth': 0,
    'xtick.major.size': 0,
    'ytick.major.size': 0,
    'xtick.minor.size': 0,
    'ytick.minor.size': 0,
    'text.color': '0.9',  # Textfarbe
    'axes.labelcolor': '0.9',  # Achsenbeschriftungsfarbe
    'xtick.color': '0.9',  # X-Achsenfarbe
    'ytick.color': '0.9',  # Y-Achsenfarbe
    'grid.color': '#2A3459',  # Gitterfarbe
    'font.sans-serif': ['Overpass', 'Helvetica', 'Helvetica Neue', 'Arial', 'Liberation Sans', 'DejaVu Sans', 'Bitstream Vera Sans', 'sans-serif'],
    'axes.prop_cycle': plt.cycler(color=['#18c0c4', '#f62196', '#A267F5', '#f3907e', '#ffe46b', '#fefeff']),
    'image.cmap': 'RdPu',
    'figure.facecolor': '#2A3459',
    'axes.facecolor': '#212946',
    'savefig.facecolor': '#2A3459'
})

#plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-light.mplstyle')

root_color = "#1D243D"
frame_color = "#212946"
button_color = "#2A3459"
entry_color = "#2A3459"

# =============================================================================
# Hauptfenster wird erstellt (für GUI)
# =============================================================================

root = ctk.CTk(fg_color=root_color)
root.title("ClusterMethods")

# ▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉
# Fenster Funktionen (root)
# ▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉

def set_minimum_window_size(root):
    # Update the app layout to calculate the required size
    root.update()

    # Get the width and height needed for the app's content
    required_width = root.winfo_reqwidth()
    required_height = root.winfo_reqheight()

    # Set the minimum size of the window to this size
    root.minsize(required_width, required_height)

def resize_window_to_screen(root, scale):
    """
    Passt die Fenstergröße basierend auf der Bildschirmauflösung an.
    
    Args:
    - root: Die Hauptanwendung (CTk-Fenster).
    - scale: Skalierungsfaktor. Standard ist 80% der Bildschirmgröße.
    """
    # Bildschirmauflösung ermitteln
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Fenstergröße basierend auf dem Skalierungsfaktor festlegen
    window_width = int(screen_width * scale)
    window_height = int(screen_height * scale)

    # Fenstergröße setzen und das Fenster mittig auf dem Bildschirm platzieren
    root.geometry(f"{window_width}x{window_height}+{int((screen_width - window_width) / 2)}+{int((screen_height - window_height) / 2)}")
    
    
# Funktion, um die Textgröße und Plot-Größe basierend auf der Bildschirmauflösung anzupassen
def adjust_text_and_plot_size():
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    
    # Berechnung der Textgröße basierend auf der Bildschirmauflösung
    headline_text_size = int(min(screen_width, screen_height) / 80)
    label_text_size = int(min(screen_width, screen_height) / 100)
    entry_text_size = int(min(screen_width, screen_height) / 120)

    # Feste Plot-Größe mit einem Seitenverhältnis von 5:3
    plot_width = screen_width * 0.3  # z.B. 30% der Bildschirmbreite
    plot_height = plot_width * 3 / 5  # Höhe basierend auf 5:3 Verhältnis
    
    return headline_text_size, label_text_size, entry_text_size, plot_width, plot_height

# Anpassung der Textgröße
headline_text_size, label_text_size, entry_text_size, plot_width, plot_height = adjust_text_and_plot_size()


# ▉▉▉▉▉▉ Custom Fonts ▉▉▉▉▉▉

headline_font = ctk.CTkFont(family="Segoe UI", size=headline_text_size, weight='bold')
label_font = ctk.CTkFont(family="Segoe UI", size=label_text_size)
entry_font = ctk.CTkFont(family="Segoe UI", size=entry_text_size)


# ▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉
# Data Input Funktionen
# ▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉

# Globale Variable für die Matplotlib-Figur
figData, axData = plt.subplots(figsize=(plot_width / 100, plot_height / 100))

def show_empty_plot_data():
    axData.clear()  # Leeren der Figur für neuen Plot
    axData.axis("off")
    axData.text(0.5, 0.5, "No Data loaded", horizontalalignment='center', verticalalignment='center', fontsize=12, transform=axData.transAxes)
    canvasData.draw()

# Datei auswählen
def open_file_dialog():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt")])
    entry_var.set(file_path)  # Den ausgewählten Dateipfad ins Entry setzen
    plot_data_points()  # Daten für den ersten Plot automatisch laden

# Funktion, die ausgeführt wird, wenn sich der Text im Entry ändert
def on_entry_change(*args):
    plot_data_points()  # Immer, wenn sich der Dateipfad ändert, wird der Plot aktualisiert

# ▉▉▉▉▉▉ Raw Data Plot ▉▉▉▉▉▉

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
            axData.scatter(X[:, 0], X[:, 1], c='blue', s=50, label='Data Points')
            axData.set_title("Datenpunkte")
            axData.set_xlabel("X-axis")
            axData.set_ylabel("Y-axis")
            axData.legend()
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

# Funktion zum Anzeigen des leeren Plots
def show_empty_plot_kmeans():
    axKmeans.clear()  # Leeren der Figur für neuen Plot
    axKmeans.axis("off")
    axKmeans.text(0.5, 0.5, "KMEANS", horizontalalignment='center', verticalalignment='center', fontsize=12, transform=axData.transAxes)
    canvasKmeans.draw()

# Funktion, um die Eingabe für k auf eine dreistellige Zahl zu beschränken
def validate_input(new_value):
    if new_value == "":  # Erlaubt leere Eingaben (falls der User löscht)
        return True
    if new_value.isdigit() and len(new_value) <= 3 and int(new_value) > 1:  # Nur Ziffern, max. 3 Stellen, kein 1
        return True
    return False

# Validierung der Eingabe für das k-Entry
k_var = StringVar()
k_var.trace_add("write", lambda *args: validate_input(k_var.get()))

# ▉▉▉▉▉▉K-Means▉▉▉▉▉▉

def kmeans():
    file_path = data_input_entry.get()  # Pfad zur Datei aus Entry
    try:
        # Den Wert von k aus dem Entry lesen
        k = int(k_var.get())
        
        # Datei laden (CSV oder TXT)
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path, delimiter=',', header=0)  # Komma als Trennzeichen
        elif file_path.endswith('.txt'):
            data = pd.read_csv(file_path, delimiter='\t', header=0)  # Tab als Trennzeichen
        else:
            raise ValueError("Bitte eine .csv oder .txt Datei auswählen.")
        
        # Überprüfen, ob Daten erfolgreich geladen wurden
        print(f'Daten geladen: {data.shape} Zeilen, {data.columns} Spalten')

        # Vorausgesetzt, dass die Datei zwei Spalten für X und Y enthält
        X = data.values  # Den gesamten Inhalt als Array nehmen
        print(f'Daten für K-Means: {X[:5]}')  # Ersten 5 Zeilen ausgeben

        # K-Means Clustering anwenden
        kmeans = KMeans(n_clusters=k-1)  # 3 Cluster als Beispiel
        kmeans.fit(X)
        y_kmeans = kmeans.predict(X)
        
        # Plot aktualisieren
        axKmeans.clear()
        axKmeans.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
        axKmeans.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                   s=200, c='red', label='Centroids')
        axKmeans.set_title(f"K-Means Clustering with k={k}")
        axKmeans.set_xlabel("X-axis")
        axKmeans.set_ylabel("Y-axis")
        axKmeans.legend()
        canvasKmeans.draw()
    except Exception as e:
        print(f"Fehler beim Laden der Datei: {e}")
        
# ▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉
# Elbow Funktionen
# ▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉

# Globale Variable für die Matplotlib-Figur
figElbow, axElbow = plt.subplots(figsize=(plot_width / 100, plot_height / 100))

# Funktion zum Anzeigen des leeren Plots
def show_empty_plot_elbow():
    axElbow.clear()  # Leeren der Figur für neuen Plot
    axElbow.axis("off")
    axElbow.text(0.5, 0.5, "Elbow", horizontalalignment='center', verticalalignment='center', fontsize=12, transform=axData.transAxes)
    canvasElbow.draw()

# Funktion, um die Eingabe für epsilon und min_samples auf gültige Werte zu beschränken
def validate_elbow(new_value):
    if new_value == "":  # Erlaubt leere Eingaben (falls der User löscht)
        return True
    if new_value.isdigit() and len(new_value) <= 3 and int(new_value) > 1:  # Nur Ziffern, max. 3 Stellen, grösser als 4
        return True
    return False

# Validierung der Eingabe für das k-Entry
elbow_var = StringVar()
elbow_var.trace_add("write", lambda *args: validate_input(elbow_var.get()))

# ▉▉▉▉▉▉Elbow Method▉▉▉▉▉▉
def elbow_method():
    """
    Führt die Elbow-Methode auf den Daten durch und plottet die WCSS für verschiedene Werte von k.
    
    Args:
    - X: Die Daten, die geclustert werden sollen (2D-Liste oder NumPy-Array)
    - max_k: Die maximale Anzahl von Clustern, die getestet werden soll (Standard: 10)
    """
    # Den Wert von k max aus dem Entry lesen
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
        
        # Überprüfen, ob Daten erfolgreich geladen wurden
        print(f'Daten geladen: {data.shape[0]} Zeilen, {data.shape[1]} Spalten')

        # Vorausgesetzt, dass die Datei zwei Spalten für X und Y enthält
        X = data.values  # Den gesamten Inhalt als Array nehmen
        print(f'Daten für K-Means: {X[:5]}')  # Ersten 5 Zeilen ausgeben
    
        wcss = []
    
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
        axElbow.grid(True)
        canvasElbow.draw()
        
    except Exception as e:
        print(f"Fehler beim Laden der Datei: {e}")

# ▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉
# DBSCAN Funktionen
# ▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉

# Globale Variable für die Matplotlib-Figur
figDBSCAN, axDBSCAN = plt.subplots(figsize=(plot_width / 100, plot_height / 100))

# Funktion zum Anzeigen des leeren Plots
def show_empty_plot_dbscan():
    axDBSCAN.clear()  # Leeren der Figur für neuen Plot
    axDBSCAN.axis("off")
    axDBSCAN.text(0.5, 0.5, "DBSCAN", horizontalalignment='center', verticalalignment='center', fontsize=12, transform=axData.transAxes)
    canvasDBSCAN.draw()


# Funktion, um die Eingabe für epsilon und min_samples auf gültige Werte zu beschränken
def validate_input_epsilon(new_value):
    if new_value == "":  # Leere Eingaben zulassen (z. B. beim Löschen)
        return True
    try:
        # Überprüfen, ob die Eingabe eine gültige Zahl ist und im gewünschten Bereich liegt
        value = float(new_value)
        if 0 <= value < 10000 and len(new_value) <= 6:
            return True
        return False
    except ValueError:
        # Falls die Eingabe nicht in eine Zahl umgewandelt werden kann, ist sie ungültig (z. B. bei Buchstaben)
        return False

epsilon_var = StringVar()
# Validierung für epsilon
epsilon_var.trace_add("write", lambda *args: validate_input_epsilon(epsilon_var.get()))

min_samples_var = StringVar()
# Validierung für min_samples
min_samples_var.trace_add("write", lambda *args: validate_input_min_samples(min_samples_var.get()))


def validate_input_min_samples(new_value):
    if new_value == "" or (new_value.isdigit() and 0 < int(new_value) < 10000):  # min_samples muss > 0 und < 10000 sein
        return True
    return False
    
# ▉▉▉▉▉▉ DBSCAN ▉▉▉▉▉▉

def dbscan():
    file_path = data_input_entry.get()  # Pfad zur Datei aus Entry
    try:
        # Den Wert von Epsilon aus dem Entry lesen
        epsilon = float(epsilon_entry.get())
        # Den Wert von min_samples aus dem Entry lesen
        min_samples = int(samples_entry.get())

        # Datei laden (CSV oder TXT)
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path, delimiter=',', header=0)  # Komma als Trennzeichen
        elif file_path.endswith('.txt'):
            data = pd.read_csv(file_path, delimiter='\t', header=0)  # Tab als Trennzeichen
        else:
            raise ValueError("Bitte eine .csv oder .txt Datei auswählen.")
            
        # Beispielhafte Daten (2D Punkte)
        X = data.values

        # DBSCAN Algorithmus ausführen
        dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
        labels = dbscan.fit_predict(X)

        # Ergebnisse plotten
        axDBSCAN.clear()
        axDBSCAN.set_xlim([0, 10])
        axDBSCAN.set_ylim([0, 10])
    
        # Unterschiedliche Cluster mit unterschiedlichen Farben plotten
        unique_labels = set(labels)
        for label in unique_labels:
            label_mask = labels == label
            axDBSCAN.scatter(X[label_mask, 0], X[label_mask, 1], label=f"Cluster {label}")
    
        axDBSCAN.set_title("DBSCAN Clustering")
        axDBSCAN.legend()
        canvasDBSCAN.draw()
    except Exception as e:
        print(f"Fehler beim Laden der Datei: {e}")
        

# ▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉
# GUI Elemente
# ▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉

# =============================================================================
# Data Input Frame
# =============================================================================

data_input_frame = ctk.CTkFrame(root, fg_color=frame_color)
data_input_frame.grid(row=0, column=0, padx=6, pady=6, sticky = 'nsew')

entry_var = StringVar()  # StringVar für das Entry, um Änderungen nachzuverfolgen
entry_var.trace_add("write", on_entry_change)  # Überwache Änderungen im Entry-Feld

data_input_label = ctk.CTkLabel(data_input_frame, text = "Data", font=headline_font)
data_input_label.grid(row=0, column=0, padx=10, pady=10, sticky = 'w')
data_input_entry = ctk.CTkEntry(data_input_frame, textvariable=entry_var, placeholder_text = "filepath", font=entry_font, fg_color=entry_color, border_width=0)
data_input_entry.grid(row=1, column=0, padx=10, pady=10, sticky = "ew")
data_input_button = ctk.CTkButton(data_input_frame, text="📂", width=30, command=open_file_dialog, font=label_font, fg_color=button_color)
data_input_button.grid(row=1, column=1, padx=5, pady=5, sticky = 'e')

# Matplotlib Canvas in das Tkinter-Fenster einfügen
canvasData = FigureCanvasTkAgg(figData, data_input_frame)
canvasData.get_tk_widget().grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

# Leeren Plot anzeigen beim Start
show_empty_plot_data()

# =============================================================================
# K-Means Input Frame
# =============================================================================

kmeans_frame = ctk.CTkFrame(root, fg_color=frame_color)
kmeans_frame.grid(row=0, column=1, padx=6, pady=6, sticky = 'nsew')
kmeans_label = ctk.CTkLabel(kmeans_frame, text = "K-Means", font=headline_font)
kmeans_label.grid(row=0, column=0, padx=10, pady=10, sticky = 'w')
compute_button = ctk.CTkButton(kmeans_frame, text="Compute", width=30, command=kmeans, font=label_font, fg_color=button_color)
compute_button.grid(row=3, column=2, padx=10, pady=10, sticky = 'w')
reset_button = ctk.CTkButton(kmeans_frame, text="Reset", width=30, command=show_empty_plot_kmeans, font=label_font, fg_color=button_color)
reset_button.grid(row=3, column=3, padx=10, pady=10, sticky = 'w')
k_parameter_label = ctk.CTkLabel(kmeans_frame, text = "cluster quantity k", font=label_font)
k_parameter_label.grid(row=1, column=2, padx=5, pady=5, sticky = 'w')
# Validierungsfunktion registrieren
vcmd = (root.register(validate_input), "%P")
k_entry = ctk.CTkEntry(kmeans_frame, textvariable=k_var, width=40, height=30, validate="key", validatecommand=vcmd, justify="center", font=entry_font, fg_color=entry_color, border_width=0)
k_entry.grid(row=1, column=3, padx=10, pady=10)

# Matplotlib Canvas in das Tkinter-Fenster einfügen
canvasKmeans = FigureCanvasTkAgg(figKmeans, kmeans_frame)
canvasKmeans.get_tk_widget().grid(row=2, column=2, columnspan=2, padx=10, pady=10, sticky="nsew")

# Leeren Plot anzeigen beim Start
show_empty_plot_kmeans()

# ======Elbow Method======
elbow_button = ctk.CTkButton(kmeans_frame, text="Compute", width=30, command=elbow_method, font=label_font, fg_color=button_color)
elbow_button.grid(row=3, column=0, padx=10, pady=10, sticky = 'w')
elbow_reset_button = ctk.CTkButton(kmeans_frame, text="Reset", width=30, command=show_empty_plot_elbow, font=label_font, fg_color=button_color)
elbow_reset_button.grid(row=3, column=1, padx=10, pady=10, sticky = 'w')

k_parameter_label = ctk.CTkLabel(kmeans_frame, text = "max. k", font=label_font)
k_parameter_label.grid(row=1, column=0, padx=10, pady=10, sticky = 'w')
# Validierungsfunktion registrieren
vcmd_elbow = (root.register(validate_elbow), "%P")
k_entry = ctk.CTkEntry(kmeans_frame, textvariable=elbow_var, width=40, height=30, validate="key", validatecommand=vcmd_elbow, justify="center", font=entry_font, fg_color=entry_color, border_width=0)
k_entry.grid(row=1, column=1, padx=10, pady=10)

# Matplotlib Canvas in das Tkinter-Fenster einfügen
canvasElbow = FigureCanvasTkAgg(figElbow, kmeans_frame)
canvasElbow.get_tk_widget().grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

# Leeren Plot anzeigen beim Start
show_empty_plot_elbow()

# =============================================================================
# DBSCAN Input Frame
# =============================================================================

dbscan_frame = ctk.CTkFrame(root, fg_color=frame_color)
dbscan_frame.grid(row=1, column=0, padx=6, pady=6, sticky = 'nsew')
dbscan_label = ctk.CTkLabel(dbscan_frame, text = "DBSCAN", font=headline_font)
dbscan_label.grid(row=0, column=0, padx=10, pady=10, sticky = 'w')

epsilon_parameter_label = ctk.CTkLabel(dbscan_frame, text = "radius ε", font=label_font)
epsilon_parameter_label.grid(row=1, column=0, padx=10, pady=10, sticky = 'w')
# Validierungsfunktion registrieren
vcmd_epsilon = (root.register(validate_input_epsilon), "%P")
epsilon_entry = ctk.CTkEntry(dbscan_frame, textvariable=epsilon_var, width=40, height=30, validate="key", validatecommand=vcmd_epsilon, justify="center", font=entry_font, fg_color=entry_color, border_width=0)
epsilon_entry.grid(row=1, column=1, padx=10, pady=10, sticky = 'e')

samples_parameter_label = ctk.CTkLabel(dbscan_frame, text = "min. Points in r", font=label_font)
samples_parameter_label.grid(row=2, column=0, padx=10, pady=10, sticky = 'w')
# Validierungsfunktion registrieren
vcmd_samples = (root.register(validate_input_min_samples), "%P")
samples_entry = ctk.CTkEntry(dbscan_frame, textvariable=min_samples_var, width=40, height=30, validate="key", validatecommand=vcmd_samples, justify="center", font=entry_font, fg_color=entry_color, border_width=0)
samples_entry.grid(row=2, column=1, padx=10, pady=10, sticky = 'e')

# Matplotlib Canvas in das Tkinter-Fenster einfügen
canvasDBSCAN = FigureCanvasTkAgg(figDBSCAN, dbscan_frame)
canvasDBSCAN.get_tk_widget().grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

compute_button = ctk.CTkButton(dbscan_frame, text="Compute", width=30, command=dbscan, font=label_font, fg_color=button_color)
compute_button.grid(row=4, column=0, padx=10, pady=10, sticky = 'w')
reset_button = ctk.CTkButton(dbscan_frame, text="Reset", width=30, command=show_empty_plot_dbscan, font=label_font, fg_color=button_color)
reset_button.grid(row=4, column=1, padx=5, pady=5, sticky = 'w')

# Leeren Plot anzeigen beim Start
show_empty_plot_dbscan()

# =============================================================================
# ILS Input Frame
# =============================================================================

ils_frame = ctk.CTkFrame(root, fg_color=frame_color)
ils_frame.grid(row=1, column=1, padx=6, pady=6, sticky = 'nsew')
ils_label = ctk.CTkLabel(ils_frame, text = "ILS", font=headline_font)
ils_label.grid(row=0, column=0, padx=10, pady=10, sticky = 'w')




# Layout-Konfiguration für flexibles Größenanpassen
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=2)
root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)

# Fenstergröße an Bildschirmauflösung anpassen (Skalierung z.B. 80% der Bildschirmgröße)
resize_window_to_screen(root, scale=0.7)
"""
                                                                                            ▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉
                                                                                            was passiert nun, wenn die rescalierung kleiner ist,
                                                                                            als die minimale fenstergröße? wie kann man das umgehen?
                                                                                            ▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉
                                                                                            """
# Nach dem Hinzufügen aller Widgets die minimale Fenstergröße festlegen
set_minimum_window_size(root)



# Hauptloop der root starten
root.mainloop()
