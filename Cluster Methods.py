import customtkinter as ctk
from tkinter import filedialog, StringVar
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
from sklearn.cluster import KMeans
import os

plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle')

# =============================================================================
# Hauptfenster wird erstellt (für GUI)
# =============================================================================

root = ctk.CTk()
root.title("ClusterMethods")
root.geometry("1000x600")
root.minsize(width = 500, height = 300)

# ▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉
# Data Input Funktionen
# ▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉

# Globale Variable für die Matplotlib-Figur
figData, axData = plt.subplots(figsize=(5, 4))

# Funktion zum Anzeigen des leeren Plots
def show_empty_plot_data():
    axData.clear()  # Leeren der Figur für neuen Plot
    axData.set_title("Empty Plot")
    axData.set_xlim(0, 10)  # X-Achse von 0 bis 10
    axData.set_ylim(0, 10)  # Y-Achse von 0 bis 10
    axData.set_xlabel("X-axis")
    axData.set_ylabel("Y-axis")
    axData.plot([], [])  # Keine Datenpunkte
    canvasData.draw()
    
# Datei auswählen
def open_file_dialog():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt")])
    entry_var.set(file_path)  # Den ausgewählten Dateipfad ins Entry setzen
    plot_data_points()  # Daten für den ersten Plot automatisch laden

# Funktion, die ausgeführt wird, wenn sich der Text im Entry ändert
def on_entry_change(*args):
    plot_data_points()  # Immer, wenn sich der Dateipfad ändert, wird der Plot aktualisiert

# ▉▉▉▉▉▉Raw Data Plot▉▉▉▉▉▉

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
figKmeans, axKmeans = plt.subplots(figsize=(5, 4))

# Funktion zum Anzeigen des leeren Plots
def show_empty_plot_kmeans():
    axKmeans.clear()  # Leeren der Figur für neuen Plot
    axKmeans.set_title("Empty Plot")
    axKmeans.set_xlim(0, 10)  # X-Achse von 0 bis 10
    axKmeans.set_ylim(0, 10)  # Y-Achse von 0 bis 10
    axKmeans.set_xlabel("X-axis")
    axKmeans.set_ylabel("Y-axis")
    axKmeans.plot([], [])  # Keine Datenpunkte
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

# ▉▉▉▉▉▉K-MEANS▉▉▉▉▉▉

def kmeans():
    file_path = data_input_entry.get()  # Pfad zur Datei aus Entry
    try:
        # Den Wert von k aus dem Entry lesen
        k = int(k_var.get())
        
        # Datei laden (CSV oder TXT)
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path, delimiter=',', header=0)  # Komma als Trennzeichen
        elif file_path.endswith('.txt'):
            data = pd.read_csv(file_path, delimiter='\t', header=0)  # Komma als Trennzeichen
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
# GUI Elemente
# ▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉

# =============================================================================
# Data Input Frame
# =============================================================================

data_input_frame = ctk.CTkFrame(root)
data_input_frame.grid(row=0, column=0, padx=6, pady=6)

entry_var = StringVar()  # StringVar für das Entry, um Änderungen nachzuverfolgen
entry_var.trace_add("write", on_entry_change)  # Überwache Änderungen im Entry-Feld

data_input_label = ctk.CTkLabel(data_input_frame, text = "Data")
data_input_label.grid(row=0, column=0, padx=5, pady=5, sticky = 'w')
data_input_entry = ctk.CTkEntry(data_input_frame, textvariable=entry_var, placeholder_text = "filepath")
data_input_entry.grid(row=1, column=0, padx=5, pady=5, sticky = "ew")
data_input_button = ctk.CTkButton(data_input_frame, text="", width=30, command=open_file_dialog)
data_input_button.grid(row=1, column=1, padx=5, pady=5, sticky = 'e')

# Matplotlib Canvas in das Tkinter-Fenster einfügen
canvasData = FigureCanvasTkAgg(figData, data_input_frame)
canvasData.get_tk_widget().grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

# Leeren Plot anzeigen beim Start
show_empty_plot_data()

# =============================================================================
# K-Means Input Frame
# =============================================================================

kmeans_frame = ctk.CTkFrame(root)
kmeans_frame.grid(row=2, column=0, padx=6, pady=6)
kmeans_label = ctk.CTkLabel(kmeans_frame, text = "K-Means")
kmeans_label.grid(row=0, column=0, padx=5, pady=5, sticky = 'w')
compute_button = ctk.CTkButton(kmeans_frame, text="Compute", width=30, command=kmeans)
compute_button.grid(row=3, column=0, padx=5, pady=5, sticky = 'w')
reset_button = ctk.CTkButton(kmeans_frame, text="Reset", width=30, command=show_empty_plot_kmeans)
reset_button.grid(row=3, column=1, padx=5, pady=5, sticky = 'w')
k_parameter_label = ctk.CTkLabel(kmeans_frame, text = "cluster quantity k")
k_parameter_label.grid(row=1, column=0, padx=5, pady=5, sticky = 'w')
# Validierungsfunktion registrieren
vcmd = (root.register(validate_input), "%P")
k_entry = ctk.CTkEntry(kmeans_frame, textvariable=k_var, width=40, height=30, validate="key", validatecommand=vcmd, justify="center")
k_entry.grid(row=1, column=1, padx=6, pady=6)

# Matplotlib Canvas in das Tkinter-Fenster einfügen
canvasKmeans = FigureCanvasTkAgg(figKmeans, kmeans_frame)
canvasKmeans.get_tk_widget().grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

# Leeren Plot anzeigen beim Start
show_empty_plot_kmeans()







# Layout-Konfiguration für flexibles Größenanpassen
root.grid_columnconfigure(0, weight=1)
root.grid_rowconfigure(2, weight=1)


# Hauptloop der root starten
root.mainloop()
