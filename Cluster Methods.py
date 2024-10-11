import customtkinter as ctk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
from sklearn.cluster import KMeans

plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle')

# =============================================================================
# Hauptfenster wird erstellt (für GUI)
# =============================================================================

root = ctk.CTk()
root.title("ClusterMethods")
root.geometry("1000x600")
root.minsize(width = 500, height = 300)

# Globale Variable für die Matplotlib-Figur
fig, ax = plt.subplots(figsize=(5, 4))

# ▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉
# DataInput Funktionen
# ▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉

# Funktion zum Anzeigen des leeren Plots
def show_empty_plot():
    ax.clear()  # Leeren der Figur für neuen Plot
    ax.set_title("Empty Plot")
    ax.set_xlim(0, 10)  # X-Achse von 0 bis 10
    ax.set_ylim(0, 10)  # Y-Achse von 0 bis 10
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.plot([], [])  # Keine Datenpunkte
    canvas.draw()
    
# Funktion zum Zurücksetzen des Plots
def reset_plot():
    show_empty_plot()

# Datei auswählen
def open_file_dialog():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt")])
    data_input_entry.delete(0, ctk.END)  # Alten Text im Entry löschen
    data_input_entry.insert(0, file_path)  # Pfad in Entry einfügen
    
# Funktion, um die Eingabe auf eine dreistellige Zahl zu beschränken
def validate_input(new_value):
    if new_value == "":  # Erlaubt leere Eingaben (falls der User löscht)
        return True
    if new_value.isdigit() and len(new_value) <= 3:  # Nur Ziffern und maximal 3 Stellen
        return True
    return False
    
# ▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉
# K-Means Funktionen
# ▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉

# ▉▉▉▉▉▉K-MEANS▉▉▉▉▉▉

def compute_kmeans():
    file_path = data_input_entry.get()  # Pfad zur Datei aus Entry
    try:
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
        kmeans = KMeans(n_clusters=3)  # 3 Cluster als Beispiel
        kmeans.fit(X)
        y_kmeans = kmeans.predict(X)
        
        # Plot aktualisieren
        ax.clear()
        ax.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
        ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                   s=200, c='red', label='Centroids')
        ax.set_title("K-Means Clustering")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.legend()
        canvas.draw()
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

data_input_label = ctk.CTkLabel(data_input_frame, text = "Data")
data_input_label.grid(row=0, column=0, padx=5, pady=5, sticky = 'w')
data_input_entry = ctk.CTkEntry(data_input_frame, placeholder_text = "filepath")
data_input_entry.grid(row=1, column=0, padx=5, pady=5, sticky = "ew")
data_input_button = ctk.CTkButton(data_input_frame, text="", width=30, command=open_file_dialog)
data_input_button.grid(row=1, column=1, padx=5, pady=5, sticky = 'e')

# =============================================================================
# K-Means Input Frame
# =============================================================================

kmeans_frame = ctk.CTkFrame(root)
kmeans_frame.grid(row=2, column=0, padx=6, pady=6)
kmeans_label = ctk.CTkLabel(kmeans_frame, text = "K-Means")
kmeans_label.grid(row=0, column=0, padx=5, pady=5, sticky = 'w')
compute_button = ctk.CTkButton(kmeans_frame, text="Compute", width=30, command=compute_kmeans)
compute_button.grid(row=3, column=0, padx=5, pady=5, sticky = 'w')
reset_button = ctk.CTkButton(kmeans_frame, text="Reset", width=30, command=show_empty_plot)
reset_button.grid(row=3, column=1, padx=5, pady=5, sticky = 'w')
k_parameter_label = ctk.CTkLabel(kmeans_frame, text = "cluster quantity k")
k_parameter_label.grid(row=1, column=0, padx=5, pady=5, sticky = 'w')
# Validierungsfunktion registrieren
vcmd = (root.register(validate_input), "%P")
k_entry = ctk.CTkEntry(kmeans_frame, width=40, height=30, validate="key", validatecommand=vcmd, justify="center")
k_entry.grid(row=1, column=1, padx=6, pady=6)

# Matplotlib Canvas in das Tkinter-Fenster einfügen
canvas = FigureCanvasTkAgg(fig, kmeans_frame)
canvas.get_tk_widget().grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

# Layout-Konfiguration für flexibles Größenanpassen
root.grid_columnconfigure(0, weight=1)
root.grid_rowconfigure(2, weight=1)

# Leeren Plot anzeigen beim Start
show_empty_plot()

# Hauptloop der root starten
root.mainloop()
