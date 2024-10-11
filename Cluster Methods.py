import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import customtkinter as ctk
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
frameData = ctk.CTkFrame(master = root)

# =============================================================================
# Funktionen
# =============================================================================

# Funktion, um die Eingabe auf eine dreistellige Zahl zu beschränken
def validate_input(new_value):
    if new_value == "":  # Erlaubt leere Eingaben (falls der User löscht)
        return True
    if new_value.isdigit() and len(new_value) <= 3:  # Nur Ziffern und maximal 3 Stellen
        return True
    return False

# Funktion, um Dateipfad zu wählen
def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt")])
    data_input_entry.delete(0, ctk.END)
    data_input_entry.insert(0, file_path)
        
# Erstellen der Matplotlib Figur
#fig, ax = plt.Figure(figsize=(5, 4), dpi=100)
fig, ax = plt.subplots(figsize=(5, 4))
# Funktion zum Anzeigen eines leeren Plots
def show_empty_plot():
    fig.clear()  # Leeren der Figur für neuen Plot
    ax = fig.add_subplot(111)
    ax.set_title("Empty Plot")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.plot([], [])  # Keine Datenpunkte für den leeren Plot
    canvas.draw()

    
# ▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉
# Funktionen für die Algorithmen
# ▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉

# ▉▉▉▉▉▉K-MEANS▉▉▉▉▉▉
def compute_kmeans():
    file_path = data_input_entry.get()  # Pfad zur Datei aus Entry
    try:
        # Datei laden (CSV oder TXT)
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
        elif file_path.endswith('.txt'):
            data = pd.read_csv(file_path, delimiter='\t', header=0)
        else:
            raise ValueError("Bitte eine .csv oder .txt Datei auswählen.")
        
        # Vorausgesetzt, dass die Datei zwei Spalten für X und Y enthält
        X = data.values  # Ersten zwei Spalten nehmen
        
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

data_input_frame = ctk.CTkFrame(master = root)
data_input_frame.grid(row=0, column=0, padx=6, pady=6)

data_input_label = ctk.CTkLabel(master = data_input_frame, text = "Data")
data_input_label.grid(row=0, column=0, padx=5, pady=5, sticky = 'w')
data_input_entry = ctk.CTkEntry(master = data_input_frame, placeholder_text = "filepath")
data_input_entry.grid(row=1, column=0, padx=5, pady=5, sticky = "ew")
data_input_button = ctk.CTkButton(master = data_input_frame, text="", width=30, command=browse_file)
data_input_button.grid(row=1, column=1, padx=5, pady=5, sticky = 'e')
# Canvas erstellen und in das Tkinter-Fenster einfügen
canvas = FigureCanvasTkAgg(fig, master = data_input_frame)  # Matplotlib Canvas
canvas.get_tk_widget().grid(row=2, column=0, columnspan = 2, padx=5, pady=5)#pack(side=ctk.TOP, fill=ctk.BOTH, expand=True)
show_empty_plot()
compute_button = ctk.CTkButton(master = data_input_frame, text="Compute", width=30, command=compute_kmeans)
compute_button.grid(row=3, column=0, padx=5, pady=5, sticky = 'w')
reset_button = ctk.CTkButton(master = data_input_frame, text="Reset", width=30, command=show_empty_plot)
reset_button.grid(row=3, column=1, padx=5, pady=5, sticky = 'w')

# =============================================================================
# K-Means Input Frame
# =============================================================================

kmeans_frame = ctk.CTkFrame(master = root)
kmeans_frame.grid(row=1, column=0, padx=6, pady=6)

kmeans_label = ctk.CTkLabel(master = kmeans_frame, text = "K-Means").grid(row=0, column=0, padx=5, pady=5, sticky = 'w')
k_parameter_label = ctk.CTkLabel(master = kmeans_frame, text = "cluster quantity k").grid(row=1, column=0, padx=5, pady=5, sticky = 'w')

# Frame als Platzhalter für Plots
plot_frame = ctk.CTkFrame(master = kmeans_frame, height = 300, width = 500)
plot_frame.grid(row=2, column=0, columnspan=3,  padx=6, pady=6)

# Validierungsfunktion registrieren
vcmd = (root.register(validate_input), "%P")

# Entry-Widget erstellen
k_entry = ctk.CTkEntry(master = kmeans_frame, width=40, height=30, validate="key", validatecommand=vcmd, justify="center")
k_entry.grid(row=1, column=1, padx=6, pady=6)

# Checkbox für Centroids
ctk.CTkCheckBox(kmeans_frame, text="Centroids").grid(row=2, column=4, padx=6, pady=6, sticky = 'e')

# Buttons
ctk.CTkButton(master = kmeans_frame, text = "Compute").grid(row=3, column=0, padx=6, pady=6, sticky = 'w')
ctk.CTkButton(master = kmeans_frame, text = "Export").grid(row=3, column=1, padx=6, pady=6, sticky = 'w')
ctk.CTkButton(master = kmeans_frame, text = "Reset").grid(row=3, column=4, padx=6, pady=6, sticky = 'e')

# =============================================================================
# DBSCAN Frame
# =============================================================================

dbscan_frame = ctk.CTkFrame(master = root)
dbscan_frame.grid(row=0, column=1, padx=6, pady=6)

data_input_label = ctk.CTkLabel(master = dbscan_frame, text = "DBSCAN").grid(row=0, column=0, padx=5, pady=5, sticky = 'w')


# =============================================================================
# ILS Frame
# =============================================================================

ils_frame = ctk.CTkFrame(master = root)
ils_frame.grid(row = 1, column = 1, padx = 6, pady = 6)

data_input_label = ctk.CTkLabel(master = ils_frame, text = "ILS").grid(row=0, column=0, padx=5, pady=5, sticky = 'w')















root.mainloop()
