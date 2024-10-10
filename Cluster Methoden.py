import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
import customtkinter as ctki

plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle')

# Hauptfenster wird erstellt
root = ctki.CTk()
root.title("ClusterMethods")
root.geometry("700x400")
root.minsize(height = 300, width = 500)

frameData = ctki.CTkFrame(master = root)

root.mainloop()