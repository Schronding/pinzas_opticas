import tkinter as tk
from tkinter import ttk
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from interfaz import gui
from visualizacion import visor_experimental

class MainSuite(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Suite de Pinzas Ópticas: Simulación vs. Experimento")
        self.geometry("1200x850") 
        
        # Estilo (Opcional, para que las pestañas se vean bien)
        style = ttk.Style()
        style.theme_use('clam') # 'clam', 'alt', 'default', 'classic'

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.tab_simulacion = gui.App(master=self.notebook)
        self.notebook.add(self.tab_simulacion, text="🔬 Simulador Teórico (Mie/Langevin)")

        self.tab_experimental = visor_experimental.ExperimentalViewer(master=self.notebook)
        self.notebook.add(self.tab_experimental, text="📊 Datos Experimentales (Lab)")

        status_frame = ttk.Frame(self)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        lbl_status = ttk.Label(status_frame, text="Proyecto de Pinzas Ópticas | Listo.", relief=tk.SUNKEN, anchor=tk.W)
        lbl_status.pack(fill=tk.X)

if __name__ == "__main__":
    app = MainSuite()
    app.mainloop()