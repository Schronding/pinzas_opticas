# -*- coding: utf-8 -*-
"""
SUITE MAESTRA: Comparación Lado a Lado
Izquierda: Simulación Teórica (Mie/Langevin)
Derecha: Datos Experimentales (Lab)
"""

import tkinter as tk
from tkinter import ttk
import sys
import os

# --- CONFIGURACIÓN DE RUTAS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Importamos los módulos (que ahora son Frames)
from interfaz import gui
from visualizacion import visor_experimental

class MainSuite(tk.Tk):
    def __init__(self):
        super().__init__()
        
        # Configuración de la Ventana Principal
        self.title("Pinzas Ópticas: Simulación Teórica vs. Datos Reales")
        self.geometry("1600x900") # Ventana ancha para que quepan los dos
        
        # Estilo visual
        style = ttk.Style()
        style.theme_use('clam') 

        # --- TÍTULO GENERAL ---
        header_frame = ttk.Frame(self)
        header_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        lbl_title = ttk.Label(
            header_frame, 
            text="COMPARACIÓN SIMULTÁNEA: MODELO NUMÉRICO VS EXPERIMENTO", 
            font=("Arial", 14, "bold"),
            anchor="center"
        )
        lbl_title.pack()

        # --- CONTENEDOR PRINCIPAL (PANEDWINDOW) ---
        # Usamos PanedWindow para que puedas ajustar el ancho de cada lado arrastrando
        paned_window = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # ==========================================
        # LADO IZQUIERDO: SIMULACIÓN
        # ==========================================
        frame_left = ttk.LabelFrame(paned_window, text=" 🔬 SIMULACIÓN TEÓRICA (Langevin) ")
        paned_window.add(frame_left, weight=1) # weight=1 significa que crece
        
        # Instanciamos la GUI de simulación dentro del marco izquierdo
        self.app_sim = gui.App(master=frame_left)

        # ==========================================
        # LADO DERECHO: EXPERIMENTO
        # ==========================================
        frame_right = ttk.LabelFrame(paned_window, text=" 📊 DATOS EXPERIMENTALES (Lab) ")
        paned_window.add(frame_right, weight=1)
        
        # Instanciamos el Visor Experimental dentro del marco derecho
        self.app_exp = visor_experimental.ExperimentalViewer(master=frame_right)

if __name__ == "__main__":
    app = MainSuite()
    app.mainloop()