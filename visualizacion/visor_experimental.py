# -*- coding: utf-8 -*-
"""
Visor de Datos Experimentales de Pinzas Ópticas.
Carga archivos crudos (Sx, Sy, Sum), los procesa y visualiza la trayectoria real.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
import os
import sys

# --- Integración de Matplotlib con Tkinter ---
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk
)
import matplotlib.animation as animation

# --- INICIO DE SOLUCIÓN DE RUTA ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
# --- FIN DE SOLUCIÓN DE RUTA ---

from utils import parametros as p

class ExperimentalViewer(tk.Tk):
    
    def __init__(self):
        super().__init__()
        
        self.title("Visor de Datos Experimentales (Pinzas Ópticas)")
        self.geometry("900x700")

        # --- Variables de datos ---
        self.traj_x_nm = None
        self.traj_y_nm = None
        self.animation = None
        self.is_running = False
        self.animation_step_size = 50 # Velocidad de reproducción

        # =====================================================================
        # 1. Panel de Control (Carga de Archivos)
        # =====================================================================
        control_frame = ttk.LabelFrame(self, text="Carga de Datos Experimentales")
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        # Botón para cargar Sx
        self.btn_load = ttk.Button(
            control_frame, 
            text="Cargar Archivos (Selecciona el archivo _Sx)", 
            command=self.load_data
        )
        self.btn_load.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.lbl_status = ttk.Label(control_frame, text="Estado: Esperando archivos...", foreground="gray")
        self.lbl_status.pack(side=tk.LEFT, padx=10)

        # Botones de reproducción
        self.btn_play = ttk.Button(control_frame, text="Reproducir", command=self.play_animation, state=tk.DISABLED)
        self.btn_play.pack(side=tk.RIGHT, padx=5)
        
        self.btn_pause = ttk.Button(control_frame, text="Pausar", command=self.pause_animation, state=tk.DISABLED)
        self.btn_pause.pack(side=tk.RIGHT, padx=5)

        # =====================================================================
        # 2. Lienzo de Matplotlib
        # =====================================================================
        plot_frame = ttk.Frame(self)
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(7, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()
        
        self.init_plot()

    def init_plot(self):
        self.ax.clear()
        self.ax.set_xlabel('Posición X (nm)')
        self.ax.set_ylabel('Posición Y (nm)')
        self.ax.set_title('Trayectoria Experimental (Datos Reales)')
        self.ax.grid(True, linestyle='--', alpha=0.5)
        self.ax.set_aspect('equal')
        self.ax.set_xlim(-300, 300)
        self.ax.set_ylim(-300, 300)
        
        self.line, = self.ax.plot([], [], 'o', markersize=6, color='crimson', label='Partícula Real')
        self.trace, = self.ax.plot([], [], '-', lw=1, alpha=0.6, color='salmon')
        self.ax.legend()
        self.canvas.draw()

    def load_data(self):
        filepath_sx = filedialog.askopenfilename(
            title="Selecciona el archivo de señal X (_Sx)",
            filetypes=[("Archivos de Datos", "*_Sx*"), ("Todos los archivos", "*.*")]
        )
        
        if not filepath_sx: return

        # Deducir los nombres de los otros archivos
        base_path = filepath_sx.replace("_Sx", "")
        filepath_sy = base_path + "_Sy"
        filepath_sum = base_path + "_Sum"
        
        # Verificar existencia
        if not (os.path.exists(filepath_sy) and os.path.exists(filepath_sum)):
            messagebox.showerror("Error", f"No se encontraron los archivos complementarios:\n{filepath_sy}\n{filepath_sum}")
            return

        try:
            self.lbl_status.config(text="Cargando y procesando datos...", foreground="blue")
            self.update()

            # 1. Cargar datos crudos (tab-separated)
            # Usamos pandas para leer eficientemente. header=None porque no tienen cabecera.
            df_sx = pd.read_csv(filepath_sx, sep='\t', header=None)
            df_sy = pd.read_csv(filepath_sy, sep='\t', header=None)
            df_sum = pd.read_csv(filepath_sum, sep='\t', header=None)

            # 2. Aplanar los datos (concatenar todas las columnas en una sola serie temporal)
            # Esto asume que los datos son una serie continua partida en columnas.
            raw_sx = df_sx.values.flatten()
            raw_sy = df_sy.values.flatten()
            raw_sum = df_sum.values.flatten()
            
            # Recortar al tamaño mínimo por seguridad
            min_len = min(len(raw_sx), len(raw_sy), len(raw_sum))
            raw_sx = raw_sx[:min_len]
            raw_sy = raw_sy[:min_len]
            raw_sum = raw_sum[:min_len]

            # 3. Normalización (Señal / Suma)
            norm_x = raw_sx / raw_sum
            norm_y = raw_sy / raw_sum
            
            # Centrar en cero (restar la media)
            norm_x = norm_x - np.mean(norm_x)
            norm_y = norm_y - np.mean(norm_y)

            # 4. Auto-Calibración (Equipartición)
            # Convertimos unidades arbitrarias a nanómetros usando la física.
            # Teoría: <x^2> = k_B * T / k_x
            var_theory_x = (p.k_B * p.T) / p.kappa_x
            var_theory_y = (p.k_B * p.T) / p.kappa_y
            
            std_theory_x_nm = np.sqrt(var_theory_x) * 1e9 # en nm
            std_theory_y_nm = np.sqrt(var_theory_y) * 1e9 # en nm
            
            std_exp_x = np.std(norm_x)
            std_exp_y = np.std(norm_y)
            
            # Factor de conversión (nm / unidad_norm)
            factor_x = std_theory_x_nm / std_exp_x
            factor_y = std_theory_y_nm / std_exp_y
            
            self.traj_x_nm = norm_x * factor_x
            self.traj_y_nm = norm_y * factor_y
            
            self.lbl_status.config(
                text=f"Datos cargados. Puntos: {min_len}. Factor Calibración X: {factor_x:.2f} nm/u", 
                foreground="green"
            )
            self.btn_play.config(state=tk.NORMAL)
            
            self.setup_animation()

        except Exception as e:
            self.lbl_status.config(text="Error al procesar datos", foreground="red")
            messagebox.showerror("Error", f"Ocurrió un error:\n{e}")
            print(e)

    def setup_animation(self):
        if self.animation:
            self.animation.event_source.stop()
        
        max_range = max(np.std(self.traj_x_nm), np.std(self.traj_y_nm)) * 4
        self.ax.set_xlim(-max_range, max_range)
        self.ax.set_ylim(-max_range, max_range)
        
        num_frames = len(self.traj_x_nm) // self.animation_step_size
        
        self.animation = animation.FuncAnimation(
            self.fig, self.animate_step, frames=num_frames,
            interval=20, blit=True, repeat=False
        )
        self.is_running = True
        self.btn_play.config(state=tk.DISABLED)
        self.btn_pause.config(state=tk.NORMAL)
        self.canvas.draw()

    def animate_step(self, i):
        step = i * self.animation_step_size
        if step >= len(self.traj_x_nm):
            return self.line, self.trace

        x = self.traj_x_nm[step]
        y = self.traj_y_nm[step]
        
        self.line.set_data([x], [y])
        
        start = max(0, step - 2)
        self.trace.set_data(self.traj_x_nm[start:step], self.traj_y_nm[start:step])
        
        return self.line, self.trace

    def play_animation(self):
        if self.animation:
            self.animation.resume()
            self.is_running = True
            self.btn_play.config(state=tk.DISABLED)
            self.btn_pause.config(state=tk.NORMAL)

    def pause_animation(self):
        if self.animation:
            self.animation.pause()
            self.is_running = False
            self.btn_play.config(state=tk.NORMAL)
            self.btn_pause.config(state=tk.DISABLED)

if __name__ == "__main__":
    app = ExperimentalViewer()
    app.mainloop()