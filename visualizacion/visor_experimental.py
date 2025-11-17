# -*- coding: utf-8 -*-
"""
Visor de Datos Experimentales de Pinzas Ópticas.
Versión 3: Corrección de nombres de archivo, detector de duplicados y animación fluida.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
import os
import sys

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk
)
import matplotlib.animation as animation

# --- INICIO DE SOLUCIÓN DE RUTA ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utils import parametros as p
# --- FIN DE SOLUCIÓN DE RUTA ---

class ExperimentalViewer(tk.Tk):
    
    def __init__(self):
        super().__init__()
        
        self.title("Visor Experimental (Corrección de Diagonal)")
        self.geometry("950x750")

        # --- Variables de datos ---
        self.traj_x_nm = None
        self.traj_y_nm = None
        self.animation = None
        self.is_running = False
        
        # CONFIGURACIÓN DE ANIMACIÓN
        self.animation_step_size = 20  # Saltos de 20 puntos (equilibrio velocidad/fluidez)
        self.trace_length_steps = 150  # Longitud de estela (en pasos, no puntos)

        # =====================================================================
        # 1. Panel de Control
        # =====================================================================
        control_frame = ttk.LabelFrame(self, text="Control de Datos")
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        self.btn_load = ttk.Button(
            control_frame, 
            text="Cargar Archivos (_Sx)", 
            command=self.load_data
        )
        self.btn_load.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.lbl_params = ttk.Label(control_frame, text="Estado: Esperando...", foreground="gray")
        self.lbl_params.pack(side=tk.LEFT, padx=10)

        self.btn_play = ttk.Button(control_frame, text="▶ Reproducir", command=self.play_animation, state=tk.DISABLED)
        self.btn_play.pack(side=tk.RIGHT, padx=5)
        
        self.btn_pause = ttk.Button(control_frame, text="⏸ Pausar", command=self.pause_animation, state=tk.DISABLED)
        self.btn_pause.pack(side=tk.RIGHT, padx=5)

        # =====================================================================
        # 2. Lienzo
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
        self.ax.set_title('Trayectoria Experimental')
        self.ax.grid(True, linestyle='--', alpha=0.5)
        self.ax.set_aspect('equal')
        self.ax.set_xlim(-300, 300)
        self.ax.set_ylim(-300, 300)
        # Colores Cian/Violeta
        self.line, = self.ax.plot([], [], 'o', markersize=6, color='cyan', markeredgecolor='blue', zorder=10)
        self.trace, = self.ax.plot([], [], '-', lw=1, alpha=0.6, color='darkviolet', zorder=5)
        self.canvas.draw()

    def parse_calibration_header(self, filepath):
        # (Misma función de lectura de metadatos que antes)
        meta = {'T': None, 'kx': None, 'ky': None}
        try:
            with open(filepath, 'r', encoding='latin-1') as f:
                lines = f.readlines()
            for i, line in enumerate(lines):
                if "Temperature (K)" in line:
                    data = lines[i+1].strip().split('\t')
                    if len(data) >= 4: meta['T'] = float(data[3])
                if "kx EP" in line and "ky EP" in line:
                    data = lines[i+1].strip().split('\t')
                    if len(data) >= 4: 
                        meta['kx'] = float(data[2])
                        meta['ky'] = float(data[3])
            return meta
        except: return meta

    def load_data(self):
        filepath_sx = filedialog.askopenfilename(title="Selecciona el archivo _Sx")
        if not filepath_sx: return

        # --- 1. DEDUCCIÓN DE NOMBRES ROBUSTA ---
        directory = os.path.dirname(filepath_sx)
        filename_sx = os.path.basename(filepath_sx)
        
        # Reemplazar _Sx por _Sy SOLAMENTE en el nombre del archivo
        if "_Sx" in filename_sx:
            filename_sy = filename_sx.replace("_Sx", "_Sy")
            filename_sum = filename_sx.replace("_Sx", "_Sum")
            # El archivo de calibración suele ser el nombre base sin _Sx
            # Probamos quitando la extensión si la tiene
            name_no_ext = os.path.splitext(filename_sx)[0]
            filename_calib = name_no_ext.replace("_Sx", "") # Intento 1
            if "." in filename_sx: # Si tiene extension .txt o similar
                 filename_calib = filename_sx.replace("_Sx", "") # Intento 2
        else:
            messagebox.showerror("Error de Nombre", "El archivo seleccionado debe contener '_Sx' en su nombre.")
            return

        filepath_sy = os.path.join(directory, filename_sy)
        filepath_sum = os.path.join(directory, filename_sum)
        filepath_calib = os.path.join(directory, filename_calib)

        if not (os.path.exists(filepath_sy) and os.path.exists(filepath_sum)):
            # Intento de fallback: buscar el archivo de calibración sin extensión
            filepath_calib_noext = os.path.splitext(filepath_calib)[0]
            if os.path.exists(filepath_calib_noext):
                filepath_calib = filepath_calib_noext
            
            if not (os.path.exists(filepath_sy) and os.path.exists(filepath_sum)):
                messagebox.showerror("Error", f"No se encontraron:\n{filename_sy}\n{filename_sum}")
                return

        try:
            # Leer metadatos
            meta = self.parse_calibration_header(filepath_calib)
            if meta['T'] is None: # Fallback a parametros.py
                T_exp, kx_exp, ky_exp = p.T, p.kappa_x, p.kappa_y
            else:
                T_exp, kx_exp, ky_exp = meta['T'], meta['kx'], meta['ky']

            # Leer Datos
            df_sx = pd.read_csv(filepath_sx, sep='\t', header=None)
            df_sy = pd.read_csv(filepath_sy, sep='\t', header=None) # <--- AQUI LEE EL ARCHIVO Y
            df_sum = pd.read_csv(filepath_sum, sep='\t', header=None)

            raw_sx = df_sx.values.flatten()
            raw_sy = df_sy.values.flatten()
            raw_sum = df_sum.values.flatten()
            
            # Recortar
            min_len = min(len(raw_sx), len(raw_sy), len(raw_sum))
            raw_sx = raw_sx[:min_len]; raw_sy = raw_sy[:min_len]; raw_sum = raw_sum[:min_len]

            # --- CHECK DE CORRELACIÓN (Detector de Diagonal) ---
            correlation = np.corrcoef(raw_sx, raw_sy)[0, 1]
            if correlation > 0.95:
                messagebox.showwarning(
                    "¡ALERTA DE DATOS!", 
                    f"Se detectó una correlación del {correlation*100:.1f}% entre X e Y.\n"
                    "Es muy probable que se esté leyendo el mismo archivo dos veces\n"
                    "o que los archivos contengan los mismos datos."
                )

            # Normalización
            norm_x = (raw_sx / raw_sum) - np.mean(raw_sx / raw_sum)
            norm_y = (raw_sy / raw_sum) - np.mean(raw_sy / raw_sum)

            # Calibración
            var_theory_x = (p.k_B * T_exp) / kx_exp
            var_theory_y = (p.k_B * T_exp) / ky_exp
            factor_x = np.sqrt(var_theory_x) * 1e9 / np.std(norm_x)
            factor_y = np.sqrt(var_theory_y) * 1e9 / np.std(norm_y)
            
            self.traj_x_nm = norm_x * factor_x
            self.traj_y_nm = norm_y * factor_y
            
            self.lbl_params.config(
                text=f"Cargado: {min_len} pts | Corr X-Y: {correlation:.2f}", 
                foreground="blue" if correlation < 0.5 else "red"
            )
            self.btn_play.config(state=tk.NORMAL)
            self.setup_animation()

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def setup_animation(self):
        if self.animation: self.animation.event_source.stop()
        
        max_range = max(np.std(self.traj_x_nm), np.std(self.traj_y_nm)) * 4
        self.ax.set_xlim(-max_range, max_range)
        self.ax.set_ylim(-max_range, max_range)
        
        num_frames = len(self.traj_x_nm) // self.animation_step_size
        self.animation = animation.FuncAnimation(
            self.fig, self.animate_step, frames=num_frames,
            interval=20, blit=True, repeat=False
        )
        self.is_running = True
        self.btn_play.config(state=tk.DISABLED); self.btn_pause.config(state=tk.NORMAL)
        self.canvas.draw()

    def animate_step(self, i):
        step = i * self.animation_step_size
        if step >= len(self.traj_x_nm): return self.line, self.trace

        # 1. Actualizar punto
        x = self.traj_x_nm[step]
        y = self.traj_y_nm[step]
        self.line.set_data([x], [y])
        
        # 2. Actualizar estela (CONECTADA AL PUNTO)
        # Calcular inicio de estela
        start = max(0, step - (self.trace_length_steps * self.animation_step_size))
        
        # IMPORTANTE: Slice hasta 'step + 1' para incluir el punto actual
        # y cerrar el hueco visual.
        self.trace.set_data(self.traj_x_nm[start:step+1], self.traj_y_nm[start:step+1])
        
        return self.line, self.trace

    def play_animation(self):
        if self.animation: self.animation.resume(); self.is_running = True; self.btn_play.config(state=tk.DISABLED); self.btn_pause.config(state=tk.NORMAL)
    def pause_animation(self):
        if self.animation: self.animation.pause(); self.is_running = False; self.btn_play.config(state=tk.NORMAL); self.btn_pause.config(state=tk.DISABLED)

if __name__ == "__main__":
    app = ExperimentalViewer()
    app.mainloop()