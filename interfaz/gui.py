# -*- coding: utf-8 -*-
"""
** Versión 4: Lógica de gráficos refactorizada a 'graficador.py'
** y con checkboxes separados para modo y visualización.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import sys
import os

# ... (integración de matplotlib) ...
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

from calculos import simulador
from utils import parametros as p
from utils import lector_datos
from visualizacion import graficador # <--- NUEVA IMPORTACIÓN

class App(tk.Tk):
    
    def __init__(self):
        super().__init__()
        
        self.title("Simulador de Pinzas Ópticas")
        self.geometry("800x700")

        # --- Variables de estado de simulación ---
        self.trajectory_data = None
        self.animation = None
        self.is_running = False
        self.animation_step_size = 50
        
        # --- Variables de estado de MODO ---
        self.anharmonic_mode = tk.BooleanVar(value=False)
        self.fx_interp = None
        self.fy_interp = None
        
        # --- NUEVA Variable de VISUALIZACIÓN ---
        self.show_force_map = tk.BooleanVar(value=True) # <-- Nuevo checkbox

        # =====================================================================
        # 1. Crear los frames
        # =====================================================================
        control_frame = ttk.Frame(self)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        mode_frame = ttk.Frame(self)
        mode_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        plot_frame = ttk.Frame(self)
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # =====================================================================
        # 2. Widgets del 'control_frame' (Botones)
        # =====================================================================
        self.btn_start = ttk.Button(control_frame, text="Iniciar", command=self.start_simulation)
        self.btn_start.pack(side=tk.LEFT, padx=5)
        self.btn_pause = ttk.Button(control_frame, text="Pausar", command=self.pause_simulation, state=tk.DISABLED)
        self.btn_pause.pack(side=tk.LEFT, padx=5)
        self.btn_reset = ttk.Button(control_frame, text="Reiniciar Simulación", command=self.reset_simulation)
        self.btn_reset.pack(side=tk.LEFT, padx=5)

        # =====================================================================
        # 3. Widgets del 'mode_frame' (Controles)
        # =====================================================================
        
        # Checkbox para MODO (Armónico / Anharmónico)
        self.check_anharmonic = ttk.Checkbutton(
            mode_frame,
            text="Modo Anharmónico",
            variable=self.anharmonic_mode,
            command=self.on_mode_change
        )
        self.check_anharmonic.pack(side=tk.LEFT, padx=5)

        self.btn_load_map = ttk.Button(mode_frame, text="Cargar Mapa...", command=self.load_force_map)
        self.btn_load_map.pack(side=tk.LEFT, padx=5)
        
        self.lbl_map_status = ttk.Label(mode_frame, text="Archivo: Ninguno")
        self.lbl_map_status.pack(side=tk.LEFT, padx=10)
        
        # --- NUEVO CHECKBOX DE VISUALIZACIÓN ---
        self.check_show_map = ttk.Checkbutton(
            mode_frame,
            text="Mostrar Haz (Fondo)",
            variable=self.show_force_map,
            command=self.reset_plot # Redibuja el fondo al cambiar
        )
        self.check_show_map.pack(side=tk.RIGHT, padx=10)

        # =====================================================================
        # 4. Configuración del lienzo de Matplotlib
        # =====================================================================
        self.fig = Figure(figsize=(7, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()
        
        self.reset_plot()
        self.update_button_states()

    # --- Callbacks de Botones y Controles ---

    def start_simulation(self):
        # ... (La lógica aquí no cambia) ...
        if self.animation and not self.is_running:
            self.animation.resume()
        else:
            if self.trajectory_data is None or self.animation is None:
                self.reset_plot() 
                if self.anharmonic_mode.get():
                    if self.fx_interp is None:
                        messagebox.showerror("Error", "Modo Anharmónico seleccionado, pero no se ha cargado ningún mapa de fuerzas.")
                        return
                    print("Iniciando nueva simulación ANHARMÓNICA...")
                    self.trajectory_data = self.generate_data(mode='anharmonic')
                else:
                    print("Iniciando nueva simulación ARMÓNICA...")
                    self.trajectory_data = self.generate_data(mode='harmonic')
                if self.trajectory_data is not None:
                    self.setup_animation()
        self.is_running = True
        self.update_button_states()

    def pause_simulation(self):
        # ... (Sin cambios) ...
        if self.animation and self.is_running:
            self.animation.pause()
            self.is_running = False
        self.update_button_states()

    def reset_simulation(self):
        # ... (Sin cambios) ...
        if self.animation:
            self.animation.event_source.stop()
            self.animation = None
        self.is_running = False
        self.trajectory_data = None
        self.line.set_data([], [])
        self.trace.set_data([], [])
        self.canvas.draw()
        print("Trayectoria de simulación reiniciada.")
        self.update_button_states()

    def load_force_map(self):
        # ... (Casi sin cambios) ...
        filepath = filedialog.askopenfilename(filetypes=[("Archivos CSV", "*.csv")])
        if not filepath: return
        try:
            self.fx_interp, self.fy_interp = lector_datos.cargar_mapa_fuerzas(filepath)
            if self.fx_interp is not None:
                filename = os.path.basename(filepath)
                self.lbl_map_status.config(text=f"Archivo: {filename}", foreground="green")
                messagebox.showinfo("Éxito", "Mapa de fuerzas cargado correctamente.")
                self.anharmonic_mode.set(True)
                self.reset_plot() # Redibuja el fondo con el nuevo mapa
            # ... (resto del try/except sin cambios) ...
        except Exception as e:
            self.lbl_map_status.config(text="Error al cargar", foreground="red")
            messagebox.showerror("Error de Carga", f"No se pudo cargar o procesar el archivo:\n{e}")
            self.fx_interp = None; self.fy_interp = None
            self.anharmonic_mode.set(False)
        self.update_button_states()

    def on_mode_change(self):
        """Callback cuando el checkbox de MODO cambia."""
        self.reset_plot() # Redibuja el fondo
        self.update_button_states()

    # --- Lógica de Gráficos (Ahora más simple) ---

    def reset_plot(self):
        """Limpia ejes y redibuja el fondo según los checkboxes."""
        if self.animation:
            self.animation.event_source.stop()
            self.animation = None
        self.is_running = False
        self.trajectory_data = None
        self.ax.clear()
        
        # --- LÓGICA DE FONDO ACTUALIZADA ---
        # Solo dibuja si el checkbox de "Mostrar Haz" está marcado
        if self.show_force_map.get():
            is_anharmonic = self.anharmonic_mode.get()
            # Validar que tengamos datos para el modo anharmónico
            if is_anharmonic and self.fx_interp is None:
                pass # No dibujar nada si no hay archivo cargado
            else:
                # Llamada a la función refactorizada
                graficador.draw_force_map(
                    self.ax,
                    anharmonic_mode=is_anharmonic,
                    fx_interp=self.fx_interp,
                    fy_interp=self.fy_interp,
                    lim_nm=300 # Límite fijo
                )
        # --- FIN DE LÓGICA DE FONDO ---

        self.line, = self.ax.plot([], [], 'o', markersize=6, color='royalblue', zorder=10)
        self.trace, = self.ax.plot([], [], '-', lw=1, alpha=0.7, color='orange', zorder=9)
        self.ax.set_xlabel('Posición X (nm)'); self.ax.set_ylabel('Posición Y (nm)')
        self.ax.set_title('Simulación de Pinza Óptica')
        self.ax.grid(True, linestyle='--', alpha=0.5, zorder=0)
        self.ax.set_aspect('equal')
        self.ax.set_xlim(-300, 300); self.ax.set_ylim(-300, 300)
        self.canvas.draw()
        self.update_button_states()
        print("Lienzo de gráfico reiniciado.")

    def generate_data(self, mode='harmonic'):
        # ... (Sin cambios) ...
        common_args = {'total_steps': p.total_steps, 'dt': p.dt, 'gamma': p.gamma, 'k_B': p.k_B, 'T': p.T}
        if mode == 'anharmonic' and self.fx_interp:
            trajectory = simulador.run_simulation(**common_args, fx_interp=self.fx_interp, fy_interp=self.fy_interp)
        else:
            trajectory = simulador.run_simulation(**common_args, k_x=p.kappa_x, k_y=p.kappa_y)
        return trajectory * 1e9

    def setup_animation(self):
        # ... (Lógica de zoom automático sin cambios) ...
        std_dev_x = np.std(self.trajectory_data[:, 0])
        std_dev_y = np.std(self.trajectory_data[:, 1])
        max_range = max(std_dev_x, std_dev_y, 10.0) * 4
        current_lim = self.ax.get_xlim()[1]
        new_lim = max(max_range, current_lim)
        self.ax.set_xlim(-new_lim, new_lim); self.ax.set_ylim(-new_lim, new_lim)
        
        num_frames = len(self.trajectory_data) // self.animation_step_size
        self.animation = animation.FuncAnimation(
            self.fig, self.animate_step, frames=num_frames,
            interval=20, blit=True, repeat=False
        )
        self.canvas.draw()

    def animate_step(self, i):
        # ... (Lógica de estela armónica/anharmónica sin cambios) ...
        step = i * self.animation_step_size
        if step >= len(self.trajectory_data):
            self.is_running = False; self.update_button_states()
            return self.line, self.trace
        x_point = self.trajectory_data[step, 0]
        y_point = self.trajectory_data[step, 1]
        self.line.set_data([x_point], [y_point])
        if self.anharmonic_mode.get():
            start_trace = max(0, step - 2000) # Estela corta
        else:
            start_trace = max(0, step - 10000) # Estela larga
        x_trace = self.trajectory_data[start_trace:step, 0]
        y_trace = self.trajectory_data[start_trace:step, 1]
        self.trace.set_data(x_trace, y_trace)
        return self.line, self.trace

    def update_button_states(self):
        # ... (Sin cambios) ...
        if self.is_running:
            self.btn_start.config(state=tk.DISABLED); self.btn_pause.config(state=tk.NORMAL)
            self.btn_reset.config(state=tk.NORMAL); self.btn_load_map.config(state=tk.DISABLED)
            self.check_anharmonic.config(state=tk.DISABLED); self.check_show_map.config(state=tk.DISABLED)
        else:
            self.btn_pause.config(state=tk.DISABLED)
            self.btn_reset.config(state=tk.NORMAL if (self.trajectory_data is not None or self.animation is not None) else tk.DISABLED)
            if self.trajectory_data is None: self.btn_start.config(text="Iniciar", state=tk.NORMAL)
            else: self.btn_start.config(text="Reanudar", state=tk.NORMAL)
            self.btn_load_map.config(state=tk.NORMAL); self.check_anharmonic.config(state=tk.NORMAL)
            self.check_show_map.config(state=tk.NORMAL)
            if self.anharmonic_mode.get() and self.fx_interp is None:
                self.btn_start.config(state=tk.DISABLED)
                if "Error" not in self.lbl_map_status.cget("text"):
                     self.lbl_map_status.config(text="Carga de archivo requerida", foreground="red")

# ... (Bloque de prueba __main__ sin cambios)
if __name__ == '__main__':
    print("Iniciando la aplicación GUI en modo de prueba...")
    app = App()
    app.mainloop()