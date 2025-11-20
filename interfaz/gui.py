# -*- coding: utf-8 -*-
"""
** Versión 5: Soporte completo para visualización de Intensidad y Fuerza.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import sys
import os

# ... (imports de matplotlib sin cambios) ...
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
from visualizacion import graficador

class App(ttk.Frame):
    
    def __init__(self, master=None):
        super().__init__(master)
        self.pack(fill=tk.BOTH, expand=True)

        # --- Estado de Simulación ---
        self.trajectory_data = None
        self.animation = None
        self.is_running = False
        self.animation_step_size = 50
        
        # --- Estado de Datos ---
        self.anharmonic_mode = tk.BooleanVar(value=False)
        self.fx_interp = None
        self.fy_interp = None
        self.int_interp = None # <--- Nuevo: Interpolador de intensidad
        
        # --- Estado de Visualización ---
        # Usaremos un string para el modo de visualización de fondo
        self.viz_mode = tk.StringVar(value="force") 

        # =====================================================================
        # FRAMES
        # =====================================================================
        control_frame = ttk.LabelFrame(self, text="Control de Simulación")
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        settings_frame = ttk.LabelFrame(self, text="Configuración del Modelo y Visualización")
        settings_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        plot_frame = ttk.Frame(self)
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0,10))

        # =====================================================================
        # WIDGETS - CONTROL
        # =====================================================================
        self.btn_start = ttk.Button(control_frame, text="Iniciar", command=self.start_simulation)
        self.btn_start.pack(side=tk.LEFT, padx=5, pady=5)
        self.btn_pause = ttk.Button(control_frame, text="Pausar", command=self.pause_simulation, state=tk.DISABLED)
        self.btn_pause.pack(side=tk.LEFT, padx=5, pady=5)
        self.btn_reset = ttk.Button(control_frame, text="Reiniciar Simulación", command=self.reset_simulation)
        self.btn_reset.pack(side=tk.LEFT, padx=5, pady=5)

        # =====================================================================
        # WIDGETS - CONFIGURACIÓN
        # =====================================================================
        
        # -- Columna Izquierda: Modelo Físico --
        model_panel = ttk.Frame(settings_frame)
        model_panel.pack(side=tk.LEFT, padx=10, pady=5)
        
        ttk.Label(model_panel, text="Modelo de Fuerza:").pack(anchor=tk.W)
        self.check_anharmonic = ttk.Checkbutton(
            model_panel, text="Usar Mapa Anharmónico (Mie)", 
            variable=self.anharmonic_mode, command=self.on_param_change
        )
        self.check_anharmonic.pack(anchor=tk.W)
        
        load_frame = ttk.Frame(model_panel)
        load_frame.pack(anchor=tk.W, pady=(5,0))
        self.btn_load = ttk.Button(load_frame, text="Cargar CSV...", command=self.load_force_map)
        self.btn_load.pack(side=tk.LEFT)
        self.lbl_file = ttk.Label(load_frame, text="(Ninguno)", font=("Arial", 8))
        self.lbl_file.pack(side=tk.LEFT, padx=5)

        # -- Separador --
        ttk.Separator(settings_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=20, pady=5)

        # -- Columna Derecha: Visualización --
        viz_panel = ttk.Frame(settings_frame)
        viz_panel.pack(side=tk.LEFT, padx=10, pady=5)
        
        ttk.Label(viz_panel, text="Fondo del Gráfico:").pack(anchor=tk.W)
        
        # Combobox para elegir qué ver en el fondo
        self.combo_viz = ttk.Combobox(
            viz_panel, 
            textvariable=self.viz_mode,
            state="readonly",
            values=["force", "intensity", "none"]
        )
        # Mapeo de nombres amigables para el usuario
        self.combo_viz['values'] = ("Fuerza (Magnitud)", "Intensidad del Haz", "Ninguno")
        self.combo_viz.current(0) # Seleccionar el primero por defecto
        self.combo_viz.bind("<<ComboboxSelected>>", self.on_viz_change)
        self.combo_viz.pack(anchor=tk.W)

        # =====================================================================
        # MATPLOTLIB
        # =====================================================================
        self.fig = Figure(figsize=(7, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()
        
        self.reset_plot()
        self.update_gui_state()

    # --- MANEJO DE EVENTOS ---

    def on_param_change(self):
        """Se llama cuando cambia un parámetro que requiere redibujar el fondo."""
        self.reset_plot()
        self.update_gui_state()
        
    def on_viz_change(self, event=None):
        """Se llama cuando el usuario cambia la selección del Combobox."""
        self.reset_plot()

    def get_viz_mode_internal(self):
        """Traduce la selección del Combobox a nuestros códigos internos."""
        selection = self.combo_viz.get()
        if selection == "Fuerza (Magnitud)": return 'force'
        if selection == "Intensidad del Haz": return 'intensity'
        return 'none'

    def load_force_map(self):
        filepath = filedialog.askopenfilename(filetypes=[("Archivos CSV", "*.csv")])
        if not filepath: return
        try:
            # Desempaquetamos los 3 interpoladores
            self.fx_interp, self.fy_interp, self.int_interp = lector_datos.cargar_mapa_fuerzas(filepath)
            
            if self.fx_interp:
                self.lbl_file.config(text=os.path.basename(filepath), foreground="green")
                messagebox.showinfo("Éxito", "Archivo cargado correctamente.")
                self.anharmonic_mode.set(True) # Activar modo anharmónico automáticamente
                
                # Si hay datos de intensidad, cambiar automáticamente a ver intensidad
                if self.int_interp:
                    self.combo_viz.current(1) # Seleccionar "Intensidad del Haz"
                
                self.reset_plot()
            else:
                raise Exception("Error al leer datos del archivo.")
        except Exception as e:
            self.lbl_file.config(text="Error", foreground="red")
            messagebox.showerror("Error", str(e))
            self.anharmonic_mode.set(False)
        self.update_gui_state()


    def reset_plot(self):
        if self.animation: self.animation.event_source.stop(); self.animation = None
        self.is_running = False; self.trajectory_data = None
        self.ax.clear()
        
        graficador.draw_background(
            self.ax,
            viz_mode=self.get_viz_mode_internal(),
            anharmonic_mode=self.anharmonic_mode.get(),
            fx_i=self.fx_interp, fy_i=self.fy_interp, int_i=self.int_interp,
            lim_nm=300
        )

        self.line, = self.ax.plot([], [], 'o', markersize=6, color='royalblue', zorder=10)
        self.trace, = self.ax.plot([], [], '-', lw=1, alpha=0.7, color='orange', zorder=9)
        self.ax.set_xlabel('X [nm]'); self.ax.set_ylabel('Y [nm]')
        self.ax.set_aspect('equal'); self.ax.grid(True, linestyle=':', alpha=0.5)
        self.ax.set_xlim(-300, 300); self.ax.set_ylim(-300, 300)
        self.canvas.draw()
        self.update_gui_state()

    def start_simulation(self):
        if self.animation and not self.is_running:
            self.animation.resume()
        else:
            if self.trajectory_data is None or self.animation is None:
                self.reset_plot()
                mode = 'anharmonic' if self.anharmonic_mode.get() else 'harmonic'
                
                if mode == 'anharmonic' and not self.fx_interp:
                    messagebox.showerror("Error", "Carga un archivo de mapa primero.")
                    return

                print(f"Iniciando simulación: {mode.upper()}")
                common = {'total_steps': p.total_steps, 'dt': p.dt, 'gamma': p.gamma, 'k_B': p.k_B, 'T': p.T}
                if mode == 'anharmonic':
                    traj = simulador.run_simulation(**common, fx_interp=self.fx_interp, fy_interp=self.fy_interp)
                else:
                    traj = simulador.run_simulation(**common, k_x=p.kappa_x, k_y=p.kappa_y)
                self.trajectory_data = traj * 1e9
                self.setup_animation()
        self.is_running = True
        self.update_gui_state()


    def pause_simulation(self):
        if self.animation and self.is_running:
             self.animation.pause()
             self.is_running = False
        self.update_gui_state()

    def reset_simulation(self):
        if self.animation:
            self.animation.event_source.stop()
            self.animation = None
        self.is_running = False
        self.trajectory_data = None
        self.line.set_data([], [])
        self.trace.set_data([], [])
        self.canvas.draw()
        self.update_gui_state()
        
    def setup_animation(self):
        std_x, std_y = np.std(self.trajectory_data[:,0]), np.std(self.trajectory_data[:,1])
        lim = max(max(std_x, std_y, 10)*4, self.ax.get_xlim()[1])
        self.ax.set_xlim(-lim, lim); self.ax.set_ylim(-lim, lim)
        self.animation = animation.FuncAnimation(self.fig, self.animate_step, 
            frames=len(self.trajectory_data)//self.animation_step_size, interval=20, blit=True, repeat=False)
        self.canvas.draw()

    def animate_step(self, i):
        step = i * self.animation_step_size
        if step >= len(self.trajectory_data):
            self.is_running = False; self.update_gui_state()
            return self.line, self.trace
        self.line.set_data([self.trajectory_data[step,0]], [self.trajectory_data[step,1]])
        start = max(0, step - (2000 if self.anharmonic_mode.get() else 10000))
        self.trace.set_data(self.trajectory_data[start:step,0], self.trajectory_data[start:step,1])
        return self.line, self.trace

    def update_gui_state(self):
        sim_active = self.is_running
        has_data = self.trajectory_data is not None
        
        self.btn_start.config(state=tk.DISABLED if sim_active else tk.NORMAL,
                              text="Reanudar" if has_data and not sim_active else "Iniciar")
        self.btn_pause.config(state=tk.NORMAL if sim_active else tk.DISABLED)
        self.btn_reset.config(state=tk.NORMAL if (has_data or self.animation) else tk.DISABLED)
        
        state_settings = tk.DISABLED if sim_active else tk.NORMAL
        self.check_anharmonic.config(state=state_settings)
        self.btn_load.config(state=state_settings)
        self.combo_viz.config(state="readonly" if not sim_active else tk.DISABLED)


if __name__ == '__main__':
    print("Iniciando la aplicación GUI en modo de prueba...")
    root = tk.Tk()
    root.title("Prueba Simulador")
    root.geometry("800x700")
    app = App(master=root)
    root.mainloop()