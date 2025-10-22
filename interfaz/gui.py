import tkinter as tk
from tkinter import ttk  

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk
)
import matplotlib.animation as animation

import numpy as np
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from calculos import simulador
from utils import parametros as p

class App(tk.Tk):
    
    def __init__(self):
        super().__init__()  

        self.title("Simulador de Pinzas Ópticas")
        self.geometry("800x650") 
        
        # Variables para manejar el estado de la simulación
        self.trajectory_data = None
        self.animation = None
        self.is_running = False
        
        # Frame para los botones
        button_frame = ttk.Frame(self)
        button_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        # Frame para el gráfico 
        plot_frame = ttk.Frame(self)
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.btn_start = ttk.Button(
            button_frame,
            text="Iniciar",
            command=self.start_simulation
        )
        self.btn_start.pack(side=tk.LEFT, padx=5)

        self.btn_pause = ttk.Button(
            button_frame,
            text="Pausar",
            command=self.pause_simulation,
            state=tk.DISABLED  # Deshabilitado al inicio
        )
        self.btn_pause.pack(side=tk.LEFT, padx=5)
        
        self.btn_reset = ttk.Button(
            button_frame,
            text="Reiniciar",
            command=self.reset_simulation,
            state=tk.DISABLED # Deshabilitado al inicio
        )
        self.btn_reset.pack(side=tk.LEFT, padx=5)

        
        self.fig = Figure(figsize=(7, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Añadir barra de herramientas de Matplotlib (zoom, pan, guardar)
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()
        
        self.init_plot()

    def init_plot(self):
        self.ax.clear() 
        self.ax.set_xlabel('Posición X (nm)')
        self.ax.set_ylabel('Posición Y (nm)')
        self.ax.set_title('Simulación de Pinza Óptica')
        self.ax.grid(True)
        self.ax.set_aspect('equal')
        
        self.line, = self.ax.plot([], [], 'o', markersize=6, color='royalblue')
        self.trace, = self.ax.plot([], [], '-', lw=1, alpha=0.5, color='orange')
        
        self.ax.set_xlim(-200, 200)
        self.ax.set_ylim(-200, 200)
        
        self.canvas.draw()

    def start_simulation(self):
        if self.animation and not self.is_running:
            self.animation.resume()
        else:
            print("Iniciando nueva simulación...")
            self.trajectory_data = self.generate_data()
            self.setup_animation()
            
        self.is_running = True
        self.update_button_states()

    def pause_simulation(self):
        if self.animation and self.is_running:
            self.animation.pause()
            self.is_running = False
        
        self.update_button_states()

    def reset_simulation(self):
        # Detener la animación existente si está corriendo
        if self.animation:
            self.animation.event_source.stop()
            self.animation = None
            
        self.is_running = False
        self.trajectory_data = None
        
        # Reiniciar el gráfico a su estado inicial
        self.init_plot()
        
        print("Simulación reiniciada.")
        self.update_button_states()

    def generate_data(self):
        trajectory = simulador.run_simulation(
            total_steps=p.total_steps,
            dt=p.dt,
            k_x=p.kappa_x,
            k_y=p.kappa_y,
            gamma=p.gamma,
            k_B=p.k_B,
            T=p.T
        )
        # Convertir a nanómetros 
        return trajectory * 1e9

    def setup_animation(self):

        std_dev_x = np.std(self.trajectory_data[:, 0])
        std_dev_y = np.std(self.trajectory_data[:, 1])
        max_range = max(std_dev_x, std_dev_y, 10.0) * 4 
        self.ax.set_xlim(-max_range, max_range)
        self.ax.set_ylim(-max_range, max_range)
        
        num_frames = len(self.trajectory_data) // 100

        self.animation = animation.FuncAnimation(
            self.fig,               # La figura de matplotlib
            self.animate_step,      # La función que actualiza cada fotograma
            frames=num_frames,      # Número total de fotogramas
            interval=20,            # Milisegundos entre fotogramas
            blit=True,              # Optimización de renderizado
            repeat=False            # No repetir al terminar
        )
        self.canvas.draw()

    def animate_step(self, i):
        step = i * 100 
        
        if step >= len(self.trajectory_data):
            self.is_running = False
            self.update_button_states()
            return self.line, self.trace

        x_point = self.trajectory_data[step, 0]
        y_point = self.trajectory_data[step, 1]
        self.line.set_data([x_point], [y_point])

        start_trace = max(0, step - 1000)
        x_trace = self.trajectory_data[start_trace:step, 0]
        y_trace = self.trajectory_data[start_trace:step, 1]
        self.trace.set_data(x_trace, y_trace)

        return self.line, self.trace # Necesario para 'blit=True'

    def update_button_states(self):
        if self.is_running:
            self.btn_start.config(state=tk.DISABLED)
            self.btn_pause.config(state=tk.NORMAL)
            self.btn_reset.config(state=tk.NORMAL)
        else:
            if self.trajectory_data is None:
                self.btn_start.config(text="Iniciar", state=tk.NORMAL)
                self.btn_pause.config(state=tk.DISABLED)
                self.btn_reset.config(state=tk.DISABLED)
            else:
                self.btn_start.config(text="Reanudar", state=tk.NORMAL)
                self.btn_pause.config(state=tk.DISABLED)
                self.btn_reset.config(state=tk.NORMAL)

if __name__ == '__main__':
    print("Iniciando la aplicación GUI en modo de prueba...")
    app = App()
    app.mainloop()