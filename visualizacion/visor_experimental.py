import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.animation as animation
import numpy as np
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__)) # carpeta visualizacion/
project_root = os.path.dirname(current_dir)              # carpeta raiz pinzas_opticas/
sys.path.append(project_root)

try:
    from calculos import procesamiento_experimental as procesador
except ImportError as e:
    messagebox.showerror("Error de Importación", 
        f"No se pudo importar el módulo de cálculo.\n"
        f"Asegúrate de que 'procesamiento_experimental.py' esté en la carpeta 'calculos'.\n\n"
        f"Detalle: {e}")
    sys.exit()

class ExperimentalViewer(ttk.Frame):    
    def __init__(self, master=None):
        super().__init__(master)
        self.pack(fill=tk.BOTH, expand=True)
        self.data = None
        self.animation = None
        
        self.run_analysis()

        tab_control = ttk.Notebook(self)
        
        self.tab_anim = ttk.Frame(tab_control)
        tab_control.add(self.tab_anim, text='Trayectoria (Animación)')
        
        self.tab_analisis = ttk.Frame(tab_control)
        tab_control.add(self.tab_analisis, text='Análisis PSD y Ajuste')
        
        tab_control.pack(expand=1, fill="both")

        # --- 3. Construir Interfaz ---
        self.build_animation_tab()
        self.build_analysis_tab()

    def run_analysis(self):
        """Llama al script de cálculo para procesar los datos fijos."""
        print("Cargando datos y ejecutando análisis...")
        
        # Llamamos a la función principal del script de cálculo
        # Esta función carga los archivos, calcula el PSD y guarda la imagen
        resultado = procesador.procesar_y_guardar()
        
        if 'error' in resultado:
            messagebox.showerror("Error de Datos", 
                f"{resultado['error']}\n\n"
                "Verifica que tus archivos .dat estén en la carpeta correcta "
                "y tengan los nombres correctos (datos_sx.dat, etc).")
            # No cerramos la app para permitir al usuario ver el error
        else:
            self.data = resultado
            print("Datos cargados y procesados exitosamente.")

    def build_animation_tab(self):
        """Construye la pestaña de animación de la partícula."""
        # Panel de información superior
        info_frame = ttk.Frame(self.tab_anim)
        info_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        if self.data:
            # Mostramos los resultados físicos calculados (k y fc)
            # Convertimos k a pN/um para que sea legible
            kx_val = self.data.get('kx_display', 0)
            ky_val = self.data.get('ky_display', 0)

            
            texto_info = f"Resultados: kx={kx_val:.2f} pN/µm | ky={ky_val:.2f} pN/µm"            
            lbl = ttk.Label(info_frame, text=texto_info, font=("Arial", 12, "bold"), foreground="#333")
            lbl.pack()

        # Lienzo de Matplotlib para la animación
        self.fig_anim = procesador.plt.figure(figsize=(6, 6))
        self.ax_anim = self.fig_anim.add_subplot(111)
        
        self.canvas_anim = FigureCanvasTkAgg(self.fig_anim, master=self.tab_anim)
        self.canvas_anim.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Barra de herramientas (Zoom, Pan)
        toolbar = NavigationToolbar2Tk(self.canvas_anim, self.tab_anim)
        toolbar.update()
        
        # Iniciar la animación si hay datos
        if self.data:
            self.start_animation()

    def start_animation(self):
        """Configura y arranca la animación."""
        self.ax_anim.clear()
        self.ax_anim.set_aspect('equal')
        self.ax_anim.grid(True, alpha=0.3, linestyle='--')
        self.ax_anim.set_title("Movimiento Browniano Experimental (Normalizado)")
        self.ax_anim.set_xlabel("X (u.a.)")
        self.ax_anim.set_ylabel("Y (u.a.)")
        
        # Usamos las trayectorias normalizadas que nos devolvió el procesador
        x_data = self.data['traj_x']
        y_data = self.data['traj_y']
        
        # Definir límites fijos para que no 'baile' el eje
        lim = max(np.std(x_data), np.std(y_data)) * 4
        self.ax_anim.set_xlim(-lim, lim)
        self.ax_anim.set_ylim(-lim, lim)
        
        line, = self.ax_anim.plot([], [], 'o', color='cyan', markeredgecolor='blue', zorder=10, label='Partícula')
        trace, = self.ax_anim.plot([], [], '.', color='darkviolet', alpha=0.5, zorder=5)
        self.ax_anim.legend(loc='upper right')
        
        def update(frame):
            step_size = 50
            idx = frame * step_size
            
            if idx >= len(x_data): 
                return line, trace
            
            line.set_data([x_data[idx]], [y_data[idx]])
            
            trace_len = 100
            start = max(0, idx - trace_len)
            trace.set_data(x_data[start:idx+1], y_data[start:idx+1])
            
            return line, trace
            
        self.ani = animation.FuncAnimation(
            self.fig_anim, 
            update, 
            frames=len(x_data)//50, 
            interval=20, 
            blit=True,
            repeat=True # Repetir al terminar
        )

    def build_analysis_tab(self):
        """Construye la pestaña con la gráfica del PSD."""
        # Si el procesador generó una figura (fig), la incrustamos aquí
        if self.data and 'fig' in self.data:
            fig_psd = self.data['fig']
            
            # Canvas para la figura estática
            canvas_psd = FigureCanvasTkAgg(fig_psd, master=self.tab_analisis)
            canvas_psd.draw()
            canvas_psd.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            
            toolbar = NavigationToolbar2Tk(canvas_psd, self.tab_analisis)
            toolbar.update()
        else:
            lbl = ttk.Label(self.tab_analisis, text="No hay datos de análisis disponibles.")
            lbl.pack(pady=20)

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Prueba Visor Experimental")
    root.geometry("1100x800")
    app = ExperimentalViewer(master=root)
    root.mainloop()