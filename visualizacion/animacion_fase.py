import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from calculos import simulador
from utils import parametros as p

def animar_espacio_fase():
    
    # Aquí fijamos la K  (1.98 pN/um)
    k_target_pN_um = 1.98 
    
    # Conversión a unidades del SI (N/m) para el simulador
    # 1 pN/um = 1e-12 N / 1e-6 m = 1e-6 N/m
    k_simulacion = k_target_pN_um * 1e-6 
    
    steps = 5000
    traj = simulador.run_simulation(
        total_steps=steps,
        dt=p.dt,
        gamma=p.gamma,
        k_B=p.k_B,
        T=p.T,
        k_x=k_simulacion, # Usamos tu K personalizada
        k_y=k_simulacion
    )
    
    # Separamos componentes y convertimos a micras y m/s
    x_um = traj[:-1, 0] * 1e6       
    v_x = np.diff(traj[:, 0]) / p.dt 
    
    fig, ax = plt.subplots(figsize=(9, 7))
    
    fig.suptitle(f"Espacio de Fase: Langevin Sobreamortiguado\n(k = {k_target_pN_um} pN/µm)", 
                 fontsize=14, color='white', fontweight='bold')
    
    # Fondo oscuro
    ax.set_facecolor('#1a1a1a')
    fig.patch.set_facecolor('#101010')
    ax.grid(True, color='gray', alpha=0.2)
    
    # Ejes en blanco
    ax.set_xlabel("Posición X ($\mu m$)", color='white', fontsize=12)
    ax.set_ylabel("Velocidad $v_x$ ($m/s$)", color='white', fontsize=12)
    ax.tick_params(axis='x', colors='white', labelsize=10)
    ax.tick_params(axis='y', colors='white', labelsize=10)
    
    # Bordes de la gráfica (spines) en blanco para que se vea el marco
    for spine in ax.spines.values():
        spine.set_edgecolor('white')

    # Límites
    lim_x = np.std(x_um) * 4
    lim_v = np.std(v_x) * 4
    ax.set_xlim(-lim_x, lim_x)
    ax.set_ylim(-lim_v, lim_v)

    
    # Pendiente m = -k / gamma
    slope = -k_simulacion / p.gamma
    x_theory = np.linspace(-lim_x, lim_x, 100)
    v_theory = (slope * (x_theory * 1e-6)) # m/s
    
    ax.plot(x_theory, v_theory, color='cyan', linestyle='--', alpha=0.5, lw=2, 
            label='Deriva Determinista ($v = -kx/\gamma$)')

    trail_len = 100
    scatter = ax.scatter([], [], s=60, c='magenta', edgecolors='white', zorder=10, 
                         label='Estado $(x, v)$')
    trail, = ax.plot([], [], color='magenta', alpha=0.3, lw=1)
    
    ax.legend(loc='upper right', facecolor='#333', edgecolor='white', labelcolor='white')

    def update(frame):
        i = frame * 10 
        if i >= len(x_um): return scatter, trail
        
        scatter.set_offsets([[x_um[i], v_x[i]]])
        
        start = max(0, i - trail_len)
        trail.set_data(x_um[start:i], v_x[start:i])
        return scatter, trail

    ani = animation.FuncAnimation(fig, update, frames=len(x_um)//10, interval=20, blit=True)
    plt.show()

if __name__ == "__main__":
    animar_espacio_fase()