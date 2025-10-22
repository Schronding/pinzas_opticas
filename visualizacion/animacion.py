import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from calculos import simulador
from utils import parametros as p

print("Generando trayectoria para la animación...")
trajectory = simulador.run_simulation(
    total_steps=p.total_steps,
    dt=p.dt,
    k_x=p.kappa_x,
    k_y=p.kappa_y,
    gamma=p.gamma,
    k_B=p.k_B,
    T=p.T
)
print("¡Trayectoria generada!")

# Convertir de metros a nanómetros (1e9) 
trajectory_nm = trajectory * 1e9

fig, ax = plt.subplots(figsize=(8, 8))

# Punto que simula a la esfera
line, = ax.plot([], [], 'o', markersize=6, color='royalblue')
# Estela que deja donde se movio
trace, = ax.plot([], [], '-', lw=1, alpha=0.5, color='orange')

def init():
    std_dev_x = np.std(trajectory_nm[:, 0])
    std_dev_y = np.std(trajectory_nm[:, 1])
    max_range = max(std_dev_x, std_dev_y, 10.0) * 4 
    
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_xlabel('Posición X (nm)')
    ax.set_ylabel('Posición Y (nm)')
    ax.set_title('Simulación de Pinza Óptica (Movimiento Browniano)')
    ax.grid(True)
    ax.set_aspect('equal') # Ejes x e y con la misma escala
    
    line.set_data([], [])
    trace.set_data([], [])
    return line, trace


def animate(i):
    step = i * 100
    
    if step >= len(trajectory_nm):
        return line, trace

    x_point = trajectory_nm[step, 0]
    y_point = trajectory_nm[step, 1]
    line.set_data([x_point], [y_point])

    start_trace = max(0, step - 1000)
    x_trace = trajectory_nm[start_trace:step, 0]
    y_trace = trajectory_nm[start_trace:step, 1]
    trace.set_data(x_trace, y_trace)

    return line, trace


num_frames = len(trajectory_nm) // 100

print("Iniciando animación... (Esto puede tardar unos segundos)")


ani = animation.FuncAnimation(
    fig,
    animate,              # La función que actualiza cada fotograma
    init_func=init,       # La función que prepara el lienzo
    frames=num_frames,    # El número total de fotogramas a dibujar
    interval=20,          # Milisegundos entre fotogramas (50 fotogramas/seg)
    blit=True,            # blit=True es una optimización para animacion mucho más fluida
    repeat=False          # No repetir la animación al terminar
)

plt.show()

print("Animación finalizada.")