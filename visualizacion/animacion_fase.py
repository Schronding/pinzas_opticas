import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# I have always used animation from matplotlib, but while it seems 
# to be very easier to use, in certain ways it seems that manim is 
# cleaner, at least with the forms, as matplotlib sometimes create
# very rough edges on what it is supposed to be round. 
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
    # I don't really recall where this value came from, as it is 
    # similar to the trap stiffness that it is present in practically
    # each one of my files, but I do think it would be better to 
    # return the calculated stiffness from the PSD method in order
    # to have a much more reliable simulation. 
    
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
    # Returning the stifness from the calculation would be better in
    # the sense that I would have a different k value for x and y. 
    
    # Separamos componentes y convertimos a micras y m/s
    x_um = traj[:-1, 0] * 1e6      
    # I don't understand this syntax that much. It is like it says 
    # that it should start from the beginning, and go to the 
    # penultim index with steps of zero ... if I interpret it as the
    # start to the end in reverse I don't see how that syntax works
    # either. Couldn't I just put a simple loop to in which in every
    # value I multiply that value for 1e6?  
    v_x = np.diff(traj[:, 0]) / p.dt 
    # I don't know what the .diff() does, but the syntax seems cryptic
    # again, as I recall that the last value represented the step 
    # but how can I have a step of zero? 
    
    fig, ax = plt.subplots(figsize=(9, 7))
    # It seems that I can get two objects from subplots, in which
    # I assume fig to be the figure that it is going to be animated 
    # and ax the axis. I probably need subplots in order to create
    # a connection between them. 
    
    fig.suptitle(f"Espacio de Fase: Langevin Sobreamortiguado\n(k = {k_target_pN_um} pN/µm)", 
                 fontsize=14, color='white', fontweight='bold')
    # Why subptitle and not subtitle? 
    
    # Fondo oscuro
    ax.set_facecolor('#1a1a1a')
    fig.patch.set_facecolor('#101010')
    ax.grid(True, color='gray', alpha=0.2)
    # I assume alpha to be the thickness. I know that when I use a 
    # function in VS Codium I can hover over it to see all the 
    # arguments, but I don't know how to do it once the line of code
    # has been written. I would like to know though, as that would 
    # allow me to not need to check google in order to know what 
    # a function does. 
    
    # Ejes en blanco
    ax.set_xlabel("Posición X ($\mu m$)", color='white', fontsize=12)
    ax.set_ylabel("Velocidad $v_x$ ($m/s$)", color='white', fontsize=12)
    ax.tick_params(axis='x', colors='white', labelsize=10)
    # What confuses me here is that the ax object has methods for 
    # its x and y axis, but on tick_params (that I am not sure what 
    # they mean) I am not putting the axis that it belongs on the 
    # method, but as an argument. Could this be the marks that 
    # separate one block of the Cartesian grid to the other? 
    ax.tick_params(axis='y', colors='white', labelsize=10)
    
    # Bordes de la gráfica (spines) en blanco para que se vea el marco
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
    # I don't really know what a spine is or what the values of the 
    # method are returned. Could these be the marks of the grid? 

    # Límites
    lim_x = np.std(x_um) * 4
    # I assume .std is the standard deviation. As the rule of 
    # the normal is 1sd - 68%, 2sd - 95% and 3sd - 99.7%, then with
    # 4 standard deviations we have enough room of possibility so the
    # limix of x and y is only surpassed 1 in a 1000 times or less. 
    lim_v = np.std(v_x) * 4
    # As the phase state was dependent on the phase variables (the 
    # ones that describe the entire system with the minimum amount
    # of variables) and the optical tweezers system is only of first
    # order, I just need two, x and dot_x, in which x is the position
    # and dot_x is the speed (the derivate respect to time of the 
    # position). I have not realize until now, it but seems that the 
    # variables of the system are assumed to have spatial symmetry, 
    # as otherwise I would have need 4 variables, 2 extra for the same
    # behavior described with x, but with y. 
    ax.set_xlim(-lim_x, lim_x)
    # I was wrong! It seems that this limit goes beyond 4 standard 
    # deviations; it goes all the way up to 8: 4 on the right, and 
    # 4 on the left. I don't know what the probability is that the 
    # bead will abandon the graph but I assume it must be as 
    # unlikely as one in a million times. 
    ax.set_ylim(-lim_v, lim_v)

    
    # Pendiente m = -k / gamma
    slope = -k_simulacion / p.gamma
    # I don't remember how we got to this derivation of the formula... 
    # I remember that one way to get the phase matrix was by doing
    # substitution of variables... But I have not visualized yet how
    # the slope could be determined by stifness and the viscosity 
    # coefficient alone, as both seem to describe the movement, 
    # but what does the slope means physically in here? 
    x_theory = np.linspace(-lim_x, lim_x, 100)
    # I assume the third argument means the width of a line, as I 
    # also assume that linspace represents a line. 
    v_theory = (slope * (x_theory * 1e-6)) # m/s
    # This is the same formula that it is written in the graph 
    # (v=(-kx)/gamma). I don't know how the 
    
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