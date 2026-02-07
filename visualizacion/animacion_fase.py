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
    # (v=(-kx)/gamma). As there is a graph of x and velocity I imagine
    # this is indeed just the behavior on one dimension, as I would need 4
    # dimensions in order to see both axis and the speed on each of them. 
    # If I assume that the speed is the same in both axis then I would reduce
    # it to 3, but the graph would probably be very boring. 
    
    ax.plot(x_theory, v_theory, color='cyan', linestyle='--', alpha=0.5, lw=2, 
            label='Deriva Determinista ($v = -kx/\gamma$)')
    # This line seems to be just the label that appears on the upper-right
    # corner. The 'label' attribute makes me think that. 

    trail_len = 50
    # I can modify this directly in order to see a longer or shorter path
    # to which the bead moves. 
    scatter = ax.scatter([], [], s=60, c='magenta', edgecolors='white', zorder=10, 
                         label='Estado $(x, v)$')
    # This function is interesting, because it seems to also be part of the 
    # labels that are in the upper-right corner, but its name is different. 
    # Instead of .plot() it uses .scatter(). I would like to know what both
    # of these mean and why I am passing two empty lists as arguments. 
    # As the 'edgecolors' attribute is white, this makes me think this 
    # describes the border of the bead, but it is interesting that it seems
    # that in one instruction I managed to do two things: movement and a label.
    trail, = ax.plot([], [], color='magenta', alpha=1, lw=1)
    # The variable name is very explicit, this is the trail. 'alpha' seems to
    # describe the saturation of the color 'magenta' as when I increased it 
    # from 0.3 to 0.7 I see a brighter color. When I tried 1.3 to see if I 
    # could get something very bright the code broke. 
    # ValueError: alpha (1.3) is outside 0-1 range. I assume that if I put it
    # on 1 I will get full saturation and if I put it in a 0, I will not 
    # see the trail (and maybe not even the bead) at all. Indeed I didn't
    # see anything with 0, but the bead stayed though. I haven't noticed but
    # there is a comma after trail that doesn't break the code. It this a way
    # to ignore a return from which I don't care losing its value? 
    # When I removed it I see 
    # AttributeError: 'FuncAnimation' object has no attribute '_resize_id'
    # It is important, but I would like to know why. 
    
    ax.legend(loc='upper right', facecolor='#333', edgecolor='white', labelcolor='white')
    # This seems to just be the border that contains the labels. What is 
    # interesting is that if I comment the line, all the square dissapears,
    # even the text that was inserted before (the formula of velocity and
    # the state (x,v)). 

    def update(frame):
        i = frame 
        # I suppose it goes from spaces of 10 to 10 to not have the animation
        # last longer. When I removed the 10 the animation turned out to be 
        # much "sharper", as while the bead seemed to move on longer steps
        # the trail had a lot of rough corners. All the animations of matplotlib
        # seem to have that characteristic of sharp corners though. 
        if i >= len(x_um): return scatter, trail
        # What I wonder about this line is how it can work without a for loop.
        # My intuition says that update() is being called inside a loop, and that
        # is the reason why we can do without one here. 
        
        scatter.set_offsets([[x_um[i], v_x[i]]])
        # I wonder what the set_offsets() method does. At least in the 
        # argument it seems that we are just selecting each stored value of
        # positions and velocities in the 'x_um' and 'v_x' lists with an index. 
        
        start = max(0, i - trail_len)
        # I think I understand this logic: we check how many steps are left
        # (as i increases each time update() is called) and then we subtract
        # that from the total steps that need to be rendered... but that is
        # not what 'trail_len' is; 'trail_len' is just how large the path
        # of the bead is being rendered... then why do we need it?  

        trail.set_data(x_um[start:i], v_x[start:i])
        # I would like to know what set_data does too. 
        return scatter, trail
        # I wonder why we need these two different paths to arrive at 
        # return scatter, trail. I don't think the if statement above is to
        # have a different behavior, it might be just to avoid the calculations
        # that are below. One reason for doing so might be that we don't want
        # to display the trail unless it is necessary; that is when we have 
        # surpassed the need length of the trail. 

    ani = animation.FuncAnimation(fig, update, frames=len(x_um)//10, interval=20, blit=True)
    # There doesn't seem to be a reason why we store the animation in a 
    # variable. I wonder what blit means. What seems interesting is that 
    plt.show()


if __name__ == "__main__":
    animar_espacio_fase()