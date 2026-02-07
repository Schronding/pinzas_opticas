import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
import os

# Path configuration
# I need to ensure Python can find the 'utils' and 'calculos' modules 
# located in the parent directory.
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from calculos import simulador
from utils import parametros as p

def animate_phase_space():
    """
    Generates an animated phase space diagram (Velocity vs. Position) 
    for a particle in an optical trap.
    """
    
    # Simulation Setup
    # Target stiffness for the simulation. 
    # I am explicitly defining this here rather than calculating it from PSD
    # because I want a clean, controlled value for the visualization.
    k_target_pN_um = 1.98 
    
    # Converting to SI units (N/m). 
    # Factor 1e-6 comes from: pN (1e-12) / um (1e-6) = 1e-6.
    k_sim = k_target_pN_um * 1e-6 
    
    steps = 5000
    
    # Running the simulation using the imported module.
    # I am passing the specific 'k' I defined above to override defaults.
    traj = simulador.run_simulation(
        total_steps=steps,
        dt=p.dt,
        gamma=p.gamma,
        k_B=p.k_B,
        T=p.T,
        k_x=k_sim,
        k_y=k_sim
    )

    # Data Pre-Proccessing 
    
    # Velocity Calculation: v = dx / dt
    # np.diff(traj) calculates (x[i+1] - x[i]). The resulting array is 
    # 1 element shorter than the original trajectory.
    v_x = np.diff(traj[:, 0]) / p.dt 
    
    # Position Slicing:
    # Since v_x has length (N-1), I must slice the position array 
    # to exclude the last point ([:-1]) so both arrays have the same length.
    x_um = traj[:-1, 0] * 1e6      

    # Figure Setup
    
    # Creating the Canvas (fig) and the Plotting Area (ax).
    fig, ax = plt.subplots(figsize=(9, 7))
    
    # 'suptitle' stands for Super Title (the title of the whole window/canvas).
    fig.suptitle(f"Phase Space: Overdamped Langevin\n(k = {k_target_pN_um} pN/µm)", 
                 fontsize=14, color='white', fontweight='bold')
    
    # Styling: Using a dark theme for better contrast of the bright bead.
    ax.set_facecolor('#1a1a1a')       # Dark gray plotting area
    fig.patch.set_facecolor('#101010') # Almost black window background
    
    # Grid configuration. 'alpha' controls transparency (0.2 = very faint).
    ax.grid(True, color='gray', alpha=0.2)
    
    # Labeling axes in white to match the dark theme.
    ax.set_xlabel("Position X ($\mu m$)", color='white', fontsize=12)
    ax.set_ylabel("Velocity $v_x$ ($m/s$)", color='white', fontsize=12)
    
    # Styling the ticks (the numbers on the axis).
    # tick_params allows me to color the numbers white.
    ax.tick_params(axis='x', colors='white', labelsize=10)
    ax.tick_params(axis='y', colors='white', labelsize=10)
    
    # Styling the Spines (the square border box of the plot).
    for spine in ax.spines.values():
        spine.set_edgecolor('white')

    # Dynamic Limits
    
    # I use Standard Deviation (sigma) to set the zoom level.
    # 4 sigma covers 99.99% of the particle's movement, ensuring it stays in frame.
    lim_x = np.std(x_um) * 4
    lim_v = np.std(v_x) * 4
    
    ax.set_xlim(-lim_x, lim_x)
    ax.set_ylim(-lim_v, lim_v)

    # Theoretical Expectation
    
    # The "Deterministic Drift" line.
    # In overdamped systems, v = Force / gamma.
    # Since Force = -kx, then v = -(k/gamma) * x.
    # This line represents where the bead "wants" to be if there were no thermal noise.
    slope = -k_sim / p.gamma
    
    # Generating X points for the theoretical line (100 points is enough for a straight line)
    x_theory = np.linspace(-lim_x, lim_x, 100)
    v_theory = (slope * (x_theory * 1e-6)) # Convert x back to meters for calc
    
    ax.plot(x_theory, v_theory, color='cyan', linestyle='--', alpha=0.5, lw=2, 
            label=r'Deterministic Drift ($v = -kx/\gamma$)')

    # Animation Elements
    
    # 1. The Head (The Bead): 
    # Created with 'scatter'. Ideally suited for moving dots.
    # I initialize it with empty lists []; data is added in the update loop.
    scatter = ax.scatter([], [], s=60, c='magenta', edgecolors='white', zorder=10, 
                         label='State $(x, v)$')
    
    # 2. The Tail (The Trail):
    # Created with 'plot'. Ideally suited for lines.
    # Note the comma "trail, =". ax.plot returns a list of lines; 
    # I unpack it to get the single Line2D object I need to modify.
    trail, = ax.plot([], [], color='magenta', alpha=0.5, lw=1)
    
    # Length of the tail (fading memory visual).
    trail_len = 50
    
    # Adding the legend. 
    # Locating it upper-right so it doesn't obscure the center activity.
    ax.legend(loc='upper right', facecolor='#333', edgecolor='white', labelcolor='white')

    # Animation Loop
    
    def update(frame):
        """
        Called for every frame of the animation.
        'frame' is the index passed by FuncAnimation.
        """
        # I check if we are out of bounds just in case.
        if frame >= len(x_um): 
            return scatter, trail
        
        # Update the Bead (Head) position
        # Scatter plots require 'offsets' in [[x, y]] format.
        scatter.set_offsets([[x_um[frame], v_x[frame]]])
        
        # Update the Trail (Tail) data
        # I calculate the start index. max(0, ...) ensures I don't use negative indices
        # at the beginning of the animation.
        start = max(0, frame - trail_len)
        
        # Line plots require separate arrays for X and Y data.
        trail.set_data(x_um[start:frame], v_x[start:frame])
        
        return scatter, trail

    # Creating the animation object.
    # frames: I use range(0, len, 10) to 'decimate' the data. 
    # This plays the animation 10x faster (skipping 9 real steps between frames)
    # so the viewer doesn't get bored watching 5000 slow steps.
    ani = animation.FuncAnimation(
        fig, 
        update, 
        frames=range(0, len(x_um), 10), 
        interval=20, # 20ms between frames = 50 fps
        blit=True    # Optimization: only redraw changed parts
    )
    
    print("Displaying animation...")
    plt.show()

if __name__ == "__main__":
    # Guard clause: This ensures the animation only runs if I execute 
    # this file directly, not if I import 'animate_phase_space' elsewhere.
    animate_phase_space()