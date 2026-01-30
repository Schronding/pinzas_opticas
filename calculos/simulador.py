"""
Simulation Engine.
Implements the Euler-Maruyama method to solve the Overdamped Langevin Equation
and calculate the particle's trajectory.

Supports both Harmonic (F = -kx) and Anharmonic modes 
(using interpolated force maps from external files).
"""

import numpy as np
import sys
import os

# Set up paths to import utility modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import parametros as p
from utils import lector_datos

def run_simulation(total_steps, dt, gamma, k_B, T, 
                   k_x=None, k_y=None, 
                   fx_interp=None, fy_interp=None):
    """
    Runs a Brownian motion simulation in a 2D optical trap.
    
    Args:
        total_steps (int): Total simulation steps.
        dt (float): Time step in seconds.
        gamma (float): Drag coefficient (Stokes).
        k_B (float): Boltzmann constant.
        T (float): Temperature in Kelvin.
        k_x, k_y (float, optional): Stiffness for Harmonic mode.
        fx_interp, fy_interp (func, optional): Interpolator functions for Anharmonic mode.
    
    Returns:
        np.ndarray: Trajectory matrix of shape (total_steps, 2).
    """

    # 1. Initialize Trajectory Matrix
    # I use a tuple (total_steps, 2) to define the shape: Rows = time, Cols = X/Y dimensions.
    # Pre-allocating with zeros is more efficient than appending to a list.
    trajectory = np.zeros((total_steps, 2))
    
    # 2. Calculate Stochastic Parameters
    # Einstein Relation: Connects the physical drag (gamma) to thermal jitter (Diffusion).
    # D = k_B * T / gamma
    diffusion_coeff = k_B * T / gamma

    # Noise Magnitude (Standard Deviation of the Wiener Process)
    # Random walks scale with sqrt(t). Therefore, the noise for a single step dt
    # must be scaled by sqrt(2*D*dt) to match the physical diffusion rate.
    noise_magnitude = np.sqrt(2 * diffusion_coeff * dt)

    # 3. Simulation Loop (Euler-Maruyama Method)
    # Starting at index 1 because index 0 is initialized to (0,0) [Trap Center].
    for i in range(1, total_steps):
        
        # Current Position (Memory of the 'now' to calculate the 'next')
        # Even though the physics is Markovian (no memory), the solver needs the 
        # previous step's location to compute the forces acting on the bead.
        x_prev, y_prev = trajectory[i-1]
        
        # --- A. Deterministic Force Calculation ---
        if fx_interp is not None and fy_interp is not None:
            # Mode: Anharmonic (Interpolated Force Map)
            # The interpolator function (LinearNDInterpolator) expects coordinates in the 
            # same units used in the CSV map. Based on the CFATA data, these are nanometers.
            # I convert meters -> nm before passing to the function.
            point_nm = [x_prev * 1e9, y_prev * 1e9]
            
            # Here, fx_interp is the 'tool' passed as an argument that calculates force.
            force_x = fx_interp(point_nm)
            force_y = fy_interp(point_nm)
            
        elif k_x is not None and k_y is not None:
            # Mode: Harmonic (Linear Spring)
            # F = -kx. The negative sign ensures the Restoring Force always points 
            # towards the equilibrium center (0,0), regardless of the bead's direction.
            force_x = -k_x * x_prev
            force_y = -k_y * y_prev
            
        else:
            raise ValueError("Simulation Error: You must provide either (k_x, k_y) "
                             "for Harmonic mode or (fx_interp, fy_interp) for Anharmonic mode.")
        
        # --- B. Stochastic Force Calculation (Thermal Noise) ---
        # np.random.randn() generates a number from a Normal Distribution N(0,1).
        # I multiply by noise_magnitude to scale it to the physical system.
        random_force_x = noise_magnitude * np.random.randn()
        random_force_y = noise_magnitude * np.random.randn()

        # --- C. Update Position (Langevin Equation) ---
        # x_new = x_old + (Drift_Velocity * dt) + (Random_Step)
        # where Drift_Velocity = Force / gamma
        trajectory[i, 0] = x_prev + (force_x / gamma) * dt + random_force_x
        trajectory[i, 1] = y_prev + (force_y / gamma) * dt + random_force_y
        
    return trajectory

if __name__ == '__main__':
    
    print("="*50)
    print("RUNNING SIMULATOR TEST (HARMONIC MODE)")
    print("="*50)
    
    # Using parameters imported from 'utils.parametros'
    try:
        trayectoria_armonica = run_simulation(
            total_steps=100000, # Reduced steps for quick testing
            dt=p.dt,
            gamma=p.gamma,
            k_B=p.k_B,
            T=p.T,
            k_x=p.kappa_x,  
            k_y=p.kappa_y
        )
        print("Harmonic simulation completed successfully.")
        print(f"Final Position: {trayectoria_armonica[-1] * 1e9} nm\n")
    except Exception as e:
        print(f"Harmonic simulation failed: {e}")

    # Path to the anharmonic force map
    ruta_mapa = os.path.join(parent_dir, 'fuerzas_particula_mie.csv')
    
    if os.path.exists(ruta_mapa):
        print("="*50)
        print("RUNNING SIMULATOR TEST (ANHARMONIC MODE)")
        print("="*50)
        
        # Note: lector_datos returns 3 values (Fx, Fy, Intensity). 
        # I use '_' to ignore the Intensity interpolator as it's not needed for motion.
        fx_i, fy_i, _ = lector_datos.cargar_mapa_fuerzas(ruta_mapa)
        
        if fx_i:
            trayectoria_anharmonica = run_simulation(
                total_steps=100000, 
                dt=p.dt,
                gamma=p.gamma,
                k_B=p.k_B,
                T=p.T,
                fx_interp=fx_i,  
                fy_interp=fy_i
            )
            print("Anharmonic simulation completed successfully.")
            print(f"Final Position: {trayectoria_anharmonica[-1] * 1e9} nm\n")
    else:
        print("ANHARMONIC TEST SKIPPED: 'fuerzas_particula_mie.csv' not found.")