# -*- coding: utf-8 -*-
"""
Este script contiene el motor de la simulación.
Implementa el algoritmo de Euler-Maruyama para resolver la ecuación de
Langevin sobreamortiguada y calcular la trayectoria de la partícula.

** Versión 2: Admite modo armónico (F = -kx) y
** modo anharmónico (usando mapa de fuerzas interpolado).
"""

import numpy as np
import sys
import os

# --- INICIO DE SOLUCIÓN DE RUTA ---
# Añadir el directorio raíz del proyecto al sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
# --- FIN DE SOLUCIÓN DE RUTA ---

from utils import parametros as p
# Importamos el lector que creamos (solo para el bloque de prueba)
from utils import lector_datos

def run_simulation(total_steps, dt, gamma, k_B, T, 
                   k_x=None, k_y=None, 
                   fx_interp=None, fy_interp=None):
    """
    Ejecuta una simulación de movimiento Browniano en una trampa óptica 2D.

    Puede operar en dos modos:
    1. Armónico: Proporcionando k_x y k_y.
    2. Anharmónico: Proporcionando fx_interp y fy_interp.

    Args:
        total_steps (int): Número total de pasos.
        dt (float): Paso de tiempo (s).
        gamma (float): Coef. de arrastre (N·s/m).
        k_B (float): Constante de Boltzmann (J/K).
        T (float): Temperatura (K).
        k_x (float, opcional): Rigidez armónica en X (N/m).
        k_y (float, opcional): Rigidez armónica en Y (N/m).
        fx_interp (func, opcional): Interpolador para la fuerza en X.
        fy_interp (func, opcional): Interpolador para la fuerza en Y.

    Returns:
        np.ndarray: Array (total_steps, 2) con posiciones [x, y] en metros.
    """
    # 1. Inicialización
    trajectory = np.zeros((total_steps, 2))
    
    # 2. Cálculo del término de la fuerza estocástica (ruido)
    diffusion_coeff = k_B * T / gamma
    noise_magnitude = np.sqrt(2 * diffusion_coeff * dt)

    # 3. Bucle de Simulación (Método de Euler-Maruyama)
    for i in range(1, total_steps):
        # Posición en el paso anterior (en metros)
        x_prev, y_prev = trajectory[i-1]

        # =================================================================
        # INICIO DE LA LÓGICA DE FUERZA ACTUALIZADA
        # =================================================================
        
        # --- MODO 2: ANHARMÓNICO (Mapa de fuerzas de MATLAB) ---
        if fx_interp is not None and fy_interp is not None:
            # Convertir posición (m) a (nm)
            x_prev_nm = x_prev * 1e9
            y_prev_nm = y_prev * 1e9
            
            # --- CORRECCIÓN CRÍTICA ---
            # El interpolador espera un único punto [x, y]
            punto_nm = [x_prev_nm, y_prev_nm]
            
            # La llamada correcta es fx_interp(punto), no fx_interp(x, y)
            force_x = fx_interp(punto_nm)
            force_y = fy_interp(punto_nm)
            
        # --- MODO 1: ARMÓNICO (Modelo de Resorte Simple) ---
        elif k_x is not None and k_y is not None:
            # F = -κ * r  (Ley de Hooke)
            force_x = -k_x * x_prev
            force_y = -k_y * y_prev
            
        # --- Error ---
        else:
            raise ValueError("Error en run_simulation: Debes proveer "
                             "(k_x, k_y) o (fx_interp, fy_interp).")
        
        # =================================================================
        # FIN DE LA LÓGICA DE FUERZA
        # =================================================================

        # b) Generación de la fuerza estocástica (Browniana)
        random_force_x = noise_magnitude * np.random.randn()
        random_force_y = noise_magnitude * np.random.randn()

        # c) Actualización de la posición (Ecuación de Langevin)
        trajectory[i, 0] = x_prev + (force_x / gamma) * dt + random_force_x
        trajectory[i, 1] = y_prev + (force_y / gamma) * dt + random_force_y
        
    return trajectory

# =============================================================================
# BLOQUE DE PRUEBA ACTUALIZADO
# =============================================================================
if __name__ == '__main__':
    
    print("="*50)
    print("EJECUTANDO PRUEBA DEL SIMULADOR (MODO ARMÓNICO)")
    print("="*50)
    
    # Prueba 1: Modo Armónico (como antes)
    trayectoria_armonica = run_simulation(
        total_steps=10000,  # Prueba corta
        dt=p.dt,
        gamma=p.gamma,
        k_B=p.k_B,
        T=p.T,
        k_x=p.kappa_x,  # Proporcionamos kappas
        k_y=p.kappa_y
    )
    print("Simulación armónica completada.")
    print(f"Posición final: {trayectoria_armonica[-1] * 1e9} nm\n")

    
    print("="*50)
    print("EJECUTANDO PRUEBA DEL SIMULADOR (MODO ANHARMÓNICO)")
    print("="*50)
    
    # Prueba 2: Modo Anharmónico
    
    # 2.1. Cargar el mapa de fuerzas
    # Asumimos que el archivo .csv está en la raíz del proyecto
    ruta_mapa = os.path.join(parent_dir, 'fuerzas_particula_mie.csv')
    fx_i, fy_i = lector_datos.cargar_mapa_fuerzas(ruta_mapa)
    
    if fx_i:
        # 2.2. Ejecutar simulación pasando los interpoladores
        trayectoria_anharmonica = run_simulation(
            total_steps=10000, # Prueba corta
            dt=p.dt,
            gamma=p.gamma,
            k_B=p.k_B,
            T=p.T,
            fx_interp=fx_i,  # Proporcionamos interpoladores
            fy_interp=fy_i
        )
        print("Simulación anharmónica completada.")
        print(f"Posición final: {trayectoria_anharmonica[-1] * 1e9} nm\n")
    else:
        print("PRUEBA ANHARMÓNICA OMITIDA: No se encontró 'fuerzas_particula_mie.csv' en la raíz.")
        print("Asegúrate de tener el archivo para probar este modo.")