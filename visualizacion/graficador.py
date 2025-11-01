# -*- coding: utf-8 -*-
"""
Este módulo contiene funciones auxiliares para graficar
elementos visuales de la simulación, como los mapas de fuerza.
"""

import numpy as np
import matplotlib.cm as cm
from utils import parametros as p

def draw_force_map(ax, anharmonic_mode=False, fx_interp=None, fy_interp=None, lim_nm=300):
    """
    Dibuja la magnitud de la fuerza (el "haz de luz")
    en el fondo de un eje (ax) de matplotlib.

    Args:
        ax (matplotlib.axes.Axes): El eje donde se dibujará.
        anharmonic_mode (bool): Flag para decidir el modo.
        fx_interp (func): Interpolador para Fx (requerido si es anharmónico).
        fy_interp (func): Interpolador para Fy (requerido si es anharmónico).
        lim_nm (float): El rango (ej. 300 para -300 a +300 nm) del mapa.
    """
    print(f"Dibujando mapa de fuerza (Modo: {'Anharmónico' if anharmonic_mode else 'Armónico'})...")
    
    # Crear una malla de 100x100 puntos para el fondo
    x = np.linspace(-lim_nm, lim_nm, 100)
    y = np.linspace(-lim_nm, lim_nm, 100)
    xx_nm, yy_nm = np.meshgrid(x, y)
    
    # Calcular la magnitud de la fuerza (Z) en cada punto
    if anharmonic_mode and fx_interp:
        # --- CORRECCIÓN CRÍTICA ---
        # 1. Preparar los puntos para el interpolador
        # Necesita una lista de (N, 2) puntos, no una malla
        points_nm = np.vstack((xx_nm.ravel(), yy_nm.ravel())).T
        
        # 2. Llamar al interpolador (con la lista de puntos)
        Fx_map_flat = fx_interp(points_nm)
        Fy_map_flat = fy_interp(points_nm)
        
        # 3. Calcular magnitud y reformar a la malla
        Z_flat = np.sqrt(Fx_map_flat**2 + Fy_map_flat**2)
        Z = Z_flat.reshape(xx_nm.shape)
        # --- FIN DE LA CORRECCIÓN ---

    else:
        # Modo Armónico: Usar k*r
        xx_m = xx_nm * 1e-9
        yy_m = yy_nm * 1e-9
        Fx_map = -p.kappa_x * xx_m
        Fy_map = -p.kappa_y * yy_m
        Z = np.sqrt(Fx_map**2 + Fy_map**2)
        
    # Dibujar el contorno de fondo
    ax.contourf(
        xx_nm, yy_nm, Z, 
        levels=20, 
        cmap=cm.plasma, # Puedes cambiar a 'viridis', 'inferno', etc.
        alpha=0.6, 
        zorder=1
    )