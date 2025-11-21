import numpy as np
import matplotlib.cm as cm
from utils import parametros as p

def draw_background(ax, viz_mode='force', anharmonic_mode=False, 
                    fx_i=None, fy_i=None, int_i=None, lim_nm=300):
    """
    Dibuja el mapa de fondo según el modo de visualización seleccionado.

    Args:
        viz_mode (str): 'force' (Magnitud de Fuerza) o 'intensity' (Intensidad del Haz).
    """
    if viz_mode == 'none':
        return

    print(f"Dibujando fondo: {viz_mode.upper()} (Modo sim: {'Anharmónico' if anharmonic_mode else 'Armónico'})")
    
    grid_points = 150 # Mayor resolución para mejor visualización
    x = np.linspace(-lim_nm, lim_nm, grid_points)
    y = np.linspace(-lim_nm, lim_nm, grid_points)
    xx, yy = np.meshgrid(x, y)

    # 2. Calcular Z (los valores a graficar) y elegir Colormap
    Z = None
    cmap = cm.viridis # Default

    if viz_mode == 'intensity':
        # --- MODO INTENSIDAD ---
        if int_i is not None:
            # Solo disponible si tenemos el interpolador de intensidad
            points = np.vstack((xx.ravel(), yy.ravel())).T
            Z = int_i(points).reshape(xx.shape)
            cmap = cm.hot # 'hot' o 'inferno' son buenos para intensidad láser
        else:
            print("Aviso: Se pidió intensidad pero no hay datos disponibles.")
            return

    elif viz_mode == 'force':
        # --- MODO FUERZA ---
        cmap = cm.plasma # 'plasma' o 'jet' para fuerzas
        if anharmonic_mode and fx_i and fy_i:
            points = np.vstack((xx.ravel(), yy.ravel())).T
            fx_map = fx_i(points).reshape(xx.shape)
            fy_map = fy_i(points).reshape(xx.shape)
            Z = np.sqrt(fx_map**2 + fy_map**2)
        else:
            # Fuerza Armónica (F = -kr)
            # Convertir nm a m para el cálculo físico
            fx_map = -p.kappa_x * (xx * 1e-9)
            fy_map = -p.kappa_y * (yy * 1e-9)
            Z = np.sqrt(fx_map**2 + fy_map**2)

    # 3. Dibujar si tenemos datos (Z)
    if Z is not None:
        ax.contourf(xx, yy, Z, levels=25, cmap=cmap, alpha=0.7, zorder=1)