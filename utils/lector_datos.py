# -*- coding: utf-8 -*-
"""
Módulo para cargar mapas de fuerza e intensidad desde archivos CSV.
"""

import numpy as np
from scipy.interpolate import LinearNDInterpolator

def cargar_mapa_fuerzas(filepath):
    """
    Carga un mapa desde un CSV. Soporta formato simple (4 cols) y completo (8 cols).

    Returns:
        tuple: (fx_interp, fy_interp, int_interp)
               Devuelve interpoladores para Fuerza X, Fuerza Y, e Intensidad.
               Si el archivo no tiene intensidad, int_interp será None.
    """
    print(f"Cargando mapa desde: {filepath}")
    try:
        # Leemos saltando el encabezado
        data = np.loadtxt(filepath, delimiter=',', skiprows=1)
        num_cols = data.shape[1]
        
        # Coordenadas (siempre son las primeras 2 columnas)
        points = data[:, [0, 1]] # Columnas 0 (X) y 1 (Y)

        # --- Detección del tipo de archivo ---
        if num_cols == 8:
            print("Formato detectado: COMPLETO (Fuerza + Intensidad)")
            # Col 3: Fx, Col 4: Fy, Col 5: Fz, Col 7: Fm, Col 8: Intensidad
            values_Fx = data[:, 3]
            values_Fy = data[:, 4]
            values_Int = data[:, 7]
        elif num_cols == 7:
            print("Formato detectado: SIMPLE (Solo Fuerza)")
            # Col 2: Fx, Col 3: Fy
            values_Fx = data[:, 3]
            values_Fy = data[:, 4]
            values_Int = None
        else:
            raise ValueError(f"Formato de archivo desconocido ({num_cols} columnas).")

        # Crear interpoladores (fill_value=0.0 para que fuera del rango la fuerza sea 0)
        fx_i = LinearNDInterpolator(points, values_Fx, fill_value=0.0)
        fy_i = LinearNDInterpolator(points, values_Fy, fill_value=0.0)
        
        int_i = None
        if values_Int is not None:
             int_i = LinearNDInterpolator(points, values_Int, fill_value=0.0)

        print("Interpoladores creados exitosamente.")
        return fx_i, fy_i, int_i

    except Exception as e:
        print(f"Error al cargar el mapa: {e}")
        return None, None, None