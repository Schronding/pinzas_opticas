# -*- coding: utf-8 -*-
"""
Este módulo se encarga de cargar y procesar los mapas de fuerza
generados externamente (ej. por MATLAB).

Utiliza interpolación para permitir que la simulación consulte la fuerza
en cualquier punto (x, y), no solo en los puntos de la malla pre-calculada.
"""

import numpy as np
from scipy.interpolate import LinearNDInterpolator
import os

def cargar_mapa_fuerzas(filepath):
    """
    Carga un mapa de fuerzas desde un archivo .csv y crea interpoladores.

    El archivo debe tener el formato: x, y, Fx, Fy

    Args:
        filepath (str): Ruta al archivo .csv del mapa de fuerzas.

    Returns:
        tuple: (interpolador_Fx, interpolador_Fy)
               Dos funciones que toman (x, y) y devuelven la fuerza.
               Devuelve (None, None) si el archivo no se encuentra.
    """
    print(f"Cargando mapa de fuerzas desde: {filepath}")
    try:
        # Cargar los datos desde el archivo de texto
        data = np.loadtxt(filepath, delimiter=',', skiprows=1)
    except IOError:
        print(f"Error: No se pudo encontrar el archivo {filepath}")
        return None, None

    # Extraer las columnas
    points = data[:, :2]  # Columnas 0 y 1 (x, y)
    values_Fx = data[:, 2] # Columna 2 (Fx)
    values_Fy = data[:, 3] # Columna 3 (Fy)

    # Crear los objetos de interpolación.
    # 'fill_value=0.0' significa que si la partícula se sale
    # del área pre-calculada, la fuerza óptica será 0.
    interpolador_Fx = LinearNDInterpolator(points, values_Fx, fill_value=0.0)
    interpolador_Fy = LinearNDInterpolator(points, values_Fy, fill_value=0.0)
    
    print("Mapa de fuerzas cargado e interpoladores creados exitosamente.")
    return interpolador_Fx, interpolador_Fy

# =============================================================================
# BLOQUE DE PRUEBA
# =============================================================================
if __name__ == '__main__':
    # Este bloque de prueba crea un archivo .csv falso
    # para simular el que generaría MATLAB.
    
    print("Ejecutando prueba de 'lector_datos.py'...")
    
    # 1. Crear un archivo .csv de prueba
    test_filepath = 'mapa_fuerzas_prueba.csv'
    
    # Creamos una malla simple de 3x3 (-1, 0, 1)
    x_coords = np.array([-1.0, 0.0, 1.0])
    y_coords = np.array([-1.0, 0.0, 1.0])
    
    with open(test_filepath, 'w') as f:
        f.write("# Archivo de prueba generado por lector_datos.py\n")
        f.write("# Formato: x, y, Fx, Fy\n")
        
        for x in x_coords:
            for y in y_coords:
                # Simular una fuerza armónica simple: F = -k*r
                # Usamos k = 0.1 para este ejemplo
                fx = -0.1 * x
                fy = -0.1 * y
                f.write(f"{x},{y},{fx},{fy}\n")
    
    print(f"Archivo de prueba '{test_filepath}' creado.")

    # 2. Probar la función de carga
    fx_interp, fy_interp = cargar_mapa_fuerzas(test_filepath)

    if fx_interp:
        # 3. Probar la interpolación
        
        # Prueba en un punto exacto de la malla
        pos_exacta = (1.0, 0.0)
        fuerza_x_exacta = fx_interp(pos_exacta)
        fuerza_y_exacta = fy_interp(pos_exacta)
        print(f"Fuerza en punto exacto (1, 0):    Fx={fuerza_x_exacta:.2f}, Fy={fuerza_y_exacta:.2f}") # Esperado: -0.1, 0.0
        
        # Prueba en un punto interpolado (en medio de la malla)
        pos_interpolada = (0.5, 0.5)
        fuerza_x_interp = fx_interp(pos_interpolada)
        fuerza_y_interp = fy_interp(pos_interpolada)
        print(f"Fuerza en punto interpolado (0.5, 0.5): Fx={fuerza_x_interp:.2f}, Fy={fuerza_y_interp:.2f}") # Esperado: -0.05, -0.05
        
        # Prueba fuera de la malla (debería dar 0.0 por 'fill_value')
        pos_fuera = (10.0, 10.0)
        fuerza_x_fuera = fx_interp(pos_fuera)
        fuerza_y_fuera = fy_interp(pos_fuera)
        print(f"Fuerza en punto fuera (10, 10):  Fx={fuerza_x_fuera:.2f}, Fy={fuerza_y_fuera:.2f}") # Esperado: 0.0, 0.0

    # 4. Limpiar (borrar el archivo de prueba)
    os.remove(test_filepath)
    print(f"Archivo de prueba '{test_filepath}' eliminado.")