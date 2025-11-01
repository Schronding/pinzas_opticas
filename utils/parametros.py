# -*- coding: utf-8 -*-
"""
Este script contiene todos los parámetros físicos y de simulación
para el proyecto de pinzas ópticas.
"""

import numpy as np

# =============================================================================
# 1. CONSTANTES FÍSICAS FUNDAMENTALES
# =============================================================================
k_B = 1.380649e-23  # J/K (Constante de Boltzmann)

# =============================================================================
# 2. PARÁMETROS DEL ENTORNO Y PARTÍCULA (Datos del Experimento)
# =============================================================================
# Estos valores se obtuvieron del archivo de calibración:
# 'CaliF_nor_100X_3W_2,5_05'

# Temperatura absoluta del medio (300 K)
T = 300.0  # K

# Radio de la microesfera (1.25 µm)
r_particula = 1.25e-6  # m

# Viscosidad dinámica del medio (Agua a ~27 °C / 300 K)
# El archivo original usa 890.000000E-3 mPa*s = 0.89e-3 Pa·s
eta_agua = 0.89e-3  # Pa·s

# =============================================================================
# 3. PARÁMETROS DERIVADOS DEL SISTEMA
# =============================================================================
# Coeficiente de arrastre de Stokes (gamma)
gamma = 6 * np.pi * eta_agua * r_particula  # N·s/m

# =============================================================================
# 4. PARÁMETROS DE LA TRAMPA ÓPTICA (Modelo Armónico)
# =============================================================================
# Valores experimentales del archivo de calibración (kx EP, ky EP)

# Rigidez en la dirección X (N/m)
kappa_x = 52.707762e-9  # N/m

# Rigidez en la dirección Y (N/m)
kappa_y = 74.882902e-9  # N/m

# =============================================================================
# 5. PARÁMETROS DE LA SIMULACIÓN NUMÉRICA
# =============================================================================
# Paso de tiempo para la integración (en segundos)
dt = 1e-5  # s (0.01 milisegundos)

# Número total de pasos de la simulación
total_steps = 200000

# Tiempo total de la simulación (en segundos)
tiempo_total = dt * total_steps

# Imprimir un resumen para verificar
if __name__ == '__main__':
    print("="*40)
    print("PARÁMETROS DE SIMULACIÓN (BASADOS EN DATOS EXP.)")
    print("="*40)
    print(f"Temperatura (T):           {T} K")
    print(f"Radio de partícula (r):    {r_particula * 1e6} µm")
    print(f"Viscosidad (eta):          {eta_agua} Pa·s")
    print(f"Coef. de arrastre (gamma): {gamma:.2e} N·s/m")
    print(f"Rigidez en X (kappa_x):    {kappa_x:.2e} N/m")
    print(f"Rigidez en Y (kappa_y):    {kappa_y:.2e} N/m")
    print(f"Paso de tiempo (dt):         {dt * 1e6} µs")
    print(f"Duración total:            {tiempo_total} s")
    print("="*40)