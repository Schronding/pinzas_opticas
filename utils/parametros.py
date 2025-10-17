import numpy as np


# Constante de Boltzmann (en Joules / Kelvin)
k_B = 1.380649e-23  # J/K

# Temperatura absoluta del medio (en Kelvin)
T = 298.15  # K

# Radio de la microesfera (en metros)
r_particula = 1.0e-6  # m

# Viscosidad dinámica del medio (Agua a 25 °C en Pascal-segundo)
eta_agua = 0.89e-3  # Pa·s


# Coeficiente de arrastre de Stokes (gamma)
# Formula: 6 * pi * eta * r
gamma = 6 * np.pi * eta_agua * r_particula  # N·s/m

# Rigidez en la dirección X (en N/m)
kappa_x = 1e-7  # N/m

# Rigidez en la dirección Y (en N/m)
kappa_y = 1e-7  # N/m

# Debe ser mucho menor que el tiempo característico de relajación (gamma/kappa).
dt = 1e-5  # s (0.01 milisegundos)

# Número total de pasos de la simulación
total_steps = 200000

# Tiempo total de la simulación (en segundos)
tiempo_total = dt * total_steps

# Imprimir un resumen para verificar
if __name__ == '__main__':
    print("="*45)
    print("PARÁMETROS DE LA SIMULACIÓN DE PINZAS ÓPTICAS")
    print("="*45)
    print(f"Temperatura (T):           {T} K")
    print(f"Radio de partícula (r):    {r_particula * 1e6} µm")
    print(f"Viscosidad del agua (eta): {eta_agua} Pa·s")
    print(f"Coef. de arrastre (gamma): {gamma:.2e} N·s/m")
    print(f"Rigidez en X (kappa_x):    {kappa_x:.2e} N/m")
    print(f"Paso de tiempo (dt):         {dt * 1e6} µs")
    print(f"Duración total:            {tiempo_total} s")
    print("="*43)