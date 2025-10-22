import numpy as np
from utils import parametros as p

def run_simulation(total_steps, dt, k_x, k_y, gamma, k_B, T):
    """
    Ejecuta una simulación de movimiento Browniano en una trampa óptica 2D.

    Args:
        total_steps (int): Número total de pasos en la simulación.
        dt (float): Paso de tiempo (s).
        k_x (float): Rigidez de la trampa en la dirección X (N/m).
        k_y (float): Rigidez de la trampa en la dirección Y (N/m).
        gamma (float): Coeficiente de arrastre de Stokes (N·s/m).
        k_B (float): Constante de Boltzmann (J/K).
        T (float): Temperatura absoluta (K).

    Returns:
        np.ndarray: Un array de forma (total_steps, 2) con las posiciones [x, y]
                    de la partícula en cada paso de tiempo.
    """
    # 1. Inicialización
    # Creamos un array vacío para almacenar la trayectoria.
    # La forma es (número de pasos, 2) para las coordenadas [x, y].
    trajectory = np.zeros((total_steps, 2))
    
    # 2. Cálculo del término de la fuerza estocástica (ruido)
    # Según la teoría, la amplitud de la fuerza Browniana depende de la
    # temperatura y el arrastre. Pre-calculamos este factor para eficiencia.
    # D es el coeficiente de difusión: D = k_B * T / gamma
    # La desviación estándar del desplazamiento por ruido es sqrt(2 * D * dt)
    diffusion_coeff = k_B * T / gamma
    noise_magnitude = np.sqrt(2 * diffusion_coeff * dt)

    # 3. Bucle de Simulación (Método de Euler-Maruyama)
    for i in range(1, total_steps):
        # Posición en el paso anterior
        x_prev, y_prev = trajectory[i-1]

        # a) Cálculo de la fuerza determinística (de la trampa óptica)
        # F = -κ * r  (Ley de Hooke)
        force_x = -k_x * x_prev
        force_y = -k_y * y_prev

        # b) Generación de la fuerza estocástica (Browniana)
        # Generamos dos números aleatorios de una distribución normal (media 0, desviación 1)
        random_force_x = noise_magnitude * np.random.randn()
        random_force_y = noise_magnitude * np.random.randn()

        # c) Actualización de la posición (Ecuación de Langevin sobreamortiguada)
        # dx = (F_trampa / gamma) * dt + dW
        # donde dW es el término de ruido que ya calculamos.
        trajectory[i, 0] = x_prev + (force_x / gamma) * dt + random_force_x
        trajectory[i, 1] = y_prev + (force_y / gamma) * dt + random_force_y
        
    return trajectory

# =============================================================================
# BLOQUE DE PRUEBA
# =============================================================================
# Este bloque solo se ejecuta cuando corres este script directamente.
# Es muy útil para probar que nuestra función `run_simulation` funciona
# correctamente antes de integrarla con la interfaz gráfica.
if __name__ == '__main__':
    print("Ejecutando prueba del simulador...")
    
    # Ejecutamos la simulación con los parámetros importados
    trayectoria_calculada = run_simulation(
        total_steps=p.total_steps,
        dt=p.dt,
        k_x=p.kappa_x,
        k_y=p.kappa_y,
        gamma=p.gamma,
        k_B=p.k_B,
        T=p.T
    )

    print(f"Simulación completada. Se generaron {len(trayectoria_calculada)} puntos.")
    print("Forma del array de la trayectoria:", trayectoria_calculada.shape)
    
    # Imprimimos los primeros 5 puntos para verificar
    print("Primeros 5 puntos de la trayectoria [x, y]:")
    print(trayectoria_calculada[:5])

    # Opcional: Visualización rápida con matplotlib
    # Si tienes matplotlib instalado, puedes descomentar las siguientes líneas
    # para ver una gráfica de la trayectoria generada.
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 8))
    plt.plot(trayectoria_calculada[:, 0] * 1e9, trayectoria_calculada[:, 1] * 1e9)
    plt.title("Trayectoria Simulada (Primeros puntos)")
    plt.xlabel("Posición X (nm)")
    plt.ylabel("Posición Y (nm)")
    plt.grid(True)
    plt.axis('equal')
    plt.show()