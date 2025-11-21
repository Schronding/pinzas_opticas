"""
Animación de la Función de Transferencia (PSD).
Muestra cómo el ruido térmico converge a la curva Lorentziana teórica
conforme pasa el tiempo.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import signal
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from calculos import simulador
from utils import parametros as p

def lorentzian(f, fc, D):
    return D / (np.pi**2 * (fc**2 + f**2))

def animar_transferencia():
    print("Calculando simulación larga para ver convergencia...")
    
    # K de 1.98 pN/um
    k_target = 1.98 * 1e-6  
    
    TOTAL_STEPS = 100000
    fs = 1 / p.dt  # Frecuencia de muestreo (~20 kHz)
    
    traj = simulador.run_simulation(
        total_steps=TOTAL_STEPS, dt=p.dt, gamma=p.gamma, k_B=p.k_B, T=p.T,
        k_x=k_target, k_y=k_target
    )
    
    x_nm = traj[:, 0] * 1e9 # Nanometros para cálculos
    
    # Teoría (La Ecuación de Transferencia / Lorentziana)
    fc_teorica = k_target / (2 * np.pi * p.gamma)
    D_teorica = p.k_B * p.T / p.gamma # relacion de Einstein
    
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(f"Convergencia a la Función de Transferencia\n(Frecuencia de Corte Teórica = {fc_teorica:.1f} Hz)", fontsize=14)
    
    ax.set_xlabel("Frecuencia (Hz)")
    ax.set_ylabel("Densidad Espectral de Potencia (PSD)")
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(10, fs/2)
    ax.set_ylim(1e-6, 100) # Ajustar según la escala de tu señal (nm^2/Hz)

    f_theory = np.logspace(1, np.log10(fs/2), 500)
    psd_theory = lorentzian(f_theory, fc_teorica, D_teorica * 1e18) # 1e18 para nm^2
    ax.plot(f_theory, psd_theory, 'r--', linewidth=3, label='Teoría (Lorentziana)')
    
    line_psd, = ax.plot([], [], 'b-', alpha=0.6, linewidth=1, label='Señal Acumulada')
    
    txt_time = ax.text(0.05, 0.1, '', transform=ax.transAxes, fontsize=12, 
                       bbox=dict(facecolor='white', alpha=0.8))

    ax.legend(loc='upper right')

    def update(frame):
        # Empezamos con 2000 puntos y añadimos 1000 en cada frame
        window_size = 2000 + frame * 1000
        
        if window_size > len(x_nm): return line_psd, txt_time
        
        fragmento = x_nm[:window_size]
        
        # Calcular PSD del fragmento actual
        f, Pxx = signal.welch(fragmento, fs, nperseg=min(window_size//2, 2048))
        
        line_psd.set_data(f, Pxx)
        txt_time.set_text(f"Datos analizados: {window_size} puntos\n({window_size*p.dt:.2f} s)")
        
        return line_psd, txt_time

    ani = animation.FuncAnimation(fig, update, frames=80, interval=100, blit=True)
    plt.show()

if __name__ == "__main__":
    animar_transferencia()